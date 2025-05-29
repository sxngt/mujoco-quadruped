#!/usr/bin/env python3
"""
RTX 4080 GPU 최대 활용 학습 스크립트
GPU 사용률 80% 목표로 최적화된 설정
"""

import numpy as np
import torch
import time
import os
import argparse
from datetime import datetime
from simple_vectorized import SimpleVectorEnv
from ppo_agent import PPOAgent
import wandb


class GPUMaxTrainer:
    """RTX 4080 최대 활용 트레이너"""
    
    def __init__(self, args):
        self.args = args
        
        # GPU 최적화 설정
        self._setup_gpu_optimization()
        
        # 대용량 벡터화 환경 생성
        self.vec_env = SimpleVectorEnv(
            num_envs=args.num_envs,
            render_mode="human" if args.render else None,
            use_reference_gait=not args.no_reference_gait
        )
        
        # GPU 최적화된 PPO 에이전트
        self.agent = PPOAgent(
            obs_dim=self.vec_env.observation_space.shape[0],
            action_dim=self.vec_env.action_space.shape[0],
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            hidden_dim=args.hidden_dim  # 더 큰 네트워크
        )
        
        # 혼합 정밀도 훈련 설정
        if args.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            print("🔥 혼합 정밀도 훈련 활성화")
        else:
            self.scaler = None
        
        # 로깅 초기화
        if args.wandb:
            wandb.init(
                project="go2-gpu-max-locomotion",
                config=vars(args),
                name=f"gpu_max_{args.num_envs}envs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # 통계 변수
        self.episode_rewards = [[] for _ in range(args.num_envs)]
        self.episode_lengths = [[] for _ in range(args.num_envs)]
        self.episode_counts = np.zeros(args.num_envs, dtype=int)
        self.best_reward = -np.inf
        
        # 성능 모니터링
        self.gpu_memory_peak = 0
        self.fps_history = []
        
        print(f"🚀 GPU 최대 활용 학습 준비 완료")
        print(f"환경 수: {args.num_envs}")
        print(f"디바이스: {self.agent.device}")
        print(f"네트워크 크기: {args.hidden_dim}")
        print(f"배치 크기: {args.batch_size}")
        print(f"롤아웃 길이: {args.rollout_length}")
        
        # GPU 메모리 사전 할당
        self._warmup_gpu()
    
    def _setup_gpu_optimization(self):
        """GPU 최적화 설정"""
        if torch.cuda.is_available():
            # CUDA 최적화 설정
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # RTX 4080 메모리 관리 최적화 (16GB)
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,expandable_segments:True,garbage_collection_threshold:0.8'
            
            print(f"🔧 GPU 최적화 설정 완료")
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"사용 가능한 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def _warmup_gpu(self):
        """GPU 메모리 사전 할당 및 워밍업"""
        print("🔥 GPU 워밍업 시작...")
        
        # 더미 데이터로 GPU 워밍업
        dummy_obs = torch.randn(
            self.args.batch_size, 
            self.vec_env.observation_space.shape[0], 
            device=self.agent.device
        )
        dummy_actions = torch.randn(
            self.args.batch_size, 
            self.vec_env.action_space.shape[0], 
            device=self.agent.device
        )
        
        # 몇 번의 forward/backward pass로 워밍업
        for _ in range(10):
            with torch.autocast(device_type='cuda', enabled=self.args.mixed_precision):
                _ = self.agent.policy(dummy_obs)
            
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated() / 1e9
        print(f"💾 초기 GPU 메모리 사용량: {initial_memory:.2f}GB")
    
    def get_vectorized_actions_optimized(self, observations):
        """GPU 최적화된 벡터화 행동 생성"""
        obs_tensor = torch.FloatTensor(observations).to(self.agent.device, non_blocking=True)
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=self.args.mixed_precision):
                actions, log_probs, values = self.agent.policy.get_action(obs_tensor)
        
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()
    
    def collect_rollout_optimized(self):
        """GPU 최적화된 롤아웃 수집"""
        observations, infos = self.vec_env.reset(seed=self.args.seed)
        current_episode_rewards = np.zeros(self.args.num_envs)
        current_episode_lengths = np.zeros(self.args.num_envs, dtype=int)
        
        rollout_start_time = time.time()
        steps_collected = 0
        target_steps = self.args.rollout_length
        
        # 배치 단위로 데이터 수집
        obs_batch = []
        action_batch = []
        reward_batch = []
        value_batch = []
        logprob_batch = []
        done_batch = []
        
        while steps_collected < target_steps:
            # 행동 생성 (GPU 가속)
            actions, log_probs, values = self.get_vectorized_actions_optimized(observations)
            
            # 환경 스텝
            next_observations, rewards, terminated, truncated, infos = self.vec_env.step(actions)
            dones = terminated | truncated
            
            # 배치에 추가
            obs_batch.append(observations.copy())
            action_batch.append(actions.copy())
            reward_batch.append(rewards.copy())
            value_batch.append(values.copy())
            logprob_batch.append(log_probs.copy())
            done_batch.append(dones.copy())
            
            # 상태 업데이트
            observations = next_observations
            current_episode_rewards += rewards
            current_episode_lengths += 1
            steps_collected += self.args.num_envs
            
            # 에피소드 완료 처리
            for env_idx in range(self.args.num_envs):
                if dones[env_idx]:
                    self.episode_rewards[env_idx].append(current_episode_rewards[env_idx])
                    self.episode_lengths[env_idx].append(current_episode_lengths[env_idx])
                    self.episode_counts[env_idx] += 1
                    
                    current_episode_rewards[env_idx] = 0
                    current_episode_lengths[env_idx] = 0
            
            # 렌더링
            if self.args.render and steps_collected % (self.args.num_envs * 4) == 0:
                self.vec_env.render()
        
        # 배치 데이터를 에이전트에 저장
        for i in range(len(obs_batch)):
            for j in range(self.args.num_envs):
                self.agent.store_transition(
                    obs_batch[i][j], action_batch[i][j], reward_batch[i][j],
                    value_batch[i][j], logprob_batch[i][j], done_batch[i][j]
                )
        
        rollout_time = time.time() - rollout_start_time
        return steps_collected, rollout_time
    
    def update_policy_optimized(self):
        """GPU 최적화된 정책 업데이트"""
        update_start_time = time.time()
        
        # 더 큰 배치로 업데이트
        losses = self.agent.update_policy(
            n_epochs=self.args.ppo_epochs,
            batch_size=self.args.batch_size
        )
        
        update_time = time.time() - update_start_time
        
        # GPU 메모리 모니터링
        current_memory = torch.cuda.memory_allocated() / 1e9
        self.gpu_memory_peak = max(self.gpu_memory_peak, current_memory)
        
        return losses, update_time
    
    def train(self):
        """메인 학습 루프"""
        start_time = time.time()
        total_timesteps = 0
        
        print(f"🎯 GPU 최대 활용 학습 시작 (목표: {self.args.total_timesteps:,} 스텝)")
        
        while total_timesteps < self.args.total_timesteps:
            # 롤아웃 수집
            steps_collected, rollout_time = self.collect_rollout_optimized()
            total_timesteps += steps_collected
            
            # 정책 업데이트  
            losses, update_time = self.update_policy_optimized()
            
            # 성능 모니터링
            fps = steps_collected / (rollout_time + update_time)
            self.fps_history.append(fps)
            
            # 통계 로깅
            self.log_progress_optimized(total_timesteps, start_time, rollout_time, update_time, fps, losses)
            
            # 모델 저장
            self.save_models(total_timesteps)
            
            # GPU 메모리 정리 (주기적)
            if total_timesteps % (self.args.save_freq * 2) < self.args.num_envs:
                torch.cuda.empty_cache()
        
        # 최종 정리
        self.cleanup()
        
        total_time = time.time() - start_time
        avg_fps = np.mean(self.fps_history[-100:]) if self.fps_history else 0
        
        print(f"✅ GPU 최대 활용 학습 완료!")
        print(f"총 시간: {total_time/3600:.1f}시간")
        print(f"평균 FPS (최근 100회): {avg_fps:.0f}")
        print(f"최대 GPU 메모리 사용량: {self.gpu_memory_peak:.2f}GB")
        print(f"최고 성능: {self.best_reward:.2f}")
    
    def log_progress_optimized(self, total_timesteps, start_time, rollout_time, update_time, fps, losses):
        """최적화된 진행상황 로깅"""
        # 통계 계산
        all_rewards = []
        all_lengths = []
        
        for env_rewards in self.episode_rewards:
            all_rewards.extend(env_rewards[-5:])  # 최근 5개 에피소드
        for env_lengths in self.episode_lengths:
            all_lengths.extend(env_lengths[-5:])
        
        if all_rewards:
            avg_reward = np.mean(all_rewards)
            avg_length = np.mean(all_lengths)
            total_episodes = np.sum(self.episode_counts)
            
            elapsed_time = time.time() - start_time
            overall_fps = total_timesteps / elapsed_time
            
            # GPU 활용률 추정 (간접적)
            gpu_memory_gb = torch.cuda.memory_allocated() / 1e9
            estimated_gpu_util = min(95, fps / 50 * 100)  # 추정치
            
            print(f"스텝 {total_timesteps:8,} | 에피소드 {total_episodes:5,} | "
                  f"평균 보상: {avg_reward:7.2f} | 평균 길이: {avg_length:5.1f} | "
                  f"FPS: {fps:5.0f} | GPU: {estimated_gpu_util:3.0f}% | "
                  f"메모리: {gpu_memory_gb:.1f}GB | "
                  f"롤아웃: {rollout_time:.2f}s | 업데이트: {update_time:.2f}s")
            
            # Weights & Biases 로깅
            if self.args.wandb:
                log_dict = {
                    "timestep": total_timesteps,
                    "total_episodes": total_episodes,
                    "avg_reward": avg_reward,
                    "avg_length": avg_length,
                    "fps": fps,
                    "overall_fps": overall_fps,
                    "estimated_gpu_util": estimated_gpu_util,
                    "gpu_memory_gb": gpu_memory_gb,
                    "rollout_time": rollout_time,
                    "update_time": update_time,
                    "num_envs": self.args.num_envs,
                }
                if losses:
                    log_dict.update(losses)
                wandb.log(log_dict)
            
            # 최고 성능 모델 저장
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                if not os.path.exists("models"):
                    os.makedirs("models")
                self.agent.save(f"models/best_gpu_max_go2_ppo.pth")
                print(f"🏆 새로운 최고 성능! 평균 보상: {self.best_reward:.2f}")
    
    def save_models(self, total_timesteps):
        """주기적 모델 저장"""
        if total_timesteps % self.args.save_freq < self.args.num_envs:
            if not os.path.exists("models"):
                os.makedirs("models")
            self.agent.save(f"models/gpu_max_go2_ppo_step_{total_timesteps}.pth")
            print(f"💾 체크포인트 저장: {total_timesteps:,} 스텝")
    
    def cleanup(self):
        """정리 작업"""
        # 최종 모델 저장
        if not os.path.exists("models"):
            os.makedirs("models")
        self.agent.save("models/gpu_max_go2_ppo_final.pth")
        
        # 환경 종료
        self.vec_env.close()
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        # Weights & Biases 종료
        if self.args.wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='RTX 4080 GPU 최대 활용 학습')
    
    # RTX 4080 GPU 최적화 설정 (16GB VRAM)
    parser.add_argument('--num_envs', type=int, default=96, 
                        help='병렬 환경 수 (RTX 4080 16GB 기준 96개 최적화)')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='혼합 정밀도 훈련 (메모리 절약 + 속도 향상)')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='네트워크 히든 레이어 크기 (RTX 4080 GPU 활용도 최대화)')
    
    # 대용량 학습 설정
    parser.add_argument('--total_timesteps', type=int, default=20000000,
                        help='총 학습 스텝 (2천만)')
    parser.add_argument('--rollout_length', type=int, default=24576,
                        help='롤아웃 길이 (RTX 4080 16GB 대용량 최적화)')
    parser.add_argument('--batch_size', type=int, default=3072,
                        help='배치 크기 (RTX 4080 16GB 메모리 최대 활용)')
    
    # PPO 하이퍼파라미터
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--gamma', type=float, default=0.99, help='할인 인수')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO 클립 비율')
    parser.add_argument('--ppo_epochs', type=int, default=20, help='PPO 업데이트 에포크 (많이)')
    
    # 환경 설정
    parser.add_argument('--no_reference_gait', action='store_true', default=True,
                        help='참조 gait 비활성화 (기본)')
    parser.add_argument('--render', action='store_true',
                        help='첫 번째 환경 렌더링 (성능 저하)')
    
    # 로깅 및 저장
    parser.add_argument('--wandb', action='store_true', help='Weights & Biases 로깅')
    parser.add_argument('--save_freq', type=int, default=500000,
                        help='체크포인트 저장 주기 (스텝 단위)')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    args = parser.parse_args()
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다. GPU 설정을 확인하세요.")
        return
    
    # 시드 설정
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"🚀 RTX 4080 GPU 최대 활용 학습 시작!")
    print(f"설정: {args.num_envs}개 환경, {args.total_timesteps:,} 스텝 목표")
    print(f"네트워크 크기: {args.hidden_dim}, 배치 크기: {args.batch_size}")
    print(f"혼합 정밀도: {'활성화' if args.mixed_precision else '비활성화'}")
    
    # 트레이너 생성 및 학습 시작
    trainer = GPUMaxTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()