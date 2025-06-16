#!/usr/bin/env python3
"""
개선된 학습 스크립트 - Isaac Lab RSL-RL 기법 적용
RTX 4080 GPU 최적화 포함
"""

import numpy as np
import torch
import time
import os
import argparse
from datetime import datetime
from environments.vectorized.improved_vectorized import ImprovedVectorEnv
from agents.ppo_agent import PPOAgent
import wandb


class ImprovedTrainer:
    """개선된 GO2 학습 트레이너"""
    
    def __init__(self, args):
        self.args = args
        
        # GPU 최적화 설정
        self._setup_gpu_optimization()
        
        # 개선된 벡터화 환경 생성
        self.vec_env = ImprovedVectorEnv(
            num_envs=args.num_envs,
            render_mode="human" if args.render else None,
            use_reference_gait=not args.no_reference_gait
        )
        
        # PPO 에이전트 (개선된 환경에 맞춤)
        self.agent = PPOAgent(
            obs_dim=self.vec_env.observation_space.shape[0],
            action_dim=self.vec_env.action_space.shape[0],
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            hidden_dim=args.hidden_dim
        )
        
        # 혼합 정밀도 훈련
        if args.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            print("🔥 혼합 정밀도 훈련 활성화")
        else:
            self.scaler = None
        
        # 로깅 초기화
        if args.wandb:
            wandb.init(
                project="go2-improved-locomotion",
                config=vars(args),
                name=f"improved_{args.num_envs}envs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # 통계 변수
        self.episode_rewards = [[] for _ in range(args.num_envs)]
        self.episode_lengths = [[] for _ in range(args.num_envs)]
        self.episode_counts = np.zeros(args.num_envs, dtype=int)
        self.best_reward = -np.inf
        
        print(f"🚀 개선된 학습 준비 완료")
        print(f"환경 수: {args.num_envs}")
        print(f"디바이스: {self.agent.device}")
        print(f"제어 모드: 관절 위치 제어")
        print(f"초기화: 참조 보행 자세")
    
    def _setup_gpu_optimization(self):
        """GPU 최적화 설정"""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # RTX 4080 메모리 최적화
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,expandable_segments:True'
            
            print(f"🔧 GPU 최적화 설정 완료")
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def get_vectorized_actions(self, observations):
        """벡터화된 행동 생성"""
        obs_tensor = torch.FloatTensor(observations).to(self.agent.device)
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=self.args.mixed_precision):
                actions, log_probs, values = self.agent.policy.get_action(obs_tensor)
        
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()
    
    def collect_rollout(self):
        """롤아웃 수집"""
        observations, infos = self.vec_env.reset(seed=self.args.seed)
        current_episode_rewards = np.zeros(self.args.num_envs)
        current_episode_lengths = np.zeros(self.args.num_envs, dtype=int)
        
        rollout_start_time = time.time()
        steps_collected = 0
        target_steps = self.args.rollout_length
        
        # 보상 구성 요소 추적
        reward_components = {}
        
        while steps_collected < target_steps:
            # 행동 생성
            actions, log_probs, values = self.get_vectorized_actions(observations)
            
            # 행동 클리핑 (안전성)
            actions = np.clip(actions, -1.0, 1.0)
            
            # 환경 스텝
            next_observations, rewards, terminated, truncated, infos = self.vec_env.step(actions)
            dones = terminated | truncated
            
            # 보상 안정성 검사
            rewards = np.clip(rewards, -100.0, 100.0)
            rewards = np.where(np.isnan(rewards) | np.isinf(rewards), -10.0, rewards)
            
            # 에이전트에 전이 저장
            for env_idx in range(self.args.num_envs):
                self.agent.store_transition(
                    observations[env_idx], actions[env_idx], rewards[env_idx],
                    values[env_idx], log_probs[env_idx], dones[env_idx]
                )
            
            # 보상 구성 요소 누적
            for env_idx, info in enumerate(infos):
                if isinstance(info, dict):
                    for key, value in info.items():
                        if key not in reward_components:
                            reward_components[key] = []
                        reward_components[key].append(value)
            
            # 상태 업데이트
            observations = next_observations
            current_episode_rewards += rewards
            current_episode_lengths += 1
            steps_collected += self.args.num_envs
            
            # 에피소드 완료 처리 (자동 리셋 없이)
            reset_envs = []
            for env_idx in range(self.args.num_envs):
                if dones[env_idx]:
                    self.episode_rewards[env_idx].append(current_episode_rewards[env_idx])
                    self.episode_lengths[env_idx].append(current_episode_lengths[env_idx])
                    self.episode_counts[env_idx] += 1
                    
                    current_episode_rewards[env_idx] = 0
                    current_episode_lengths[env_idx] = 0
                    reset_envs.append(env_idx)
                    
                    print(f"환경 {env_idx} 에피소드 종료: 보상 {current_episode_rewards[env_idx]:.2f}, 길이 {current_episode_lengths[env_idx]}")
            
            # 필요시만 리셋 (벡터화 환경에서 자동 처리됨)
            if len(reset_envs) > 0:
                print(f"{len(reset_envs)}개 환경이 자동 리셋되었습니다.")
            
            # 렌더링
            if self.args.render and steps_collected % (self.args.num_envs * 10) == 0:
                self.vec_env.render()
        
        rollout_time = time.time() - rollout_start_time
        
        # 평균 보상 구성 요소 계산
        avg_reward_components = {}
        for key, values in reward_components.items():
            avg_reward_components[key] = np.mean(values)
        
        return steps_collected, rollout_time, avg_reward_components
    
    def update_policy(self):
        """정책 업데이트"""
        update_start_time = time.time()
        
        losses = self.agent.update_policy(
            n_epochs=self.args.ppo_epochs,
            batch_size=self.args.batch_size
        )
        
        update_time = time.time() - update_start_time
        
        return losses, update_time
    
    def train(self):
        """메인 학습 루프"""
        start_time = time.time()
        total_timesteps = 0
        
        print(f"🎯 개선된 학습 시작 (목표: {self.args.total_timesteps:,} 스텝)")
        
        while total_timesteps < self.args.total_timesteps:
            # 롤아웃 수집
            steps_collected, rollout_time, reward_components = self.collect_rollout()
            total_timesteps += steps_collected
            
            # 정책 업데이트
            losses, update_time = self.update_policy()
            
            # 통계 로깅
            self.log_progress(total_timesteps, start_time, rollout_time, update_time, 
                            losses, reward_components)
            
            # 모델 저장
            self.save_models(total_timesteps)
        
        # 최종 정리
        self.cleanup()
        
        total_time = time.time() - start_time
        print(f"✅ 개선된 학습 완료!")
        print(f"총 시간: {total_time/3600:.1f}시간")
        print(f"최고 성능: {self.best_reward:.2f}")
    
    def log_progress(self, total_timesteps, start_time, rollout_time, update_time, 
                    losses, reward_components):
        """진행상황 로깅"""
        # 통계 계산
        all_rewards = []
        all_lengths = []
        
        for env_rewards in self.episode_rewards:
            all_rewards.extend(env_rewards[-10:])  # 최근 10개 에피소드
        for env_lengths in self.episode_lengths:
            all_lengths.extend(env_lengths[-10:])
        
        if all_rewards:
            avg_reward = np.mean(all_rewards)
            avg_length = np.mean(all_lengths)
            total_episodes = np.sum(self.episode_counts)
            
            elapsed_time = time.time() - start_time
            fps = total_timesteps / elapsed_time
            
            print(f"스텝 {total_timesteps:8,} | 에피소드 {total_episodes:5,} | "
                  f"평균 보상: {avg_reward:7.2f} | 평균 길이: {avg_length:5.1f} | "
                  f"FPS: {fps:5.0f}")
            
            # 보상 구성 요소 출력
            if reward_components:
                print("  보상 구성: ", end="")
                for key, value in sorted(reward_components.items())[:5]:  # 상위 5개만
                    print(f"{key}: {value:.3f}, ", end="")
                print()
            
            # Weights & Biases 로깅
            if self.args.wandb:
                log_dict = {
                    "timestep": total_timesteps,
                    "total_episodes": total_episodes,
                    "avg_reward": avg_reward,
                    "avg_length": avg_length,
                    "fps": fps,
                    "rollout_time": rollout_time,
                    "update_time": update_time,
                }
                
                # 보상 구성 요소 로깅
                for key, value in reward_components.items():
                    log_dict[f"reward/{key}"] = value
                
                if losses:
                    log_dict.update(losses)
                    
                wandb.log(log_dict)
            
            # 최고 성능 모델 저장
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                if not os.path.exists("models"):
                    os.makedirs("models")
                self.agent.save(f"models/best_improved_go2_ppo.pth")
                print(f"🏆 새로운 최고 성능! 평균 보상: {self.best_reward:.2f}")
    
    def save_models(self, total_timesteps):
        """주기적 모델 저장"""
        if total_timesteps % self.args.save_freq < self.args.num_envs:
            if not os.path.exists("models"):
                os.makedirs("models")
            self.agent.save(f"models/improved_go2_ppo_step_{total_timesteps}.pth")
            print(f"💾 체크포인트 저장: {total_timesteps:,} 스텝")
    
    def cleanup(self):
        """정리 작업"""
        # 최종 모델 저장
        if not os.path.exists("models"):
            os.makedirs("models")
        self.agent.save("models/improved_go2_ppo_final.pth")
        
        # 환경 종료
        self.vec_env.close()
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        # Weights & Biases 종료
        if self.args.wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='개선된 GO2 학습 (Isaac Lab RSL-RL 기법)')
    
    # 환경 설정
    parser.add_argument('--num_envs', type=int, default=64,
                        help='병렬 환경 수')
    parser.add_argument('--no_reference_gait', action='store_true',
                        help='참조 gait 비활성화')
    parser.add_argument('--render', action='store_true',
                        help='첫 번째 환경 렌더링')
    
    # 학습 설정
    parser.add_argument('--total_timesteps', type=int, default=10000000,
                        help='총 학습 스텝')
    parser.add_argument('--rollout_length', type=int, default=8192,
                        help='롤아웃 길이')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='배치 크기')
    
    # PPO 하이퍼파라미터
    parser.add_argument('--lr', type=float, default=3e-4, help='학습률')
    parser.add_argument('--gamma', type=float, default=0.99, help='할인 인수')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO 클립 비율')
    parser.add_argument('--ppo_epochs', type=int, default=5, help='PPO 업데이트 에포크')
    parser.add_argument('--hidden_dim', type=int, default=512, help='네트워크 히든 레이어 크기')
    
    # GPU 최적화
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='혼합 정밀도 훈련')
    
    # 로깅 및 저장
    parser.add_argument('--wandb', action='store_true', help='Weights & Biases 로깅')
    parser.add_argument('--save_freq', type=int, default=100000,
                        help='체크포인트 저장 주기')
    
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
    
    print(f"🚀 개선된 GO2 학습 시작!")
    print(f"주요 개선사항:")
    print(f"  ✅ 관절 위치 제어 (안정적)")
    print(f"  ✅ 참조 보행 자세 초기화")
    print(f"  ✅ 액션 스무싱")
    print(f"  ✅ 모듈화된 보상 함수")
    print(f"  ✅ 발 접촉 이력 추적")
    
    # 트레이너 생성 및 학습 시작
    trainer = ImprovedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()