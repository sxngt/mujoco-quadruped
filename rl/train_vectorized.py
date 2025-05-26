#!/usr/bin/env python3
"""
벡터화 환경을 사용한 고속 PPO 학습
16개 환경 병렬 실행으로 학습 효율성 극대화
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


class VectorizedTrainer:
    """벡터화 환경 전용 PPO 트레이너"""
    
    def __init__(self, args):
        self.args = args
        
        # 벡터화 환경 생성
        self.vec_env = SimpleVectorEnv(
            num_envs=args.num_envs,
            render_mode="human" if args.render else None,
            use_reference_gait=not args.no_reference_gait
        )
        
        # PPO 에이전트 생성
        self.agent = PPOAgent(
            obs_dim=self.vec_env.observation_space.shape[0],
            action_dim=self.vec_env.action_space.shape[0],
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio
        )
        
        # 로깅 초기화
        if args.wandb:
            wandb.init(
                project="go2-vectorized-locomotion",
                config=vars(args),
                name=f"vec_ppo_{args.num_envs}envs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # 통계 변수
        self.episode_rewards = [[] for _ in range(args.num_envs)]
        self.episode_lengths = [[] for _ in range(args.num_envs)]
        self.episode_counts = np.zeros(args.num_envs, dtype=int)
        self.best_reward = -np.inf
        
        print(f"🚀 벡터화 학습 준비 완료")
        print(f"환경 수: {args.num_envs}")
        print(f"디바이스: {self.agent.device}")
        print(f"롤아웃 길이: {args.rollout_length}")
        print(f"배치 크기: {args.batch_size}")
    
    def get_vectorized_actions(self, observations):
        """
        벡터화된 관찰에 대해 행동 생성
        
        Args:
            observations: (num_envs, obs_dim) 관찰 배열
            
        Returns:
            actions: (num_envs, action_dim) 행동 배열
            log_probs: (num_envs,) 로그 확률 배열  
            values: (num_envs,) 가치 추정 배열
        """
        obs_tensor = torch.FloatTensor(observations).to(self.agent.device)
        
        with torch.no_grad():
            # PPOAgent의 PolicyNetwork 사용
            actions, log_probs, values = self.agent.policy.get_action(obs_tensor)
        
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()
    
    def collect_rollout(self):
        """롤아웃 데이터 수집"""
        observations, infos = self.vec_env.reset(seed=self.args.seed)
        current_episode_rewards = np.zeros(self.args.num_envs)
        current_episode_lengths = np.zeros(self.args.num_envs, dtype=int)
        
        steps_collected = 0
        target_steps = self.args.rollout_length
        
        while steps_collected < target_steps:
            # 행동 생성
            actions, log_probs, values = self.get_vectorized_actions(observations)
            
            # 환경 스텝
            next_observations, rewards, terminated, truncated, infos = self.vec_env.step(actions)
            dones = terminated | truncated
            
            # 데이터 저장 (각 환경별로)
            for i in range(self.args.num_envs):
                self.agent.store_transition(
                    observations[i], actions[i], rewards[i], 
                    values[i], log_probs[i], dones[i]
                )
            
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
            if self.args.render:
                self.vec_env.render()
        
        return steps_collected
    
    def train(self):
        """메인 학습 루프"""
        start_time = time.time()
        total_timesteps = 0
        
        print(f"🎯 학습 시작 (목표: {self.args.total_timesteps:,} 스텝)")
        
        while total_timesteps < self.args.total_timesteps:
            # 롤아웃 수집
            rollout_start = time.time()
            steps_collected = self.collect_rollout()
            rollout_time = time.time() - rollout_start
            
            total_timesteps += steps_collected
            
            # 정책 업데이트
            update_start = time.time()
            losses = self.agent.update_policy(
                n_epochs=self.args.ppo_epochs,
                batch_size=self.args.batch_size
            )
            update_time = time.time() - update_start
            
            # 통계 계산
            self.log_progress(total_timesteps, start_time, rollout_time, update_time, losses)
            
            # 모델 저장
            self.save_models(total_timesteps)
        
        # 최종 정리
        self.cleanup()
        
        total_time = time.time() - start_time
        final_fps = total_timesteps / total_time
        
        print(f"✅ 학습 완료!")
        print(f"총 시간: {total_time/3600:.1f}시간")
        print(f"평균 FPS: {final_fps:.0f}")
        print(f"최고 성능: {self.best_reward:.2f}")
    
    def log_progress(self, total_timesteps, start_time, rollout_time, update_time, losses):
        """학습 진행상황 로깅"""
        # 최근 에피소드들의 통계
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
            
            print(f"스텝 {total_timesteps:7,} | 에피소드 {total_episodes:4,} | "
                  f"평균 보상: {avg_reward:7.2f} | 평균 길이: {avg_length:5.1f} | "
                  f"FPS: {fps:5.0f} | 롤아웃: {rollout_time:.2f}s | 업데이트: {update_time:.2f}s")
            
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
                self.agent.save(f"models/best_vectorized_go2_ppo.pth")
                print(f"🏆 새로운 최고 성능! 평균 보상: {self.best_reward:.2f}")
    
    def save_models(self, total_timesteps):
        """주기적 모델 저장"""
        if total_timesteps % self.args.save_freq < self.args.num_envs:
            if not os.path.exists("models"):
                os.makedirs("models")
            self.agent.save(f"models/vectorized_go2_ppo_step_{total_timesteps}.pth")
            print(f"💾 체크포인트 저장: {total_timesteps:,} 스텝")
    
    def cleanup(self):
        """정리 작업"""
        # 최종 모델 저장
        if not os.path.exists("models"):
            os.makedirs("models")
        self.agent.save("models/vectorized_go2_ppo_final.pth")
        
        # 환경 종료
        self.vec_env.close()
        
        # Weights & Biases 종료
        if self.args.wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='벡터화 PPO 학습')
    
    # 벡터화 환경 설정
    parser.add_argument('--num_envs', type=int, default=16, 
                        help='병렬 환경 수 (기본: 16)')
    
    # 학습 설정
    parser.add_argument('--total_timesteps', type=int, default=2000000,
                        help='총 학습 스텝')
    parser.add_argument('--rollout_length', type=int, default=4096,
                        help='롤아웃 길이')
    
    # PPO 하이퍼파라미터
    parser.add_argument('--lr', type=float, default=3e-4, help='학습률')
    parser.add_argument('--gamma', type=float, default=0.99, help='할인 인수')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO 클립 비율')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='PPO 업데이트 에포크')
    parser.add_argument('--batch_size', type=int, default=256, help='배치 크기')
    
    # 환경 설정
    parser.add_argument('--no_reference_gait', action='store_true',
                        help='참조 gait 비활성화')
    parser.add_argument('--render', action='store_true',
                        help='첫 번째 환경 렌더링')
    
    # 로깅 및 저장
    parser.add_argument('--wandb', action='store_true', help='Weights & Biases 로깅')
    parser.add_argument('--save_freq', type=int, default=100000,
                        help='체크포인트 저장 주기 (스텝 단위)')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    args = parser.parse_args()
    
    # 시드 설정
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    print(f"🎮 벡터화 PPO 학습 시작")
    print(f"설정: {args.num_envs}개 환경, {args.total_timesteps:,} 스텝 목표")
    
    # 트레이너 생성 및 학습 시작
    trainer = VectorizedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()