#!/usr/bin/env python3
"""
벡터화 환경을 사용한 고속 PPO 학습
16개 환경 병렬 실행으로 학습 효율성 극대화
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from ppo_agent import PPOAgent
from vectorized_env import create_vectorized_env, SyncVectorEnv
import wandb
import argparse
import os
import time
from datetime import datetime


class VectorizedPPOAgent(PPOAgent):
    """
    벡터화 환경용 PPO 에이전트
    여러 환경에서 동시에 데이터 수집
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 벡터화 환경용 메모리 관리
        self.num_envs = None
        self.env_observations = []
        self.env_actions = []
        self.env_rewards = []
        self.env_values = []
        self.env_log_probs = []
        self.env_dones = []
        
    def reset_memory(self):
        """메모리 초기화"""
        super().reset_memory()
        self.env_observations = []
        self.env_actions = []
        self.env_rewards = []
        self.env_values = []
        self.env_log_probs = []
        self.env_dones = []
    
    def store_vectorized_transition(self, obs, actions, rewards, values, log_probs, dones):
        """
        벡터화된 transition 저장
        
        Args:
            obs: (num_envs, obs_dim) 관찰
            actions: (num_envs, action_dim) 행동
            rewards: (num_envs,) 보상
            values: (num_envs,) 가치 추정
            log_probs: (num_envs,) 로그 확률
            dones: (num_envs,) 종료 플래그
        """
        # 각 환경별로 개별 저장
        for i in range(len(obs)):
            self.observations.append(obs[i])
            self.actions.append(actions[i])
            self.rewards.append(rewards[i])
            self.values.append(values[i])
            self.log_probs.append(log_probs[i])
            self.dones.append(dones[i])
    
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
        batch_size = observations.shape[0]
        
        # 배치로 처리
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        
        with torch.no_grad():
            # Actor 출력 (평균, 로그 표준편차)
            action_mean, action_log_std = self.actor(obs_tensor)
            action_std = torch.exp(action_log_std)
            
            # 정규분포에서 샘플링
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)  # 다차원 행동의 합
            
            # Critic 출력
            values = self.critic(obs_tensor).squeeze(-1)
        
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()


def train_vectorized_ppo(args):
    """
    벡터화 환경에서 PPO 학습
    """
    print(f"🚀 벡터화 PPO 학습 시작 ({args.num_envs}개 환경)")
    
    # 벡터화 환경 생성
    if args.sync_env:
        # 동기식 환경 (디버깅용)
        from vectorized_env import make_env
        env_fns = [make_env(i, None, not args.no_reference_gait) for i in range(args.num_envs)]
        vec_env = SyncVectorEnv(env_fns)
        print("🔄 동기식 벡터 환경 사용")
    else:
        # 비동기식 환경 (고성능)
        vec_env = create_vectorized_env(
            num_envs=args.num_envs,
            render_mode="human" if args.render else None,
            use_reference_gait=not args.no_reference_gait
        )
        print("⚡ 비동기식 벡터 환경 사용")
    
    # 벡터화 에이전트 생성
    agent = VectorizedPPOAgent(
        obs_dim=vec_env.observation_space.shape[0],
        action_dim=vec_env.action_space.shape[0],
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio
    )
    
    agent.num_envs = args.num_envs
    
    # 로깅 초기화
    if args.wandb:
        wandb.init(
            project="go2-vectorized-locomotion",
            config=vars(args),
            name=f"vec_ppo_{args.num_envs}envs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # 학습 변수
    episode_rewards = [[] for _ in range(args.num_envs)]
    episode_lengths = [[] for _ in range(args.num_envs)]
    episode_counts = np.zeros(args.num_envs, dtype=int)
    best_reward = -np.inf
    
    print(f"디바이스: {agent.device}")
    print(f"관찰 공간: {vec_env.observation_space}")
    print(f"행동 공간: {vec_env.action_space}")
    
    # 환경 리셋
    observations, infos = vec_env.reset()
    current_episode_rewards = np.zeros(args.num_envs)
    current_episode_lengths = np.zeros(args.num_envs)
    
    start_time = time.time()
    timestep = 0
    
    while timestep < args.total_timesteps:
        # 롤아웃 수집
        for step in range(args.rollout_length // args.num_envs):
            # 벡터화된 행동 생성
            actions, log_probs, values = agent.get_vectorized_actions(observations)
            
            # 환경 스텝
            next_observations, rewards, terminated, truncated, infos = vec_env.step(actions)
            dones = terminated | truncated
            
            # 데이터 저장
            agent.store_vectorized_transition(
                observations, actions, rewards, values, log_probs, dones
            )
            
            # 상태 업데이트
            observations = next_observations
            current_episode_rewards += rewards
            current_episode_lengths += 1
            timestep += args.num_envs
            
            # 에피소드 완료 처리
            for env_idx in range(args.num_envs):
                if dones[env_idx]:
                    episode_rewards[env_idx].append(current_episode_rewards[env_idx])
                    episode_lengths[env_idx].append(current_episode_lengths[env_idx])
                    episode_counts[env_idx] += 1
                    
                    current_episode_rewards[env_idx] = 0
                    current_episode_lengths[env_idx] = 0
            
            # 렌더링
            if args.render and hasattr(vec_env, 'render'):
                vec_env.render()
        
        # 정책 업데이트
        if len(agent.observations) >= args.rollout_length:
            losses = agent.update_policy(
                n_epochs=args.ppo_epochs,
                batch_size=args.batch_size
            )
            
            # 통계 계산
            all_rewards = [r for env_rewards in episode_rewards for r in env_rewards[-10:]]
            all_lengths = [l for env_lengths in episode_lengths for l in env_lengths[-10:]]
            
            if all_rewards:
                avg_reward = np.mean(all_rewards)
                avg_length = np.mean(all_lengths)
                total_episodes = np.sum(episode_counts)
                
                elapsed_time = time.time() - start_time
                fps = timestep / elapsed_time
                
                print(f"스텝 {timestep:7d} | 에피소드 {total_episodes:4d} | "
                      f"평균 보상: {avg_reward:7.2f} | 평균 길이: {avg_length:5.1f} | "
                      f"FPS: {fps:5.0f}")
                
                # 로깅
                if args.wandb:
                    log_dict = {
                        "timestep": timestep,
                        "total_episodes": total_episodes,
                        "avg_reward": avg_reward,
                        "avg_length": avg_length,
                        "fps": fps,
                        "num_envs": args.num_envs,
                    }
                    if losses:
                        log_dict.update(losses)
                    wandb.log(log_dict)
                
                # 최고 모델 저장
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    if not os.path.exists("models"):
                        os.makedirs("models")
                    agent.save(f"models/best_vectorized_go2_ppo.pth")
                    print(f"🏆 새로운 최고 성능 모델 저장! 평균 보상: {best_reward:.2f}")
        
        # 주기적 체크포인트
        if timestep % (args.save_freq * args.num_envs) < args.num_envs:
            if not os.path.exists("models"):
                os.makedirs("models")
            agent.save(f"models/vectorized_go2_ppo_step_{timestep}.pth")
    
    # 최종 저장
    if not os.path.exists("models"):
        os.makedirs("models")
    agent.save("models/vectorized_go2_ppo_final.pth")
    
    # 정리
    vec_env.close()
    if args.wandb:
        wandb.finish()
    
    total_time = time.time() - start_time
    final_fps = timestep / total_time
    
    print(f"✅ 학습 완료!")
    print(f"총 시간: {total_time:.1f}초")
    print(f"평균 FPS: {final_fps:.0f}")
    print(f"최고 성능: {best_reward:.2f}")
    
    return agent


def main():
    parser = argparse.ArgumentParser(description='벡터화 PPO 학습')
    
    # 벡터화 환경 설정
    parser.add_argument('--num_envs', type=int, default=16, 
                        help='병렬 환경 수 (기본: 16)')
    parser.add_argument('--sync_env', action='store_true',
                        help='동기식 환경 사용 (디버깅용)')
    
    # 학습 설정
    parser.add_argument('--total_timesteps', type=int, default=2000000,
                        help='총 학습 스텝')
    parser.add_argument('--rollout_length', type=int, default=4096,
                        help='롤아웃 길이 (num_envs의 배수여야 함)')
    
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
    parser.add_argument('--save_freq', type=int, default=50000,
                        help='체크포인트 저장 주기 (스텝 단위)')
    
    # 기타
    parser.add_argument('--seed', type=int, default=None, help='랜덤 시드')
    
    args = parser.parse_args()
    
    # 시드 설정
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # 롤아웃 길이 유효성 검사
    if args.rollout_length % args.num_envs != 0:
        args.rollout_length = (args.rollout_length // args.num_envs) * args.num_envs
        print(f"⚠️  롤아웃 길이를 {args.rollout_length}로 조정 (num_envs의 배수)")
    
    # 학습 시작
    train_vectorized_ppo(args)


if __name__ == "__main__":
    main()