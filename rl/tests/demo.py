#!/usr/bin/env python3
"""
GO2 로봇 학습된 모델 데모 스크립트
간단하게 로봇의 걷기를 시연합니다.
"""

import argparse
import numpy as np
import torch
from environment import GO2ForwardEnv
from ppo_agent import PPOAgent
import time
import os


def run_demo(model_path, num_episodes=5, slow_motion=False):
    """학습된 모델을 실행하여 로봇의 보행을 시연"""
    
    # 환경 생성 (렌더링 켜기)
    env = GO2ForwardEnv(render_mode="human")
    
    # 에이전트 생성 및 모델 로드
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✅ 모델 로드 완료: {model_path}")
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    print("\n🤖 GO2 로봇 보행 데모 시작!")
    print("=" * 50)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\n에피소드 {episode + 1}/{num_episodes} 시작...")
        
        while True:
            # 행동 선택 (학습된 정책 사용)
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            
            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # 렌더링
            env.render()
            
            # 슬로우 모션 옵션
            if slow_motion:
                time.sleep(0.01)  # 10ms 지연
            
            # 상태 출력 (10스텝마다)
            if step_count % 10 == 0:
                print(f"  스텝 {step_count}: 보상={reward:.2f}, 누적={episode_reward:.2f}")
            
            if terminated or truncated:
                break
        
        print(f"에피소드 {episode + 1} 종료:")
        print(f"  - 총 스텝: {step_count}")
        print(f"  - 총 보상: {episode_reward:.2f}")
        print(f"  - 평균 보상: {episode_reward/step_count:.2f}")
    
    env.close()
    print("\n✨ 데모 완료!")


def main():
    parser = argparse.ArgumentParser(description='GO2 로봇 학습 모델 데모')
    parser.add_argument('--model', type=str, default='models/best_go2_ppo.pth',
                        help='모델 파일 경로')
    parser.add_argument('--episodes', type=int, default=5,
                        help='실행할 에피소드 수')
    parser.add_argument('--slow', action='store_true',
                        help='슬로우 모션으로 실행')
    
    args = parser.parse_args()
    
    # 데모 실행
    run_demo(args.model, args.episodes, args.slow)


if __name__ == "__main__":
    main()