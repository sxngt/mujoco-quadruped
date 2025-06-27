#!/usr/bin/env python3
"""
Generation Final 빠른 데모 - 짧은 훈련으로 동작 확인
"""

import os
import sys
import numpy as np
import torch
import time

# 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from integrated.integrated_go2_env import IntegratedGO2Env
from agents.ppo_agent import PPOAgent

def quick_demo():
    print("🚀 Generation Final - 빠른 데모 시작")
    print("="*50)
    
    # 환경 생성
    env = IntegratedGO2Env(render_mode=None)
    print(f"✅ 환경 생성: 관찰({env.observation_space.shape}), 액션({env.action_space.shape})")
    
    # 에이전트 생성 (작은 네트워크로 빠른 테스트)
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr=3e-4,
        hidden_dim=64,  # 작게 설정
        gamma=0.99,
        gae_lambda=0.95
    )
    print(f"✅ PPO 에이전트 생성 (디바이스: {agent.device})")
    
    # 짧은 훈련 루프 (100 스텝만)
    obs, _ = env.reset()
    total_reward = 0
    episode_count = 0
    step_count = 0
    
    print("\n🎯 훈련 시작 (100 스텝 데모)")
    start_time = time.time()
    
    for step in range(100):
        # 액션 선택
        action, log_prob, value = agent.get_action(obs)
        
        # 환경 스텝
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 경험 저장
        agent.store_transition(obs, action, reward, value, log_prob, terminated)
        
        obs = next_obs
        total_reward += reward
        step_count += 1
        
        # 진행 상황 출력
        if step % 20 == 0:
            print(f"   Step {step}: 보상={reward:.2f}, 값={value:.2f}, 총보상={total_reward:.1f}")
        
        # 에피소드 종료
        if terminated or truncated:
            episode_count += 1
            print(f"💫 에피소드 {episode_count} 종료: 총 보상={total_reward:.2f}, 스텝={step_count}")
            
            # 에피소드 리셋
            obs, _ = env.reset()
            total_reward = 0
            step_count = 0
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ 100 스텝 완료 시간: {elapsed_time:.2f}초")
    
    # 정책 업데이트 테스트
    if len(agent.observations) > 10:
        print("\n🔄 정책 업데이트 테스트...")
        update_info = agent.update_policy(n_epochs=2, batch_size=32)
        print(f"   정책 손실: {update_info['policy_loss']:.4f}")
        print(f"   가치 손실: {update_info['value_loss']:.4f}")
        print("✅ 정책 업데이트 성공")
    
    # 보상 구조 분석
    print(f"\n📊 보상 구조 분석:")
    print(f"   최근 보상 정보: {info}")
    
    env.close()
    print("\n🎉 Generation Final 데모 완료!")
    print("="*50)
    print("✅ 환경이 정상 작동하며 훈련 준비 완료")
    print(f"💡 전체 훈련: python integrated/train_integrated.py")

if __name__ == "__main__":
    quick_demo()