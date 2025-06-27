#!/usr/bin/env python3
"""
훈련 렌더링 문제 디버그 스크립트
"""

import os
import sys
import time

# 프로젝트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.join(current_dir, "rl")
sys.path.append(rl_dir)

from environments.integrated import IntegratedGO2Env
from agents.ppo_agent import PPOAgent

def debug_training_render():
    print("=== 훈련 렌더링 디버그 ===")
    
    # 1. 환경 생성 테스트 (렌더링 모드)
    print("1. 렌더링 모드로 환경 생성...")
    try:
        env = IntegratedGO2Env(render_mode="human")
        print("✅ 환경 생성 성공")
    except Exception as e:
        print(f"❌ 환경 생성 실패: {e}")
        return False
    
    # 2. 에이전트 생성 테스트
    print("2. 에이전트 생성...")
    try:
        agent = PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            lr=3e-4,
            hidden_dim=128
        )
        print("✅ 에이전트 생성 성공")
    except Exception as e:
        print(f"❌ 에이전트 생성 실패: {e}")
        env.close()
        return False
    
    # 3. 훈련 루프 시뮬레이션 (짧게)
    print("3. 훈련 루프 시뮬레이션 (50 스텝)...")
    try:
        obs, _ = env.reset()
        print("   리셋 완료")
        
        for step in range(50):
            print(f"   Step {step+1}/50")
            
            # 에이전트 액션
            action, log_prob, value = agent.get_action(obs)
            
            # 환경 스텝
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 렌더링 시도
            print(f"     렌더링 시도...")
            render_result = env.render()
            print(f"     렌더링 결과: {render_result}")
            
            # 경험 저장
            agent.store_transition(obs, action, reward, value, log_prob, terminated)
            
            obs = next_obs
            time.sleep(0.05)  # 관찰하기 위한 지연
            
            if step % 10 == 0:
                print(f"   Step {step}: 보상={reward:.3f}")
            
            if terminated or truncated:
                print(f"   에피소드 종료 (step {step})")
                obs, _ = env.reset()
        
        print("✅ 훈련 루프 시뮬레이션 완료")
        
    except Exception as e:
        print(f"❌ 훈련 루프 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()
    
    return True

def debug_render_difference():
    print("\n=== 테스트 vs 훈련 렌더링 차이 분석 ===")
    
    print("1. 테스트 방식 (test_integrated_env.py와 동일)")
    try:
        env1 = IntegratedGO2Env(render_mode="human")
        obs, _ = env1.reset()
        
        for i in range(10):
            action = env1.action_space.sample() * 0.5
            obs, reward, terminated, truncated, info = env1.step(action)
            render_result = env1.render()
            print(f"   테스트 방식 step {i}: 렌더링 결과 = {render_result}")
            time.sleep(0.1)
            
            if terminated or truncated:
                obs, _ = env1.reset()
        
        env1.close()
        print("✅ 테스트 방식 완료")
        
    except Exception as e:
        print(f"❌ 테스트 방식 실패: {e}")
    
    print("\n2. 훈련 방식 (PPO 에이전트 사용)")
    try:
        env2 = IntegratedGO2Env(render_mode="human")
        agent = PPOAgent(
            obs_dim=env2.observation_space.shape[0],
            action_dim=env2.action_space.shape[0],
            lr=3e-4,
            hidden_dim=64
        )
        obs, _ = env2.reset()
        
        for i in range(10):
            action, log_prob, value = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env2.step(action)
            render_result = env2.render()
            print(f"   훈련 방식 step {i}: 렌더링 결과 = {render_result}")
            time.sleep(0.1)
            
            if terminated or truncated:
                obs, _ = env2.reset()
        
        env2.close()
        print("✅ 훈련 방식 완료")
        
    except Exception as e:
        print(f"❌ 훈련 방식 실패: {e}")
        import traceback
        traceback.print_exc()

def debug_environment_state():
    print("\n=== 환경 상태 디버그 ===")
    
    try:
        # 렌더링 모드로 환경 생성
        env = IntegratedGO2Env(render_mode="human")
        print(f"환경 렌더링 모드: {env.render_mode}")
        print(f"뷰어 초기 상태: {env.viewer}")
        
        # 리셋
        obs, _ = env.reset()
        print(f"리셋 후 뷰어 상태: {env.viewer}")
        
        # 첫 번째 렌더링
        print("첫 번째 렌더링 시도...")
        result1 = env.render()
        print(f"첫 번째 렌더링 결과: {result1}")
        print(f"렌더링 후 뷰어 상태: {env.viewer}")
        
        # 몇 스텝 실행
        for i in range(5):
            action = env.action_space.sample() * 0.3
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {i} 렌더링 시도...")
            result = env.render()
            print(f"Step {i} 렌더링 결과: {result}")
            
            time.sleep(0.2)
        
        env.close()
        
    except Exception as e:
        print(f"❌ 환경 상태 디버그 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("훈련 렌더링 문제 진단 시작\n")
    
    # 기본 훈련 렌더링 테스트
    success = debug_training_render()
    
    # 테스트 vs 훈련 방식 차이 분석
    debug_render_difference()
    
    # 환경 상태 세부 분석
    debug_environment_state()
    
    print(f"\n진단 완료. 기본 테스트 {'성공' if success else '실패'}")