#!/usr/bin/env python3
"""
에피소드 조기 종료 문제 디버깅 스크립트
"""

import sys
import os
import numpy as np

# rl 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.getcwd())

def test_single_environment():
    """단일 환경에서 종료 원인 테스트"""
    print("🔍 단일 환경 종료 원인 분석")
    
    try:
        from environments.improved.improved_environment import ImprovedGO2Env
        
        env = ImprovedGO2Env(render_mode=None, use_reference_gait=False)
        print("✅ 환경 생성 성공")
        
        obs, info = env.reset()
        print(f"📍 초기 상태: 높이 {env.data.qpos[2]:.3f}m")
        
        episode_count = 0
        max_episodes = 3
        
        while episode_count < max_episodes:
            step_count = 0
            total_reward = 0
            
            print(f"\n🏁 에피소드 {episode_count + 1} 시작")
            
            while True:
                # 매우 작은 액션 (거의 움직이지 않음)
                action = np.random.uniform(-0.1, 0.1, env.action_space.shape[0])
                
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                total_reward += reward
                
                # 매 50스텝마다 상태 출력
                if step_count % 50 == 0:
                    height = env.data.qpos[2]
                    forward_vel = env.data.qvel[0]
                    vertical_vel = env.data.qvel[2]
                    print(f"   스텝 {step_count}: 높이 {height:.3f}m, 전진 {forward_vel:.3f}m/s, 수직 {vertical_vel:.3f}m/s")
                
                # 종료 확인
                if terminated or truncated:
                    print(f"🚨 에피소드 종료!")
                    print(f"   terminated: {terminated}, truncated: {truncated}")
                    print(f"   총 스텝: {step_count}, 총 보상: {total_reward:.2f}")
                    
                    height = env.data.qpos[2]
                    quat = env.data.qpos[3:7]
                    z_axis = np.array([
                        2*(quat[1]*quat[3] + quat[0]*quat[2]),
                        2*(quat[2]*quat[3] - quat[0]*quat[1]),
                        quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2
                    ])
                    
                    print(f"   최종 높이: {height:.3f}m")
                    print(f"   최종 z축: {z_axis[2]:.3f}")
                    print(f"   최종 위치: x={env.data.qpos[0]:.2f}, y={env.data.qpos[1]:.2f}")
                    
                    break
                
                # 무한 루프 방지
                if step_count > 2000:
                    print(f"⏰ 2000스텝 도달 - 강제 종료")
                    break
            
            episode_count += 1
            obs, info = env.reset()
        
        env.close()
        print("✅ 단일 환경 테스트 완료")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

def test_vectorized_environment():
    """벡터화 환경에서 종료 원인 테스트"""
    print("\n🔍 벡터화 환경 종료 원인 분석")
    
    try:
        from environments.vectorized.improved_vectorized import ImprovedVectorEnv
        
        vec_env = ImprovedVectorEnv(num_envs=2, render_mode=None, use_reference_gait=False)
        print("✅ 벡터화 환경 생성 성공")
        
        observations, infos = vec_env.reset()
        print(f"📍 초기 상태 설정 완료")
        
        step_count = 0
        episodes_terminated = [False] * 2
        
        while not all(episodes_terminated) and step_count < 1000:
            # 매우 작은 액션
            actions = np.random.uniform(-0.1, 0.1, (2, vec_env.action_space.shape[0]))
            
            observations, rewards, terminated, truncated, infos = vec_env.step(actions)
            dones = terminated | truncated
            
            step_count += 1
            
            # 에피소드 종료 확인
            for env_idx in range(2):
                if dones[env_idx] and not episodes_terminated[env_idx]:
                    print(f"🚨 환경 {env_idx} 에피소드 종료!")
                    print(f"   스텝: {step_count}")
                    print(f"   terminated: {terminated[env_idx]}, truncated: {truncated[env_idx]}")
                    print(f"   보상: {rewards[env_idx]:.2f}")
                    episodes_terminated[env_idx] = True
            
            # 주기적 상태 출력
            if step_count % 100 == 0:
                print(f"📊 스텝 {step_count}: 환경별 보상 {rewards}")
        
        vec_env.close()
        print("✅ 벡터화 환경 테스트 완료")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚨 에피소드 조기 종료 문제 디버깅 시작")
    print("="*60)
    
    # 1. 단일 환경 테스트
    test_single_environment()
    
    # 2. 벡터화 환경 테스트  
    test_vectorized_environment()
    
    print("\n🔚 디버깅 완료")
    print("="*60)

if __name__ == "__main__":
    main()