#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from go2_mujoco_env import GO2MujocoEnv
import numpy as np

def test_env():
    print("🔍 새로운 GO2 환경 테스트")
    
    try:
        env = GO2MujocoEnv()
        print("✅ 환경 생성 성공")
        
        print(f"📊 관찰 공간 크기: {env.observation_space.shape}")
        print(f"🎮 액션 공간 크기: {env.action_space.shape}")
        
        print("\n=== 초기 리셋 ===")
        obs, info = env.reset()
        print(f"관찰 차원: {obs.shape}")
        print(f"초기 healthy 상태: {env.is_healthy}")
        print(f"초기 높이: {env.data.qpos[2]:.3f}m")
        
        print("\n=== 몇 스텝 실행 ===")
        for step in range(5):
            action = np.random.uniform(-1, 1, 12)  # 랜덤 액션
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step}: reward={reward:.2f}, terminated={terminated}, healthy={env.is_healthy}")
            
            if terminated:
                print("에피소드 종료!")
                break
        
        env.close()
        print("✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env()