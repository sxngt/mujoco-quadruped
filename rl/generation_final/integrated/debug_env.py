#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from integrated_go2_env import IntegratedGO2Env
import numpy as np

def test_env():
    print("🔍 환경 디버그 테스트")
    
    env = IntegratedGO2Env()
    
    print("\n=== 초기 리셋 ===")
    obs, info = env.reset()
    print(f"관찰 차원: {obs.shape}")
    print(f"초기 healthy 상태: {env.is_healthy}")
    print(f"초기 높이: {env.data.qpos[2]:.3f}m")
    
    quat = env.data.qpos[3:7]
    w, x, y, z = quat
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(2*(w*y - z*x))
    print(f"초기 roll: {np.degrees(roll):.1f}°, pitch: {np.degrees(pitch):.1f}°")
    
    print("\n=== 스텝 테스트 ===")
    for step in range(10):
        action = np.zeros(12)  # 무동작
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: reward={reward:.2f}, terminated={terminated}, healthy={env.is_healthy}, height={env.data.qpos[2]:.3f}m")
        
        if terminated:
            print("에피소드 종료!")
            break
    
    print("\n=== 강한 액션 테스트 ===")
    obs, info = env.reset()
    for step in range(10):
        action = np.random.uniform(-10, 10, 12)  # 강한 랜덤 액션
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: reward={reward:.2f}, terminated={terminated}, healthy={env.is_healthy}, height={env.data.qpos[2]:.3f}m")
        
        if terminated:
            print("에피소드 종료!")
            break
    
    env.close()

if __name__ == "__main__":
    test_env()