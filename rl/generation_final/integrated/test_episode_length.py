#!/usr/bin/env python3
"""에피소드 길이 테스트"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from go2_mujoco_env import GO2MujocoEnv
import numpy as np

def test_episode_length():
    print("🔍 에피소드 길이 테스트")
    
    env = GO2MujocoEnv(render_mode="human")
    
    for episode in range(3):
        print(f"\n=== 에피소드 {episode + 1} ===")
        obs, info = env.reset()
        
        step = 0
        episode_reward = 0
        
        while True:
            # 작은 랜덤 액션
            action = np.random.uniform(-0.1, 0.1, 12)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # 매 50스텝마다 상태 출력
            if step % 50 == 0:
                print(f"Step {step}: z={info['z_position']:.3f}m, reward={reward:.2f}, total={episode_reward:.2f}")
            
            if terminated or truncated:
                print(f"\n종료! 이유: {'terminated' if terminated else 'truncated'}")
                print(f"총 스텝: {step}")
                print(f"총 보상: {episode_reward:.2f}")
                print(f"최종 높이: {info['z_position']:.3f}m")
                break
        
        input("Enter를 눌러 다음 에피소드 시작...")
    
    env.close()

if __name__ == "__main__":
    test_episode_length()