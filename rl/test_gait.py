#!/usr/bin/env python3
"""
참조 보행 패턴 시연 스크립트
사족보행 로봇이 어떻게 걸어야 하는지 보여줍니다.
"""

from environment import GO2ForwardEnv
from gait_generator import GaitGenerator
import numpy as np
import time


def demo_reference_gait():
    """참조 보행 패턴을 시연"""
    
    env = GO2ForwardEnv(render_mode="human")
    gait = GaitGenerator(gait_type="trot", frequency=1.5)
    
    print("🐕 참조 보행 패턴 시연 시작!")
    print("이것이 로봇이 학습해야 할 목표 동작입니다.")
    
    obs, _ = env.reset()
    
    for step in range(2000):  # 4초간 시연
        # 참조 동작 가져오기
        sim_time = step * 0.002
        target_angles, target_contacts = gait.get_joint_targets(sim_time)
        
        # 참조 동작을 액션으로 변환 (단순 PD 제어)
        current_angles = env.data.qpos[7:19]
        angle_error = target_angles - current_angles
        action = angle_error * 50.0  # P gain
        action = np.clip(action, -20, 20)
        
        # 환경 스텝
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        # 상태 출력
        if step % 100 == 0:
            print(f"스텝 {step}: 목표 발 접촉 {target_contacts}")
        
        if terminated or truncated:
            obs, _ = env.reset()
            
        time.sleep(0.01)  # 천천히 관찰
    
    env.close()
    print("✨ 참조 보행 시연 완료!")


if __name__ == "__main__":
    demo_reference_gait()