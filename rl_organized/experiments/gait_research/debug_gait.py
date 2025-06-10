#!/usr/bin/env python3
"""
보행 패턴 디버깅 스크립트
실제 로봇 모델과 참조 패턴의 차이를 분석합니다.
"""

from environment import GO2ForwardEnv
from gait_generator import GaitGenerator
import numpy as np
import mujoco as mj


def debug_robot_info():
    """로봇 모델 정보 분석"""
    env = GO2ForwardEnv()
    
    print("=== GO2 로봇 모델 정보 ===")
    print(f"전체 관절 수 (nq): {env.model.nq}")
    print(f"속도 차원 (nv): {env.model.nv}")
    print(f"액추에이터 수 (nu): {env.model.nu}")
    
    print("\n=== 관절 이름들 ===")
    for i in range(env.model.njnt):
        joint_name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            print(f"Joint {i}: {joint_name}")
    
    print("\n=== 액추에이터 이름들 ===")
    for i in range(env.model.nu):
        actuator_name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name:
            print(f"Actuator {i}: {actuator_name}")
    
    print("\n=== 초기 관절 위치 ===")
    joint_pos = env.data.qpos[7:7+12]  # Free joint 제외
    print(f"초기 qpos[7:19]: {joint_pos}")
    
    print("\n=== 토크 제한 ===")
    if env.model.actuator_forcerange is not None:
        for i in range(env.model.nu):
            force_range = env.model.actuator_forcerange[i]
            actuator_name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"{actuator_name}: {force_range}")
    
    env.close()


def test_reference_motion():
    """참조 동작의 적절성 테스트"""
    gait = GaitGenerator(gait_type="trot", frequency=1.5)
    
    print("\n=== 참조 동작 분석 ===")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        angles, contacts = gait.get_joint_targets(t)
        print(f"\n시간 {t:.2f}s:")
        print(f"  관절 각도: {angles}")
        print(f"  발 접촉: {contacts}")
        print(f"  관절 각도 범위: [{angles.min():.3f}, {angles.max():.3f}]")


def compare_with_model():
    """모델과 참조 동작 비교"""
    env = GO2ForwardEnv()
    gait = GaitGenerator(gait_type="trot", frequency=1.5)
    
    print("\n=== 모델과 참조 동작 비교 ===")
    
    # 현재 모델 상태
    current_angles = env.data.qpos[7:19]
    print(f"현재 모델 관절 각도: {current_angles}")
    
    # 참조 동작
    target_angles, target_contacts = gait.get_joint_targets(0.0)
    print(f"참조 관절 각도: {target_angles}")
    
    # 차이
    diff = np.abs(current_angles - target_angles)
    print(f"각도 차이: {diff}")
    print(f"평균 차이: {diff.mean():.3f} rad")
    print(f"최대 차이: {diff.max():.3f} rad")
    
    # 관절 범위 확인
    print(f"\n모델 관절 범위:")
    for i in range(env.model.njnt):
        joint_name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_JOINT, i)
        if joint_name and i >= 1:  # Free joint 제외
            joint_range = env.model.jnt_range[i]
            print(f"  {joint_name}: [{joint_range[0]:.3f}, {joint_range[1]:.3f}]")
    
    env.close()


def test_simple_motion():
    """간단한 움직임 테스트"""
    env = GO2ForwardEnv(render_mode="human")
    
    print("\n=== 간단한 움직임 테스트 ===")
    print("로봇을 키프레임 자세로 설정합니다...")
    
    # 홈 키프레임으로 설정
    if hasattr(env.model, 'key_qpos') and len(env.model.key_qpos) > 0:
        env.data.qpos[:] = env.model.key_qpos[0]
        mj.mj_forward(env.model, env.data)
        print("키프레임 자세로 설정 완료")
    
    obs, _ = env.reset()
    
    # 10초간 아무 움직임 없이 서있기
    for step in range(5000):
        action = np.zeros(env.action_space.shape[0])  # 모든 토크 0
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if step % 1000 == 0:
            height = env.data.qpos[2]
            print(f"스텝 {step}: 높이 {height:.3f}m, 보상 {reward:.2f}")
        
        if terminated or truncated:
            print("에피소드 종료!")
            break
    
    env.close()


if __name__ == "__main__":
    print("🔍 GO2 보행 패턴 디버깅 시작\n")
    
    debug_robot_info()
    test_reference_motion()
    compare_with_model()
    
    print("\n간단한 움직임 테스트를 시작합니다...")
    print("로봇이 10초간 서있는지 확인하세요.")
    input("Enter를 눌러 시작...")
    
    test_simple_motion()
    
    print("\n🔍 디버깅 완료!")