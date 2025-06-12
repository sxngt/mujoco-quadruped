"""
사족보행 로봇을 위한 기본 보행 패턴 생성기
Trot, Walk, Pace 등의 기본 gait pattern을 제공
"""

import numpy as np
import math


class GaitGenerator:
    """기본적인 사족보행 패턴을 생성하는 클래스"""
    
    def __init__(self, gait_type="trot", frequency=2.0, swing_height=0.08):
        """
        Args:
            gait_type: "trot", "walk", "pace", "gallop" 중 선택
            frequency: 보행 주파수 (Hz)
            swing_height: 발을 들어올리는 높이 (m)
        """
        self.gait_type = gait_type
        self.frequency = frequency
        self.swing_height = swing_height
        self.phase = 0.0
        
        # 각 보행 패턴의 위상 차이 정의
        # [FL, FR, RL, RR] 순서
        self.phase_offsets = {
            "trot": [0.0, 0.5, 0.5, 0.0],      # 대각선 다리가 함께 움직임
            "walk": [0.0, 0.25, 0.75, 0.5],    # 한 번에 한 다리씩
            "pace": [0.0, 0.0, 0.5, 0.5],      # 같은 쪽 다리가 함께
            "gallop": [0.0, 0.1, 0.5, 0.6]     # 앞다리 후 뒷다리
        }
        
        # 각 다리의 기본 위치 (hip에서의 상대 위치)
        self.default_foot_positions = {
            "FL": np.array([0.0, 0.0, -0.3]),
            "FR": np.array([0.0, 0.0, -0.3]),
            "RL": np.array([0.0, 0.0, -0.3]),
            "RR": np.array([0.0, 0.0, -0.3])
        }
        
    def get_foot_trajectory(self, leg_name, phase):
        """
        특정 다리의 목표 궤적 계산
        
        Args:
            leg_name: "FL", "FR", "RL", "RR" 중 하나
            phase: 현재 보행 위상 (0~1)
            
        Returns:
            foot_position: 발끝의 목표 위치 (x, y, z)
            in_swing: 스윙 단계인지 여부
        """
        leg_idx = ["FL", "FR", "RL", "RR"].index(leg_name)
        leg_phase = (phase + self.phase_offsets[self.gait_type][leg_idx]) % 1.0
        
        # Swing phase (공중) vs Stance phase (지면)
        swing_ratio = 0.4  # 40% 시간은 공중에
        
        if leg_phase < swing_ratio:
            # Swing phase: 발을 들어올림
            in_swing = True
            swing_progress = leg_phase / swing_ratio
            
            # 사인 곡선으로 부드러운 궤적
            z_offset = self.swing_height * math.sin(swing_progress * math.pi)
            x_offset = 0.05 * (swing_progress - 0.5)  # 약간 앞으로
            
        else:
            # Stance phase: 지면 접촉
            in_swing = False
            stance_progress = (leg_phase - swing_ratio) / (1 - swing_ratio)
            
            z_offset = 0.0  # 지면에 닿음
            x_offset = -0.05 * stance_progress  # 뒤로 밀기
        
        # 기본 위치에 오프셋 추가
        foot_pos = self.default_foot_positions[leg_name].copy()
        foot_pos[0] += x_offset
        foot_pos[2] += z_offset
        
        return foot_pos, in_swing
    
    def get_joint_targets(self, current_time):
        """
        현재 시간에 따른 모든 관절의 목표 각도 계산
        
        Returns:
            joint_angles: 12개 관절 각도 [FL_hip, FL_thigh, FL_calf, FR_hip, ...]
            foot_contacts: 각 발의 접촉 여부 [FL, FR, RL, RR]
        """
        # 현재 위상 계산 (0~1)
        self.phase = (current_time * self.frequency) % 1.0
        
        joint_angles = []
        foot_contacts = []
        
        for leg in ["FL", "FR", "RL", "RR"]:
            foot_pos, in_swing = self.get_foot_trajectory(leg, self.phase)
            
            # 간단한 역기구학 (실제로는 더 복잡함)
            # 여기서는 근사치 사용
            hip_angle = 0.0  # 고관절 (좌우)
            thigh_angle = 0.3 if not in_swing else 0.5  # 대퇴부
            calf_angle = -0.6 if not in_swing else -1.0  # 종아리
            
            # 다리별 미세 조정
            if leg.startswith("R"):  # 뒷다리
                thigh_angle += 0.1
                
            joint_angles.extend([hip_angle, thigh_angle, calf_angle])
            foot_contacts.append(not in_swing)
        
        return np.array(joint_angles), np.array(foot_contacts)
    
    def get_reference_motion(self, duration=5.0, dt=0.002):
        """
        전체 참조 동작 시퀀스 생성
        
        Args:
            duration: 생성할 동작의 총 시간 (초)
            dt: 시간 간격
            
        Returns:
            times: 시간 배열
            joint_angles_seq: 시간에 따른 관절 각도 시퀀스
            foot_contacts_seq: 시간에 따른 발 접촉 시퀀스
        """
        times = np.arange(0, duration, dt)
        joint_angles_seq = []
        foot_contacts_seq = []
        
        for t in times:
            angles, contacts = self.get_joint_targets(t)
            joint_angles_seq.append(angles)
            foot_contacts_seq.append(contacts)
        
        return times, np.array(joint_angles_seq), np.array(foot_contacts_seq)


class CyclicGaitReward:
    """주기적인 보행 패턴을 유도하는 보상 함수"""
    
    def __init__(self, target_frequency=2.0):
        self.target_frequency = target_frequency
        self.last_foot_contacts = None
        self.phase_estimator = 0.0
        
    def compute_gait_reward(self, foot_contacts, dt=0.002):
        """
        현재 발 접촉 패턴이 목표 주기와 얼마나 일치하는지 계산
        """
        if self.last_foot_contacts is None:
            self.last_foot_contacts = foot_contacts
            return 0.0
        
        # 발 상태 변화 감지
        foot_changes = foot_contacts != self.last_foot_contacts
        
        # 주기성 보상
        if np.any(foot_changes):
            # 이상적인 주기와의 차이
            expected_phase_change = dt * self.target_frequency * 2 * np.pi
            phase_reward = np.cos(self.phase_estimator)
            self.phase_estimator += expected_phase_change
        else:
            phase_reward = 0.0
        
        self.last_foot_contacts = foot_contacts
        
        return phase_reward * 2.0  # 스케일링


# 사용 예시
if __name__ == "__main__":
    # Trot gait 생성
    gait = GaitGenerator(gait_type="trot", frequency=2.0)
    
    # 5초간의 참조 동작 생성
    times, angles, contacts = gait.get_reference_motion(duration=5.0)
    
    print(f"생성된 참조 동작:")
    print(f"- 총 프레임: {len(times)}")
    print(f"- 관절 각도 shape: {angles.shape}")
    print(f"- 발 접촉 shape: {contacts.shape}")
    
    # 첫 몇 프레임 출력
    for i in range(10):
        print(f"\n프레임 {i} (t={times[i]:.3f}s):")
        print(f"  관절 각도: {angles[i][:6]}")  # 처음 6개만
        print(f"  발 접촉: {contacts[i]}")