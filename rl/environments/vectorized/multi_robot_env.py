#!/usr/bin/env python3
"""
멀티 로봇 환경
하나의 MuJoCo 시뮬레이션에서 여러 GO2 로봇이 동시에 학습
"""

import mujoco as mj
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import xml.etree.ElementTree as ET
import tempfile
import os


class MultiRobotGO2Env(gym.Env):
    """
    하나의 시뮬레이션에서 여러 GO2 로봇이 동시에 학습하는 환경
    """
    
    def __init__(self, num_robots=16, render_mode=None, robot_spacing=3.0):
        self.num_robots = num_robots
        self.robot_spacing = robot_spacing  # 로봇 간 간격 (미터)
        self.render_mode = render_mode
        self.viewer = None
        
        # 멀티 로봇 scene.xml 생성
        self.model_path = self._create_multi_robot_scene()
        
        self.model = mj.MjModel.from_xml_path(self.model_path)
        self.data = mj.MjData(self.model)
        
        # 시뮬레이션 초기화
        mj.mj_forward(self.model, self.data)
        
        # 각 로봇의 관절 인덱스 매핑
        self._setup_robot_indices()
        
        # 행동/관찰 공간 정의 (모든 로봇의 합)
        single_robot_action_dim = 12  # GO2 관절 수
        single_robot_obs_dim = 34     # GO2 관찰 차원
        
        self.action_space = spaces.Box(
            low=-20.0, high=20.0,
            shape=(num_robots * single_robot_action_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_robots * single_robot_obs_dim,),
            dtype=np.float32
        )
        
        # 에피소드 관련 변수
        self.max_episode_steps = float('inf')  # 무제한 에피소드 - 오직 넘어질 때만 종료
        self.current_step = 0
        
        print(f"🤖 {num_robots}개 로봇 멀티 환경 생성 완료")
        print(f"총 행동 차원: {self.action_space.shape[0]}")
        print(f"총 관찰 차원: {self.observation_space.shape[0]}")
    
    def _create_multi_robot_scene(self):
        """
        여러 로봇이 포함된 scene.xml 파일 생성
        """
        # 기본 GO2 scene.xml 읽기 (assets 폴더에서)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # rl 디렉토리
        base_scene_path = os.path.join(base_dir, "assets", "go2_scene.xml")
        if not os.path.exists(base_scene_path):
            raise FileNotFoundError(f"기본 scene 파일을 찾을 수 없습니다: {base_scene_path}")
        
        # XML 파싱
        tree = ET.parse(base_scene_path)
        root = tree.getroot()
        
        # worldbody 찾기
        worldbody = root.find('worldbody')
        if worldbody is None:
            raise ValueError("worldbody를 찾을 수 없습니다")
        
        # 기존 로봇 제거 (GO2 관련 body들)
        bodies_to_remove = []
        for body in worldbody.findall('body'):
            if body.get('name') and 'GO2' in body.get('name'):
                bodies_to_remove.append(body)
        
        for body in bodies_to_remove:
            worldbody.remove(body)
        
        # 그리드 배치로 여러 로봇 추가
        grid_size = int(np.ceil(np.sqrt(self.num_robots)))
        
        for robot_idx in range(self.num_robots):
            row = robot_idx // grid_size
            col = robot_idx % grid_size
            
            # 로봇 위치 계산
            x_pos = (col - grid_size//2) * self.robot_spacing
            y_pos = (row - grid_size//2) * self.robot_spacing
            z_pos = 0.4  # 지면 위 높이
            
            # 로봇 body 생성
            robot_body = ET.SubElement(worldbody, 'body')
            robot_body.set('name', f'GO2_robot_{robot_idx}')
            robot_body.set('pos', f'{x_pos} {y_pos} {z_pos}')
            
            # freejoint 추가
            freejoint = ET.SubElement(robot_body, 'freejoint')
            freejoint.set('name', f'GO2_freejoint_{robot_idx}')
            
            # 로봇 mesh/geom 추가 (간단한 박스로 대체)
            geom = ET.SubElement(robot_body, 'geom')
            geom.set('name', f'GO2_torso_{robot_idx}')
            geom.set('type', 'box')
            geom.set('size', '0.3 0.15 0.1')
            geom.set('rgba', f'{0.2 + robot_idx*0.05} {0.3 + robot_idx*0.03} {0.8 - robot_idx*0.02} 1')
            geom.set('mass', '15')
            
            # 다리 관절들 추가
            leg_names = ['FL', 'FR', 'RL', 'RR']
            leg_positions = [
                [0.2, 0.1, -0.1],   # Front Left
                [0.2, -0.1, -0.1],  # Front Right  
                [-0.2, 0.1, -0.1],  # Rear Left
                [-0.2, -0.1, -0.1]  # Rear Right
            ]
            
            for leg_idx, (leg_name, leg_pos) in enumerate(zip(leg_names, leg_positions)):
                # Hip joint
                hip_body = ET.SubElement(robot_body, 'body')
                hip_body.set('name', f'{leg_name}_hip_link_{robot_idx}')
                hip_body.set('pos', f'{leg_pos[0]} {leg_pos[1]} {leg_pos[2]}')
                
                hip_joint = ET.SubElement(hip_body, 'joint')
                hip_joint.set('name', f'{leg_name}_hip_joint_{robot_idx}')
                hip_joint.set('type', 'hinge')
                hip_joint.set('axis', '1 0 0')
                hip_joint.set('range', '-1.047 1.047')
                
                hip_geom = ET.SubElement(hip_body, 'geom')
                hip_geom.set('name', f'{leg_name}_hip_geom_{robot_idx}')
                hip_geom.set('type', 'cylinder')
                hip_geom.set('size', '0.03 0.05')
                hip_geom.set('rgba', '0.5 0.5 0.5 1')
                
                # Thigh joint
                thigh_body = ET.SubElement(hip_body, 'body')
                thigh_body.set('name', f'{leg_name}_thigh_link_{robot_idx}')
                thigh_body.set('pos', '0 0 -0.1')
                
                thigh_joint = ET.SubElement(thigh_body, 'joint')
                thigh_joint.set('name', f'{leg_name}_thigh_joint_{robot_idx}')
                thigh_joint.set('type', 'hinge')
                thigh_joint.set('axis', '0 1 0')
                thigh_joint.set('range', '-1.571 3.491')
                
                thigh_geom = ET.SubElement(thigh_body, 'geom')
                thigh_geom.set('name', f'{leg_name}_thigh_geom_{robot_idx}')
                thigh_geom.set('type', 'capsule')
                thigh_geom.set('size', '0.02 0.1')
                thigh_geom.set('rgba', '0.3 0.3 0.8 1')
                
                # Calf joint
                calf_body = ET.SubElement(thigh_body, 'body')
                calf_body.set('name', f'{leg_name}_calf_link_{robot_idx}')
                calf_body.set('pos', '0 0 -0.2')
                
                calf_joint = ET.SubElement(calf_body, 'joint')
                calf_joint.set('name', f'{leg_name}_calf_joint_{robot_idx}')
                calf_joint.set('type', 'hinge')
                calf_joint.set('axis', '0 1 0')
                calf_joint.set('range', '-2.723 -0.838')
                
                calf_geom = ET.SubElement(calf_body, 'geom')
                calf_geom.set('name', f'{leg_name}_calf_geom_{robot_idx}')
                calf_geom.set('type', 'capsule')
                calf_geom.set('size', '0.015 0.1')
                calf_geom.set('rgba', '0.2 0.8 0.3 1')
                
                # Foot
                foot_geom = ET.SubElement(calf_body, 'geom')
                foot_geom.set('name', f'{leg_name}_foot_{robot_idx}')
                foot_geom.set('type', 'sphere')
                foot_geom.set('size', '0.03')
                foot_geom.set('pos', '0 0 -0.1')
                foot_geom.set('rgba', '0.8 0.2 0.2 1')
                foot_geom.set('contype', '1')
                foot_geom.set('conaffinity', '1')
        
        # actuator 섹션에 모든 로봇의 액추에이터 추가
        actuator_section = root.find('actuator')
        if actuator_section is None:
            actuator_section = ET.SubElement(root, 'actuator')
        
        # 기존 액추에이터 제거
        for motor in list(actuator_section):
            actuator_section.remove(motor)
        
        # 새 액추에이터 추가
        for robot_idx in range(self.num_robots):
            for leg_name in ['FL', 'FR', 'RL', 'RR']:
                for joint_type in ['hip', 'thigh', 'calf']:
                    motor = ET.SubElement(actuator_section, 'motor')
                    motor.set('name', f'{leg_name}_{joint_type}_{robot_idx}')
                    motor.set('joint', f'{leg_name}_{joint_type}_joint_{robot_idx}')
                    motor.set('gear', '150')
                    motor.set('ctrllimited', 'true')
                    motor.set('ctrlrange', '-20 20')
        
        # 임시 파일로 저장
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        tree.write(temp_file.name, encoding='unicode')
        temp_file.close()
        
        return temp_file.name
    
    def _setup_robot_indices(self):
        """각 로봇의 관절 인덱스 매핑 설정"""
        self.robot_joint_indices = {}
        self.robot_actuator_indices = {}
        
        for robot_idx in range(self.num_robots):
            joint_indices = []
            actuator_indices = []
            
            # 각 로봇의 관절 찾기
            for leg_name in ['FL', 'FR', 'RL', 'RR']:
                for joint_type in ['hip', 'thigh', 'calf']:
                    joint_name = f'{leg_name}_{joint_type}_joint_{robot_idx}'
                    actuator_name = f'{leg_name}_{joint_type}_{robot_idx}'
                    
                    try:
                        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
                        actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                        
                        joint_indices.append(joint_id)
                        actuator_indices.append(actuator_id)
                    except:
                        print(f"경고: {joint_name} 또는 {actuator_name}을 찾을 수 없습니다")
            
            self.robot_joint_indices[robot_idx] = joint_indices
            self.robot_actuator_indices[robot_idx] = actuator_indices
            
            print(f"로봇 {robot_idx}: {len(joint_indices)}개 관절, {len(actuator_indices)}개 액추에이터")
    
    def _get_robot_observation(self, robot_idx):
        """특정 로봇의 관찰값 계산"""
        # 로봇의 freejoint 인덱스 계산
        freejoint_qpos_start = robot_idx * 7
        freejoint_qvel_start = robot_idx * 6
        
        # Body position and orientation (freejoint)
        if freejoint_qpos_start + 7 <= len(self.data.qpos):
            body_pos = self.data.qpos[freejoint_qpos_start:freejoint_qpos_start+3]
            body_quat = self.data.qpos[freejoint_qpos_start+3:freejoint_qpos_start+7]
        else:
            body_pos = np.zeros(3)
            body_quat = np.array([1, 0, 0, 0])
        
        if freejoint_qvel_start + 6 <= len(self.data.qvel):
            body_linvel = self.data.qvel[freejoint_qvel_start:freejoint_qvel_start+3]
            body_angvel = self.data.qvel[freejoint_qvel_start+3:freejoint_qvel_start+6]
        else:
            body_linvel = np.zeros(3)
            body_angvel = np.zeros(3)
        
        # Joint positions and velocities
        joint_indices = self.robot_joint_indices.get(robot_idx, [])
        joint_pos = np.zeros(12)
        joint_vel = np.zeros(12)
        
        for i, joint_idx in enumerate(joint_indices[:12]):
            if joint_idx < len(self.data.qpos):
                # freejoint를 제외한 인덱스 계산
                qpos_idx = joint_idx + robot_idx * 7  # freejoint 오프셋 추가
                qvel_idx = joint_idx + robot_idx * 6
                
                if qpos_idx < len(self.data.qpos):
                    joint_pos[i] = self.data.qpos[qpos_idx]
                if qvel_idx < len(self.data.qvel):
                    joint_vel[i] = self.data.qvel[qvel_idx]
        
        return np.concatenate([
            joint_pos,      # 12
            joint_vel,      # 12  
            body_quat,      # 4
            body_angvel,    # 3
            body_linvel     # 3
        ])  # 총 34차원
    
    def _get_observation(self):
        """모든 로봇의 관찰값"""
        observations = []
        for robot_idx in range(self.num_robots):
            obs = self._get_robot_observation(robot_idx)
            observations.append(obs)
        return np.concatenate(observations)
    
    def _get_robot_reward(self, robot_idx):
        """특정 로봇의 보상 계산"""
        # 로봇의 freejoint에서 속도 정보 가져오기
        freejoint_qvel_start = robot_idx * 6
        
        if freejoint_qvel_start + 3 <= len(self.data.qvel):
            forward_vel = self.data.qvel[freejoint_qvel_start]  # x 속도
            lateral_vel = abs(self.data.qvel[freejoint_qvel_start + 1])  # y 속도
            vertical_vel = abs(self.data.qvel[freejoint_qvel_start + 2])  # z 속도
        else:
            forward_vel = 0.0
            lateral_vel = 0.0
            vertical_vel = 0.0
        
        # 로봇 높이
        freejoint_qpos_start = robot_idx * 7
        if freejoint_qpos_start + 3 <= len(self.data.qpos):
            body_height = self.data.qpos[freejoint_qpos_start + 2]
        else:
            body_height = 0.4
        
        # 간단한 보상 함수
        forward_reward = forward_vel * 10.0 if forward_vel > 0 else forward_vel * 2.0
        survival_reward = 5.0 if body_height > 0.2 else -20.0
        direction_penalty = -lateral_vel * 5.0
        stability_penalty = -vertical_vel * 2.0
        
        return forward_reward + survival_reward + direction_penalty + stability_penalty
    
    def _get_reward(self):
        """모든 로봇의 보상 합계"""
        total_reward = 0.0
        for robot_idx in range(self.num_robots):
            total_reward += self._get_robot_reward(robot_idx)
        return total_reward / self.num_robots  # 평균 보상
    
    def _is_terminated(self):
        """최대한 관대한 종료 조건 - 로봇들이 충분히 보행을 시도할 수 있도록"""
        severely_failed_count = 0
        
        for robot_idx in range(self.num_robots):
            freejoint_qpos_start = robot_idx * 7
            if freejoint_qpos_start + 3 <= len(self.data.qpos):
                body_height = self.data.qpos[freejoint_qpos_start + 2]
                # 지면 아래로 뚫고 들어간 경우만 실패로 간주
                if body_height < -0.05:
                    severely_failed_count += 1
        
        # 대부분의 로봇이 심각하게 실패한 경우만 종료
        return severely_failed_count > self.num_robots * 0.8
    
    def reset(self, seed=None, options=None):
        """환경 리셋"""
        super().reset(seed=seed)
        
        # 모든 로봇을 초기 위치로 리셋
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        
        # 각 로봇의 초기 위치 설정
        grid_size = int(np.ceil(np.sqrt(self.num_robots)))
        
        for robot_idx in range(self.num_robots):
            row = robot_idx // grid_size
            col = robot_idx % grid_size
            
            freejoint_qpos_start = robot_idx * 7
            if freejoint_qpos_start + 7 <= len(self.data.qpos):
                # Position
                x_pos = (col - grid_size//2) * self.robot_spacing
                y_pos = (row - grid_size//2) * self.robot_spacing
                z_pos = 0.4
                
                self.data.qpos[freejoint_qpos_start] = x_pos + np.random.normal(0, 0.1)
                self.data.qpos[freejoint_qpos_start + 1] = y_pos + np.random.normal(0, 0.1)
                self.data.qpos[freejoint_qpos_start + 2] = z_pos + np.random.normal(0, 0.05)
                
                # Orientation (quaternion w, x, y, z)
                self.data.qpos[freejoint_qpos_start + 3] = 1.0  # w
                self.data.qpos[freejoint_qpos_start + 4] = 0.0  # x
                self.data.qpos[freejoint_qpos_start + 5] = 0.0  # y
                self.data.qpos[freejoint_qpos_start + 6] = 0.0  # z
        
        self.current_step = 0
        mj.mj_forward(self.model, self.data)
        
        observation = self._get_observation()
        info = {'num_robots': self.num_robots}
        
        return observation, info
    
    def step(self, action):
        """환경 스텝"""
        # 행동을 각 로봇의 액추에이터에 적용
        action = np.clip(action, -20, 20)
        
        action_idx = 0
        for robot_idx in range(self.num_robots):
            actuator_indices = self.robot_actuator_indices.get(robot_idx, [])
            robot_action_dim = len(actuator_indices)
            
            if action_idx + robot_action_dim <= len(action):
                robot_action = action[action_idx:action_idx + robot_action_dim]
                
                for i, actuator_idx in enumerate(actuator_indices):
                    if actuator_idx < len(self.data.ctrl) and i < len(robot_action):
                        self.data.ctrl[actuator_idx] = robot_action[i]
                
                action_idx += robot_action_dim
        
        # 물리 시뮬레이션 스텝
        mj.mj_step(self.model, self.data)
        
        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = False  # 시간 제한 없음 - 오직 넘어질 때만 에피소드 종료
        
        self.current_step += 1
        
        info = {
            'step': self.current_step,
            'num_robots': self.num_robots,
            'avg_height': np.mean([self.data.qpos[i*7 + 2] for i in range(self.num_robots) 
                                 if i*7 + 2 < len(self.data.qpos)])
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """렌더링"""
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # 카메라를 전체 필드가 보이도록 설정
                self.viewer.cam.distance = max(15.0, self.num_robots * 0.8)
                self.viewer.cam.elevation = -30
                self.viewer.cam.azimuth = 45
                self.viewer.cam.lookat[0] = 0
                self.viewer.cam.lookat[1] = 0
                self.viewer.cam.lookat[2] = 0.5
            self.viewer.sync()
    
    def close(self):
        """환경 정리"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        # 임시 파일 삭제
        if hasattr(self, 'model_path') and os.path.exists(self.model_path):
            os.unlink(self.model_path)


if __name__ == "__main__":
    # 테스트
    print("🤖 멀티 로봇 환경 테스트")
    
    env = MultiRobotGO2Env(num_robots=4, render_mode="human")
    
    obs, info = env.reset()
    print(f"관찰 크기: {obs.shape}")
    print(f"행동 크기: {env.action_space.shape}")
    
    for step in range(1000):
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if step % 100 == 0:
            print(f"스텝 {step}: 보상 {reward:.2f}, 평균 높이 {info.get('avg_height', 0):.2f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("테스트 완료!")