#!/usr/bin/env python3
"""
간단한 멀티 로봇 환경
하나의 MuJoCo 시뮬레이션에서 여러 간단한 사족보행 로봇이 동시에 학습
"""

import mujoco as mj
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tempfile


class SimpleMultiRobotEnv(gym.Env):
    """
    하나의 시뮬레이션에서 여러 간단한 사족보행 로봇이 동시에 학습하는 환경
    """
    
    def __init__(self, num_robots=16, render_mode=None, robot_spacing=2.0):
        self.num_robots = num_robots
        self.robot_spacing = robot_spacing
        self.render_mode = render_mode
        self.viewer = None
        
        # 멀티 로봇 XML 생성
        self.model_path = self._create_multi_robot_xml()
        
        self.model = mj.MjModel.from_xml_path(self.model_path)
        self.data = mj.MjData(self.model)
        
        # 시뮬레이션 초기화
        mj.mj_forward(self.model, self.data)
        
        # 행동/관찰 공간 정의
        single_robot_action_dim = 12  # 각 로봇당 12개 관절
        single_robot_obs_dim = 25     # 간소화된 관찰
        
        self.action_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(num_robots * single_robot_action_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_robots * single_robot_obs_dim,),
            dtype=np.float32
        )
        
        # 에피소드 관련
        self.max_episode_steps = 1000
        self.current_step = 0
        
        print(f"🤖 {num_robots}개 간단한 로봇 멀티 환경 생성 완료")
        print(f"총 행동 차원: {self.action_space.shape[0]}")
        print(f"총 관찰 차원: {self.observation_space.shape[0]}")
    
    def _create_multi_robot_xml(self):
        """간단한 사족보행 로봇들의 XML 생성"""
        
        xml_content = f'''
        <mujoco model="multi_quadruped">
            <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
            
            <option gravity="0 0 -9.81" integrator="RK4" timestep="0.002"/>
            
            <worldbody>
                <!-- Ground -->
                <geom name="ground" type="plane" pos="0 0 0" size="50 50 0.1" 
                      rgba="0.5 0.5 0.5 1" friction="1.0 0.1 0.1"/>
                
                <!-- Light -->
                <light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" 
                       specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1"/>
        '''
        
        # 그리드 배치
        grid_size = int(np.ceil(np.sqrt(self.num_robots)))
        
        for robot_idx in range(self.num_robots):
            row = robot_idx // grid_size
            col = robot_idx % grid_size
            
            x_pos = (col - grid_size//2) * self.robot_spacing
            y_pos = (row - grid_size//2) * self.robot_spacing
            z_pos = 0.3
            
            # 로봇별 색상
            hue = robot_idx / self.num_robots
            r = 0.5 + 0.5 * np.sin(2 * np.pi * hue)
            g = 0.5 + 0.5 * np.sin(2 * np.pi * hue + 2 * np.pi / 3)
            b = 0.5 + 0.5 * np.sin(2 * np.pi * hue + 4 * np.pi / 3)
            
            xml_content += f'''
                <!-- Robot {robot_idx} -->
                <body name="robot_{robot_idx}" pos="{x_pos} {y_pos} {z_pos}">
                    <freejoint name="robot_{robot_idx}_freejoint"/>
                    
                    <!-- Main body -->
                    <geom name="body_{robot_idx}" type="box" size="0.2 0.1 0.05" 
                          rgba="{r:.3f} {g:.3f} {b:.3f} 1" mass="2"/>
                    
                    <!-- Front Left Leg -->
                    <body name="FL_hip_{robot_idx}" pos="0.15 0.08 -0.02">
                        <joint name="FL_hip_{robot_idx}" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="FL_hip_geom_{robot_idx}" type="cylinder" size="0.02 0.03" rgba="0.3 0.3 0.3 1"/>
                        
                        <body name="FL_thigh_{robot_idx}" pos="0 0 -0.08">
                            <joint name="FL_thigh_{robot_idx}" type="hinge" axis="0 1 0" range="-90 60"/>
                            <geom name="FL_thigh_geom_{robot_idx}" type="capsule" size="0.015 0.06" rgba="0.2 0.4 0.6 1"/>
                            
                            <body name="FL_calf_{robot_idx}" pos="0 0 -0.12">
                                <joint name="FL_calf_{robot_idx}" type="hinge" axis="0 1 0" range="-120 -20"/>
                                <geom name="FL_calf_geom_{robot_idx}" type="capsule" size="0.01 0.08" rgba="0.1 0.6 0.3 1"/>
                                <geom name="FL_foot_{robot_idx}" type="sphere" size="0.02" pos="0 0 -0.08" 
                                      rgba="0.8 0.2 0.2 1" friction="1.5 0.1 0.1"/>
                            </body>
                        </body>
                    </body>
                    
                    <!-- Front Right Leg -->
                    <body name="FR_hip_{robot_idx}" pos="0.15 -0.08 -0.02">
                        <joint name="FR_hip_{robot_idx}" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="FR_hip_geom_{robot_idx}" type="cylinder" size="0.02 0.03" rgba="0.3 0.3 0.3 1"/>
                        
                        <body name="FR_thigh_{robot_idx}" pos="0 0 -0.08">
                            <joint name="FR_thigh_{robot_idx}" type="hinge" axis="0 1 0" range="-90 60"/>
                            <geom name="FR_thigh_geom_{robot_idx}" type="capsule" size="0.015 0.06" rgba="0.2 0.4 0.6 1"/>
                            
                            <body name="FR_calf_{robot_idx}" pos="0 0 -0.12">
                                <joint name="FR_calf_{robot_idx}" type="hinge" axis="0 1 0" range="-120 -20"/>
                                <geom name="FR_calf_geom_{robot_idx}" type="capsule" size="0.01 0.08" rgba="0.1 0.6 0.3 1"/>
                                <geom name="FR_foot_{robot_idx}" type="sphere" size="0.02" pos="0 0 -0.08" 
                                      rgba="0.8 0.2 0.2 1" friction="1.5 0.1 0.1"/>
                            </body>
                        </body>
                    </body>
                    
                    <!-- Rear Left Leg -->
                    <body name="RL_hip_{robot_idx}" pos="-0.15 0.08 -0.02">
                        <joint name="RL_hip_{robot_idx}" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="RL_hip_geom_{robot_idx}" type="cylinder" size="0.02 0.03" rgba="0.3 0.3 0.3 1"/>
                        
                        <body name="RL_thigh_{robot_idx}" pos="0 0 -0.08">
                            <joint name="RL_thigh_{robot_idx}" type="hinge" axis="0 1 0" range="-90 60"/>
                            <geom name="RL_thigh_geom_{robot_idx}" type="capsule" size="0.015 0.06" rgba="0.2 0.4 0.6 1"/>
                            
                            <body name="RL_calf_{robot_idx}" pos="0 0 -0.12">
                                <joint name="RL_calf_{robot_idx}" type="hinge" axis="0 1 0" range="-120 -20"/>
                                <geom name="RL_calf_geom_{robot_idx}" type="capsule" size="0.01 0.08" rgba="0.1 0.6 0.3 1"/>
                                <geom name="RL_foot_{robot_idx}" type="sphere" size="0.02" pos="0 0 -0.08" 
                                      rgba="0.8 0.2 0.2 1" friction="1.5 0.1 0.1"/>
                            </body>
                        </body>
                    </body>
                    
                    <!-- Rear Right Leg -->
                    <body name="RR_hip_{robot_idx}" pos="-0.15 -0.08 -0.02">
                        <joint name="RR_hip_{robot_idx}" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="RR_hip_geom_{robot_idx}" type="cylinder" size="0.02 0.03" rgba="0.3 0.3 0.3 1"/>
                        
                        <body name="RR_thigh_{robot_idx}" pos="0 0 -0.08">
                            <joint name="RR_thigh_{robot_idx}" type="hinge" axis="0 1 0" range="-90 60"/>
                            <geom name="RR_thigh_geom_{robot_idx}" type="capsule" size="0.015 0.06" rgba="0.2 0.4 0.6 1"/>
                            
                            <body name="RR_calf_{robot_idx}" pos="0 0 -0.12">
                                <joint name="RR_calf_{robot_idx}" type="hinge" axis="0 1 0" range="-120 -20"/>
                                <geom name="RR_calf_geom_{robot_idx}" type="capsule" size="0.01 0.08" rgba="0.1 0.6 0.3 1"/>
                                <geom name="RR_foot_{robot_idx}" type="sphere" size="0.02" pos="0 0 -0.08" 
                                      rgba="0.8 0.2 0.2 1" friction="1.5 0.1 0.1"/>
                            </body>
                        </body>
                    </body>
                    
                </body>
            '''
        
        # 액추에이터 섹션
        xml_content += '''
            </worldbody>
            
            <actuator>
        '''
        
        for robot_idx in range(self.num_robots):
            for leg in ['FL', 'FR', 'RL', 'RR']:
                for joint in ['hip', 'thigh', 'calf']:
                    xml_content += f'''
                        <motor name="{leg}_{joint}_motor_{robot_idx}" 
                               joint="{leg}_{joint}_{robot_idx}" 
                               gear="100" ctrllimited="true" ctrlrange="-5 5"/>
                    '''
        
        xml_content += '''
            </actuator>
        </mujoco>
        '''
        
        # 임시 파일로 저장
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        temp_file.write(xml_content)
        temp_file.close()
        
        return temp_file.name
    
    def _get_robot_observation(self, robot_idx):
        """특정 로봇의 관찰값"""
        # 로봇의 freejoint 위치와 속도
        freejoint_qpos_start = robot_idx * 7
        freejoint_qvel_start = robot_idx * 6
        
        # Body state
        body_pos = self.data.qpos[freejoint_qpos_start:freejoint_qpos_start+3]
        body_quat = self.data.qpos[freejoint_qpos_start+3:freejoint_qpos_start+7]
        body_linvel = self.data.qvel[freejoint_qvel_start:freejoint_qvel_start+3]
        body_angvel = self.data.qvel[freejoint_qvel_start+3:freejoint_qvel_start+6]
        
        # Joint positions (12개 관절)
        joint_start = self.num_robots * 7 + robot_idx * 12
        joint_pos = self.data.qpos[joint_start:joint_start+12]
        
        return np.concatenate([
            body_pos,       # 3
            body_quat,      # 4  
            body_linvel,    # 3
            body_angvel,    # 3
            joint_pos       # 12
        ])  # 총 25차원
    
    def _get_observation(self):
        """모든 로봇의 관찰값"""
        observations = []
        for robot_idx in range(self.num_robots):
            obs = self._get_robot_observation(robot_idx)
            observations.append(obs)
        return np.concatenate(observations)
    
    def _get_robot_reward(self, robot_idx):
        """특정 로봇의 보상"""
        freejoint_qpos_start = robot_idx * 7
        freejoint_qvel_start = robot_idx * 6
        
        # 전진 속도
        forward_vel = self.data.qvel[freejoint_qvel_start]
        
        # 로봇 높이 
        body_height = self.data.qpos[freejoint_qpos_start + 2]
        
        # 측면 속도
        lateral_vel = abs(self.data.qvel[freejoint_qvel_start + 1])
        
        # 보상 계산
        forward_reward = forward_vel * 10.0 if forward_vel > 0 else forward_vel * 2.0
        survival_reward = 3.0 if body_height > 0.15 else -10.0
        direction_penalty = -lateral_vel * 3.0
        
        return forward_reward + survival_reward + direction_penalty
    
    def _get_reward(self):
        """모든 로봇의 평균 보상"""
        total_reward = 0.0
        for robot_idx in range(self.num_robots):
            total_reward += self._get_robot_reward(robot_idx)
        return total_reward / self.num_robots
    
    def _is_terminated(self):
        """종료 조건"""
        fallen_count = 0
        for robot_idx in range(self.num_robots):
            freejoint_qpos_start = robot_idx * 7
            body_height = self.data.qpos[freejoint_qpos_start + 2]
            if body_height < 0.1:
                fallen_count += 1
        
        return fallen_count > self.num_robots // 2
    
    def reset(self, seed=None, options=None):
        """환경 리셋"""
        super().reset(seed=seed)
        
        # 모든 상태 초기화
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        
        # 각 로봇 초기 위치 설정
        grid_size = int(np.ceil(np.sqrt(self.num_robots)))
        
        for robot_idx in range(self.num_robots):
            row = robot_idx // grid_size
            col = robot_idx % grid_size
            
            freejoint_qpos_start = robot_idx * 7
            
            # 위치
            x_pos = (col - grid_size//2) * self.robot_spacing + np.random.normal(0, 0.05)
            y_pos = (row - grid_size//2) * self.robot_spacing + np.random.normal(0, 0.05)  
            z_pos = 0.3 + np.random.normal(0, 0.02)
            
            self.data.qpos[freejoint_qpos_start:freejoint_qpos_start+3] = [x_pos, y_pos, z_pos]
            
            # 쿼터니언 (w, x, y, z)
            self.data.qpos[freejoint_qpos_start+3:freejoint_qpos_start+7] = [1, 0, 0, 0]
        
        self.current_step = 0
        mj.mj_forward(self.model, self.data)
        
        observation = self._get_observation()
        info = {'num_robots': self.num_robots}
        
        return observation, info
    
    def step(self, action):
        """환경 스텝"""
        # 행동 클리핑
        action = np.clip(action, -5, 5)
        
        # 각 로봇의 액추에이터에 행동 적용
        for robot_idx in range(self.num_robots):
            robot_action_start = robot_idx * 12
            robot_action = action[robot_action_start:robot_action_start+12]
            
            actuator_start = robot_idx * 12
            self.data.ctrl[actuator_start:actuator_start+12] = robot_action
        
        # 물리 시뮬레이션
        mj.mj_step(self.model, self.data)
        
        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        self.current_step += 1
        
        # 평균 높이 계산
        avg_height = np.mean([self.data.qpos[i*7 + 2] for i in range(self.num_robots)])
        
        info = {
            'step': self.current_step,
            'num_robots': self.num_robots,
            'avg_height': avg_height
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """렌더링"""
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # 전체 필드가 보이도록 카메라 설정
                distance = max(20.0, self.num_robots * 0.5)
                self.viewer.cam.distance = distance
                self.viewer.cam.elevation = -20
                self.viewer.cam.azimuth = 45
                self.viewer.cam.lookat[0] = 0
                self.viewer.cam.lookat[1] = 0
                self.viewer.cam.lookat[2] = 0.3
            self.viewer.sync()
    
    def close(self):
        """환경 정리"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        # 임시 파일 삭제
        if hasattr(self, 'model_path'):
            import os
            if os.path.exists(self.model_path):
                os.unlink(self.model_path)


if __name__ == "__main__":
    # 테스트
    print("🤖 간단한 멀티 로봇 환경 테스트")
    
    env = SimpleMultiRobotEnv(num_robots=9, render_mode="human")
    
    obs, info = env.reset()
    print(f"관찰 크기: {obs.shape}")
    print(f"행동 크기: {env.action_space.shape}")
    
    for step in range(2000):
        # 랜덤 행동 (약간의 전진 편향)
        action = np.random.uniform(-0.5, 0.5, env.action_space.shape[0])
        # 전진을 위한 편향 추가
        for robot_idx in range(env.num_robots):
            base_idx = robot_idx * 12
            action[base_idx+3] += 0.5   # FR thigh
            action[base_idx+9] += 0.5   # RR thigh
            action[base_idx+4] -= 0.3   # FR calf  
            action[base_idx+10] -= 0.3  # RR calf
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if step % 200 == 0:
            print(f"스텝 {step}: 보상 {reward:.2f}, 평균 높이 {info.get('avg_height', 0):.2f}")
        
        if terminated or truncated:
            print(f"에피소드 종료! 스텝 {step}")
            obs, info = env.reset()
    
    env.close()
    print("✅ 테스트 완료!")