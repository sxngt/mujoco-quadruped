import mujoco as mj
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from common.gait_generator import GaitGenerator, CyclicGaitReward


class GO2ForwardEnv(gym.Env):
    """MuJoCo environment for training Unitree GO2 forward locomotion"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None, use_reference_gait=True):
        # XML 파일 경로 설정 (assets 폴더에서 찾기)
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # rl 디렉토리
        self.model_path = os.path.join(base_dir, "assets", "go2_scene.xml")
        
        self.model = mj.MjModel.from_xml_path(self.model_path)
        self.data = mj.MjData(self.model)
        
        # Initialize simulation state
        mj.mj_forward(self.model, self.data)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # Get actuated joint indices
        self.actuated_joint_ids = []
        
        # Get number of actuated joints from model
        self.n_actions = self.model.nu  # Number of actuators
        
        # Get torque limits from model
        if self.model.actuator_forcerange is not None:
            torque_limits = self.model.actuator_forcerange[:, 1]
        else:
            torque_limits = np.full(self.n_actions, 20.0)  # Default fallback
            
        self.action_space = spaces.Box(
            low=-torque_limits.astype(np.float32), 
            high=torque_limits.astype(np.float32), 
            shape=(self.n_actions,), dtype=np.float32
        )
        
        # Observation space: joint positions, velocities, body orientation, velocity
        n_joints = self.model.nq - 7  # Subtract 7 for free joint (3 pos + 4 quat)
        n_velocities = self.model.nv - 6  # Subtract 6 for free joint velocities
        obs_dim = n_joints + n_velocities + 4 + 3 + 3  # pos + vel + quat + ang_vel + lin_vel
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32), 
            high=np.full(obs_dim, np.inf, dtype=np.float32), 
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Store initial position from keyframe (standing pose)
        home_keyframe = 0  # Index of 'home' keyframe
        self.initial_qpos = self.model.key_qpos[home_keyframe].copy()
        self.initial_qvel = np.zeros(self.model.nv)
        
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Contact tracking for foot contact detection
        self.foot_geom_ids = []
        for name in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']:
            self.foot_geom_ids.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name))
        
        # Gait guidance system (선택적)
        self.use_reference_gait = use_reference_gait
        if self.use_reference_gait:
            self.gait_generator = GaitGenerator(gait_type="trot", frequency=1.5)
            self.gait_reward_calculator = CyclicGaitReward(target_frequency=1.5)
        self.simulation_time = 0.0
        
    def _get_observation(self):
        # Joint positions and velocities (excluding free joint)
        n_joints = self.model.nq - 7
        n_velocities = self.model.nv - 6
        
        joint_pos = self.data.qpos[7:7+n_joints].copy()  # Skip free joint position/orientation
        joint_vel = self.data.qvel[6:6+n_velocities].copy()  # Skip free joint velocity
        
        # Body orientation (quaternion) and angular velocity
        body_quat = self.data.qpos[3:7].copy()  # Quaternion [w, x, y, z]
        body_angvel = self.data.qvel[3:6].copy()  # Angular velocity
        
        # Linear velocity
        body_linvel = self.data.qvel[0:3].copy()  # Linear velocity
        
        return np.concatenate([
            joint_pos, joint_vel, body_quat, body_angvel, body_linvel
        ])
    
    def _get_contact_info(self):
        """Get information about foot contacts with ground"""
        contacts = {}
        foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        
        for i, foot_name in enumerate(foot_names):
            if i < len(self.foot_geom_ids):
                geom_id = self.foot_geom_ids[i]
                in_contact = False
                contact_force = 0.0
                
                # Check all contacts
                for j in range(self.data.ncon):
                    contact = self.data.contact[j]
                    if contact.geom1 == geom_id or contact.geom2 == geom_id:
                        in_contact = True
                        # Get contact force
                        contact_force += np.linalg.norm(contact.frame[0:3])  # Normal force
                        break
                
                contacts[foot_name] = {
                    'in_contact': in_contact,
                    'force': contact_force
                }
        
        return contacts
    
    def _get_reward(self):
        # === 최우선 목표: 전진 보행 ===
        forward_vel = self.data.qvel[0]  # x-velocity
        # 전진 속도에 따른 기하급수적 보상 (최대 목표)
        if forward_vel > 0:
            forward_reward = forward_vel * 20.0  # 대폭 강화!
            # 빠른 전진에 보너스 (0.5m/s 이상시)
            if forward_vel > 0.5:
                forward_reward += (forward_vel - 0.5) * 30.0
        else:
            forward_reward = forward_vel * 5.0  # 뒤로 가면 페널티
        
        # === 전진을 위한 필수 조건들 (보조 역할) ===
        
        # 1. 기본 생존: 넘어지지 않기 (전진의 전제조건)
        body_height = self.data.qpos[2]
        if body_height < 0.15:
            survival_reward = -50.0  # 넘어지면 큰 페널티
        else:
            survival_reward = 2.0  # 서있으면 기본 보상
        
        # 2. 전진 방향 유지 (옆으로 벗어나지 않기)
        lateral_vel = abs(self.data.qvel[1])  # y-velocity
        direction_bonus = max(0, 2.0 - lateral_vel * 10.0)  # 직진할수록 보너스
        
        # 3. 보행 패턴 장려 (발 접촉 다양성)
        contacts = self._get_contact_info()
        num_contacts = sum(1 for contact in contacts.values() if contact['in_contact'])
        
        # 전진시에만 보행 패턴 보상
        if forward_vel > 0.1:  # 전진할 때만
            if 1 <= num_contacts <= 3:  # 적절한 보행 패턴
                gait_reward = 3.0
            elif num_contacts == 0:  # 점프는 위험하지만 때로 필요
                gait_reward = -1.0
            else:  # 4발 모두 땅에 (정적)
                gait_reward = 1.0
        else:
            gait_reward = 0.0  # 전진 안하면 보행 패턴 신경쓰지 않음
        
        # 4. 에너지 효율성 (너무 많은 힘 쓰지 않기)
        energy_penalty = -0.001 * np.sum(np.square(self.current_action))
        
        # 5. 안정성 (과도한 회전 방지)
        body_quat = self.data.qpos[3:7]
        z_axis = np.array([2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
                          2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
                          body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2])
        stability_reward = max(0, z_axis[2]) * 1.0  # 직립 보너스 (적당히)
        
        # 6. 점프/호핑 방지 페널티
        vertical_vel = abs(self.data.qvel[2])  # z축 속도
        hop_penalty = -10.0 * max(0, vertical_vel - 0.15)  # 과도한 수직 움직임 페널티
        
        # 7. 다리 균형 사용 (앞뒤 다리 균등 사용)
        if hasattr(self, 'current_action') and len(self.current_action) >= 12:
            # 앞다리 토크 (FL, FR)
            front_torques = np.abs(self.current_action[0:6])
            # 뒷다리 토크 (RL, RR) 
            rear_torques = np.abs(self.current_action[6:12])
            # 뒷다리만 과도하게 사용하면 페널티
            leg_imbalance = np.mean(rear_torques) - np.mean(front_torques)
            leg_balance_penalty = -5.0 * max(0, leg_imbalance - 0.3)
        else:
            leg_balance_penalty = 0.0
        
        # 8. 관절 안전성 페널티 (과도한 관절 굽힘 방지)
        joint_angles = self.data.qpos[7:19]  # 12개 관절
        joint_safety_penalty = 0.0
        
        # 각 관절의 허용 범위와 현재 각도 확인
        for i in range(len(joint_angles)):
            joint_idx = i + 1  # Free joint 다음부터
            if joint_idx < self.model.njnt:
                joint_range = self.model.jnt_range[joint_idx]
                current_angle = joint_angles[i]
                
                # 관절 범위의 80% 이상 사용하면 페널티
                range_center = (joint_range[0] + joint_range[1]) / 2
                range_width = joint_range[1] - joint_range[0]
                safe_range = range_width * 0.8
                
                # 안전 범위를 벗어난 정도에 따라 점진적 페널티
                danger_threshold_low = joint_range[0] + 0.05 * range_width
                danger_threshold_high = joint_range[1] - 0.05 * range_width
                
                if current_angle < danger_threshold_low:
                    excess = abs(current_angle - danger_threshold_low)
                    joint_safety_penalty += -2.0 * excess  # 완만한 페널티
                elif current_angle > danger_threshold_high:
                    excess = abs(current_angle - danger_threshold_high) 
                    joint_safety_penalty += -2.0 * excess  # 완만한 페널티
        
        # 9. 급격한 관절 움직임 페널티 (부드러운 움직임 장려)
        joint_velocities = self.data.qvel[6:18]  # 12개 관절 속도
        violent_motion_penalty = -0.01 * np.sum(np.square(joint_velocities))  # 약한 페널티
        
        # 10. 부드러운 보행 보상 (지면 접촉 유지)
        min_contact_reward = 2.0 if num_contacts >= 2 else -3.0
        
        # 11. 참조 동작 모방 보상 (선택적)
        if self.use_reference_gait:
            target_angles, target_contacts = self.gait_generator.get_joint_targets(self.simulation_time)
            
            # 관절 각도 유사성 보상
            current_angles = self.data.qpos[7:19]  # 12개 관절
            angle_diff = np.abs(current_angles - target_angles)
            angle_similarity = np.exp(-angle_diff.mean() * 5.0) * 5.0  # 유사할수록 높은 보상
            
            # 발 접촉 패턴 유사성
            current_contacts = [contact['in_contact'] for contact in contacts.values()]
            contact_match = np.sum(np.array(current_contacts) == target_contacts)
            contact_similarity = contact_match * 1.0  # 매칭되는 발당 1점
            
            # 주기적 보행 보상
            gait_rhythm = self.gait_reward_calculator.compute_gait_reward(
                np.array(current_contacts), dt=0.002
            )
        else:
            # 참조 보행 없이 기본 강화학습
            angle_similarity = 0.0
            contact_similarity = 0.0
            gait_rhythm = 0.0
        
        # === 총 보상 (전진 + 안전한 걷기 패턴) ===
        total_reward = (forward_reward +           # 최대 ~50+ (전진의 핵심)
                       survival_reward +          # ±50 (생존 필수)
                       direction_bonus +          # 0~2 (직진 보너스)
                       gait_reward +              # 0~3 (보행 패턴)
                       energy_penalty +           # 작은 페널티
                       stability_reward +         # 0~1 (안정성)
                       hop_penalty +              # 점프 방지
                       leg_balance_penalty +      # 다리 균형 사용
                       joint_safety_penalty +     # 관절 안전성 (새로 추가)
                       violent_motion_penalty +   # 급격한 움직임 방지 (새로 추가)
                       min_contact_reward +       # 지면 접촉 유지
                       angle_similarity +         # 0~5 (참조 동작 모방)
                       contact_similarity +       # 0~4 (발 접촉 패턴)
                       gait_rhythm)               # 주기적 보행
        
        return total_reward, {
            'forward': forward_reward,
            'survival': survival_reward,
            'direction': direction_bonus,
            'gait': gait_reward,
            'energy': energy_penalty,
            'stability': stability_reward,
            'hop_penalty': hop_penalty,
            'leg_balance': leg_balance_penalty,
            'joint_safety': joint_safety_penalty,
            'violent_motion': violent_motion_penalty,
            'contact': min_contact_reward,
            'angle_imitation': angle_similarity,
            'contact_imitation': contact_similarity,
            'rhythm': gait_rhythm,
            'total': total_reward
        }
    
    def _is_terminated(self):
        # === 전진 학습을 위한 관대한 종료 조건 ===
        
        body_height = self.data.qpos[2]
        body_quat = self.data.qpos[3:7]  # [w, x, y, z]
        
        # 1. 완전히 넘어졌을 때만 종료 (학습 기회 최대화)
        if body_height < 0.10:  # 매우 관대한 높이 제한
            return True
            
        # 2. 뒤집어졌을 때만 종료
        z_axis = np.array([2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
                          2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
                          body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2])
        
        if z_axis[2] < -0.1:  # 거의 뒤집어진 상태
            return True
        
        # 3. 너무 멀리 옆으로 벗어났을 때
        if abs(self.data.qpos[1]) > 5.0:  # 매우 관대한 측면 제한
            return True
        
        # 4. 뒤로 너무 많이 갔을 때
        if self.data.qpos[0] < -2.0:  # 뒤로 2m 이상 가면 종료
            return True
            
        return False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset to initial state with small random perturbations
        self.data.qpos[:] = self.initial_qpos.copy()
        self.data.qvel[:] = self.initial_qvel.copy()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Add small random perturbations to joint positions only
        joint_noise = np.random.normal(0, 0.05, self.model.nq - 7)  # Exclude free joint
        self.data.qpos[7:] += joint_noise
        
        # Add small random perturbations to initial position
        self.data.qpos[0] += np.random.normal(0, 0.02)  # x position
        self.data.qpos[1] += np.random.normal(0, 0.02)  # y position
        self.data.qpos[2] += np.random.normal(0, 0.01)  # z position (height)
        
        # Small random orientation perturbation
        quat_noise = np.random.normal(0, 0.01, 4)
        quat_noise[0] = 1.0 + quat_noise[0]  # w component should be close to 1
        quat_noise = quat_noise / np.linalg.norm(quat_noise)  # Normalize
        self.data.qpos[3:7] = quat_noise
        
        self.current_step = 0
        self.current_action = np.zeros(self.n_actions)
        self.simulation_time = 0.0  # 시뮬레이션 시간 리셋
        
        # Set the robot properly on the ground
        mj.mj_forward(self.model, self.data)
        
        observation = self._get_observation()
        info = {'contacts': self._get_contact_info()}
        
        return observation, info
    
    def step(self, action):
        self.current_action = np.clip(action, -20, 20)
        
        # Apply actions to actuated joints
        self.data.ctrl[:] = self.current_action
        
        # Simulate physics with multiple substeps for stability
        for _ in range(1):  # Single step is fine with proper timestep
            mj.mj_step(self.model, self.data)
        
        observation = self._get_observation()
        reward, reward_info = self._get_reward()
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        self.current_step += 1
        self.simulation_time += 0.002  # MuJoCo timestep
        
        info = {**reward_info, 'step': self.current_step, 'contacts': self._get_contact_info()}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # 카메라 위치를 로봇 측후방에서 관찰하도록 설정
                self.viewer.cam.distance = 3.5  # 거리
                self.viewer.cam.elevation = -25  # 각도 (위에서 아래로)
                self.viewer.cam.azimuth = 135   # 방위각 (측후방 45도)
                self.viewer.cam.lookat[0] = 0   # x축 중심
                self.viewer.cam.lookat[1] = 0   # y축 중심 
                self.viewer.cam.lookat[2] = 0.3 # z축 중심 (로봇 높이)
            else:
                # 로봇을 따라다니는 카메라
                robot_x = self.data.qpos[0]
                robot_y = self.data.qpos[1]
                self.viewer.cam.lookat[0] = robot_x
                self.viewer.cam.lookat[1] = robot_y
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mj.Renderer(self.model, width=1024, height=768)
            self.viewer.update_scene(self.data)
            return self.viewer.render()
        elif self.render_mode == "depth_array":
            if self.viewer is None:
                self.viewer = mj.Renderer(self.model, width=1024, height=768)
            self.viewer.update_scene(self.data)
            return self.viewer.render(depth=True)
        return None
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None