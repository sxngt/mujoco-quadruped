import mujoco as mj
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GO2ForwardEnv(gym.Env):
    """MuJoCo environment for training Unitree GO2 forward locomotion"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None):
        self.model_path = "go2_scene.xml"  # Use our custom scene with ground
        
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
        
        # === 총 보상 (전진이 압도적 비중) ===
        total_reward = (forward_reward +           # 최대 ~50+ (전진의 핵심)
                       survival_reward +          # ±50 (생존 필수)
                       direction_bonus +          # 0~2 (직진 보너스)
                       gait_reward +              # 0~3 (보행 패턴)
                       energy_penalty +           # 작은 페널티
                       stability_reward)          # 0~1 (안정성)
        
        return total_reward, {
            'forward': forward_reward,
            'survival': survival_reward,
            'direction': direction_bonus,
            'gait': gait_reward,
            'energy': energy_penalty,
            'stability': stability_reward,
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