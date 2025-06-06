import mujoco as mj
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gait_generator import GaitGenerator, CyclicGaitReward
from collections import deque


class ImprovedGO2Env(gym.Env):
    """개선된 MuJoCo GO2 환경 - Isaac Lab RSL-RL 기법 적용"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None, use_reference_gait=True):
        self.model_path = "go2_scene.xml"
        
        self.model = mj.MjModel.from_xml_path(self.model_path)
        self.data = mj.MjData(self.model)
        
        # Initialize simulation state
        mj.mj_forward(self.model, self.data)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # Control mode: position control instead of torque
        self.n_actions = self.model.nu  # Number of actuators
        
        # 관절 위치 제어를 위한 설정
        self.joint_position_limits = self.model.jnt_range[1:self.n_actions+1].copy()
        self.default_joint_pos = np.zeros(self.n_actions)
        
        # 액션 공간: 관절 위치 명령 (-1 to 1 normalized)
        self.action_space = spaces.Box(
            low=-np.ones(self.n_actions, dtype=np.float32),
            high=np.ones(self.n_actions, dtype=np.float32),
            shape=(self.n_actions,), dtype=np.float32
        )
        
        # 현실적인 물리 설정
        self._setup_realistic_physics()
        
        # 관찰 공간 개선: 이전 액션 + 관절 이력 포함
        n_joints = self.model.nq - 7
        n_velocities = self.model.nv - 6
        # pos + vel + quat + ang_vel + lin_vel + prev_action + contact_forces
        obs_dim = n_joints + n_velocities + 4 + 3 + 3 + self.n_actions + 4
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            shape=(obs_dim,), dtype=np.float32
        )
        
        # 참조 보행 자세 초기화 설정
        self.standing_height = 0.28  # 목표 서있는 높이
        self.init_joint_pos = np.array([
            # FL (앞왼쪽): hip, thigh, calf
            0.0, 0.8, -1.6,
            # FR (앞오른쪽): hip, thigh, calf  
            0.0, 0.8, -1.6,
            # RL (뒤왼쪽): hip, thigh, calf
            0.0, 0.8, -1.6,
            # RR (뒤오른쪽): hip, thigh, calf
            0.0, 0.8, -1.6
        ])
        
        # 액션 스무싱을 위한 버퍼
        self.action_history = deque(maxlen=3)
        self.prev_action = np.zeros(self.n_actions)
        self.action_smoothing_alpha = 0.7  # 스무싱 강도
        
        # PD 제어기 게인 (현실적인 물리를 위한 조정)
        self.kp = 30.0  # Proportional gain (현실적으로 감소)
        self.kd = 0.8   # Derivative gain (현실적으로 감소)
        
        # 발 접촉 추적
        self.foot_geom_ids = []
        for name in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']:
            self.foot_geom_ids.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name))
        
        # 발 접촉 이력 (발 공중 시간 계산용)
        self.foot_contact_history = {i: deque(maxlen=50) for i in range(4)}
        self.last_contact_state = np.zeros(4, dtype=bool)
        
        # 보상 함수 가중치 (현실적인 물리를 위한 조정)
        self.reward_weights = {
            'forward_velocity': 15.0,  # 전진 보상 약간 감소
            'survival': 2.0,           # 생존 보상 증가
            'orientation': 5.0,        # 자세 유지 중요도 증가
            'base_height': 8.0,        # 높이 유지 중요도 증가
            'feet_air_time': 1.0,      # 발 공중시간 보상 감소 (더 현실적)
            'action_smoothness': 0.1,  # 부드러운 움직임 중요도 증가
            'energy': 0.002,           # 에너지 효율성 중요도 증가
            'lateral_velocity': -6.0,  # 측면 이동 페널티 증가
            'angular_velocity': -1.0,  # 회전 페널티 증가
            'joint_acceleration': -5e-4, # 관절 가속도 페널티 증가
            'feet_stumble': -3.0,      # 발 걸림 페널티 증가
            'joint_limits': -8.0,      # 관절 한계 페널티 증가
            'gravity_compensation': 3.0, # 중력 보상 추가
        }
        
        self.max_episode_steps = 5000  # 더 긴 에피소드 (원래 1000)
        self.current_step = 0
        self.dt = self.model.opt.timestep
        
        # Gait guidance
        self.use_reference_gait = use_reference_gait
        if self.use_reference_gait:
            self.gait_generator = GaitGenerator(gait_type="trot", frequency=1.5)
            self.gait_reward_calculator = CyclicGaitReward(target_frequency=1.5)
        self.simulation_time = 0.0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # 참조 보행 자세로 초기화
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        
        # 로봇 위치 설정
        self.data.qpos[0] = 0.0  # x
        self.data.qpos[1] = 0.0  # y
        self.data.qpos[2] = self.standing_height  # z (서있는 높이)
        
        # 방향 (quaternion w,x,y,z)
        self.data.qpos[3] = 1.0  # w
        self.data.qpos[4] = 0.0  # x
        self.data.qpos[5] = 0.0  # y
        self.data.qpos[6] = 0.0  # z
        
        # 관절 위치를 참조 자세로 설정
        joint_start_idx = 7
        self.data.qpos[joint_start_idx:joint_start_idx+self.n_actions] = self.init_joint_pos
        
        # 약간의 무작위성 추가 (안정성 유지)
        if self.np_random is not None:
            # 위치에 작은 노이즈
            self.data.qpos[0:2] += self.np_random.uniform(-0.05, 0.05, 2)
            # 관절에 작은 노이즈
            self.data.qpos[joint_start_idx:joint_start_idx+self.n_actions] += \
                self.np_random.uniform(-0.1, 0.1, self.n_actions)
        
        # 시뮬레이션 전진
        mj.mj_forward(self.model, self.data)
        
        # 현실적인 물리 설정 적용
        self._apply_physics_settings()
        
        # 상태 초기화
        self.current_step = 0
        self.simulation_time = 0.0
        self.action_history.clear()
        self.prev_action = np.zeros(self.n_actions)
        
        # 발 접촉 이력 초기화
        for i in range(4):
            self.foot_contact_history[i].clear()
        self.last_contact_state = self._get_foot_contacts()
        
        return self._get_observation(), {}
    
    def _get_foot_contacts(self):
        """각 발의 접촉 상태 반환"""
        contacts = np.zeros(4, dtype=bool)
        
        for j in range(self.data.ncon):
            contact = self.data.contact[j]
            for i, geom_id in enumerate(self.foot_geom_ids):
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    contacts[i] = True
        
        return contacts
    
    def _get_contact_forces(self):
        """각 발의 접촉력 반환"""
        forces = np.zeros(4)
        
        for j in range(self.data.ncon):
            contact = self.data.contact[j]
            for i, geom_id in enumerate(self.foot_geom_ids):
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    # 수직 접촉력
                    forces[i] += np.abs(contact.frame[2])
        
        return forces
    
    def _get_observation(self):
        # 기본 관절 정보
        n_joints = self.model.nq - 7
        n_velocities = self.model.nv - 6
        
        joint_pos = self.data.qpos[7:7+n_joints].copy()
        joint_vel = self.data.qvel[6:6+n_velocities].copy()
        
        # 몸체 상태
        body_quat = self.data.qpos[3:7].copy()
        body_angvel = self.data.qvel[3:6].copy()
        body_linvel = self.data.qvel[0:3].copy()
        
        # 이전 액션
        prev_action = self.prev_action.copy()
        
        # 발 접촉력
        contact_forces = self._get_contact_forces()
        
        return np.concatenate([
            joint_pos, joint_vel, body_quat, body_angvel, body_linvel,
            prev_action, contact_forces
        ])
    
    def _compute_pd_torque(self, target_pos, current_pos, current_vel):
        """PD 제어기로 토크 계산 (현실적인 토크 제한)"""
        # 입력값 클리핑 (더 작은 범위)
        target_pos = np.clip(target_pos, -3.0, 3.0)
        current_pos = np.clip(current_pos, -3.0, 3.0)
        current_vel = np.clip(current_vel, -30.0, 30.0)
        
        pos_error = target_pos - current_pos
        pos_error = np.clip(pos_error, -0.5, 0.5)  # 더 작은 오차 제한
        
        torque = self.kp * pos_error - self.kd * current_vel
        
        # 토크 한계 더 엄격하게 (현실적인 값)
        max_torque = np.abs(self.model.actuator_forcerange[:, 1])
        max_torque = np.where(max_torque > 0, max_torque, 15.0)  # 기본값 15Nm
        max_torque = np.minimum(max_torque, 25.0)  # 최대 25Nm로 제한
        
        return np.clip(torque, -max_torque, max_torque)
    
    def step(self, action):
        # 액션 스무싱
        if len(self.action_history) > 0:
            smoothed_action = (self.action_smoothing_alpha * action + 
                             (1 - self.action_smoothing_alpha) * self.action_history[-1])
        else:
            smoothed_action = action
        
        self.action_history.append(smoothed_action)
        
        # 정규화된 액션을 관절 위치 명령으로 변환
        joint_ranges = self.joint_position_limits
        joint_mid = (joint_ranges[:, 0] + joint_ranges[:, 1]) / 2
        joint_span = (joint_ranges[:, 1] - joint_ranges[:, 0]) / 2
        
        # 현재 관절 위치 기준 상대적 변화 (더 작은 스케일링)
        current_joint_pos = self.data.qpos[7:7+self.n_actions]
        target_joint_pos = current_joint_pos + smoothed_action * 0.02  # 더더 작은 델타
        
        # 관절 한계 내로 클리핑
        target_joint_pos = np.clip(target_joint_pos, joint_ranges[:, 0], joint_ranges[:, 1])
        
        # PD 제어로 토크 계산
        current_joint_vel = self.data.qvel[6:6+self.n_actions]
        torques = self._compute_pd_torque(target_joint_pos, current_joint_pos, current_joint_vel)
        
        # 토크 적용
        self.data.ctrl[:] = torques
        self.current_action = torques  # 보상 계산용
        
        # 시뮬레이션 스텝
        mj.mj_step(self.model, self.data)
        
        # 상태 업데이트
        self.current_step += 1
        self.simulation_time += self.dt
        self.prev_action = smoothed_action
        
        # 발 접촉 이력 업데이트
        current_contacts = self._get_foot_contacts()
        for i in range(4):
            self.foot_contact_history[i].append(current_contacts[i])
        
        # 관찰, 보상, 종료 계산
        observation = self._get_observation()
        reward, reward_info = self._compute_modular_reward(smoothed_action, current_contacts)
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        # 렌더링
        if self.render_mode == "human":
            self.render()
        
        self.last_contact_state = current_contacts
        
        return observation, reward, terminated, truncated, reward_info
    
    def _setup_realistic_physics(self):
        """현실적인 물리 시뮬레이션과 시각적 설정"""
        # 중력 설정 확인 (지구 중력: -9.81 m/s²)
        if hasattr(self.model, 'opt') and hasattr(self.model.opt, 'gravity'):
            self.model.opt.gravity[2] = -9.81
        
        # 시뮬레이션 시간 스텝 (기본값 사용)
        if hasattr(self.model, 'opt') and hasattr(self.model.opt, 'timestep'):
            self.model.opt.timestep = 0.002  # 기본값 유지
        
        # 솔버 설정 (안정성 우선)
        if hasattr(self.model, 'opt'):
            self.model.opt.iterations = 20  # 적당한 반복
            self.model.opt.ls_iterations = 6   # 기본값
        
        # 접촉 설정 개선 (바닥 침투 방지)
        if hasattr(self.model, 'opt'):
            self.model.opt.tolerance = 1e-6      # 더 엄격한 허용오차
            self.model.opt.impratio = 1.0        # 개선 비율
        
        # 바닥 접촉 물리 설정
        self._setup_contact_physics()
    
    def _setup_contact_physics(self):
        """바닥 접촉 물리 설정"""
        # 바닥과 발의 접촉 파라미터 조정
        for i in range(self.model.ngeom):
            geom_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i)
            if geom_name:
                # 발 기하체에 대한 접촉 설정
                if 'foot' in geom_name:
                    # 발의 마찰 계수 증가
                    if hasattr(self.model, 'geom_friction'):
                        self.model.geom_friction[i, 0] = 1.0   # 마찰 계수
                        self.model.geom_friction[i, 1] = 0.1   # 롤링 마찰
                        self.model.geom_friction[i, 2] = 0.1   # 비틀림 마찰
                    
                    # 발의 반발력 설정
                    if hasattr(self.model, 'geom_solimp'):
                        self.model.geom_solimp[i, 0] = 0.9    # 반발 계수
                        self.model.geom_solimp[i, 1] = 0.95   # 반발 안정성
                    
                    # 접촉 강성 설정
                    if hasattr(self.model, 'geom_solref'):
                        self.model.geom_solref[i, 0] = 0.01   # 시간 상수
                        self.model.geom_solref[i, 1] = 1.0    # 댐핑 비율
                
                # 바닥 기하체에 대한 설정
                elif 'ground' in geom_name or 'floor' in geom_name:
                    if hasattr(self.model, 'geom_friction'):
                        self.model.geom_friction[i, 0] = 1.2   # 바닥 마찰
    
    def _apply_physics_settings(self):
        """매 에피소드마다 적용할 물리 설정"""
        # 발의 위치가 바닥 아래로 가지 않도록 보정
        foot_height_threshold = 0.02  # 2cm
        
        for i, foot_geom_id in enumerate(self.foot_geom_ids):
            # 발의 현재 위치 확인
            foot_pos = self.data.geom_xpos[foot_geom_id]
            if foot_pos[2] < foot_height_threshold:  # z 좌표가 너무 낮으면
                # 발을 살짝 위로 올려줌 (물리적으로 자연스럽게)
                pass  # MuJoCo 내부 접촉 해결기에 맡김
        
        # 관절 마찰 설정
        if hasattr(self.model, 'dof_frictionloss'):
            self.model.dof_frictionloss[:] = 0.05  # 적당한 관절 마찰
    
    def _compute_modular_reward(self, action, current_contacts):
        """모듈화된 보상 함수"""
        rewards = {}
        
        # 1. 전진 속도 보상
        forward_vel = np.clip(self.data.qvel[0], -10.0, 10.0)  # 속도 클리핑
        rewards['forward_velocity'] = forward_vel * self.reward_weights['forward_velocity']
        
        # 2. 생존 보상
        rewards['survival'] = self.reward_weights['survival']
        
        # 3. 방향 유지 보상
        body_quat = self.data.qpos[3:7]
        z_axis = np.array([
            2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
            2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
            body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2
        ])
        orientation_error = 1.0 - z_axis[2]
        rewards['orientation'] = -orientation_error * self.reward_weights['orientation']
        
        # 4. 기본 높이 유지
        height_error = abs(self.data.qpos[2] - self.standing_height)
        rewards['base_height'] = -height_error * self.reward_weights['base_height']
        
        # 5. 발 공중 시간 보상
        air_time_reward = 0
        for i in range(4):
            if len(self.foot_contact_history[i]) > 10:
                # 최근 10스텝 중 공중에 있던 비율
                air_ratio = 1.0 - np.mean(list(self.foot_contact_history[i])[-10:])
                if 0.2 < air_ratio < 0.8:  # 적절한 공중 시간
                    air_time_reward += 0.25
        rewards['feet_air_time'] = air_time_reward * self.reward_weights['feet_air_time']
        
        # 6. 액션 부드러움
        if len(self.action_history) > 1:
            action_diff = np.clip(action - self.action_history[-2], -5.0, 5.0)  # 차이 클리핑
            action_diff_norm = np.sum(np.square(action_diff))
            action_diff_norm = np.clip(action_diff_norm, 0, 100)  # 결과 클리핑
            rewards['action_smoothness'] = -action_diff_norm * self.reward_weights['action_smoothness']
        else:
            rewards['action_smoothness'] = 0
        
        # 7. 에너지 효율성
        clipped_action = np.clip(self.current_action, -100.0, 100.0)  # 토크 클리핑
        energy = np.sum(np.square(clipped_action))
        energy = np.clip(energy, 0, 10000)  # 에너지 클리핑
        rewards['energy'] = -energy * self.reward_weights['energy']
        
        # 8. 측면 속도 페널티
        lateral_vel = np.clip(abs(self.data.qvel[1]), 0, 10.0)  # 속도 클리핑
        rewards['lateral_velocity'] = lateral_vel * self.reward_weights['lateral_velocity']
        
        # 9. 회전 속도 페널티
        ang_vel_xy = self.data.qvel[3:5]
        ang_vel_xy = np.clip(ang_vel_xy, -20.0, 20.0)  # 각속도 클리핑
        ang_vel_norm = np.linalg.norm(ang_vel_xy)
        rewards['angular_velocity'] = ang_vel_norm * self.reward_weights['angular_velocity']
        
        # 10. 관절 가속도 페널티
        if hasattr(self, 'prev_joint_vel'):
            current_vel = np.clip(self.data.qvel[6:6+self.n_actions], -50.0, 50.0)
            prev_vel = np.clip(self.prev_joint_vel, -50.0, 50.0)
            joint_acc = (current_vel - prev_vel) / max(self.dt, 1e-6)  # dt가 0인 경우 방지
            joint_acc = np.clip(joint_acc, -1000.0, 1000.0)  # 가속도 클리핑
            acc_penalty = np.sum(np.square(joint_acc))
            acc_penalty = np.clip(acc_penalty, 0, 1000000)  # 페널티 클리핑
            rewards['joint_acceleration'] = -acc_penalty * self.reward_weights['joint_acceleration']
        else:
            rewards['joint_acceleration'] = 0
        self.prev_joint_vel = self.data.qvel[6:6+self.n_actions].copy()
        
        # 11. 발 걸림 페널티
        stumble_penalty = 0
        for i in range(4):
            if current_contacts[i] and not self.last_contact_state[i]:
                # 착지 시 수평 속도가 크면 페널티
                foot_vel = self._get_foot_velocity(i)
                foot_vel_clipped = np.clip(foot_vel[:2], -10.0, 10.0)
                if np.linalg.norm(foot_vel_clipped) > 0.5:
                    stumble_penalty += 1
        rewards['feet_stumble'] = stumble_penalty * self.reward_weights['feet_stumble']
        
        # 12. 관절 한계 페널티
        joint_pos = self.data.qpos[7:7+self.n_actions]
        joint_limit_penalty = 0
        for i in range(self.n_actions):
            range_span = self.joint_position_limits[i, 1] - self.joint_position_limits[i, 0]
            normalized_pos = (joint_pos[i] - self.joint_position_limits[i, 0]) / range_span
            if normalized_pos < 0.1 or normalized_pos > 0.9:
                joint_limit_penalty += 1
        rewards['joint_limits'] = joint_limit_penalty * self.reward_weights['joint_limits']
        
        # 13. 중력 보상 (높이 유지 시 보상)
        if self.data.qpos[2] > self.standing_height * 0.9:  # 목표 높이의 90% 이상
            gravity_reward = 1.0
        else:
            gravity_reward = 0.0
        rewards['gravity_compensation'] = gravity_reward * self.reward_weights['gravity_compensation']
        
        # 총 보상 (오버플로우 방지)
        total_reward = sum(rewards.values())
        total_reward = np.clip(total_reward, -1000.0, 1000.0)  # 보상 클리핑
        
        # NaN 체크
        if np.isnan(total_reward) or np.isinf(total_reward):
            total_reward = -100.0  # 페널티로 대체
            print(f"Warning: Invalid reward detected, replacing with penalty")
        
        return float(total_reward), rewards
    
    def _get_foot_velocity(self, foot_idx):
        """발의 속도 계산"""
        foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        foot_body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, foot_names[foot_idx])
        return self.data.cvel[foot_body * 6: foot_body * 6 + 3]
    
    def _is_terminated(self):
        # 실제 넘어짐만 감지 (더 관대한 조건)
        
        # 1. 극심한 낮은 높이 (완전히 바닥에 붙었을 때만)
        if self.data.qpos[2] < 0.08:  # 8cm 이하 (원래 15cm)
            print(f"에피소드 종료: 높이 너무 낮음 ({self.data.qpos[2]:.3f}m)")
            return True
        
        # 2. 완전히 뒤집어졌을 때만 (더 관대하게)
        body_quat = self.data.qpos[3:7]
        z_axis = np.array([
            2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
            2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
            body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2
        ])
        if z_axis[2] < -0.1:  # 완전히 뒤집어진 경우 (원래 0.5)
            print(f"에피소드 종료: 완전히 뒤집어짐 (z_axis: {z_axis[2]:.3f})")
            return True
        
        # 3. 극도로 빠른 회전 (회전하며 넘어지는 경우)
        angular_speed = np.linalg.norm(self.data.qvel[3:6])
        if angular_speed > 20.0:  # 매우 빠른 회전
            print(f"에피소드 종료: 과도한 회전 속도 ({angular_speed:.3f} rad/s)")
            return True
        
        # 4. 몸체와 바닥의 접촉 (몸체 기하체와 바닥이 닿은 경우)
        body_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # 몸체 관련 기하체 확인 (발이 아닌)
            geom1_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, contact.geom1) if contact.geom1 >= 0 else ""
            geom2_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, contact.geom2) if contact.geom2 >= 0 else ""
            
            # 몸체나 다리(발 제외)가 바닥에 닿으면 넘어진 것으로 판단
            body_parts = ['torso', 'hip', 'thigh', 'calf']
            for part in body_parts:
                if (geom1_name and part in geom1_name) or (geom2_name and part in geom2_name):
                    if 'foot' not in geom1_name and 'foot' not in geom2_name:  # 발이 아닌 경우
                        body_contact = True
                        break
            if body_contact:
                break
        
        if body_contact:
            print(f"에피소드 종료: 몸체가 바닥에 접촉")
            return True
        
        return False
    
    def render(self):
        if self.render_mode == "human":
            try:
                # MuJoCo 3.x 방식
                if self.viewer is None:
                    import mujoco.viewer
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    # 시각적 설정
                    self._setup_viewer_visuals()
                self.viewer.sync()
            except (AttributeError, ImportError):
                try:
                    # MuJoCo 2.x 방식
                    import mujoco_viewer
                    if self.viewer is None:
                        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                        self._setup_viewer_visuals()
                    self.viewer.render()
                except ImportError:
                    print("Warning: No viewer available. Install mujoco-viewer: pip install mujoco-viewer")
        elif self.render_mode == "rgb_array":
            pass
    
    def _setup_viewer_visuals(self):
        """뷰어 시각적 설정"""
        if hasattr(self.viewer, 'opt'):
            # 접촉력 표시
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            
            # 관절 표시
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
            
            # 질량 중심 표시
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_COM] = True
            
        print("시각적 표시 활성화: 접촉점, 접촉력, 관절, 질량중심")
    
    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
            self.viewer = None