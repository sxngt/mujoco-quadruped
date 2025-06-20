import mujoco as mj
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from common.gait_generator import GaitGenerator, CyclicGaitReward
from collections import deque


class ImprovedGO2Env(gym.Env):
    """개선된 MuJoCo GO2 환경 - Isaac Lab RSL-RL 기법 적용"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None, use_reference_gait=True):
        # XML 파일 경로 설정 및 동적 수정
        import os
        import tempfile
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # rl 디렉토리
        xml_template_path = os.path.join(base_dir, "assets", "go2_scene.xml")
        
        # 메시 디렉토리 절대 경로 설정
        mesh_dir = os.path.join(os.path.dirname(base_dir), "mujoco_menagerie", "unitree_go2", "assets")
        
        # XML 내용 읽기 및 수정
        with open(xml_template_path, 'r') as f:
            xml_content = f.read()
        
        # meshdir을 절대 경로로 교체
        xml_content = xml_content.replace(
            'meshdir="/Users/sxngt/Research/mujoco_quadruped/mujoco_menagerie/unitree_go2/assets"',
            f'meshdir="{mesh_dir}"'
        )
        
        # 임시 XML 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            self.model_path = f.name
        
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
        
        # 관절별 차별화된 PD 제어기 게인 (연구 기반)
        # GO2 로봇의 실제 관절 유형: hip, thigh, calf
        self.joint_types = ['hip', 'thigh', 'calf'] * 4  # 4개 다리
        
        self.kp_gains = {
            'hip': 60.0,    # Hip 관절: 중간 강성
            'thigh': 80.0,  # Thigh 관절: 높은 강성 (체중 지지)
            'calf': 100.0   # Calf 관절: 최고 강성 (접촉 제어)
        }
        
        self.kd_gains = {
            'hip': 3.0,     # Hip 관절: 적당한 댓핑
            'thigh': 4.0,   # Thigh 관절: 높은 댓핑
            'calf': 5.0     # Calf 관절: 최고 댓핑
        }
        
        self.torque_limits = {
            'hip': 23.7,    # ±23.7 Nm (GO2 실제 스펙)
            'thigh': 23.7,  # ±23.7 Nm
            'calf': 35.0    # ±35.0 Nm (안전 마진)
        }
        
        # 발 접촉 추적
        self.foot_geom_ids = []
        for name in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']:
            self.foot_geom_ids.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name))
        
        # 발 접촉 이력 (발 공중 시간 계산용)
        self.foot_contact_history = {i: deque(maxlen=50) for i in range(4)}
        self.last_contact_state = np.zeros(4, dtype=bool)
        
        # 연구 기반 보상 함수 가중치 (점프 방지 및 보행 강화)
        self.reward_weights = {
            # === 핵심 목표 (전진) ===
            'forward_velocity': 25.0,    # 전진 보상 (점프 방지를 위해 감소)
            'target_velocity': 15.0,     # 목표 속도 추적 보상
            
            # === 보행 품질 (지상 접촉 강조) ===
            'gait_pattern': 30.0,        # 보행 패턴 중요도 대폭 증가
            'ground_contact': 1.0,       # 지상 접촉 유지 보상 (새로 추가)
            'energy_efficiency': 5.0,    # 에너지 효율적 보행
            
            # === 자세 안정성 ===
            'height_tracking': 20.0,     # 높이 유지 중요도 증가
            'orientation': 15.0,         # 자세 유지 중요도 증가
            
            # === 방향 제어 ===
            'direction_control': 10.0,   # 직진 보상 증가
            
            # === 안전성 및 부드러움 ===
            'action_smoothness': 0.5,    # 부드러운 움직임 증가
            'joint_safety': -20.0,       # 관절 안전성 페널티 강화
            'vertical_control': 3.0,     # 수직 움직임 제어 강화
            'stability': 10.0,           # 전반적 안정성 대폭 증가
        }
        
        # 목표 전진 속도 설정
        self.target_forward_velocity = 0.8  # 0.8 m/s 목표
        
        self.max_episode_steps = 8000  # 더 긴 에피소드로 보행 학습 기회 증가
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
        """관절별 차별화된 PD 제어기 (연구 기반)"""
        torques = []
        
        for i in range(len(target_pos)):
            joint_type = self.joint_types[i]
            
            # 관절별 PD 게인
            kp = self.kp_gains[joint_type]
            kd = self.kd_gains[joint_type]
            
            # 안전한 입력값 범위
            target = np.clip(target_pos[i], -3.0, 3.0)
            current = np.clip(current_pos[i], -3.0, 3.0)
            velocity = np.clip(current_vel[i], -20.0, 20.0)
            
            # 오차 계산
            pos_error = target - current
            pos_error = np.clip(pos_error, -0.3, 0.3)  # 작은 오차로 안정성 향상
            
            # PD 제어 법칙
            torque = kp * pos_error - kd * velocity
            
            # 관절별 토크 제한
            max_torque = self.torque_limits[joint_type]
            torque = np.clip(torque, -max_torque, max_torque)
            
            torques.append(torque)
        
        return np.array(torques)
    
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
        
        # 현재 관절 위치 기준 상대적 변화 (매우 안전한 스케일링)
        current_joint_pos = self.data.qpos[7:7+self.n_actions]
        target_joint_pos = current_joint_pos + smoothed_action * 0.008  # 점프 방지를 위한 더 작은 델타
        
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
        """연구 기반 재설계된 보상 함수"""
        rewards = {}
        
        # === 1. 핵심 목표: 전진 보상 (가장 중요) ===
        forward_vel = np.clip(self.data.qvel[0], -5.0, 5.0)
        
        # 1-1. 기본 전진 보상 (속도에 비례)
        if forward_vel > 0:
            rewards['forward_velocity'] = forward_vel * self.reward_weights['forward_velocity']
        else:
            rewards['forward_velocity'] = forward_vel * self.reward_weights['forward_velocity'] * 2  # 후진 페널티 강화
        
        # 1-2. 목표 속도 추적 보상 (0.8 m/s 목표)
        vel_error = abs(forward_vel - self.target_forward_velocity)
        target_bonus = np.exp(-3.0 * vel_error)  # 지수적 보상
        rewards['target_velocity'] = target_bonus * self.reward_weights['target_velocity']
        
        # === 2. 보행 품질 (지상 접촉 강조) ===
        # 2-1. 지상 접촉 유지 보상 (점프 방지)
        num_contacts = sum(1 for contact in current_contacts if contact)
        
        # 지상 접촉 강조: 항상 최소 1개 발은 땅에 닿아야 함
        if num_contacts == 0:
            gait_reward = -20.0  # 점프 매우 강한 페널티
        elif num_contacts == 1:
            gait_reward = 0.3   # 한 발 접촉 (어려우나 허용)
        elif num_contacts == 2:
            gait_reward = 1.5   # 이상적 동적 보행 (강화)
        elif num_contacts == 3:
            gait_reward = 1.0   # 좋은 보행
        else:  # 4개 모두 접촉
            gait_reward = 2.0   # 정적 보행 (점프 방지를 위해 더 보상)
        
        rewards['gait_pattern'] = gait_reward * self.reward_weights['gait_pattern']
        
        # 2-2. 지상 유지 보너스 (점프 방지)
        ground_contact_bonus = min(num_contacts, 4) / 4.0  # 모든 발 접촉 장려
        rewards['ground_contact'] = ground_contact_bonus * 15.0  # 지상 접촉 강화
        
        # 2-3. 에너지 효율성 (낮은 토크 사용 보상)
        torque_efficiency = max(0, 1.0 - np.mean(np.abs(self.current_action)) / 15.0)
        rewards['energy_efficiency'] = torque_efficiency * self.reward_weights['energy_efficiency']
        
        # === 3. 자세 안정성 ===
        # 3-1. 높이 추적
        height_error = abs(self.data.qpos[2] - self.standing_height)
        height_reward = max(0, 1.0 - height_error * 10.0)  # 높이 차이에 민감
        rewards['height_tracking'] = height_reward * self.reward_weights['height_tracking']
        
        # 3-2. 자세 유지 (upright orientation)
        body_quat = self.data.qpos[3:7]
        z_axis = np.array([
            2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
            2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
            body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2
        ])
        upright_reward = max(0, z_axis[2])  # z축이 위를 향할수록 보상
        rewards['orientation'] = upright_reward * self.reward_weights['orientation']
        
        # === 4. 방향 제어 ===
        # 직진 보상 (측면 속도 및 회전 최소화)
        lateral_vel = abs(self.data.qvel[1])  # y축 속도
        yaw_vel = abs(self.data.qvel[5])      # z축 회전 속도
        
        direction_reward = max(0, 1.0 - lateral_vel) * 0.7 + max(0, 1.0 - yaw_vel) * 0.3
        rewards['direction_control'] = direction_reward * self.reward_weights['direction_control']
        
        # === 5. 안전성 및 부드러움 ===
        # 5-1. 액션 부드러움
        if len(self.action_history) > 1:
            action_diff = np.clip(action - self.action_history[-2], -2.0, 2.0)
            smoothness = max(0, 1.0 - np.mean(np.abs(action_diff)))
            rewards['action_smoothness'] = smoothness * self.reward_weights['action_smoothness']
        else:
            rewards['action_smoothness'] = 0
        
        # 5-2. 관절 안전성 (관절 한계 및 과도한 사용 방지)
        joint_pos = self.data.qpos[7:7+self.n_actions]
        joint_safety_penalty = 0
        
        for i in range(self.n_actions):
            range_span = self.joint_position_limits[i, 1] - self.joint_position_limits[i, 0]
            normalized_pos = (joint_pos[i] - self.joint_position_limits[i, 0]) / range_span
            if normalized_pos < 0.15 or normalized_pos > 0.85:  # 위험 영역
                joint_safety_penalty += 1
        
        rewards['joint_safety'] = joint_safety_penalty * self.reward_weights['joint_safety']
        
        # 5-3. 수직 움직임 제어 (점프 방지)
        vertical_vel = self.data.qvel[2]  # z축 속도
        
        # 위로 점프하는 것 매우 강하게 페널티
        if vertical_vel > 0.1:  # 위로 0.1m/s 이상 (더 엄격)
            jump_penalty = -50.0 * (vertical_vel - 0.1)  # 더 강한 페널티
        else:
            jump_penalty = 0.0
        
        # 급격한 낙하도 페널티
        if vertical_vel < -0.3:  # 아래로 0.3m/s 이상 (더 엄격)
            fall_penalty = -25.0 * abs(vertical_vel + 0.3)
        else:
            fall_penalty = 0.0
        
        # 수직 속도가 거의 0에 가까우면 보너스 (안정적 보행)
        if abs(vertical_vel) < 0.05:
            stability_bonus = 5.0
        else:
            stability_bonus = 0.0
        
        rewards['vertical_control'] = jump_penalty + fall_penalty + stability_bonus
        
        # 5-4. 전반적 안정성 (급격한 변화 최소화)
        body_stability = 1.0
        
        # 수직 속도 안정성 (작은 수직 움직임 선호)
        vertical_stability = max(0, 1.0 - abs(vertical_vel) * 2.0)
        body_stability *= vertical_stability
        
        # 각속도 안정성
        total_ang_vel = np.linalg.norm(self.data.qvel[3:6])
        body_stability *= max(0, 1.0 - total_ang_vel * 0.5)
        
        rewards['stability'] = body_stability * self.reward_weights['stability']
        
        # === 6. 보행 노력 보상 (지상 접촉하며 전진) ===
        # 땅에 발이 닿아있으면서 동시에 전진할 때만 보상
        walking_effort = 0.0
        if num_contacts >= 2 and forward_vel > 0.05:  # 최소 2발 접촉 + 전진
            walking_effort = forward_vel * num_contacts * 5.0  # 접촉 많을수록, 빠를수록 보상
        elif num_contacts >= 1 and forward_vel > 0.02:  # 최소 1발 접촉 + 천천히 전진
            walking_effort = forward_vel * 2.0
        
        rewards['walking_effort'] = walking_effort
        
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
        """연구 기반 단계적 종료 조건"""
        
        body_height = self.data.qpos[2]
        body_quat = self.data.qpos[3:7]
        
        # 1. 즉시 종료: 심각한 실패 상황
        # 1-1. 극도로 낮은 높이 (완전히 바닥에 누움)
        if body_height < 0.06:  # 6cm 이하 (매우 관대)
            print(f"에피소드 종료: 높이 {body_height:.3f}m (너무 낮음)")
            return True
        
        # 1-2. 완전히 뒤집어진 상태
        z_axis = np.array([
            2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
            2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
            body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2
        ])
        if z_axis[2] < 0.1:  # 거의 뒤집어진 상태
            print(f"에피소드 종료: 뒤집어짐 (z_axis: {z_axis[2]:.3f})")
            return True
        
        # 2. 학습 영역 벗어남 (무한히 굴러가는 것 방지)
        if abs(self.data.qpos[1]) > 5.0:  # 좌우 5m
            print(f"에피소드 종료: 학습 영역 벗어남 (y: {self.data.qpos[1]:.3f}m)")
            return True
        
        if self.data.qpos[0] < -3.0:  # 뒤로 3m
            print(f"에피소드 종료: 후진 한계 (x: {self.data.qpos[0]:.3f}m)")
            return True
        
        # 3. 관절 안전성 (관절 파손 방지)
        joint_pos = self.data.qpos[7:7+self.n_actions]
        for i in range(self.n_actions):
            if i + 1 < self.model.njnt:
                joint_range = self.model.jnt_range[i + 1]
                if joint_pos[i] <= joint_range[0] + 0.005 or joint_pos[i] >= joint_range[1] - 0.005:
                    print(f"에피소드 종료: 관절 {i} 한계 근처 ({joint_pos[i]:.3f})")
                    return True
        
        # 4. 과도한 회전 (제어 불가능한 상황)
        angular_speed = np.linalg.norm(self.data.qvel[3:6])
        if angular_speed > 25.0:  # 매우 빠른 회전
            print(f"에피소드 종료: 과도한 회전 ({angular_speed:.3f} rad/s)")
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
        
        # 임시 XML 파일 정리
        if hasattr(self, 'model_path') and self.model_path.startswith('/tmp'):
            try:
                import os
                os.unlink(self.model_path)
            except:
                pass