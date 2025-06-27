import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from pathlib import Path


DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}


class IntegratedGO2Env(MujocoEnv):
    """
    Unitree GO2 환경 - 성공적인 참조 레포지터리 기법을 GO2에 적용
    참조: nimazareian/quadruped-rl-locomotion
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None):
        # XML 파일 설정
        import os
        import tempfile
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        xml_template_path = os.path.join(base_dir, "assets", "go2_scene.xml")
        mesh_dir = os.path.join(os.path.dirname(base_dir), "mujoco_menagerie", "unitree_go2", "assets")
        
        # XML 경로 수정
        with open(xml_template_path, 'r') as f:
            xml_content = f.read()
        
        xml_content = xml_content.replace(
            'meshdir="/Users/sxngt/Research/mujoco_quadruped/mujoco_menagerie/unitree_go2/assets"',
            f'meshdir="{mesh_dir}"'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            self.model_path = f.name
        
        self.model = mj.MjModel.from_xml_path(self.model_path)
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # 참조 레포지터리 방식: 직접 토크 제어
        self.n_actions = 12  # GO2의 12개 관절
        
        # 토크 한계 (GO2 실제 스펙)
        torque_limits = np.array([23.7] * 12)  # ±23.7 Nm for all joints
        
        self.action_space = spaces.Box(
            low=-torque_limits.astype(np.float32),
            high=torque_limits.astype(np.float32),
            shape=(self.n_actions,), dtype=np.float32
        )
        
        # 참조 레포지터리와 동일한 45차원 관찰 공간
        # [lin_vel(3) + ang_vel(3) + gravity(3) + commands(2) + joint_pos(12) + joint_vel(12) + prev_actions(10)] = 45차원
        obs_dim = 45
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            shape=(obs_dim,), dtype=np.float32
        )
        
        # 참조 보행 자세 (GO2 키프레임 기반)
        home_keyframe = 0
        self.initial_qpos = self.model.key_qpos[home_keyframe].copy()
        self.initial_qvel = np.zeros(self.model.nv)
        
        # 발 감지 (접촉 추적용)
        self.foot_geom_ids = []
        for name in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']:
            self.foot_geom_ids.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name))
        
        # 상태 변수
        self.current_step = 0
        self.max_episode_steps = 1000  # 최대 에피소드 길이
        self.current_action = np.zeros(self.n_actions)
        self.prev_action = np.zeros(self.n_actions)
        
        # 참조 방식: 목표 속도 설정
        self.target_velocity = np.array([0.8, 0.0])  # [forward, lateral] m/s
        
        # 관찰 스케일링 (참조 방식)
        self.obs_scales = {
            'lin_vel': 2.0,
            'ang_vel': 0.25, 
            'dof_pos': 1.0,
            'dof_vel': 0.05,
            'quat': 1.0,
            'actions': 1.0
        }
        
    def _get_observation(self):
        """참조 레포지터리 방식의 관찰 공간"""
        # 1. 선형 속도 (body frame) - 3차원
        body_quat = self.data.qpos[3:7]
        body_linvel_world = self.data.qvel[0:3]
        
        # 쿼터니언을 회전 행렬로 변환하여 body frame으로 변환
        quat_w, quat_x, quat_y, quat_z = body_quat
        
        # body frame으로 변환하는 회전 행렬 (간단화)
        body_linvel = body_linvel_world.copy()  # 단순화: world frame 사용
        
        # 2. 각속도 (body frame) - 3차원  
        body_angvel = self.data.qvel[3:6]
        
        # 3. 중력 투영 벡터 - 3차원
        gravity_vec = np.array([0, 0, -1])
        # 몸체 방향으로 중력 벡터 회전 (단순화)
        projected_gravity = gravity_vec.copy()
        
        # 4. 명령 속도 - 2차원 (forward, lateral)
        commands = self.target_velocity.copy()
        
        # 5. 관절 위치 (정규화) - 12차원
        joint_pos = self.data.qpos[7:7+12]
        
        # 6. 관절 속도 - 12차원
        joint_vel = self.data.qvel[6:6+12]
        
        # 7. 이전 액션 - 12차원
        prev_actions = self.prev_action.copy()
        
        # 참조 방식 스케일링 적용
        body_linvel *= self.obs_scales['lin_vel']
        body_angvel *= self.obs_scales['ang_vel'] 
        joint_pos *= self.obs_scales['dof_pos']
        joint_vel *= self.obs_scales['dof_vel']
        prev_actions *= self.obs_scales['actions']
        
        # 관찰 벡터 구성 (45차원)
        obs = np.concatenate([
            body_linvel,        # 3
            body_angvel,        # 3  
            projected_gravity,  # 3
            commands,           # 2
            joint_pos,          # 12
            joint_vel,          # 12
            prev_actions[:10]   # 10 (45차원 맞추기 위해 10개만)
        ])
        
        # 클리핑 (참조 방식)
        obs = np.clip(obs, -5.0, 5.0)
        
        return obs.astype(np.float32)
    
    def _get_contact_info(self):
        """발 접촉 정보"""
        contacts = {}
        foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        
        for i, foot_name in enumerate(foot_names):
            if i < len(self.foot_geom_ids):
                geom_id = self.foot_geom_ids[i]
                in_contact = False
                contact_force = 0.0
                
                for j in range(self.data.ncon):
                    contact = self.data.contact[j]
                    if contact.geom1 == geom_id or contact.geom2 == geom_id:
                        in_contact = True
                        contact_force += np.linalg.norm(contact.frame[0:3])
                        break
                
                contacts[foot_name] = {
                    'in_contact': in_contact,
                    'force': contact_force
                }
        
        return contacts
    
    def _compute_reward(self):
        """참조 레포지터리의 보상 구조를 GO2에 적용"""
        
        # === 핵심 보상: 전진 속도 ===
        forward_vel = self.data.qvel[0]  # x축 속도
        
        # 선형 속도 추적 보상 (참조 방식)
        lin_vel_error = abs(forward_vel - self.target_velocity[0])
        lin_vel_reward = np.exp(-lin_vel_error * 2.0) * 10.0
        
        # === 정지 방지 페널티 (강화) ===
        total_vel = np.linalg.norm(self.data.qvel[:3])  # 전체 선형 속도
        if total_vel < 0.1:  # 거의 정지 상태
            stationary_penalty = -50.0  # 강한 페널티
        elif total_vel < 0.3:  # 느린 움직임
            stationary_penalty = -20.0  # 중간 페널티
        else:
            stationary_penalty = 0.0
            
        # 전진 방향 보상
        if forward_vel > 0.1:
            forward_bonus = 5.0
        elif forward_vel < 0:  # 후진 페널티
            forward_bonus = -10.0
        else:
            forward_bonus = 0.0
        
        # === 각속도 추적 보상 ===
        target_ang_vel = 0.0  # 직진
        ang_vel_error = abs(self.data.qvel[5] - target_ang_vel)  # yaw 속도
        ang_vel_reward = np.exp(-ang_vel_error * 2.0) * 2.0
        
        # === 발 공중 시간 보상 (참조 방식) ===
        contacts = self._get_contact_info()
        feet_air_time = 0.0
        for contact in contacts.values():
            if not contact['in_contact']:
                feet_air_time += 1.0
        feet_air_reward = feet_air_time * 0.5
        
        # === 생존 보상 ===
        # 높이 체크
        body_height = self.data.qpos[2]
        if body_height > 0.15:  # 최소 높이 유지
            alive_reward = 2.0
        else:
            alive_reward = -5.0
        
        # === 비용 (참조 방식) ===
        # 토크 사용량 페널티
        torque_cost = -0.0001 * np.sum(np.square(self.current_action))
        
        # 액션 변화율 페널티 
        action_rate_cost = -0.01 * np.sum(np.square(self.current_action - self.prev_action))
        
        # 수직 속도 페널티 (점프 방지)
        vertical_vel = abs(self.data.qvel[2])
        vertical_cost = -2.0 * max(0, vertical_vel - 0.3)
        
        # 관절 한계 페널티
        joint_pos = self.data.qpos[7:7+12]
        joint_limit_cost = 0.0
        for i in range(12):
            joint_idx = i + 1
            if joint_idx < self.model.njnt:
                joint_range = self.model.jnt_range[joint_idx]
                if joint_pos[i] < joint_range[0] * 0.9 or joint_pos[i] > joint_range[1] * 0.9:
                    joint_limit_cost -= 1.0
        
        # 충돌 페널티 (몸체 접촉)
        collision_cost = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # 발이 아닌 다른 부위 접촉 감지 (간단화)
            if contact.geom1 not in self.foot_geom_ids and contact.geom2 not in self.foot_geom_ids:
                collision_cost -= 2.0
        
        # 방향 유지 페널티
        body_quat = self.data.qpos[3:7]
        z_axis = np.array([
            2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
            2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
            body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2
        ])
        orientation_cost = -2.0 * max(0, 0.7 - z_axis[2])  # z축이 위를 향하도록
        
        # === 총 보상 (개선된 가중치) ===
        total_reward = (
            lin_vel_reward +      # 선형 속도 추적
            ang_vel_reward +      # 각속도 추적  
            feet_air_reward +     # 발 공중 시간
            alive_reward +        # 생존 보상
            stationary_penalty +  # 정지 방지 페널티 (새로 추가)
            forward_bonus +       # 전진 보너스 (새로 추가)
            torque_cost +         # 토크 비용
            action_rate_cost +    # 액션 변화율 비용
            vertical_cost +       # 수직 속도 비용
            joint_limit_cost +    # 관절 한계 비용
            collision_cost +      # 충돌 비용
            orientation_cost      # 방향 비용
        )
        
        return total_reward, {
            'lin_vel_reward': lin_vel_reward,
            'ang_vel_reward': ang_vel_reward,
            'feet_air_reward': feet_air_reward,
            'alive_reward': alive_reward,
            'stationary_penalty': stationary_penalty,  # 새로 추가
            'forward_bonus': forward_bonus,            # 새로 추가
            'torque_cost': torque_cost,
            'action_rate_cost': action_rate_cost,
            'vertical_cost': vertical_cost,
            'joint_limit_cost': joint_limit_cost,
            'collision_cost': collision_cost,
            'orientation_cost': orientation_cost,
            'total': total_reward
        }
    
    @property
    def is_healthy(self):
        """참조 레포지터리와 동일한 건강 상태 체크"""
        # 모든 상태값이 유한한지 확인
        state = self._get_observation()
        if not np.all(np.isfinite(state)):
            return False
        
        # Z 위치 체크 (참조: 0.22-0.65)
        z_pos = self.data.qpos[2]
        if not (0.22 <= z_pos <= 0.65):
            return False
        
        # Roll, Pitch 각도 체크 (±10도)
        quat = self.data.qpos[3:7]
        # 쿼터니언을 오일러 각도로 변환
        w, x, y, z = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        
        max_angle = np.radians(10)  # 10도를 라디안으로
        if abs(roll) > max_angle or abs(pitch) > max_angle:
            return False
        
        # 정지 상태 체크 (50스텝 후)
        if self.current_step > 50:
            total_vel = np.linalg.norm(self.data.qvel[:3])
            if total_vel < 0.1:
                return False
        
        return True
    
    def _is_terminated(self):
        """참조 레포지터리 방식의 종료 조건"""
        return not self.is_healthy
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 초기 상태로 리셋 (참조 방식)
        self.data.qpos[:] = self.initial_qpos.copy()
        self.data.qvel[:] = self.initial_qvel.copy()
        
        if seed is not None:
            np.random.seed(seed)
        
        # 작은 무작위 perturbation (참조 방식)
        joint_noise = np.random.normal(0, 0.02, 12)  # 관절 위치 노이즈
        self.data.qpos[7:7+12] += joint_noise
        
        # 위치 노이즈
        self.data.qpos[0] += np.random.normal(0, 0.01)  # x
        self.data.qpos[1] += np.random.normal(0, 0.01)  # y
        
        # 초기 모멘텀 추가 (정지 방지)
        self.data.qvel[0] = np.random.uniform(0.1, 0.3)  # 전진 속도
        self.data.qvel[1] = np.random.normal(0, 0.05)    # 측면 속도 약간
        
        # 관절 속도에도 약간의 초기 움직임
        joint_vel_noise = np.random.normal(0, 0.1, 12)
        self.data.qvel[6:6+12] = joint_vel_noise
        self.data.qpos[2] += np.random.normal(0, 0.005) # z
        
        # 방향 노이즈 (쿼터니언)
        small_angle = np.random.normal(0, 0.05, 3)
        angle_magnitude = np.linalg.norm(small_angle)
        if angle_magnitude > 0:
            axis = small_angle / angle_magnitude
            quat_noise = np.array([
                np.cos(angle_magnitude/2),
                axis[0] * np.sin(angle_magnitude/2),
                axis[1] * np.sin(angle_magnitude/2), 
                axis[2] * np.sin(angle_magnitude/2)
            ])
            # 기존 쿼터니언과 곱하기 (간단화: 작은 노이즈만 추가)
            self.data.qpos[3:7] += quat_noise * 0.01
            self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])
        
        # 상태 변수 초기화
        self.current_step = 0
        self.current_action = np.zeros(self.n_actions)
        self.prev_action = np.zeros(self.n_actions)
        
        mj.mj_forward(self.model, self.data)
        
        observation = self._get_observation()
        info = {'contacts': self._get_contact_info()}
        
        return observation, info
    
    def step(self, action):
        # 액션 클리핑 (참조 방식)
        action = np.clip(action, -20.0, 20.0)
        self.current_action = action.copy()
        
        # 토크 직접 적용 (참조 방식)
        self.data.ctrl[:12] = action
        
        # 물리 시뮬레이션 (참조 방식: 단일 스텝)
        mj.mj_step(self.model, self.data)
        
        # 상태 업데이트
        observation = self._get_observation()
        reward, reward_info = self._compute_reward()
        terminated = self._is_terminated()
        
        # 에피소드 길이 제한 (truncated 조건)
        truncated = self.current_step >= self.max_episode_steps
        
        self.current_step += 1
        self.prev_action = self.current_action.copy()
        
        info = {**reward_info, 'step': self.current_step, 'contacts': self._get_contact_info()}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 3.0
                self.viewer.cam.elevation = -20
                self.viewer.cam.azimuth = 135
                self.viewer.cam.lookat[0] = 0
                self.viewer.cam.lookat[1] = 0
                self.viewer.cam.lookat[2] = 0.3
            else:
                # 로봇 추적
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
        
        return None
    
    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
            self.viewer = None
        
        # 임시 파일 정리
        if hasattr(self, 'model_path') and self.model_path.startswith('/tmp'):
            try:
                import os
                os.unlink(self.model_path)
            except:
                pass