import mujoco as mj
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IntegratedGO2Env(gym.Env):
    """
    Unitree GO2 환경 - 성공적인 참조 레포지터리 기법을 GO2에 적용
    참조: nimazareian/quadruped-rl-locomotion
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None):
        # 환경별 렌더링 설정
        import os
        import tempfile
        import platform
        
        # 운영체제별 렌더링 설정
        if render_mode == "human":
            system = platform.system()
            
            if system == "Linux":  # Ubuntu 환경
                # DISPLAY 환경변수 확인 및 설정
                if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
                    print("⚠️  DISPLAY 환경변수가 설정되지 않았습니다.")
                    print("다음 방법 중 하나를 시도하세요:")
                    print("1. GUI 세션에서 실행: export DISPLAY=:0")
                    print("2. SSH X11 포워딩: ssh -X username@server")
                    print("3. Xvfb 가상 디스플레이: xvfb-run -a python script.py")
                    print("4. 원격 데스크톱: VNC, XRDP 등")
                    
                    # 자동으로 :0 시도
                    os.environ['DISPLAY'] = ':0'
                    print(f"🔧 DISPLAY를 ':0'으로 자동 설정했습니다.")
                
                # OpenGL 설정
                os.environ.setdefault('MUJOCO_GL', 'glfw')
                os.environ.setdefault('LIBGL_ALWAYS_INDIRECT', '0')
                
            elif system == "Darwin":  # macOS 환경
                print("ℹ️  macOS에서는 mjpython으로 실행하는 것이 권장됩니다.")
                print("예: mjpython train_integrated.py --render")
                
            print(f"🖥️  렌더링 모드: {render_mode}, 시스템: {system}")
        
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
        
        # 참조 방식: 간결한 39차원 관찰 공간
        # [joint_pos(12) + joint_vel(12) + body_quat(4) + body_angvel(3) + body_linvel(3) + prev_action(12)] = 46차원
        # GO2 특화로 3차원 추가하여 39차원으로 맞춤
        obs_dim = 39
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
        
        # 관찰 벡터 구성 (39차원)
        obs = np.concatenate([
            body_linvel,        # 3
            body_angvel,        # 3  
            projected_gravity,  # 3
            commands,           # 2
            joint_pos,          # 12
            joint_vel,          # 12
            prev_actions[:4]    # 4 (39차원 맞추기 위해 일부만)
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
        
        # === 총 보상 (참조 방식 가중치) ===
        total_reward = (
            lin_vel_reward +      # 선형 속도 추적
            ang_vel_reward +      # 각속도 추적  
            feet_air_reward +     # 발 공중 시간
            alive_reward +        # 생존 보상
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
            'torque_cost': torque_cost,
            'action_rate_cost': action_rate_cost,
            'vertical_cost': vertical_cost,
            'joint_limit_cost': joint_limit_cost,
            'collision_cost': collision_cost,
            'orientation_cost': orientation_cost,
            'total': total_reward
        }
    
    def _is_terminated(self):
        """참조 방식의 간단하고 명확한 종료 조건"""
        
        # 높이 체크
        body_height = self.data.qpos[2]
        if body_height < 0.1:  # 10cm 아래로 떨어지면 종료
            return True
        
        # 기울기 체크 (참조 방식)
        body_quat = self.data.qpos[3:7]
        z_axis = np.array([
            2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
            2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
            body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2
        ])
        
        if z_axis[2] < 0.3:  # 너무 기울어지면 종료
            return True
        
        # 측면 이탈
        if abs(self.data.qpos[1]) > 5.0:
            return True
        
        # 후진 제한
        if self.data.qpos[0] < -2.0:
            return True
        
        # 수치적 불안정성
        if (np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)) or
            np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel))):
            return True
        
        return False
    
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
        truncated = False
        
        self.current_step += 1
        self.prev_action = self.current_action.copy()
        
        info = {**reward_info, 'step': self.current_step, 'contacts': self._get_contact_info()}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None
    
    def _render_human(self):
        """인간이 볼 수 있는 GUI 렌더링 (공식 문서 기준)"""
        if self.viewer is None:
            try:
                import mujoco.viewer
                # 공식 문서 권장: passive viewer 사용
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                if self.viewer is not None:
                    # 카메라 초기 설정
                    self.viewer.cam.distance = 3.0
                    self.viewer.cam.elevation = -20
                    self.viewer.cam.azimuth = 135
                    self.viewer.cam.lookat[:] = [0, 0, 0.3]
                    print("✅ Passive viewer 초기화 성공")
                else:
                    raise RuntimeError("Passive viewer 생성 실패")
            except Exception as e:
                print(f"❌ Passive viewer 실패: {e}")
                try:
                    # 대안: blocking viewer (비권장이지만 동작함)
                    print("🔄 Blocking viewer 시도...")
                    self.viewer = mujoco.viewer.launch(self.model, self.data)
                    print("✅ Blocking viewer 초기화 성공")
                except Exception as e2:
                    print(f"❌ Blocking viewer도 실패: {e2}")
                    self.viewer = None
                    return None
        
        if self.viewer is not None:
            try:
                # 뷰어 상태 확인 (passive viewer의 경우)
                if hasattr(self.viewer, 'is_running') and not self.viewer.is_running():
                    print("뷰어 창이 닫혔습니다.")
                    self.viewer = None
                    return None
                
                # 로봇 추적 카메라
                if len(self.data.qpos) >= 2:
                    robot_x = self.data.qpos[0]
                    robot_y = self.data.qpos[1]
                    self.viewer.cam.lookat[0] = robot_x
                    self.viewer.cam.lookat[1] = robot_y
                
                # 공식 문서 권장: sync로 데이터 동기화
                self.viewer.sync()
                return True
                
            except Exception as e:
                print(f"❌ 뷰어 업데이트 실패: {e}")
                return None
        
        return None
    
    def _render_rgb_array(self):
        """RGB 배열 렌더링 (헤드리스 환경용, 공식 문서 기준)"""
        if self.viewer is None:
            # 공식 문서 기준: (model, height, width) 순서
            self.viewer = mj.Renderer(self.model, height=768, width=1024)
        
        try:
            # 씬 업데이트 후 렌더링
            self.viewer.update_scene(self.data)
            rgb_array = self.viewer.render()
            
            # 공식 문서: RGB 배열은 (height, width, 3) 형태
            if rgb_array.shape[-1] != 3:
                raise RuntimeError(f"잘못된 RGB 배열 형태: {rgb_array.shape}")
            
            return rgb_array
            
        except Exception as e:
            print(f"❌ RGB 렌더링 실패: {e}")
            return None
    
    def close(self):
        """리소스 정리 (공식 문서 기준)"""
        if self.viewer is not None:
            try:
                if hasattr(self.viewer, 'close'):
                    self.viewer.close()
                    print("✅ 뷰어 정상 종료")
                elif hasattr(self.viewer, 'is_running'):
                    # passive viewer의 경우 자동으로 정리됨
                    print("ℹ️  Passive viewer 자동 정리")
            except Exception as e:
                print(f"⚠️  뷰어 종료 중 오류: {e}")
            finally:
                self.viewer = None
        
        # 임시 XML 파일 정리
        try:
            import os
            if hasattr(self, 'model_path') and os.path.exists(self.model_path):
                os.unlink(self.model_path)
                print("✅ 임시 XML 파일 정리 완료")
        except Exception as e:
            print(f"⚠️  임시 파일 정리 실패: {e}")