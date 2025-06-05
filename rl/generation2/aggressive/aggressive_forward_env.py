"""
공격적 전진 환경 - 무조건 앞으로 가려고 시도하는 환경
"""

import mujoco as mj
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import os
import tempfile


class AggressiveForwardGO2Env(gym.Env):
    """전진만을 목표로 하는 공격적 학습 환경"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None):
        # XML 파일 경로 설정 및 동적 수정
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
        
        # 시뮬레이션 초기화
        mj.mj_forward(self.model, self.data)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # 제어 설정
        self.n_actions = self.model.nu
        
        # 토크 제어 (더 강한 토크 허용)
        self.action_space = spaces.Box(
            low=-40.0, high=40.0,  # 더 강한 토크
            shape=(self.n_actions,), dtype=np.float32
        )
        
        # 관찰 공간
        n_joints = self.model.nq - 7
        n_velocities = self.model.nv - 6
        obs_dim = n_joints + n_velocities + 4 + 3 + 3  # pos + vel + quat + ang_vel + lin_vel
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
        # 초기 위치 (XML keyframe)
        home_keyframe = 0
        self.initial_qpos = self.model.key_qpos[home_keyframe].copy()
        self.initial_qvel = np.zeros(self.model.nv)
        
        self.current_step = 0
        self.dt = self.model.opt.timestep
        
        # 발 geom IDs
        self.foot_geom_ids = []
        for name in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']:
            self.foot_geom_ids.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name))
        
        print("🚀 공격적 전진 환경 생성 완료")
        print("목표: 오직 전진! 넘어져도 괜찮아!")
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # 표준 위치로 리셋
        self.data.qpos[:] = self.initial_qpos.copy()
        self.data.qvel[:] = self.initial_qvel.copy()
        
        # 매우 작은 랜덤성 (안정적 시작)
        if seed is not None:
            np.random.seed(seed)
            # 위치 노이즈
            self.data.qpos[0:2] += np.random.uniform(-0.02, 0.02, 2)
            # 관절 노이즈
            self.data.qpos[7:] += np.random.uniform(-0.02, 0.02, self.model.nq - 7)
        
        self.current_step = 0
        mj.mj_forward(self.model, self.data)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # 관절 정보
        n_joints = self.model.nq - 7
        n_velocities = self.model.nv - 6
        
        joint_pos = self.data.qpos[7:7+n_joints].copy()
        joint_vel = self.data.qvel[6:6+n_velocities].copy()
        
        # 몸체 정보
        body_quat = self.data.qpos[3:7].copy()
        body_angvel = self.data.qvel[3:6].copy()
        body_linvel = self.data.qvel[0:3].copy()
        
        return np.concatenate([
            joint_pos, joint_vel, body_quat, body_angvel, body_linvel
        ])
    
    def _get_reward(self):
        """공격적 전진 보상 - 오직 앞으로만!"""
        
        # === 1. 전진이 전부! ===
        forward_vel = self.data.qvel[0]  # x 속도
        
        # 전진 보상 (매우 강력)
        if forward_vel > 0:
            forward_reward = forward_vel * 50.0  # 매우 강한 보상
            # 빠를수록 추가 보너스
            if forward_vel > 0.5:
                forward_reward += (forward_vel - 0.5) * 100.0
            if forward_vel > 1.0:
                forward_reward += (forward_vel - 1.0) * 200.0
        else:
            # 후진하면 강한 페널티
            forward_reward = forward_vel * 100.0
        
        # === 2. 넘어지지 않으면 작은 보너스 (하지만 중요하지 않음) ===
        body_height = self.data.qpos[2]
        if body_height > 0.15:
            survival_bonus = 1.0  # 아주 작은 보너스
        else:
            survival_bonus = 0.0  # 넘어져도 괜찮아
        
        # === 3. 움직이지 않으면 큰 페널티 ===
        total_vel = np.linalg.norm(self.data.qvel[:3])
        if total_vel < 0.1:  # 거의 정지
            static_penalty = -10.0
        else:
            static_penalty = 0.0
        
        # === 4. 측면 이동 약간의 페널티 (하지만 전진이 더 중요) ===
        lateral_vel = abs(self.data.qvel[1])
        lateral_penalty = -lateral_vel * 2.0  # 약한 페널티
        
        # === 5. 에너지는 신경쓰지 마! (높은 토크 OK) ===
        # 에너지 페널티 없음 - 얼마든지 힘을 써도 됨
        
        total_reward = (
            forward_reward +      # 핵심! 
            survival_bonus +      # 부가적
            static_penalty +      # 정지 방지
            lateral_penalty       # 직진 유도
        )
        
        reward_info = {
            'forward': forward_reward,
            'survival': survival_bonus,
            'static': static_penalty,
            'lateral': lateral_penalty,
            'total': total_reward
        }
        
        return total_reward, reward_info
    
    def _is_terminated(self):
        """매우 관대한 종료 - 정말 심각한 경우만"""
        
        body_height = self.data.qpos[2]
        
        # 1. 완전히 땅에 파묻힌 경우만
        if body_height < -0.1:  # 지면 아래 10cm
            return True
        
        # 2. 너무 멀리 간 경우
        if abs(self.data.qpos[0]) > 100.0:  # 100m 이상
            return True
        
        if abs(self.data.qpos[1]) > 50.0:  # 옆으로 50m
            return True
        
        # 3. NaN 발생
        if (np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)) or
            np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel))):
            return True
        
        # 넘어져도 OK, 뒤집어져도 OK - 계속 시도!
        return False
    
    def step(self, action):
        # 강한 액션 허용
        action = np.clip(action, -40.0, 40.0)
        self.data.ctrl[:] = action
        
        # 시뮬레이션
        mj.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        observation = self._get_observation()
        reward, reward_info = self._get_reward()
        terminated = self._is_terminated()
        truncated = False  # 시간 제한 없음
        
        # 매 100스텝마다 상태 출력
        if self.current_step % 100 == 0:
            forward_vel = self.data.qvel[0]
            height = self.data.qpos[2]
            print(f"스텝 {self.current_step}: 전진속도 {forward_vel:.3f}m/s, 높이 {height:.3f}m, 보상 {reward:.1f}")
        
        # 렌더링
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, reward_info
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # 카메라 설정
                self.viewer.cam.distance = 3.5
                self.viewer.cam.elevation = -20
                self.viewer.cam.azimuth = 135
            else:
                # 로봇 따라가기
                self.viewer.cam.lookat[0] = self.data.qpos[0]
                self.viewer.cam.lookat[1] = self.data.qpos[1]
            self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        # 임시 파일 정리
        if hasattr(self, 'model_path') and self.model_path.startswith('/tmp'):
            try:
                os.unlink(self.model_path)
            except:
                pass


if __name__ == "__main__":
    print("🔥 공격적 전진 환경 테스트")
    
    env = AggressiveForwardGO2Env(render_mode="human")
    
    obs, info = env.reset()
    
    for step in range(1000):
        # 랜덤 액션 (강한 토크)
        action = np.random.uniform(-20, 20, env.action_space.shape[0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"에피소드 종료! 총 스텝: {step}")
            obs, info = env.reset()
    
    env.close()