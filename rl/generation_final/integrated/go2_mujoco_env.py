from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco
import numpy as np
from pathlib import Path
import os
import tempfile


DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "trackbodyid": -1,
    "type": 2,
}


class GO2MujocoEnv(MujocoEnv):
    """GO2 환경 - 참조 레포지터리 완전 모방"""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, ctrl_type="torque", **kwargs):
        # XML 파일 준비
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
            model_path = f.name

        # MujocoEnv 초기화 (참조 레포지터리와 동일)
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=10,  # 핵심! 참조와 동일한 프레임 스킵
            observation_space=None,  # 나중에 수동 설정
            **kwargs,
        )

        # 메타데이터 업데이트
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array", 
                "depth_array",
            ],
            "render_fps": 60,
        }
        self._last_render_time = -1.0
        self._max_episode_time_sec = 30.0  # 30초로 증가
        self._step = 0

        # 보상/페널티 가중치 (참조 레포지터리와 동일)
        self.reward_weights = {
            "linear_vel_tracking": 2.0,
            "angular_vel_tracking": 1.0,
            "healthy": 0.0,
            "feet_airtime": 1.0,
        }
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 2.0,
            "xy_angular_vel": 0.05,
            "action_rate": 0.01,
            "joint_limit": 10.0,
            "joint_velocity": 0.01,
            "joint_acceleration": 2.5e-7,
            "orientation": 1.0,
            "collision": 1.0,
            "default_joint_position": 0.1
        }

        self._curriculum_base = 0.3
        self._gravity_vector = np.array(self.model.opt.gravity)
        
        # 키프레임에서 기본 관절 위치 가져오기
        try:
            self._default_joint_position = np.array(self.model.key_ctrl[0])
        except:
            # 키프레임이 없으면 0으로 설정
            self._default_joint_position = np.zeros(12)

        # 목표 속도 설정 (참조와 동일)
        self._desired_velocity_min = np.array([0.5, -0.0, -0.0])
        self._desired_velocity_max = np.array([0.5, 0.0, 0.0])
        self._desired_velocity = self._sample_desired_vel()

        # 관찰 스케일링 (참조와 동일)
        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        self._tracking_velocity_sigma = 0.25

        # 건강 상태 범위 (GO2에 맞게 조정)
        self._healthy_z_range = (0.15, 0.45)  # GO2가 더 낮음
        self._healthy_pitch_range = (-np.deg2rad(30), np.deg2rad(30))  # 더 관대하게
        self._healthy_roll_range = (-np.deg2rad(30), np.deg2rad(30))  # 더 관대하게

        # 발 관련 추적
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        
        # GO2용 발 인덱스 (cfrc_ext 배열에서)
        self._cfrc_ext_feet_indices = []
        self._cfrc_ext_contact_indices = []
        
        # GO2 발 이름으로 인덱스 찾기
        feet_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        for name in feet_names:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                self._cfrc_ext_feet_indices.append(body_id)
            except:
                pass
        
        # 관절 범위 설정 (참조 방식)
        dof_position_limit_multiplier = 0.9
        ctrl_range_offset = (
            0.5
            * (1 - dof_position_limit_multiplier)
            * (
                self.model.actuator_ctrlrange[:, 1]
                - self.model.actuator_ctrlrange[:, 0]
            )
        )
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1

        # 액션: 12개 토크 값
        self._last_action = np.zeros(12)

        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        # 발 사이트 이름-ID 매핑
        feet_site = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._feet_site_name_to_id = {}
        for f in feet_site:
            try:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f)
                self._feet_site_name_to_id[f] = site_id
            except:
                pass

        # 메인 몸체 ID
        try:
            self._main_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
            )
        except:
            self._main_body_id = 1  # 기본값

    def step(self, action):
        self._step += 1
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action)
        terminated = not self.is_healthy
        truncated = self._step >= (self._max_episode_time_sec / self.dt)
        
        # 디버그 정보
        if terminated and self._step < 100:  # 100스텝 이내 종료 시 출력
            z_pos = self.data.qpos[2]
            quat = self.data.qpos[3:7]
            roll, pitch, yaw = self.euler_from_quaternion(*quat)
            print(f"⚠️ 조기 종료: step={self._step}, z={z_pos:.3f}m, roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")
        
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "episode_length": self._step,
            "z_position": self.data.qpos[2],
            **reward_info,
        }

        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (
            1.0 / self.metadata["render_fps"]
        ):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action

        return observation, reward, terminated, truncated, info

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z

        # 오일러 각도로 변환하여 체크
        quat = self.data.qpos[3:7]
        roll, pitch, yaw = self.euler_from_quaternion(*quat)
        
        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= roll <= max_roll

        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= pitch <= max_pitch

        return is_healthy

    @property
    def projected_gravity(self):
        w, x, y, z = self.data.qpos[3:7]
        euler_orientation = np.array(self.euler_from_quaternion(w, x, y, z))
        projected_gravity_not_normalized = (
            np.dot(self._gravity_vector, euler_orientation) * euler_orientation
        )
        if np.linalg.norm(projected_gravity_not_normalized) == 0:
            return projected_gravity_not_normalized
        else:
            return projected_gravity_not_normalized / np.linalg.norm(
                projected_gravity_not_normalized
            )

    @property
    def feet_contact_forces(self):
        if len(self._cfrc_ext_feet_indices) > 0:
            feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices]
            return np.linalg.norm(feet_contact_forces, axis=1)
        else:
            return np.zeros(4)

    ######### 양의 보상 함수들 #########
    @property
    def linear_velocity_tracking_reward(self):
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - self.data.qvel[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def angular_velocity_tracking_reward(self):
        vel_sqr_error = np.square(self._desired_velocity[2] - self.data.qvel[5])
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def feet_air_time_reward(self):
        """공중 시간에 따른 보상"""
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        air_time_reward = np.sum((self._feet_air_time - 1.0) * first_contact)
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1

        self._feet_air_time *= ~contact_filter

        return air_time_reward

    @property
    def healthy_reward(self):
        return self.is_healthy

    ######### 음의 보상 함수들 #########
    @property
    def non_flat_base_cost(self):
        return np.sum(np.square(self.projected_gravity[:2]))

    @property
    def collision_cost(self):
        if len(self._cfrc_ext_contact_indices) > 0:
            return np.sum(
                1.0
                * (np.linalg.norm(self.data.cfrc_ext[self._cfrc_ext_contact_indices]) > 0.1)
            )
        else:
            return 0.0

    @property
    def joint_limit_cost(self):
        out_of_range = (self._soft_joint_range[:, 0] - self.data.qpos[7:]).clip(
            min=0.0
        ) + (self.data.qpos[7:] - self._soft_joint_range[:, 1]).clip(min=0.0)
        return np.sum(out_of_range)

    @property
    def torque_cost(self):
        return np.sum(np.square(self.data.qfrc_actuator[-12:]))

    @property
    def vertical_velocity_cost(self):
        return np.square(self.data.qvel[2])

    @property
    def xy_angular_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[3:5]))

    def action_rate_cost(self, action):
        return np.sum(np.square(self._last_action - action))

    @property
    def joint_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[6:]))

    @property
    def acceleration_cost(self):
        return np.sum(np.square(self.data.qacc[6:]))

    @property
    def default_joint_position_cost(self):
        return np.sum(np.square(self.data.qpos[7:] - self._default_joint_position))

    def _calc_reward(self, action):
        # 양의 보상
        linear_vel_tracking_reward = (
            self.linear_velocity_tracking_reward
            * self.reward_weights["linear_vel_tracking"]
        )
        angular_vel_tracking_reward = (
            self.angular_velocity_tracking_reward
            * self.reward_weights["angular_vel_tracking"]
        )
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_time_reward = (
            self.feet_air_time_reward * self.reward_weights["feet_airtime"]
        )
        rewards = (
            linear_vel_tracking_reward
            + angular_vel_tracking_reward
            + healthy_reward
            + feet_air_time_reward
        )

        # 음의 비용
        ctrl_cost = self.torque_cost * self.cost_weights["torque"]
        action_rate_cost = (
            self.action_rate_cost(action) * self.cost_weights["action_rate"]
        )
        vertical_vel_cost = (
            self.vertical_velocity_cost * self.cost_weights["vertical_vel"]
        )
        xy_angular_vel_cost = (
            self.xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"]
        )
        joint_limit_cost = self.joint_limit_cost * self.cost_weights["joint_limit"]
        joint_velocity_cost = (
            self.joint_velocity_cost * self.cost_weights["joint_velocity"]
        )
        joint_acceleration_cost = (
            self.acceleration_cost * self.cost_weights["joint_acceleration"]
        )
        orientation_cost = self.non_flat_base_cost * self.cost_weights["orientation"]
        collision_cost = self.collision_cost * self.cost_weights["collision"]
        default_joint_position_cost = (
            self.default_joint_position_cost
            * self.cost_weights["default_joint_position"]
        )
        costs = (
            ctrl_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
            + joint_acceleration_cost
            + orientation_cost
            + default_joint_position_cost
        )

        reward = max(0.0, rewards - costs)  # 참조와 동일하게 0 이하 클리핑
        reward_info = {
            "linear_vel_tracking_reward": linear_vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        # 관절 위치 (기본 위치 대비 상대적)
        dofs_position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]

        # 속도 정보
        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]
        dofs_velocity = velocity[6:]

        desired_vel = self._desired_velocity
        last_action = self._last_action
        projected_gravity = self.projected_gravity

        curr_obs = np.concatenate(
            (
                base_linear_velocity * self._obs_scale["linear_velocity"],
                base_angular_velocity * self._obs_scale["angular_velocity"],
                projected_gravity,
                desired_vel * self._obs_scale["linear_velocity"],
                dofs_position * self._obs_scale["dofs_position"],
                dofs_velocity * self._obs_scale["dofs_velocity"],
                last_action,
            )
        ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)

        return curr_obs

    def reset_model(self):
        # 위치와 제어값을 노이즈와 함께 리셋
        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        self.data.ctrl[:] = self.model.key_ctrl[
            0
        ] + self._reset_noise_scale * self.np_random.standard_normal(
            *self.data.ctrl.shape
        )

        # 변수 리셋 및 새 목표 속도 샘플링
        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def _sample_desired_vel(self):
        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        return desired_vel

    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        """쿼터니언을 오일러 각도로 변환"""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z