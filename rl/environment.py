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
        # Primary reward: forward velocity
        forward_vel = self.data.qvel[0]  # x-velocity
        forward_reward = np.clip(forward_vel * 1.5, 0, 4.0)  # Encourage forward movement
        
        # Stability reward: maintain upright orientation
        body_quat = self.data.qpos[3:7]  # [w, x, y, z]
        # Calculate uprightness using z-axis alignment
        z_axis = np.array([2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
                          2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
                          body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2])
        upright_reward = max(0, z_axis[2])  # Only positive when upright
        
        # Energy penalty: penalize high torques
        ctrl_penalty = -0.0005 * np.sum(np.square(self.current_action))
        
        # Height reward: maintain reasonable height
        body_height = self.data.qpos[2]
        target_height = 0.27  # Target standing height
        height_reward = 1.0 - abs(body_height - target_height) * 10.0
        height_reward = max(height_reward, -2.0)  # Cap negative reward
        
        # Angular velocity penalty: discourage excessive rotation
        ang_vel_penalty = -0.005 * np.sum(np.square(self.data.qvel[3:6]))
        
        # Foot contact reward: encourage proper gait
        contacts = self._get_contact_info()
        num_contacts = sum(1 for contact in contacts.values() if contact['in_contact'])
        contact_reward = 0.1 if 1 <= num_contacts <= 3 else -0.1  # Prefer 1-3 feet in contact
        
        # Lateral velocity penalty: discourage sideways drift
        lateral_vel_penalty = -0.1 * abs(self.data.qvel[1])  # y-velocity
        
        # Joint smoothness reward: penalize rapid joint changes
        joint_vel_penalty = -0.001 * np.sum(np.square(self.data.qvel[6:]))
        
        total_reward = (forward_reward + upright_reward + ctrl_penalty + 
                       height_reward + ang_vel_penalty + contact_reward + 
                       lateral_vel_penalty + joint_vel_penalty)
        
        return total_reward, {
            'forward': forward_reward,
            'upright': upright_reward, 
            'control': ctrl_penalty,
            'height': height_reward,
            'ang_vel': ang_vel_penalty,
            'contact': contact_reward,
            'lateral': lateral_vel_penalty,
            'joint_smooth': joint_vel_penalty
        }
    
    def _is_terminated(self):
        # Terminate if robot falls or flips
        body_height = self.data.qpos[2]
        body_quat = self.data.qpos[3:7]  # [w, x, y, z]
        
        # Check if fallen (too low)
        if body_height < 0.12:  # More forgiving height limit
            return True
            
        # Check orientation using z-axis alignment
        z_axis = np.array([2*(body_quat[1]*body_quat[3] + body_quat[0]*body_quat[2]),
                          2*(body_quat[2]*body_quat[3] - body_quat[0]*body_quat[1]),
                          body_quat[0]**2 - body_quat[1]**2 - body_quat[2]**2 + body_quat[3]**2])
        
        # Terminate if robot is too tilted (z-component < 0.2)
        if z_axis[2] < 0.2:  # More forgiving tilt limit
            return True
        
        # Check for excessive lateral displacement
        if abs(self.data.qpos[1]) > 2.0:  # Too far sideways
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
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mj.Renderer(self.model, width=640, height=480)
            self.viewer.update_scene(self.data)
            return self.viewer.render()
        elif self.render_mode == "depth_array":
            if self.viewer is None:
                self.viewer = mj.Renderer(self.model, width=640, height=480)
            self.viewer.update_scene(self.data)
            return self.viewer.render(depth=True)
        return None
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None