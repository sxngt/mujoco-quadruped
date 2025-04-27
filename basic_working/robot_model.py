import mujoco as mj
import numpy as np
from config import Config

class RobotModel:
    def __init__(self):
        self.model = mj.MjModel.from_xml_path(Config.MODEL_PATH)
        self.data = mj.MjData(self.model)
        self.joint_angles = {}
        self.joint_indices = {}
        
        for joint_name in Config.JOINT_NAMES:
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.joint_indices[joint_name] = joint_id
                self.joint_angles[joint_name] = 0.0
        
        self.reset_to_standing_pose()
    
    def reset_to_standing_pose(self):
        # "home" 키프레임 사용 (인덱스 0)
        mj.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # 조인트 각도 업데이트
        for i, joint_name in enumerate(Config.JOINT_NAMES):
            if joint_name in self.joint_indices:
                joint_idx = self.joint_indices[joint_name]
                # qpos에서 조인트 위치 읽기 (free joint 7개 + 조인트 ID는 1부터 시작)
                qpos_idx = joint_idx + 6  # 7 - 1 = 6
                if qpos_idx < len(self.data.qpos):
                    self.joint_angles[joint_name] = self.data.qpos[qpos_idx]
        
        mj.mj_forward(self.model, self.data)
    
    def update_joint_angle(self, joint_name, delta):
        if joint_name not in self.joint_indices:
            return False
        
        current_angle = self.joint_angles[joint_name]
        new_angle = current_angle + delta
        
        joint_range = Config.JOINT_RANGES.get(joint_name, (-np.pi, np.pi))
        new_angle = np.clip(new_angle, joint_range[0], joint_range[1])
        
        if new_angle != current_angle:
            self.joint_angles[joint_name] = new_angle
            joint_idx = self.joint_indices[joint_name]
            qpos_idx = joint_idx + 6  # 7 - 1 = 6
            if qpos_idx < len(self.data.qpos):
                self.data.qpos[qpos_idx] = new_angle
                mj.mj_forward(self.model, self.data)
                return True
        return False
    
    def get_joint_angle(self, joint_name):
        return self.joint_angles.get(joint_name, 0.0)
    
    def get_all_joint_angles(self):
        return self.joint_angles.copy()
    
    def set_joint_angles(self, angles_dict):
        for joint_name, angle in angles_dict.items():
            if joint_name in self.joint_indices:
                joint_range = Config.JOINT_RANGES.get(joint_name, (-np.pi, np.pi))
                angle = np.clip(angle, joint_range[0], joint_range[1])
                self.joint_angles[joint_name] = angle
                joint_idx = self.joint_indices[joint_name]
                qpos_idx = joint_idx + 6  # 7 - 1 = 6
                if qpos_idx < len(self.data.qpos):
                    self.data.qpos[qpos_idx] = angle
        mj.mj_forward(self.model, self.data)