import os

class Config:
    MODEL_PATH = "scene.xml"
    
    JOINT_NAMES = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
    ]
    
    JOINT_RANGES = {
        # Hip joints (abduction) - X축 회전, 좌우 벌림/모음
        "FL_hip_joint": (-0.436, 0.524),      # -25° ~ 30°
        "FR_hip_joint": (-0.436, 0.524),      # -25° ~ 30°
        "RL_hip_joint": (-0.436, 0.524),      # -25° ~ 30°  
        "RR_hip_joint": (-0.436, 0.524),      # -25° ~ 30°
        
        # Thigh joints - Y축 회전, 앞뒤 움직임
        "FL_thigh_joint": (-2.094, 1.396),    # -120° ~ 80°
        "FR_thigh_joint": (-2.094, 1.396),    # -120° ~ 80°
        "RL_thigh_joint": (-2.094, 1.396),    # -120° ~ 80°
        "RR_thigh_joint": (-2.094, 1.396),    # -120° ~ 80°
        
        # Calf joints (knee) - Y축 회전, 무릎 굽히기
        "FL_calf_joint": (-0.785, 0.960),     # -45° ~ 55°
        "FR_calf_joint": (-0.785, 0.960),     # -45° ~ 55°
        "RL_calf_joint": (-0.785, 0.960),     # -45° ~ 55°
        "RR_calf_joint": (-0.785, 0.960)      # -45° ~ 55°
    }
    
    # 키프레임 "home"에서 가져온 실제 서있는 포즈
    DEFAULT_STANDING_POSE = [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]
    
    ANGLE_STEP = 0.1
    
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    
    KEY_MAPPINGS = {
        # Front Left Leg (FL)
        'q': ('FL_hip_joint', -1),    # 좌우 벌림 (안쪽으로)
        'w': ('FL_hip_joint', 1),     # 좌우 벌림 (바깥쪽으로) 
        'e': ('FL_thigh_joint', -1),  # 앞뒤 움직임 (뒤로)
        'r': ('FL_thigh_joint', 1),   # 앞뒤 움직임 (앞으로)
        't': ('FL_calf_joint', -1),   # 무릎 굽히기 (접기)
        'y': ('FL_calf_joint', 1),    # 무릎 펴기 (펴기)
        
        # Front Right Leg (FR)
        'u': ('FR_hip_joint', -1),    # 좌우 벌림 (안쪽으로)
        'i': ('FR_hip_joint', 1),     # 좌우 벌림 (바깥쪽으로)
        'o': ('FR_thigh_joint', -1),  # 앞뒤 움직임 (뒤로)
        'p': ('FR_thigh_joint', 1),   # 앞뒤 움직임 (앞으로)
        '[': ('FR_calf_joint', -1),   # 무릎 굽히기 (접기)
        ']': ('FR_calf_joint', 1),    # 무릎 펴기 (펴기)
        
        # Rear Left Leg (RL)  
        'a': ('RL_hip_joint', -1),    # 좌우 벌림 (안쪽으로)
        's': ('RL_hip_joint', 1),     # 좌우 벌림 (바깥쪽으로)
        'd': ('RL_thigh_joint', -1),  # 앞뒤 움직임 (뒤로)
        'f': ('RL_thigh_joint', 1),   # 앞뒤 움직임 (앞으로)
        'g': ('RL_calf_joint', -1),   # 무릎 굽히기 (접기)
        'h': ('RL_calf_joint', 1),    # 무릎 펴기 (펴기)
        
        # Rear Right Leg (RR)
        'z': ('RR_hip_joint', -1),    # 좌우 벌림 (안쪽으로)
        'x': ('RR_hip_joint', 1),     # 좌우 벌림 (바깥쪽으로)
        'c': ('RR_thigh_joint', -1),  # 앞뒤 움직임 (뒤로)
        'v': ('RR_thigh_joint', 1),   # 앞뒤 움직임 (앞으로)
        'b': ('RR_calf_joint', -1),   # 무릎 굽히기 (접기)
        'n': ('RR_calf_joint', 1),    # 무릎 펴기 (펴기)
        
        ' ': 'reset'
    }