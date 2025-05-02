import threading
import time
from config import Config

class KeyboardController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.running = True
        self.input_thread = None
        self.command_queue = []
        self.queue_lock = threading.Lock()
        
        # 입력 스레드 시작
        self.start_input_thread()
    
    def start_input_thread(self):
        """별도 스레드에서 키보드 입력 받기"""
        def input_worker():
            print("Press Enter after each command:")
            while self.running:
                try:
                    command = input().strip().lower()
                    if command:
                        with self.queue_lock:
                            self.command_queue.append(command)
                except (EOFError, KeyboardInterrupt):
                    self.running = False
                    break
        
        self.input_thread = threading.Thread(target=input_worker, daemon=True)
        self.input_thread.start()
    
    def handle_command(self, command):
        """명령 처리"""
        if command == 'exit' or command == 'quit':
            print("Exiting...")
            self.running = False
            return True
        
        # 연속 명령 처리 (예: "qqq" -> q를 3번)
        for char in command:
            if char in Config.KEY_MAPPINGS:
                mapping = Config.KEY_MAPPINGS[char]
                
                if mapping == 'reset':
                    self.robot_model.reset_to_standing_pose()
                    print("Robot reset to standing pose")
                else:
                    joint_name, direction = mapping
                    delta = direction * Config.ANGLE_STEP
                    
                    if self.robot_model.update_joint_angle(joint_name, delta):
                        angle = self.robot_model.get_joint_angle(joint_name)
                        print(f"{joint_name}: {angle:.3f} rad")
        return False
    
    def process_input(self):
        """큐에서 명령 처리"""
        with self.queue_lock:
            while self.command_queue:
                command = self.command_queue.pop(0)
                self.handle_command(command)
    
    def is_running(self):
        return self.running
    
    def print_help(self):
        print("\n=== Unitree GO2 Joint Controls ===")
        print("Front Left Leg (FL):")
        print("  q/w - Hip 좌우벌림 (안쪽/바깥쪽)")
        print("  e/r - Thigh 앞뒤움직임 (뒤/앞)")  
        print("  t/y - Calf 무릎 (굽히기/펴기)")
        print()
        print("Front Right Leg (FR):")
        print("  u/i - Hip 좌우벌림 (안쪽/바깥쪽)")
        print("  o/p - Thigh 앞뒤움직임 (뒤/앞)")
        print("  [/] - Calf 무릎 (굽히기/펴기)")
        print()
        print("Rear Left Leg (RL):")
        print("  a/s - Hip 좌우벌림 (안쪽/바깥쪽)")
        print("  d/f - Thigh 앞뒤움직임 (뒤/앞)")
        print("  g/h - Calf 무릎 (굽히기/펴기)")
        print()
        print("Rear Right Leg (RR):")
        print("  z/x - Hip 좌우벌림 (안쪽/바깥쪽)")
        print("  c/v - Thigh 앞뒤움직임 (뒤/앞)")
        print("  b/n - Calf 무릎 (굽히기/펴기)")
        print()
        print("SPACE: Reset to standing pose")
        print("Multiple chars: qqq (repeat 3 times)")
        print("Type 'exit' or 'quit' to end")
        print("====================================\n")