import mujoco as mj
import mujoco.viewer
import time
from robot_model import RobotModel
from keyboard_controller import KeyboardController
from config import Config

def main():
    print("Starting Unitree GO2 Interactive Simulation...")
    
    try:
        robot = RobotModel()
        print("Robot model loaded successfully")
        
        controller = KeyboardController(robot)
        controller.print_help()
        
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            
            print("Simulation started. Use keyboard controls in this terminal.")
            
            while viewer.is_running() and controller.is_running():
                step_start = time.time()
                
                # 터미널 키 입력 처리
                controller.process_input()
                
                viewer.sync()
                
                time_until_next_step = robot.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure MuJoCo and the robot model files are properly installed.")
        return 1
    finally:
        # 컨트롤러 정리
        if 'controller' in locals():
            controller.running = False
    
    print("Simulation ended.")
    return 0

if __name__ == "__main__":
    exit(main())