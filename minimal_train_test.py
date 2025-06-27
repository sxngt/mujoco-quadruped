#!/usr/bin/env python3
"""
최소 훈련 테스트 - 렌더링 문제 디버그
"""

import sys
import os
import time

# 올바른 경로 설정
rl_path = os.path.join(os.path.dirname(__file__), 'rl')
sys.path.append(rl_path)

from training.integrated.train_integrated import IntegratedTrainer

def minimal_train_test():
    print("=== 최소 훈련 테스트 ===")
    
    # 최소 설정
    trainer = IntegratedTrainer(
        total_timesteps=20,  # 아주 짧게
        eval_freq=1000,      # 평가 건너뛰기
        save_freq=1000,      # 저장 건너뛰기
        log_freq=5,          # 자주 로그
        render_training=True # 렌더링 활성화
    )
    
    print(f"환경 렌더링 모드: {trainer.env.render_mode}")
    print(f"렌더링 설정: {trainer.render_training}")
    print(f"뷰어 초기 상태: {trainer.env.viewer}")
    
    # 수동으로 훈련 루프 시뮬레이션
    obs, _ = trainer.env.reset()
    print(f"리셋 후 뷰어 상태: {trainer.env.viewer}")
    
    print("수동 훈련 루프 시작...")
    for step in range(5):  # 5스텝만
        print(f"\n=== Step {step+1} ===")
        
        # 에이전트 액션
        action, log_prob, value = trainer.agent.get_action(obs)
        print(f"액션 생성: {action[:3]}... (크기: {action.shape})")
        
        # 환경 스텝
        next_obs, reward, terminated, truncated, info = trainer.env.step(action)
        print(f"환경 스텝: 보상={reward:.3f}, 종료={terminated}")
        
        # 렌더링 확인
        if trainer.render_training:
            print("렌더링 시도...")
            try:
                render_result = trainer.env.render()
                print(f"렌더링 결과: {render_result}")
                print(f"뷰어 상태: {trainer.env.viewer}")
                
                # 뷰어가 있는지 확인
                if trainer.env.viewer is not None:
                    if hasattr(trainer.env.viewer, 'is_running'):
                        print(f"뷰어 실행 상태: {trainer.env.viewer.is_running()}")
                    print("렌더링 성공으로 보임")
                else:
                    print("❌ 뷰어가 None입니다")
                    
            except Exception as e:
                print(f"❌ 렌더링 실패: {e}")
                import traceback
                traceback.print_exc()
        
        obs = next_obs
        time.sleep(0.5)  # 관찰하기 위한 지연
        
        if terminated or truncated:
            print("에피소드 종료, 리셋...")
            obs, _ = trainer.env.reset()
    
    print("\n=== 테스트 완료 ===")
    trainer.env.close()

if __name__ == "__main__":
    try:
        minimal_train_test()
    except KeyboardInterrupt:
        print("사용자 중단")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()