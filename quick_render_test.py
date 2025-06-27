#!/usr/bin/env python3
"""
빠른 렌더링 테스트
"""

import os
import sys
import time

# 프로젝트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.join(current_dir, "rl")
sys.path.append(rl_dir)

from training.integrated.train_integrated import IntegratedTrainer

def quick_render_test():
    print("=== 빠른 렌더링 테스트 ===")
    
    # 매우 짧은 설정으로 훈련
    trainer = IntegratedTrainer(
        total_timesteps=200,  # 200 스텝만
        eval_freq=1000,  # 평가 건너뛰기
        save_freq=1000,  # 저장 건너뛰기  
        log_freq=50,     # 자주 로그
        render_training=True  # 렌더링 활성화
    )
    
    print("훈련 시작...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("사용자 중단")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.env.close()
    
    print("테스트 완료")

if __name__ == "__main__":
    quick_render_test()