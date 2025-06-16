#!/usr/bin/env python3
"""
개선된 보행 학습 실행 스크립트
"""

import sys
import os

# rl 디렉토리를 Python 경로에 추가
rl_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, rl_dir)

def main():
    # 절대 import 사용
    from training.vectorized.improved_train import main as train_main
    train_main()

if __name__ == "__main__":
    main()