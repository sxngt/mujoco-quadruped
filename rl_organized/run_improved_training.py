#!/usr/bin/env python3
"""
개선된 보행 학습 실행 스크립트
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    from training.vectorized.improved_train import main as train_main
    train_main()

if __name__ == "__main__":
    main()