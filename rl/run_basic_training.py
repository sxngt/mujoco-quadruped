#!/usr/bin/env python3
"""
기본 학습 실행 스크립트
상대 경로 문제 해결을 위한 메인 실행 파일
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    from training.basic.train import main as train_main
    train_main()

if __name__ == "__main__":
    main()