#!/usr/bin/env python3
"""
테스트 실행 스크립트
"""

import sys
import os
import argparse

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_walking_test():
    """보행 개선 테스트"""
    from experiments.walking_improvements.test_walking_improvements import test_walking_behavior
    return test_walking_behavior()

def run_gait_test():
    """보행 패턴 테스트"""
    from experiments.gait_research.test_gait import main as test_gait_main
    test_gait_main()

def run_gpu_test():
    """GPU 최적화 테스트"""
    from experiments.gpu_optimization.test_gpu_optimization import main as test_gpu_main
    test_gpu_main()

def main():
    parser = argparse.ArgumentParser(description='RL 테스트 실행')
    parser.add_argument('--test', choices=['walking', 'gait', 'gpu', 'all'], 
                       default='all', help='실행할 테스트 선택')
    
    args = parser.parse_args()
    
    if args.test == 'walking' or args.test == 'all':
        print("🚶‍♀️ 보행 개선 테스트 실행...")
        run_walking_test()
    
    if args.test == 'gait' or args.test == 'all':
        print("👣 보행 패턴 테스트 실행...")
        run_gait_test()
    
    if args.test == 'gpu' or args.test == 'all':
        print("⚡ GPU 최적화 테스트 실행...")
        run_gpu_test()

if __name__ == "__main__":
    main()