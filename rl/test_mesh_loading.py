#!/usr/bin/env python3
"""
메시 파일 로딩 테스트
"""

import sys
import os

# rl 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.getcwd())

def test_mesh_loading():
    print("🔧 메시 파일 경로 테스트")
    
    # 경로 확인
    base_dir = os.getcwd()
    xml_path = os.path.join(base_dir, "assets", "go2_scene.xml")
    mesh_dir = os.path.join(os.path.dirname(base_dir), "mujoco_menagerie", "unitree_go2", "assets")
    
    print(f"RL 디렉토리: {base_dir}")
    print(f"XML 파일: {xml_path}")
    print(f"메시 디렉토리: {mesh_dir}")
    print(f"XML 파일 존재: {os.path.exists(xml_path)}")
    print(f"메시 디렉토리 존재: {os.path.exists(mesh_dir)}")
    
    if os.path.exists(mesh_dir):
        mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj')]
        print(f"메시 파일 개수: {len(mesh_files)}")
        print(f"base_0.obj 존재: {'base_0.obj' in mesh_files}")
    
    print("\n🚀 환경 로딩 테스트")
    try:
        from environments.improved.improved_environment import ImprovedGO2Env
        print("✅ 클래스 import 성공")
        
        env = ImprovedGO2Env(render_mode=None, use_reference_gait=False)
        print("✅ 환경 생성 성공!")
        print(f"모델 로딩 성공")
        
        env.close()
        print("✅ 환경 정리 완료")
        
    except Exception as e:
        print(f"❌ 환경 로딩 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mesh_loading()