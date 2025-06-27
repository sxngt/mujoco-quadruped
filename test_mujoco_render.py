#!/usr/bin/env python3
"""
MuJoCo 렌더링 기본 테스트
macOS M1 환경에서 렌더링이 작동하는지 확인
"""

import mujoco as mj
import numpy as np
import time

def test_basic_rendering():
    print("=== MuJoCo 렌더링 테스트 ===")
    print(f"MuJoCo 버전: {mj.__version__}")
    
    # 간단한 XML 모델 생성
    xml_string = """
    <mujoco>
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
            <body pos="0 0 1">
                <joint type="free"/>
                <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    try:
        # 모델 생성
        print("1. 모델 생성 중...")
        model = mj.MjModel.from_xml_string(xml_string)
        data = mj.MjData(model)
        print("✅ 모델 생성 성공")
        
        # 시뮬레이션 초기화
        print("2. 시뮬레이션 초기화 중...")
        mj.mj_forward(model, data)
        print("✅ 시뮬레이션 초기화 성공")
        
        # 렌더링 테스트 1: rgb_array
        print("3. RGB 배열 렌더링 테스트...")
        renderer = mj.Renderer(model, width=640, height=480)
        renderer.update_scene(data)
        img = renderer.render()
        print(f"✅ RGB 렌더링 성공 - 이미지 크기: {img.shape}")
        
        # 렌더링 테스트 2: viewer (GUI)
        print("4. GUI 뷰어 테스트...")
        try:
            # macOS에서 viewer 테스트
            import mujoco.viewer
            print("  - mujoco.viewer 모듈 로드 성공")
            
            # passive viewer 테스트
            print("  - passive viewer 시도...")
            viewer = mujoco.viewer.launch_passive(model, data)
            if viewer is not None:
                print("  ✅ passive viewer 생성 성공")
                
                # 몇 스텝 실행하며 렌더링
                for i in range(10):
                    mj.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.1)
                print("  ✅ 렌더링 루프 성공")
                
                # 뷰어 종료
                viewer.close()
                print("  ✅ 뷰어 종료 성공")
            else:
                print("  ❌ passive viewer 생성 실패")
                
        except Exception as e:
            print(f"  ❌ GUI 뷰어 테스트 실패: {e}")
            
            # 대안 테스트
            try:
                print("  - 대안 viewer 시도...")
                viewer = mujoco.viewer.launch(model, data)
                print("  ✅ 대안 viewer 성공")
                time.sleep(1)
            except Exception as e2:
                print(f"  ❌ 대안 viewer도 실패: {e2}")
        
        print("\n=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_rendering()