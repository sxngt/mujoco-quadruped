#!/usr/bin/env python3
"""
MuJoCo 공식 문서 기반 올바른 렌더링 구현 예제
"""

import mujoco as mj
import numpy as np
import time

class CorrectMuJoCoRenderer:
    """공식 문서 기반 올바른 MuJoCo 렌더링 구현"""
    
    def __init__(self, model, render_mode=None):
        self.model = model
        self.data = mj.MjData(model)
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        
    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None
    
    def _render_human(self):
        """인간이 볼 수 있는 GUI 렌더링"""
        if self.viewer is None:
            try:
                import mujoco.viewer
                # 공식 문서 권장: passive viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                if self.viewer is not None:
                    # 카메라 설정
                    self.viewer.cam.distance = 3.0
                    self.viewer.cam.elevation = -20
                    self.viewer.cam.azimuth = 135
                    self.viewer.cam.lookat[:] = [0, 0, 0.3]
                else:
                    raise RuntimeError("passive viewer 생성 실패")
            except Exception as e:
                print(f"Passive viewer 실패: {e}")
                try:
                    # 대안: blocking viewer (권장하지 않음)
                    print("Blocking viewer 시도...")
                    self.viewer = mujoco.viewer.launch(self.model, self.data)
                except Exception as e2:
                    print(f"Blocking viewer도 실패: {e2}")
                    return None
        
        if self.viewer is not None:
            try:
                # 공식 문서 권장 방법
                if hasattr(self.viewer, 'is_running') and not self.viewer.is_running():
                    return None
                
                # 데이터 동기화
                self.viewer.sync()
                return True
            except Exception as e:
                print(f"뷰어 동기화 실패: {e}")
                return None
        
        return None
    
    def _render_rgb_array(self):
        """RGB 배열 렌더링 (헤드리스 환경용)"""
        if self.renderer is None:
            # 공식 문서 기준: (model, height, width) 순서
            self.renderer = mj.Renderer(self.model, height=480, width=640)
        
        # 공식 문서 기준 렌더링 방법
        self.renderer.update_scene(self.data)
        rgb_array = self.renderer.render()
        
        # RGB 배열은 (height, width, 3) 형태여야 함
        if rgb_array.shape[-1] != 3:
            raise RuntimeError(f"잘못된 RGB 배열 형태: {rgb_array.shape}")
        
        return rgb_array
    
    def close(self):
        """리소스 정리"""
        if self.viewer is not None:
            try:
                if hasattr(self.viewer, 'close'):
                    self.viewer.close()
                elif hasattr(self.viewer, 'is_running'):
                    # passive viewer의 경우
                    pass  # 자동으로 정리됨
            except Exception as e:
                print(f"뷰어 종료 실패: {e}")
            finally:
                self.viewer = None
        
        if self.renderer is not None:
            try:
                if hasattr(self.renderer, 'close'):
                    self.renderer.close()
            except Exception as e:
                print(f"렌더러 종료 실패: {e}")
            finally:
                self.renderer = None


def test_correct_rendering():
    """올바른 렌더링 구현 테스트"""
    
    # 간단한 테스트 모델
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
    
    model = mj.MjModel.from_xml_string(xml_string)
    
    print("=== RGB 배열 렌더링 테스트 ===")
    renderer = CorrectMuJoCoRenderer(model, render_mode="rgb_array")
    
    for i in range(5):
        mj.mj_step(model, renderer.data)
        rgb = renderer.render()
        if rgb is not None:
            print(f"프레임 {i}: RGB 배열 크기 {rgb.shape}, dtype {rgb.dtype}")
        else:
            print(f"프레임 {i}: 렌더링 실패")
    
    renderer.close()
    
    print("\n=== GUI 렌더링 테스트 ===")
    gui_renderer = CorrectMuJoCoRenderer(model, render_mode="human")
    
    for i in range(20):
        mj.mj_step(model, gui_renderer.data)
        result = gui_renderer.render()
        if result:
            print(f"프레임 {i}: GUI 렌더링 성공")
        else:
            print(f"프레임 {i}: GUI 렌더링 실패")
        time.sleep(0.1)
    
    gui_renderer.close()
    print("테스트 완료")


if __name__ == "__main__":
    test_correct_rendering()