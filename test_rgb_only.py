#!/usr/bin/env python3
import mujoco as mj

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
data = mj.MjData(model)

print("=== Renderer 클래스 테스트 ===")
try:
    # 올바른 방법 시도
    renderer = mj.Renderer(model, height=480, width=640)
    print("✅ Renderer 생성 성공")
    
    # update_scene 메서드 확인
    if hasattr(renderer, 'update_scene'):
        renderer.update_scene(data)
        print("✅ update_scene 메서드 존재")
    else:
        print("❌ update_scene 메서드 없음")
        print("사용 가능한 메서드:", [m for m in dir(renderer) if not m.startswith('_')])
    
    # render 메서드 확인
    if hasattr(renderer, 'render'):
        img = renderer.render()
        print(f"✅ render 성공: {img.shape}")
    else:
        print("❌ render 메서드 없음")
        
except Exception as e:
    print(f"❌ Renderer 테스트 실패: {e}")
    import traceback
    traceback.print_exc()