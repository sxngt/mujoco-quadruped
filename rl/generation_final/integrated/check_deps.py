#!/usr/bin/env python3
"""의존성 체크 스크립트"""

import sys
import importlib.util

def check_package(name, import_name=None):
    """패키지 설치 상태 확인"""
    if import_name is None:
        import_name = name
    
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"❌ {name}: 설치되지 않음")
        return False
    else:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
            return True
        except Exception as e:
            print(f"⚠️  {name}: 설치됨 but 오류 ({e})")
            return False

print("🔍 의존성 체크 중...\n")

required_packages = [
    ("gymnasium", None),
    ("mujoco", None),
    ("stable_baselines3", "stable_baselines3"),
    ("tensorboard", None),
    ("torch", None),
    ("tqdm", None),
    ("numpy", None),
    ("matplotlib", None),
]

missing = []
for package, import_name in required_packages:
    if not check_package(package, import_name):
        missing.append(package)

print("\n" + "="*40)
if missing:
    print(f"❌ 누락된 패키지: {', '.join(missing)}")
    print("\n설치 방법:")
    print(f"  pip install {' '.join(missing)}")
else:
    print("✅ 모든 필수 패키지가 설치되어 있습니다!")

# Python 버전 체크
print(f"\n🐍 Python 버전: {sys.version}")

# 가상환경 체크
print(f"🗂️  실행 경로: {sys.executable}")
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✅ 가상환경 활성화됨")
else:
    print("⚠️  가상환경이 아닙니다")