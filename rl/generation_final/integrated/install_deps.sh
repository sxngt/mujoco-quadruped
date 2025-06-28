#!/bin/bash
# 의존성 설치 스크립트

echo "🔧 GO2 환경 의존성 설치 중..."
echo ""

# Python 버전 체크
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "🐍 Python 버전: $python_version"

# 필수 패키지 목록
PACKAGES=(
    "gymnasium[mujoco]"
    "stable-baselines3[extra]"
    "tensorboard"
    "tqdm"
    "mujoco"
    "torch"
    "numpy"
    "matplotlib"
)

# uv가 있으면 uv 사용
if command -v uv &> /dev/null; then
    echo "📦 uv로 설치..."
    echo ""
    
    # 먼저 프로젝트 루트에서 sync 시도
    PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
    if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        echo "프로젝트 루트에서 uv sync 실행..."
        (cd "$PROJECT_ROOT" && uv sync)
    fi
    
    # 개별 패키지 설치
    for package in "${PACKAGES[@]}"; do
        echo "설치 중: $package"
        uv pip install "$package"
    done
else
    echo "📦 pip로 설치..."
    echo ""
    
    # pip 업그레이드
    pip install --upgrade pip
    
    # 패키지 설치
    for package in "${PACKAGES[@]}"; do
        echo "설치 중: $package"
        pip install "$package"
    done
fi

echo ""
echo "🔍 설치 확인 중..."
python3 check_deps.py

echo ""
echo "✅ 설치 완료!"
echo ""
echo "다음 명령으로 훈련을 시작하세요:"
echo "  uv run python train_sb3.py --run train --total_timesteps 100000"
echo ""
echo "TensorBoard 없이 실행하려면:"
echo "  uv run python train_sb3.py --run train --no_tensorboard --total_timesteps 100000"