#!/bin/bash
# 의존성 설치 스크립트

echo "🔧 GO2 환경 의존성 설치 중..."

# uv가 있으면 uv 사용
if command -v uv &> /dev/null; then
    echo "📦 uv로 설치..."
    uv pip install tensorboard tqdm
else
    echo "📦 pip로 설치..."
    pip install tensorboard tqdm
fi

echo "✅ 설치 완료!"
echo ""
echo "다음 명령으로 훈련을 시작하세요:"
echo "  uv run python train_sb3.py --run train --total_timesteps 100000"
echo ""
echo "TensorBoard 없이 실행하려면:"
echo "  uv run python train_sb3.py --run train --no_tensorboard --total_timesteps 100000"