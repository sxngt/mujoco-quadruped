#!/bin/bash
"""
RTX 4080 GPU 최대 활용 학습 스크립트
GPU 사용률 80% 목표
"""

echo "🚀 RTX 4080 최대 활용 학습 시작"
echo "GPU 정보 확인 중..."

# GPU 정보 출력
nvidia-smi

echo ""
echo "🔥 GPU 최적화 환경 변수 설정"

# CUDA 최적화
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 4080 Ampere 아키텍처
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# cuDNN 최적화
export CUDNN_BENCHMARK=1
export CUDNN_DETERMINISTIC=0

# PyTorch 최적화
export OMP_NUM_THREADS=8
export PYTORCH_TENSOREXPR_FALLBACK=0

echo "환경 변수 설정 완료"
echo ""

# 사용자 선택 메뉴
echo "RTX 4080 최적화 학습 모드 선택:"
echo "1) 🔥 GPU 최대 성능 모드 (64 환경, GPU 80% 목표)"
echo "2) ⚡ GPU 고성능 모드 (48 환경, GPU 60% 목표)"  
echo "3) 🎯 GPU 안정 모드 (32 환경, GPU 40% 목표)"
echo "4) 🚀 벡터화 최대 모드 (64 환경, CPU+GPU 혼합)"
echo "5) ⚡ 벡터화 고성능 모드 (32 환경, CPU+GPU 혼합)"
echo "6) 🧪 테스트 모드 (16 환경, 단시간 테스트)"

read -p "모드를 선택하세요 (1-6): " mode

case $mode in
    1)
        echo "🔥 최대 성능 모드 시작"
        uv run python gpu_max_train.py \
            --num_envs 64 \
            --total_timesteps 20000000 \
            --rollout_length 16384 \
            --batch_size 2048 \
            --hidden_dim 512 \
            --lr 1e-4 \
            --ppo_epochs 20 \
            --mixed_precision \
            --wandb \
            --save_freq 500000
        ;;
    2)
        echo "⚡ 고성능 모드 시작"
        uv run python gpu_max_train.py \
            --num_envs 48 \
            --total_timesteps 15000000 \
            --rollout_length 12288 \
            --batch_size 1536 \
            --hidden_dim 512 \
            --lr 1e-4 \
            --ppo_epochs 15 \
            --mixed_precision \
            --wandb \
            --save_freq 400000
        ;;
    3)
        echo "🎯 안정 모드 시작"
        uv run python gpu_max_train.py \
            --num_envs 32 \
            --total_timesteps 10000000 \
            --rollout_length 8192 \
            --batch_size 1024 \
            --hidden_dim 512 \
            --lr 1e-4 \
            --ppo_epochs 12 \
            --mixed_precision \
            --wandb \
            --save_freq 300000
        ;;
    4)
        echo "🚀 벡터화 최대 모드 시작"
        uv run python train_vectorized.py \
            --num_envs 64 \
            --total_timesteps 20000000 \
            --rollout_length 8192 \
            --batch_size 1024 \
            --lr 1e-4 \
            --ppo_epochs 15 \
            --no_reference_gait \
            --wandb \
            --save_freq 500000
        ;;
    5)
        echo "⚡ 벡터화 고성능 모드 시작"
        uv run python train_vectorized.py \
            --num_envs 32 \
            --total_timesteps 15000000 \
            --rollout_length 4096 \
            --batch_size 512 \
            --lr 1e-4 \
            --ppo_epochs 12 \
            --no_reference_gait \
            --wandb \
            --save_freq 400000
        ;;
    6)
        echo "🧪 테스트 모드 시작"
        uv run python gpu_max_train.py \
            --num_envs 16 \
            --total_timesteps 1000000 \
            --rollout_length 4096 \
            --batch_size 512 \
            --hidden_dim 256 \
            --lr 3e-4 \
            --ppo_epochs 10 \
            --mixed_precision \
            --wandb \
            --save_freq 100000
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "✅ 학습 완료!"
echo "GPU 상태 확인:"
nvidia-smi