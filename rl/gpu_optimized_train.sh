#!/bin/bash

# RTX 4080 최적화 학습 스크립트

echo "🚀 RTX 4080 최적화 학습 시작"
echo "================================"

# GPU 정보 출력
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "================================"

# CUDA 최적화 환경 변수
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# 텐서 코어 활성화
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_BENCHMARK=1

# 시스템 최적화
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 학습 모드 선택
echo "학습 모드를 선택하세요:"
echo "1) 빠른 프로토타이핑 (2-3시간)"
echo "2) 표준 학습 (5-6시간)"
echo "3) 정밀 학습 (10-12시간)"
echo "4) 사용자 정의"
read -p "선택 (1-4): " mode

case $mode in
    1)
        echo "⚡ 빠른 프로토타이핑 모드"
        uv run python train.py \
            --mode train \
            --total_timesteps 3000000 \
            --rollout_length 4096 \
            --batch_size 256 \
            --lr 8e-4 \
            --ppo_epochs 8 \
            --save_freq 50 \
            --wandb
        ;;
    2)
        echo "🎯 표준 학습 모드"
        uv run python train.py \
            --mode train \
            --total_timesteps 8000000 \
            --rollout_length 8192 \
            --batch_size 512 \
            --lr 5e-4 \
            --ppo_epochs 12 \
            --save_freq 100 \
            --wandb
        ;;
    3)
        echo "💎 정밀 학습 모드"
        uv run python train.py \
            --mode train \
            --total_timesteps 20000000 \
            --rollout_length 16384 \
            --batch_size 1024 \
            --lr 3e-4 \
            --lr_schedule linear \
            --ppo_epochs 20 \
            --gamma 0.995 \
            --gae_lambda 0.97 \
            --clip_ratio 0.15 \
            --save_freq 200 \
            --wandb
        ;;
    4)
        echo "🔧 사용자 정의 모드"
        read -p "Total timesteps (default: 10000000): " timesteps
        read -p "Batch size (default: 512): " batch
        read -p "Learning rate (default: 5e-4): " lr
        
        timesteps=${timesteps:-10000000}
        batch=${batch:-512}
        lr=${lr:-5e-4}
        
        uv run python train.py \
            --mode train \
            --total_timesteps $timesteps \
            --rollout_length 8192 \
            --batch_size $batch \
            --lr $lr \
            --ppo_epochs 15 \
            --save_freq 100 \
            --wandb
        ;;
esac

echo "================================"
echo "학습 완료!"
echo "모델 평가를 실행하려면:"
echo "uv run python train.py --mode eval --render"