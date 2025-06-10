# 🏋️‍♀️ 학습 스크립트들

## 📁 폴더 구조

### basic/
- `train.py`: 단일 환경 기본 PPO 학습
  - CPU/GPU 기본 설정
  - 표준 하이퍼파라미터

### gpu_optimized/
- `gpu_max_train.py`: RTX 4080 최적화 학습
  - 96개 환경 병렬 처리
  - 3072 배치 사이즈
  - 24576 롤아웃 길이
  - Mixed precision 학습
- `gpu_optimized_train.sh`: GPU 최적화 실행 스크립트
- `rtx4080_max.sh`: RTX 4080 전용 실행 스크립트

### vectorized/
- `train_vectorized.py`: 기본 벡터화 학습
- `vectorized_train.py`: 개선된 벡터화 학습
- `improved_train.py`: 최신 개선사항 적용 학습

## 🚀 실행 방법

### 기본 학습
```bash
cd training/basic
python train.py
```

### GPU 최적화 학습
```bash
cd training/gpu_optimized
bash rtx4080_max.sh
```

### 벡터화 학습
```bash
cd training/vectorized
python improved_train.py --envs 16 --render
```

## ⚡ 성능 비교

| 방식 | 환경 수 | 배치 크기 | FPS | 메모리 사용량 |
|------|---------|-----------|-----|---------------|
| Basic | 1 | 64 | ~1000 | 2GB |
| Vectorized | 16 | 512 | ~8000 | 8GB |
| GPU Optimized | 96 | 3072 | ~25000 | 22GB |