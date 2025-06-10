# 🤖 강화학습 폴더 구조

## 📁 디렉토리 구조

```
rl_organized/
├── 📂 environments/          # 환경 구현체들
│   ├── basic/               # 기본 환경
│   ├── improved/            # 개선된 환경 (Isaac Lab 기법 적용)
│   └── vectorized/          # 벡터화 환경
├── 📂 training/             # 학습 스크립트들
│   ├── basic/               # 기본 학습
│   ├── gpu_optimized/       # GPU 최적화 학습
│   └── vectorized/          # 벡터화 학습
├── 📂 agents/               # 에이전트 구현
├── 📂 experiments/          # 실험별 분류
│   ├── gait_research/       # 보행 패턴 연구
│   ├── gpu_optimization/    # GPU 최적화 실험
│   └── walking_improvements/ # 보행 개선 실험
├── 📂 documentation/        # 문서들
├── 📂 scripts/              # 유틸리티 스크립트
├── 📂 assets/               # 모델 파일, 설정 등
├── 📂 models/               # 학습된 모델들
│   ├── checkpoints/         # 중간 체크포인트
│   ├── final/              # 최종 모델
│   └── experimental/        # 실험용 모델
└── 📂 tests/               # 테스트 파일들
```

## 🎯 각 폴더의 역할

### Environments
- **basic/**: 초기 GO2 환경 구현
- **improved/**: Isaac Lab 기법 적용한 개선 환경
- **vectorized/**: 병렬 처리를 위한 벡터화 환경

### Training
- **basic/**: 단일 환경 기본 학습
- **gpu_optimized/**: RTX 4080 최적화 학습
- **vectorized/**: 다중 환경 병렬 학습

### Experiments
- **gait_research/**: 보행 패턴 및 gait generator 연구
- **gpu_optimization/**: GPU 메모리 및 성능 최적화
- **walking_improvements/**: 점프 방지 및 실제 보행 개선

### Models
- **checkpoints/**: 에피소드별 중간 저장 모델
- **final/**: 완전히 학습된 최종 모델
- **experimental/**: 실험 중인 모델들

## 📋 이주 계획

1. **환경 파일들 분류 이동**
2. **학습 스크립트 목적별 정리**
3. **모델 파일 버전별 체계화**
4. **문서 및 테스트 파일 정리**
5. **설정 파일 중앙화**

## 🚀 사용법

각 실험은 해당 디렉토리에서 독립적으로 실행 가능하며,
공통 컴포넌트는 상위 디렉토리에서 import하여 사용합니다.