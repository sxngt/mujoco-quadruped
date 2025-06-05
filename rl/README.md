# Unitree GO2 Quadruped Locomotion Research

Unitree GO2 사족보행 로봇의 강화학습 기반 전진 보행 연구 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 MuJoCo 물리 시뮬레이션 환경에서 Proximal Policy Optimization(PPO) 알고리즘을 사용하여 Unitree GO2 로봇의 자연스러운 전진 보행을 학습시키는 연구입니다.

## 연구 진화 과정

### 🏗️ [Generation 1: 기초 환경 구축](generation1/)
- 기본 PPO 구현 및 물리 환경 구축
- 텐서 차원 문제 해결
- 지면, 중력, 마찰력 등 기본 물리 요소 추가

### 🎯 [Generation 2: 보행 중심 보상 설계](generation2/)
- 전진 보상 대폭 강화 (1.5 → 20.0)
- 호핑 치팅 행동 방지
- 시각화 개선 (카메라, 조명, 바닥)

### 🤖 [Generation 3: Imitation Learning 도입](generation3/)
- 참조 Gait 패턴 기반 모방 학습
- GPU 최적화 및 벡터화 구현
- Trot, Walk 등 자연스러운 보행 패턴 제공

### 🛡️ [Generation 4: 안정성 집중 개선](generation4/)
- 현실적 물리 파라미터 적용
- 관절 안전성 보호 시스템
- 조기 종료 문제 해결

### 🚀 [Generation 5: 최종 완성도 향상](generation5/)
- Gait 패턴 강제 적용
- 정지 상태 방지 메커니즘
- 극도로 관대한 종료 조건

### 🏆 [Generation Final: 통합 솔루션](generation_final/)
- **추천**: 참조 레포지터리 방법론 적용
- 39차원 최적화된 관찰 공간
- 검증된 토크 기반 직접 제어
- 단순하고 효과적인 보상 구조

## 빠른 시작

### 최신 통합 솔루션 사용 (권장)
```bash
cd generation_final
python test_integrated_env.py  # 환경 테스트
python integrated/train_integrated.py  # 훈련 시작
```

### 특정 세대 실험
```bash
cd generation[1-5]
# 각 세대별 README.md 참조
```

## 주요 학습 결과

### 핵심 통찰
1. **보상 함수의 중요성**: 단순히 목표를 보상하는 것이 아니라 원하는 행동을 유도하는 설계 필요
2. **참조 동작의 효과**: 무작위 탐색보다 모방 학습이 훨씬 효율적
3. **치팅 행동 대비**: 호핑, 슬라이딩 등 예상치 못한 해결책에 대한 사전 대비 필요
4. **물리 현실성**: 정확한 물리 파라미터가 의미 있는 학습의 전제 조건

### 해결한 주요 문제
- ❌ 텐서 차원 불일치 → ✅ 안전한 NumPy 변환
- ❌ 무중력 환경 → ✅ 현실적 물리 환경
- ❌ 소극적 서있기 → ✅ 전진 보상 강화
- ❌ 호핑 치팅 → ✅ 수직 움직임 페널티
- ❌ 무작위 탐색 → ✅ Gait 패턴 모방 학습
- ❌ 조기 종료 → ✅ 관대한 종료 조건

## 기술 스택

- **시뮬레이션**: MuJoCo 3.x
- **강화학습**: PyTorch + PPO
- **환경**: Gymnasium
- **모델**: Unitree GO2 (MuJoCo Menagerie)
- **GPU**: RTX 4080 최적화

## 프로젝트 구조

```
rl/
├── generation1/           # 기초 환경 구축
├── generation2/           # 보행 중심 보상
├── generation3/           # Imitation Learning
├── generation4/           # 안정성 개선
├── generation5/           # 최종 완성도
├── generation_final/      # 통합 솔루션 (권장)
├── agents/               # PPO 에이전트
├── assets/              # GO2 모델 및 장면
└── documentation/       # 연구 보고서
```

## 참조 자료

- [상세 연구 보고서](documentation/RESEARCH_REPORT.md)
- [참조 레포지터리](https://github.com/nimazareian/quadruped-rl-locomotion)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

## 라이선스

MIT License

---

*Unitree GO2 Quadruped Locomotion Research Project*  
*SClab 윤상현 연구원*