# Unitree GO2 강화학습 기반 전진보행 연구

## 연구 개요

본 연구는 Unitree GO2 사족보행 로봇의 전진보행 능력을 Proximal Policy Optimization(PPO) 강화학습 알고리즘을 통해 학습시키는 것을 목표로 한다. MuJoCo 물리 시뮬레이션 환경에서 로봇이 안정적인 전진보행을 수행하도록 정책을 학습한다.

## 시스템 구성

- `environment.py`: GO2 로봇의 MuJoCo Gymnasium 환경 구현
- `ppo_agent.py`: PPO 알고리즘 및 Actor-Critic 네트워크 구현
- `train.py`: 학습 및 평가를 위한 메인 실행 스크립트

## 환경 설정

프로젝트 루트에서 uv를 사용하여 의존성을 설치합니다:

```bash
cd /Users/sxngt/Research/mujoco_quadruped
uv sync
```

## 실험 실행

rl 디렉토리에서 실행합니다:

```bash
cd rl
```

### 기본 학습
```bash
uv run mjpython train.py --mode train
```

### 시각화를 포함한 학습
```bash
uv run mjpython train.py --mode train --render
```

### 로깅을 포함한 학습
```bash
uv run mjpython train.py --mode train --wandb
```

### 하이퍼파라미터 조정 학습
```bash
uv run mjpython train.py --mode train --lr 1e-4 --total_timesteps 2000000
```

## 모델 평가

```bash
uv run mjpython train.py --mode eval --model_path models/best_go2_ppo.pth
```

## 실험 설계

### 관찰 공간 (34차원)
- 관절 위치 (12차원): 각 다리의 고관절, 대퇴부, 종아리 관절
- 관절 속도 (12차원): 해당 관절들의 각속도
- 몸체 자세 (4차원): 쿼터니언 표현
- 몸체 각속도 (3차원): 롤, 피치, 요 방향 각속도
- 몸체 선속도 (3차원): x, y, z 방향 선속도

### 행동 공간 (12차원)
- 12개 구동 관절에 대한 토크 제어
- 토크 범위: 각 관절당 [-20, 20] Nm

### 보상 함수 설계
- **전진 속도 보상**: x축 방향 이동 속도에 비례한 양의 보상
- **자세 안정성 보상**: 직립 자세 유지 시 양의 보상
- **제어 비용**: 과도한 토크 사용에 대한 음의 보상
- **높이 유지 보상**: 최소 지면 높이 유지 시 양의 보상

### 종료 조건
- 로봇 높이 < 0.15m (낙상)
- 과도한 회전 (쿼터니언 w < 0.5)
- 최대 에피소드 스텝 도달

### 하이퍼파라미터
- 학습률: 3e-4
- 할인 인수 (γ): 0.99
- GAE λ: 0.95
- 클립 비율: 0.2
- PPO 업데이트 에포크: 10
- 롤아웃 길이: 2048
- 배치 크기: 64

## 실험 결과물
- `models/`: 학습된 모델 체크포인트
- `training_curves.png`: 학습 진행 시각화
- `wandb/`: Weights & Biases 로그 데이터