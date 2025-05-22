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

### GPU 최적화 학습 (RTX 4080 등)
```bash
# 인터랙티브 모드 선택
./gpu_optimized_train.sh

# 또는 직접 실행
uv run python train.py --mode train --total_timesteps 10000000 --rollout_length 8192 --batch_size 512 --lr 5e-4 --ppo_epochs 15 --save_freq 100 --wandb
```

## 모델 평가

### 기본 평가
```bash
uv run python train.py --mode eval --model_path models/best_go2_ppo.pth --render
```

### 간단한 데모 실행
```bash
# 최고 성능 모델로 5회 실행
uv run python demo.py

# 특정 모델로 실행
uv run python demo.py --model models/go2_ppo_episode_1000.pth

# 슬로우 모션으로 자세히 관찰
uv run python demo.py --slow --episodes 3
```

### 참조 보행 패턴 확인
```bash
# 로봇이 학습해야 할 올바른 걷기 패턴 시연
uv run python test_gait.py
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
- **전진 속도 보상**: x축 방향 이동 속도에 비례한 양의 보상 (최대 50+)
- **생존 보상**: 넘어지지 않고 서있을 때 기본 보상 (±50)
- **참조 동작 모방**: 올바른 보행 패턴과의 유사성 보상 (0~15)
- **발 접촉 패턴**: 적절한 보행 주기에 따른 발 접촉 보상 (0~4)
- **호핑 방지**: 과도한 점프 움직임에 대한 페널티 (-10)
- **다리 균형**: 앞뒤 다리 균등 사용 유도 (-5~0)
- **안정성 보상**: 직립 자세 유지 시 보상 (0~1)
- **에너지 효율**: 과도한 토크 사용 방지 (작은 페널티)

### 종료 조건
- 로봇 높이 < 0.10m (완전한 낙상)
- 과도한 기울어짐 (z축 정렬 < -0.1)
- 측면 이탈 (±5m 초과)
- 후진 이동 (-2m 초과)
- 최대 에피소드 스텝 도달 (1000 스텝)

### 하이퍼파라미터
- 학습률: 3e-4
- 할인 인수 (γ): 0.99
- GAE λ: 0.95
- 클립 비율: 0.2
- PPO 업데이트 에포크: 10
- 롤아웃 길이: 2048
- 배치 크기: 64

## 주요 기능

### 참조 보행 패턴 시스템
- **Trot Gait**: 대각선 다리가 함께 움직이는 자연스러운 보행 패턴
- **주기적 학습**: 1.5Hz 주파수로 일정한 보행 리듬 학습
- **Imitation Learning**: 올바른 걷기 패턴을 모방하여 학습 효율성 향상

### 치팅 방지 시스템
- **호핑 억제**: 과도한 수직 움직임 페널티로 깡충뛰기 방지
- **다리 균형**: 뒷다리만 사용하는 bunny hop 방지
- **지면 접촉**: 최소 2발 이상 지면 접촉 유지 강제

### GPU 최적화
- **RTX 4080 최적화**: 대용량 배치 처리 및 텐서 코어 활용
- **3가지 학습 모드**: 빠른 프로토타이핑 / 표준 학습 / 정밀 학습
- **메모리 최적화**: CUDA 환경 변수 및 배치 크기 최적화

## 실험 결과물
- `models/`: 학습된 모델 체크포인트 (로컬에만 저장, Git 추적 안함)
- `training_curves.png`: 학습 진행 시각화 (로컬 생성)
- `wandb/`: Weights & Biases 로그 데이터 (로컬 생성)
- `RESEARCH_REPORT.md`: 구현 과정 연구 보고서

> **참고**: 학습된 모델 파일들(`models/` 폴더)은 각 실행 환경별로 다르게 생성되므로 Git에서 제외됩니다. 학습 후 로컬 `rl/models/` 폴더에서 확인할 수 있습니다.

## 파일 구조
```
rl/
├── train.py              # 메인 훈련 스크립트
├── environment.py        # GO2 환경 구현
├── ppo_agent.py          # PPO 알고리즘 구현
├── gait_generator.py     # 참조 보행 패턴 생성기
├── demo.py               # 학습된 모델 데모
├── test_gait.py          # 참조 패턴 시연
├── gpu_optimized_train.sh # GPU 최적화 학습 스크립트
├── go2_scene.xml         # 시뮬레이션 환경 설정
└── RESEARCH_REPORT.md    # 연구 과정 보고서
```