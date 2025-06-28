# GO2 Quadruped Locomotion - Generation Final

MujocoEnv 기반 Unitree GO2 사족보행 로봇의 강화학습 환경입니다.

## 설치 방법

```bash
# 프로젝트 루트에서
uv sync

# 또는 pip 사용 시
pip install -r requirements.txt
```

## 필수 패키지

- `gymnasium[mujoco]>=1.1.1` - 환경 프레임워크
- `stable-baselines3[extra]>=2.6.0` - PPO 알고리즘
- `mujoco>=3.3.3` - 물리 시뮬레이션
- `torch>=2.7.1` - 신경망 학습
- `tqdm>=4.66.0` - 진행률 표시
- `numpy>=2.3.0` - 수치 계산
- `matplotlib>=3.10.3` - 시각화

## 환경 구조

### go2_mujoco_env.py
- MujocoEnv 상속 기반 GO2 환경
- 50Hz 제어율 (frame_skip=10)
- 48차원 관찰 공간
- 12차원 액션 공간 (토크 제어)

### train_sb3.py
- Stable Baselines3 PPO 훈련 스크립트
- 병렬 환경 지원
- 실시간 렌더링 옵션

## 사용법

### 훈련

```bash
# 빠른 병렬 훈련 (권장)
uv run python train_sb3.py --run train \
    --total_timesteps 1000000 \
    --num_parallel_envs 8 \
    --run_name "go2_walking"

# 실시간 렌더링 훈련 (시각화)
uv run python train_sb3.py --run train \
    --total_timesteps 10000 \
    --render_training \
    --run_name "visual_debug"
```

### 테스트

```bash
# 학습된 모델 테스트
uv run python train_sb3.py --run test \
    --model_path models_sb3/[모델폴더]/best_model.zip \
    --num_test_episodes 5
```

## 주요 파라미터

- `--total_timesteps`: 총 훈련 스텝 수 (기본: 5,000,000)
- `--num_parallel_envs`: 병렬 환경 수 (기본: 12)
- `--eval_frequency`: 평가 주기 (기본: 10,000)
- `--render_training`: 훈련 중 실시간 렌더링
- `--ctrl_type`: 제어 타입 (torque/position)

## 보상 함수

### 양의 보상
- **선형 속도 추적**: 목표 속도 (0.5 m/s) 추적
- **각속도 추적**: 목표 방향 유지
- **발 공중 시간**: 자연스러운 보행 리듬

### 음의 보상 (페널티)
- **토크 사용량**: 에너지 효율성
- **수직 속도**: 점프 방지
- **관절 한계**: 하드웨어 보호
- **충돌**: 몸체 접촉 방지
- **자세**: 안정적인 몸체 유지

## 건강 상태 체크

- 높이 범위: 0.22m - 0.65m
- 기울기 범위: ±10도 (Roll, Pitch)
- 상태값 유한성 검사

## 파일 구조

```
integrated/
├── go2_mujoco_env.py      # 메인 환경 클래스
├── train_sb3.py           # 훈련 스크립트
├── test_new_env.py        # 환경 테스트
├── models_sb3/            # 학습된 모델
├── logs_sb3/              # 텐서보드 로그
└── README.md              # 이 문서
```

## 문제 해결

### "Package not found" 오류
```bash
# 프로젝트 루트에서
uv sync
```

### 렌더링 오류
- macOS: `mjpython` 대신 `python` 사용
- 카메라 오류: 무시 가능 (자동 해결됨)

### 메모리 부족
- `--num_parallel_envs` 수 감소
- 단일 환경으로 테스트

## 참고 자료

- [Stable Baselines3 문서](https://stable-baselines3.readthedocs.io/)
- [MuJoCo 문서](https://mujoco.readthedocs.io/)
- [Gymnasium 문서](https://gymnasium.farama.org/)