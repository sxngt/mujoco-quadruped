# 🌍 환경 구현체들

## 📁 폴더 구조

### basic/
- `environment.py`: 초기 GO2 환경 구현
  - 토크 제어 기반
  - 기본적인 보상 함수
  - 단순한 종료 조건

### improved/
- `improved_environment.py`: Isaac Lab 기법 적용 환경
  - PD 위치 제어 방식
  - 관절별 차별화된 PD 게인
  - 점프 방지 및 보행 강화 시스템
  - 모듈화된 보상 함수

### vectorized/
- `vectorized_env.py`: 기본 벡터화 환경
- `simple_vectorized.py`: 간단한 벡터화 구현
- `improved_vectorized.py`: 개선된 벡터화 환경
- `multi_robot_env.py`: 다중 로봇 환경
- `simple_multi_env.py`: 간단한 다중 환경

## 🎯 사용 지침

```python
# 기본 환경
from environments.basic.environment import GO2ForwardEnv

# 개선된 환경
from environments.improved.improved_environment import ImprovedGO2Env

# 벡터화 환경
from environments.vectorized.simple_vectorized import SimpleVectorEnv
```

## 🔄 발전 과정

1. **Basic** → 토크 제어, 기본 보상
2. **Improved** → PD 제어, 점프 방지, 보상 개선
3. **Vectorized** → 병렬 처리, 학습 효율성 증대