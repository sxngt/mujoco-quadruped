# 사족보행 로봇 강화학습 구현 과정에서의 문제점 분석 및 해결 방안

## 1. 서론

본 보고서는 Unitree GO2 사족보행 로봇의 전진 보행 학습을 위한 강화학습 시스템 구현 과정에서 발생한 주요 문제점들과 그에 대한 해결 방안을 체계적으로 정리한 것이다. MuJoCo 물리 시뮬레이션 환경에서 Proximal Policy Optimization(PPO) 알고리즘을 활용하여 로봇의 자연스러운 보행을 학습시키는 과정에서 직면한 기술적 도전과제들을 분석하고, 각 문제에 대한 해결책을 제시한다.

## 2. 주요 문제점 및 해결 방안

### 2.1 텐서 차원 불일치 문제

#### 2.1.1 문제 현상
```
RuntimeError: The size of tensor a (64) must match the size of tensor b (2074) at non-singleton dimension 1
```
PPO 알고리즘의 정책 업데이트 과정에서 배치 처리 시 텐서의 차원이 일치하지 않아 학습이 중단되는 문제가 발생하였다.

#### 2.1.2 원인 분석
- 관찰값 리스트를 직접 PyTorch 텐서로 변환 시 차원 불일치 발생
- 배치 처리 과정에서 log probability와 advantage의 shape 불일치
- PolicyNetwork의 출력 차원 관리 미흡

#### 2.1.3 해결 방안
```python
# 1. NumPy 배열을 통한 안전한 텐서 변환
obs_array = np.array(self.observations, dtype=np.float32)
obs_tensor = torch.from_numpy(obs_array).to(self.device)

# 2. 일관된 차원 관리
log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
log_probs = log_probs.squeeze(-1)

# 3. 배열 reshape를 통한 1차원 통일
log_probs_array = np.array(self.log_probs, dtype=np.float32).reshape(-1)
```

### 2.2 물리 환경 부재 문제

#### 2.2.1 문제 현상
로봇이 공중에 떠 있는 상태에서 학습이 진행되어 현실적인 보행 학습이 불가능하였다. 지면과의 상호작용이 없어 물리적으로 의미 있는 보행 패턴을 학습할 수 없었다.

#### 2.2.2 원인 분석
- 원본 GO2 모델 파일에 지면 geometry가 정의되지 않음
- 중력과 접촉 물리가 제대로 구현되지 않음
- 발과 지면 간의 마찰력 부재

#### 2.2.3 해결 방안
```xml
<!-- 지면 추가 -->
<geom name="ground" type="plane" pos="0 0 0" size="50 50 0.1" 
      material="groundplane" contype="1" conaffinity="1" friction="1.0 0.1 0.1"/>

<!-- 물리 설정 개선 -->
<option timestep="0.002" iterations="50" solver="PGS" cone="elliptic" 
        impratio="100" gravity="0 0 -9.81" integrator="RK4"/>

<!-- 발 접촉 재질 설정 -->
<default class="foot">
  <geom size="0.022" pos="-0.002 0 -0.213" priority="1" 
        solimp="0.9 0.95 0.001 0.5 2" solref="0.01 1" 
        condim="6" friction="1.0 0.1 0.1"/>
</default>
```

### 2.3 보상 함수 설계 문제

#### 2.3.1 문제 현상
초기 보상 함수 설계에서 로봇이 전진하지 않고 제자리에 서있거나 넘어지는 행동만 반복하는 문제가 발생하였다.

#### 2.3.2 원인 분석
- 전진 보상(최대 4.0)이 생존 페널티(-2.0)보다 약함
- 넘어지지 않고 서있는 것이 최적 전략이 됨
- 보상 간의 균형이 맞지 않아 전진 동기 부족

#### 2.3.3 해결 방안
```python
# 전진 보상 대폭 강화
if forward_vel > 0:
    forward_reward = forward_vel * 20.0  # 기존 1.5에서 20.0으로 증가
    if forward_vel > 0.5:
        forward_reward += (forward_vel - 0.5) * 30.0  # 추가 보너스
else:
    forward_reward = forward_vel * 5.0  # 후진 페널티

# 보상 우선순위 재정립
# 1순위: 전진 (최대 50+)
# 2순위: 생존 (±50)
# 3순위: 보조 보상들 (0~3)
```

### 2.4 호핑(Hopping) 치팅 문제

#### 2.4.1 문제 현상
로봇이 자연스러운 4족 보행 대신 뒷다리로 강하게 차면서 깡충깡충 뛰는 방식으로 전진하는 치팅 행동을 보였다.

#### 2.4.2 원인 분석
- 단순 전진 속도만을 보상하여 비현실적 움직임 발생
- 점프가 걷기보다 물리적으로 더 효율적
- 4개 다리 협응보다 2개 다리 사용이 학습하기 쉬움

#### 2.4.3 해결 방안
```python
# 1. 수직 속도 페널티
vertical_vel = abs(self.data.qvel[2])
hop_penalty = -10.0 * max(0, vertical_vel - 0.15)

# 2. 다리 균형 사용 강제
front_torques = np.abs(self.current_action[0:6])
rear_torques = np.abs(self.current_action[6:12])
leg_imbalance = np.mean(rear_torques) - np.mean(front_torques)
leg_balance_penalty = -5.0 * max(0, leg_imbalance - 0.3)

# 3. 지면 접촉 유지 보상
min_contact_reward = 2.0 if num_contacts >= 2 else -3.0
```

### 2.5 학습 방향성 부재 문제

#### 2.5.1 문제 현상
로봇이 "걷기"가 무엇인지 전혀 모르는 상태에서 무작위 행동으로 학습을 시작하여 비효율적이고 부자연스러운 움직임만 반복하는 문제가 발생하였다.

#### 2.5.2 원인 분석
- 강화학습 에이전트가 목표 행동에 대한 사전 지식 없음
- 무작위 탐색으로만 보행 패턴 발견해야 하는 비효율성
- 올바른 보행이 무엇인지에 대한 참조 기준 부재
- 복잡한 4족 보행의 다리 협응 학습의 어려움

#### 2.5.3 해결 방안: Imitation Learning 시스템 구현

**1. 참조 보행 패턴 생성기 (GaitGenerator)**
```python
class GaitGenerator:
    def __init__(self, gait_type="trot", frequency=1.5):
        # Trot: 대각선 다리가 함께 움직이는 자연스러운 보행
        self.phase_offsets = {
            "trot": [0.0, 0.5, 0.5, 0.0],      # FL, FR, RL, RR
            "walk": [0.0, 0.25, 0.75, 0.5],    # 한 번에 한 다리씩
        }
    
    def get_joint_targets(self, current_time):
        # 매 시간스텝마다 목표 관절 각도와 발 접촉 제공
        phase = (current_time * self.frequency) % 1.0
        return target_angles, target_contacts
```

**2. 모방 학습 보상 함수**
```python
# 관절 각도 유사성 보상
target_angles, target_contacts = self.gait_generator.get_joint_targets(self.simulation_time)
current_angles = self.data.qpos[7:19]
angle_diff = np.abs(current_angles - target_angles)
angle_similarity = np.exp(-angle_diff.mean() * 5.0) * 5.0

# 발 접촉 패턴 매칭 보상
current_contacts = [contact['in_contact'] for contact in contacts.values()]
contact_match = np.sum(np.array(current_contacts) == target_contacts)
contact_similarity = contact_match * 1.0

# 주기적 보행 리듬 보상
gait_rhythm = self.gait_reward_calculator.compute_gait_reward(
    np.array(current_contacts), dt=0.002
)
```

**3. 학습 효과**
- 학습 초기부터 올바른 보행 패턴 인식
- 무작위 탐색 대신 목표 지향적 학습
- 자연스러운 Trot gait 패턴으로 수렴
- 학습 속도 및 안정성 대폭 향상

### 2.6 시각화 및 관찰성 문제

#### 2.6.1 문제 현상
- 투명한 바닥으로 인한 시각적 혼란
- 너무 가까운 카메라 시점으로 전체적인 움직임 파악 어려움
- 정면 시점으로 인한 보행 패턴 관찰 제한

#### 2.6.2 해결 방안
```python
# 카메라 설정 개선
self.viewer.cam.distance = 3.5      # 거리
self.viewer.cam.elevation = -25     # 고도각
self.viewer.cam.azimuth = 135       # 방위각 (측후방)

# 무광 바닥 재질
<material name="groundplane" rgba="0.4 0.4 0.4 1" 
          specular="0.1" shininess="0.1"/>

# 3점 조명 시스템
<light name="spotlight" mode="targetbody" target="base" 
       pos="0 -2 3" diffuse="0.8 0.8 0.8"/>
<light name="ambient" pos="0 0 5" dir="0 0 -1" 
       directional="true" diffuse="0.4 0.4 0.4"/>
<light name="fill" pos="2 2 2" dir="-1 -1 -1" 
       directional="true" diffuse="0.3 0.3 0.3"/>
```

## 3. 학습된 교훈

### 3.1 보상 함수 설계의 중요성
강화학습에서 보상 함수는 학습의 방향을 결정하는 가장 중요한 요소이다. 단순히 목표만을 보상하는 것이 아니라, 원하지 않는 행동을 억제하고 자연스러운 행동을 유도하는 섬세한 설계가 필요하다.

### 3.2 참조 동작의 필요성
복잡한 운동 기능 학습에서는 무작위 탐색만으로는 비효율적이다. 올바른 행동 패턴에 대한 사전 지식이나 참조 동작을 제공하는 것이 학습 효율성을 크게 향상시킨다. Imitation Learning은 강화학습의 탐색 공간을 크게 줄여주는 효과적인 방법이다.

### 3.3 물리 시뮬레이션의 현실성
현실적인 학습을 위해서는 정확한 물리 파라미터 설정이 필수적이다. 지면, 중력, 마찰력 등의 기본적인 물리 환경이 제대로 구현되지 않으면 의미 있는 학습이 불가능하다.

### 3.4 치팅 행동의 예상과 대비
강화학습 에이전트는 예상치 못한 방법으로 보상을 최대화하려 시도한다. 호핑, 슬라이딩 등의 치팅 행동을 미리 예상하고 이를 방지하는 보상 설계가 중요하다.

### 3.5 디버깅과 시각화의 필요성
강화학습 과정에서 발생하는 문제를 빠르게 파악하고 해결하기 위해서는 적절한 시각화와 로깅이 필수적이다. 카메라 시점, 조명, 재질 등의 시각적 요소들이 문제 진단에 큰 도움을 준다.

### 3.6 점진적 개선의 중요성
복잡한 시스템에서는 한 번에 모든 문제를 해결하려 하지 말고, 문제를 작은 단위로 나누어 점진적으로 해결하는 것이 효과적이다.

## 4. 향후 개선 방향

### 4.1 다양한 보행 패턴 확장
- Walk, Pace, Gallop 등 다른 gait 패턴 추가 구현
- 속도에 따른 적응적 보행 패턴 자동 전환
- 실시간 보행 패턴 선택 메커니즘 개발

### 4.2 고급 Imitation Learning 기법
- Generative Adversarial Imitation Learning (GAIL) 적용
- Behavioral Cloning으로 초기 정책 사전 훈련
- Motion Matching 기반 참조 동작 다양화

### 4.3 커리큘럼 학습 도입
- 단계별 난이도 조절: 서기 → 느린 걷기 → 빠른 달리기
- 지형 복잡도 점진적 증가
- 동적 보상 가중치 조절

### 4.4 지형 적응성 향상
- 경사면, 계단, 장애물 등 다양한 지형 추가
- 환경 인식 센서 정보 통합
- 지형별 최적 보행 패턴 학습

### 4.5 실제 로봇 Transfer Learning
- Sim-to-Real 도메인 적응 기법 적용
- 실제 로봇 하드웨어 제약 사항 반영
- 센서 노이즈 및 지연 시간 시뮬레이션

### 4.6 에너지 효율성 최적화
- 실제 로봇의 배터리 소모를 고려한 보상 설계
- 최소 에너지로 최대 이동거리 달성
- 동적 보행 최적화 알고리즘 통합

## 5. 결론

사족보행 로봇의 강화학습 구현은 단순한 알고리즘 적용을 넘어서는 복잡한 시스템 설계 문제이다. 물리 시뮬레이션 환경 구축, 보상 함수 설계, 학습 안정성 확보 등 다양한 측면에서의 세심한 고려가 필요하다. 

본 프로젝트를 통해 얻은 가장 중요한 통찰은 다음과 같다:

1. **강화학습의 본질**: 단순히 "보상을 최대화하는" 과정이 아니라, "원하는 행동을 유도하는 환경을 설계하는" 과정이다.

2. **참조 동작의 중요성**: 복잡한 운동 기능에서는 무작위 탐색보다 모방 학습이 훨씬 효과적이다. Trot gait와 같은 자연스러운 보행 패턴을 참조로 제공하는 것이 학습 성능을 크게 향상시켰다.

3. **치팅 행동의 교육적 가치**: 로봇이 찾아낸 예상치 못한 해결책(호핑, 슬라이딩 등)들은 오히려 우리의 보상 설계가 불완전했음을 보여주는 귀중한 피드백이었다.

4. **시스템적 접근의 필요성**: 알고리즘, 환경, 보상, 시각화 등 모든 구성 요소가 조화롭게 설계되어야 성공적인 학습이 가능하다.

향후 이러한 경험과 구현된 Imitation Learning 시스템을 바탕으로 더욱 자연스럽고 효율적인 사족보행 로봇 제어 시스템을 개발할 수 있을 것으로 기대한다. 특히 다양한 보행 패턴과 지형 적응성을 갖춘 범용 사족보행 에이전트 개발이 가능할 것이다.

---

*작성일: 2025년 6월 26일*  
*작성자: SClab 윤상현 연구원*