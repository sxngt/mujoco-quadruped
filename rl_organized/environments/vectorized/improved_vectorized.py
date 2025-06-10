#!/usr/bin/env python3
"""
개선된 벡터화 환경 - Isaac Lab RSL-RL 기법 적용
"""

import numpy as np
import time
from improved_environment import ImprovedGO2Env
from typing import List, Tuple, Any, Dict


class ImprovedVectorEnv:
    """
    개선된 벡터화 환경
    관절 위치 제어와 향상된 보상 함수 적용
    """
    
    def __init__(self, num_envs: int = 16, render_mode: str = None, use_reference_gait: bool = True):
        self.num_envs = num_envs
        
        # 환경들 생성
        self.envs = []
        for i in range(num_envs):
            # 첫 번째 환경만 렌더링
            env_render_mode = render_mode if i == 0 else None
            env = ImprovedGO2Env(
                render_mode=env_render_mode,
                use_reference_gait=use_reference_gait
            )
            self.envs.append(env)
        
        # 환경 스펙
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        print(f"✅ {num_envs}개 개선된 벡터화 환경 생성 완료")
        print(f"🎮 제어 모드: 관절 위치 제어 (PD 컨트롤러)")
        print(f"📊 관찰 공간: {self.observation_space.shape}")
        print(f"🎯 행동 공간: {self.action_space.shape} (정규화된 관절 위치 명령)")
        print(f"🏃 초기화: 참조 보행 자세")
        print(f"💰 보상: 모듈화된 보상 함수 (발 공중시간 추적 포함)")
    
    def reset(self, seed=None):
        """모든 환경 리셋"""
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        
        return np.array(observations, dtype=np.float32), infos
    
    def step(self, actions):
        """모든 환경에서 동시 스텝"""
        observations = []
        rewards = []
        terminated = []
        truncated = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
        
        return (np.array(observations, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(terminated, dtype=bool),
                np.array(truncated, dtype=bool),
                infos)
    
    def render(self):
        """첫 번째 환경만 렌더링"""
        if self.envs and hasattr(self.envs[0], 'render'):
            return self.envs[0].render()
    
    def close(self):
        """모든 환경 종료"""
        for env in self.envs:
            env.close()
        print(f"🔒 {self.num_envs}개 개선된 환경 종료 완료")
    
    def get_reward_info(self):
        """현재 보상 가중치 정보 반환"""
        if self.envs:
            return self.envs[0].reward_weights
        return {}


def test_improved_environment():
    """개선된 환경 테스트"""
    print("🧪 개선된 GO2 환경 테스트 시작\n")
    
    # 단일 환경 테스트
    print("1️⃣ 단일 환경 테스트")
    env = ImprovedGO2Env(render_mode="human")
    
    obs, info = env.reset()
    print(f"초기 관찰 차원: {obs.shape}")
    print(f"초기 로봇 높이: {env.data.qpos[2]:.3f}m")
    print(f"초기 관절 위치: {env.data.qpos[7:19]}")
    
    # 몇 스텝 실행
    total_reward = 0
    reward_breakdown = {}
    
    for step in range(100):
        # 작은 랜덤 액션 (위치 제어)
        action = np.random.uniform(-0.3, 0.3, env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # 보상 구성 요소 누적
        for key, value in info.items():
            if key not in reward_breakdown:
                reward_breakdown[key] = 0
            reward_breakdown[key] += value
        
        if terminated or truncated:
            print(f"에피소드 종료: 스텝 {step+1}")
            break
        
        if step % 20 == 0:
            print(f"스텝 {step}: 보상 {reward:.3f}, 높이 {env.data.qpos[2]:.3f}m")
    
    print(f"\n총 보상: {total_reward:.2f}")
    print("보상 구성요소 분석:")
    for key, value in reward_breakdown.items():
        print(f"  {key}: {value:.3f}")
    
    env.close()
    
    # 벡터화 환경 테스트
    print("\n\n1️⃣6️⃣ 벡터화 환경 테스트")
    vec_env = ImprovedVectorEnv(num_envs=4, render_mode=None)
    
    observations, infos = vec_env.reset()
    print(f"관찰 배치 형태: {observations.shape}")
    
    # 몇 스텝 실행
    for step in range(50):
        actions = np.random.uniform(-0.3, 0.3, (4, vec_env.action_space.shape[0]))
        observations, rewards, terminated, truncated, infos = vec_env.step(actions)
        
        if step % 10 == 0:
            print(f"스텝 {step}: 평균 보상 {np.mean(rewards):.3f}, "
                  f"최대 보상 {np.max(rewards):.3f}, 최소 보상 {np.min(rewards):.3f}")
    
    vec_env.close()
    
    print("\n✅ 개선된 환경 테스트 완료!")
    print("\n주요 개선사항:")
    print("1. ✅ 관절 위치 제어 (PD 컨트롤러)")
    print("2. ✅ 참조 보행 자세 초기화")
    print("3. ✅ 액션 스무싱")
    print("4. ✅ 모듈화된 보상 함수")
    print("5. ✅ 발 접촉 이력 추적")
    print("6. ✅ 발 공중 시간 보상")


if __name__ == "__main__":
    test_improved_environment()