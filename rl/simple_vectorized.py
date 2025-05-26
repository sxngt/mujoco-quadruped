#!/usr/bin/env python3
"""
간단한 벡터화 환경 구현
16개 GO2 환경을 동시에 실행하여 학습 효율성 극대화
"""

import numpy as np
import time
from environment import GO2ForwardEnv
from typing import List, Tuple, Any, Dict


class SimpleVectorEnv:
    """
    간단한 벡터화 환경
    여러 환경을 동일 프로세스에서 순차 실행
    """
    
    def __init__(self, num_envs: int = 16, render_mode: str = None, use_reference_gait: bool = True):
        self.num_envs = num_envs
        
        # 환경들 생성
        self.envs = []
        for i in range(num_envs):
            # 첫 번째 환경만 렌더링
            env_render_mode = render_mode if i == 0 else None
            env = GO2ForwardEnv(
                render_mode=env_render_mode,
                use_reference_gait=use_reference_gait
            )
            self.envs.append(env)
        
        # 환경 스펙 (첫 번째 환경 기준)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        print(f"✅ {num_envs}개 벡터화 환경 생성 완료")
        print(f"관찰 공간: {self.observation_space.shape}")
        print(f"행동 공간: {self.action_space.shape}")
    
    def reset(self, seed=None):
        """
        모든 환경 리셋
        
        Returns:
            observations: (num_envs, obs_dim) 관찰
            infos: List[Dict] 정보
        """
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        
        return np.array(observations, dtype=np.float32), infos
    
    def step(self, actions):
        """
        모든 환경에서 동시 스텝
        
        Args:
            actions: (num_envs, action_dim) 행동 배열
            
        Returns:
            observations: (num_envs, obs_dim) 다음 관찰
            rewards: (num_envs,) 보상
            terminated: (num_envs,) 종료 플래그
            truncated: (num_envs,) 잘림 플래그  
            infos: List[Dict] 정보
        """
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
        print(f"🔒 {self.num_envs}개 환경 종료 완료")


def test_vectorized_performance():
    """벡터화 환경 성능 테스트"""
    
    print("🔥 벡터화 환경 성능 테스트")
    
    # 단일 환경 vs 벡터화 환경 비교
    num_steps = 1000
    
    # === 단일 환경 테스트 ===
    print("\n1️⃣ 단일 환경 테스트")
    single_env = GO2ForwardEnv(use_reference_gait=False)
    
    start_time = time.time()
    obs, _ = single_env.reset()
    
    for step in range(num_steps):
        action = np.random.uniform(-1, 1, single_env.action_space.shape[0])
        obs, reward, terminated, truncated, info = single_env.step(action)
        
        if terminated or truncated:
            obs, _ = single_env.reset()
    
    single_time = time.time() - start_time
    single_fps = num_steps / single_time
    
    single_env.close()
    print(f"단일 환경: {single_time:.2f}초, {single_fps:.0f} FPS")
    
    # === 벡터화 환경 테스트 ===
    print("\n1️⃣6️⃣ 벡터화 환경 테스트 (16개)")
    vec_env = SimpleVectorEnv(num_envs=16, use_reference_gait=False)
    
    start_time = time.time()
    observations, infos = vec_env.reset()
    
    total_env_steps = 0
    for step in range(num_steps // 16):  # 16배 적은 스텝 (총 스텝 수 동일)
        actions = np.random.uniform(-1, 1, (16, vec_env.action_space.shape[0]))
        observations, rewards, terminated, truncated, infos = vec_env.step(actions)
        total_env_steps += 16
        
        # 종료된 환경이 있으면 자동으로 리셋됨 (환경 내부에서)
    
    vec_time = time.time() - start_time
    vec_fps = total_env_steps / vec_time
    
    vec_env.close()
    print(f"벡터화 환경: {vec_time:.2f}초, {vec_fps:.0f} FPS")
    
    # === 성능 비교 ===
    print(f"\n📊 성능 비교")
    print(f"속도 향상: {vec_fps / single_fps:.1f}배")
    print(f"처리량 향상: {(vec_fps * 16) / single_fps:.1f}배 (16개 환경)")
    
    if vec_fps > single_fps * 8:  # 최소 8배 이상 빨라야 함
        print("✅ 벡터화 성능 우수!")
    else:
        print("⚠️ 벡터화 성능 개선 필요")


if __name__ == "__main__":
    # 기본 기능 테스트
    print("🚀 벡터화 환경 기본 테스트")
    
    vec_env = SimpleVectorEnv(num_envs=4, use_reference_gait=False)
    
    # 리셋 테스트
    observations, infos = vec_env.reset(seed=42)
    print(f"리셋 완료: 관찰 배열 크기 {observations.shape}")
    
    # 스텝 테스트
    for step in range(10):
        actions = np.random.uniform(-1, 1, (4, vec_env.action_space.shape[0]))
        observations, rewards, terminated, truncated, infos = vec_env.step(actions)
        
        avg_reward = np.mean(rewards)
        num_done = np.sum(terminated | truncated)
        print(f"스텝 {step:2d}: 평균 보상 {avg_reward:6.1f}, 종료 환경 {num_done}개")
    
    vec_env.close()
    
    # 성능 테스트
    test_vectorized_performance()