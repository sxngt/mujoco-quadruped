#!/usr/bin/env python3
"""
Vectorized Environment for GO2 Forward Locomotion
16개의 환경을 병렬로 실행하여 학습 효율성 극대화
"""

import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv
from ..basic.environment import GO2ForwardEnv
import multiprocessing as mp
from typing import List, Tuple, Any, Dict
import time


class SubprocVecEnv(VectorEnv):
    """
    서브프로세스 기반 벡터화 환경
    각 환경을 별도 프로세스에서 실행하여 GIL 우회
    """
    
    def __init__(self, env_fns: List[callable], start_method: str = 'spawn'):
        self.waiting = False
        self.closed = False
        
        nenvs = len(env_fns)
        
        if start_method is None:
            # macOS에서는 spawn이 안전함
            start_method = 'spawn'
        
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.processes = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                         for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        
        for process in self.processes:
            process.daemon = True  # 메인 프로세스 종료 시 함께 종료
            process.start()
        for remote in self.work_remotes:
            remote.close()
        
        # 첫 번째 환경에서 스펙 가져오기
        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, spec = self.remotes[0].recv()
        
        VectorEnv.__init__(self, nenvs, observation_space, action_space)
        
        self.spec = spec
        
    def step_async(self, actions):
        self._assert_is_running()
        
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        
    def step_wait(self):
        self._assert_is_running()
        
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        obs, rews, terminated, truncated, infos = zip(*results)
        
        return np.stack(obs), np.stack(rews), np.stack(terminated), np.stack(truncated), infos
    
    def reset(self, seed=None, options=None):
        self._assert_is_running()
        
        if seed is None:
            seed = [None] * self.num_envs
        if options is None:
            options = [None] * self.num_envs
            
        for remote, single_seed, single_options in zip(self.remotes, seed, options):
            remote.send(('reset', (single_seed, single_options)))
            
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        
        return np.stack(obs), infos
    
    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
            
    def get_attr(self, attr_name, indices=None):
        """속성 값 가져오기"""
        self._assert_is_running()
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
            
        for i, remote in enumerate(self.remotes):
            if i in indices:
                remote.send(('get_attr', attr_name))
                
        results = []
        for i, remote in enumerate(self.remotes):
            if i in indices:
                results.append(remote.recv())
                
        return results
    
    def set_attr(self, attr_name, values, indices=None):
        """속성 값 설정"""
        self._assert_is_running()
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
            
        for i, remote in enumerate(self.remotes):
            if i in indices:
                remote.send(('set_attr', (attr_name, values)))
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """환경 메서드 호출"""
        self._assert_is_running()
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
            
        for i, remote in enumerate(self.remotes):
            if i in indices:
                remote.send(('env_method', (method_name, method_args, method_kwargs)))
                
        results = []
        for i, remote in enumerate(self.remotes):
            if i in indices:
                results.append(remote.recv())
                
        return results
    
    def _assert_is_running(self):
        if self.closed:
            raise AssertionError("Trying to operate on a SubprocVecEnv after calling `close()`.")


def worker(remote, parent_remote, env_fn_wrapper):
    """
    워커 프로세스 함수
    각 환경을 독립적으로 실행
    """
    parent_remote.close()
    env = env_fn_wrapper.var()
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                ob, reward, terminated, truncated, info = env.step(data)
                remote.send((ob, reward, terminated, truncated, info))
                
            elif cmd == 'reset':
                seed, options = data if data is not None else (None, None)
                ob, info = env.reset(seed=seed, options=options)
                remote.send((ob, info))
                
            elif cmd == 'render':
                remote.send(env.render())
                
            elif cmd == 'close':
                env.close()
                remote.close()
                break
                
            elif cmd == 'get_spaces_spec':
                remote.send((env.observation_space, env.action_space, env.spec))
                
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
                
            elif cmd == 'set_attr':
                attr_name, value = data
                setattr(env, attr_name, value)
                remote.send(None)
                
            elif cmd == 'env_method':
                method_name, method_args, method_kwargs = data
                method = getattr(env, method_name)
                remote.send(method(*method_args, **method_kwargs))
                
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class CloudpickleWrapper:
    """
    환경 함수를 pickle 가능하게 래핑
    multiprocessing에서 함수 전달용
    """
    def __init__(self, var):
        self.var = var
        
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.var)
        
    def __setstate__(self, var):
        import cloudpickle
        self.var = cloudpickle.loads(var)


def make_env(env_id: int, render_mode: str = None, use_reference_gait: bool = True):
    """
    단일 환경 생성 함수
    """
    def _init():
        env = GO2ForwardEnv(
            render_mode=render_mode if env_id == 0 else None,  # 첫 번째 환경만 렌더링
            use_reference_gait=use_reference_gait
        )
        return env
    return _init


def create_vectorized_env(num_envs: int = 16, render_mode: str = None, use_reference_gait: bool = True):
    """
    벡터화된 환경 생성
    
    Args:
        num_envs: 병렬 환경 수 (기본 16개)
        render_mode: 렌더링 모드 (첫 번째 환경만 적용)
        use_reference_gait: 참조 gait 사용 여부
    
    Returns:
        SubprocVecEnv: 벡터화된 환경
    """
    env_fns = [make_env(i, render_mode, use_reference_gait) for i in range(num_envs)]
    return SubprocVecEnv(env_fns)


# 단순한 동기식 벡터 환경 (GIL 제한 있지만 구현 간단)
class SyncVectorEnv(VectorEnv):
    """
    동기식 벡터 환경 - 모든 환경을 같은 프로세스에서 순차 실행
    GIL 때문에 성능 제한 있지만 디버깅에 유용
    """
    
    def __init__(self, env_fns: List[callable]):
        self.envs = [env_fn() for env_fn in env_fns]
        
        # 첫 번째 환경에서 스펙 가져오기
        observation_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space
        
        VectorEnv.__init__(self, len(env_fns), observation_space, action_space)
        
        self.actions = None
        
    def reset(self, seed=None, options=None):
        if seed is None:
            seed = [None] * self.num_envs
        if options is None:
            options = [None] * self.num_envs
            
        observations, infos = [], []
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed[i], options=options[i])
            observations.append(obs)
            infos.append(info)
            
        return np.array(observations), infos
    
    def step(self, actions):
        observations, rewards, terminated, truncated, infos = [], [], [], [], []
        
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
            
        return (np.array(observations), np.array(rewards), 
                np.array(terminated), np.array(truncated), infos)
    
    def render(self):
        # 첫 번째 환경만 렌더링
        if len(self.envs) > 0:
            return self.envs[0].render()
    
    def close(self):
        for env in self.envs:
            env.close()


if __name__ == "__main__":
    # 테스트 코드
    print("🚀 벡터화 환경 테스트 시작")
    
    # 간단한 동기식 환경으로 먼저 테스트
    env_fns = [make_env(i, None, False) for i in range(4)]
    vec_env = SyncVectorEnv(env_fns)
    
    print(f"벡터 환경 생성 완료: {vec_env.num_envs}개 환경")
    print(f"관찰 공간: {vec_env.observation_space}")
    print(f"행동 공간: {vec_env.action_space}")
    
    # 리셋 테스트
    start_time = time.time()
    observations, infos = vec_env.reset()
    reset_time = time.time() - start_time
    
    print(f"리셋 완료: {reset_time:.3f}초")
    print(f"관찰 배열 크기: {observations.shape}")
    
    # 스텝 테스트
    start_time = time.time()
    for step in range(100):
        actions = np.random.uniform(-1, 1, (vec_env.num_envs, vec_env.action_space.shape[0]))
        observations, rewards, terminated, truncated, infos = vec_env.step(actions)
        
        if step % 25 == 0:
            avg_reward = np.mean(rewards)
            print(f"스텝 {step}: 평균 보상 {avg_reward:.2f}")
    
    step_time = time.time() - start_time
    print(f"100 스텝 완료: {step_time:.3f}초 (스텝당 {step_time/100:.4f}초)")
    
    vec_env.close()
    print("✅ 벡터화 환경 테스트 완료")