#!/usr/bin/env python3
"""
통합 GO2 환경 빠른 테스트 스크립트
환경이 올바르게 작동하는지 확인
"""

import sys
import os
import numpy as np
import time

# 프로젝트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from environments.integrated import IntegratedGO2Env
    from agents.ppo_agent import PPOAgent
except ImportError as e:
    print(f"Import 오류: {e}")
    print("rl 디렉토리에서 실행하세요.")
    sys.exit(1)


def test_environment_basic():
    """기본 환경 테스트"""
    print("=== 기본 환경 테스트 ===")
    
    try:
        # 환경 생성
        env = IntegratedGO2Env(render_mode=None)
        print(f"✅ 환경 생성 성공")
        print(f"   관찰 공간: {env.observation_space.shape}")
        print(f"   액션 공간: {env.action_space.shape}")
        
        # 리셋 테스트
        obs, info = env.reset()
        print(f"✅ 리셋 성공")
        print(f"   관찰 크기: {obs.shape}")
        print(f"   관찰 범위: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # 무작위 액션 테스트
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   Step {step+1}: 보상={reward:.3f}, 종료={terminated}, 자른됨={truncated}")
            
            if terminated or truncated:
                print(f"   에피소드 종료 (step {step+1})")
                obs, info = env.reset()
                break
        
        env.close()
        print("✅ 기본 환경 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 환경 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_integration():
    """PPO 에이전트와 환경 통합 테스트"""
    print("\n=== 에이전트 통합 테스트 ===")
    
    try:
        # 환경과 에이전트 생성
        env = IntegratedGO2Env(render_mode=None)
        agent = PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            lr=3e-4,
            hidden_dim=128  # 테스트용으로 작게
        )
        
        print(f"✅ 에이전트 생성 성공")
        print(f"   디바이스: {agent.device}")
        
        # 짧은 훈련 루프 테스트
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(50):  # 50 스텝만 테스트
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 경험 저장
            agent.store_transition(obs, action, reward, value, log_prob, terminated)
            
            obs = next_obs
            total_reward += reward
            
            if step % 10 == 0:
                print(f"   Step {step}: 보상={reward:.3f}, 값={value:.3f}")
            
            if terminated or truncated:
                print(f"   에피소드 종료 (step {step+1}), 총 보상: {total_reward:.3f}")
                obs, _ = env.reset()
                total_reward = 0
        
        # 정책 업데이트 테스트
        if len(agent.observations) > 10:
            print("   정책 업데이트 테스트...")
            update_info = agent.update_policy(n_epochs=2, batch_size=32)
            print(f"   ✅ 정책 업데이트 성공: {update_info}")
        
        env.close()
        print("✅ 에이전트 통합 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 에이전트 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_structure():
    """보상 구조 테스트"""
    print("\n=== 보상 구조 테스트 ===")
    
    try:
        env = IntegratedGO2Env(render_mode=None)
        obs, _ = env.reset()
        
        print("다양한 행동에 대한 보상 테스트:")
        
        # 1. 정지 상태 (보상이 낮아야 함)
        action = np.zeros(env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   정지: 보상={reward:.3f}")
        
        # 2. 전진 시도 (보상이 높아야 함)
        action = np.array([5.0] * 12)  # 앞다리에 토크
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(action)
        print(f"   전진 시도: 보상={reward:.3f}")
        
        # 3. 보상 상세 분석
        print("   보상 구성 요소:")
        for key, value in info.items():
            if isinstance(value, (int, float)) and 'reward' in key.lower() or 'cost' in key.lower():
                print(f"     {key}: {value:.3f}")
        
        env.close()
        print("✅ 보상 구조 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 보상 구조 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_consistency():
    """관찰 일관성 테스트"""
    print("\n=== 관찰 일관성 테스트 ===")
    
    try:
        env = IntegratedGO2Env(render_mode=None)
        
        # 여러 번 리셋하여 관찰 일관성 확인
        obs_shapes = []
        obs_ranges = []
        
        for i in range(5):
            obs, _ = env.reset()
            obs_shapes.append(obs.shape)
            obs_ranges.append((obs.min(), obs.max()))
            
            # 몇 스텝 실행
            for _ in range(3):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        
        # 일관성 확인
        unique_shapes = set(obs_shapes)
        if len(unique_shapes) == 1:
            print(f"✅ 관찰 형태 일관성: {list(unique_shapes)[0]}")
        else:
            print(f"❌ 관찰 형태 불일치: {unique_shapes}")
            return False
        
        print(f"   관찰 값 범위: {obs_ranges}")
        
        # NaN/Inf 체크
        obs, _ = env.reset()
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"❌ 관찰에 NaN/Inf 발견 (step {step})")
                return False
            
            if np.isnan(reward) or np.isinf(reward):
                print(f"❌ 보상에 NaN/Inf 발견 (step {step})")
                return False
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
        print("✅ 관찰 일관성 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 관찰 일관성 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rendering():
    """렌더링 테스트 (선택적)"""
    print("\n=== 렌더링 테스트 (3초) ===")
    
    try:
        env = IntegratedGO2Env(render_mode="human")
        obs, _ = env.reset()
        
        start_time = time.time()
        step_count = 0
        
        while time.time() - start_time < 3.0:  # 3초 동안
            action = env.action_space.sample() * 0.5  # 약한 액션
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            step_count += 1
            time.sleep(0.02)  # 50 FPS
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
        print(f"✅ 렌더링 테스트 완료 ({step_count} 스텝)")
        return True
        
    except Exception as e:
        print(f"⚠️ 렌더링 테스트 건너뜀: {e}")
        return True  # 렌더링 실패는 치명적이지 않음


def main():
    """모든 테스트 실행"""
    print("통합 GO2 환경 테스트 시작\n")
    
    tests = [
        ("기본 환경", test_environment_basic),
        ("에이전트 통합", test_agent_integration),
        ("보상 구조", test_reward_structure),
        ("관찰 일관성", test_observation_consistency),
        ("렌더링", test_rendering),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"테스트 실행: {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} 테스트 통과")
            else:
                print(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            print(f"❌ {test_name} 테스트 예외: {e}")
    
    print(f"\n{'='*50}")
    print(f"테스트 완료: {passed}/{total} 통과")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 환경이 준비되었습니다.")
        print("\n다음 단계:")
        print("  python training/integrated/train_integrated.py")
    else:
        print("⚠️ 일부 테스트 실패. 문제를 해결하고 다시 실행하세요.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)