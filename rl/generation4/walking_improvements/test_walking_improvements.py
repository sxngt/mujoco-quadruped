#!/usr/bin/env python3
"""
점프 방지 및 보행 강화 시스템 테스트
로봇이 점프하지 않고 실제 보행을 시도하는지 확인
"""

import numpy as np
import time
from collections import defaultdict

def test_walking_behavior():
    """보행 개선 사항 테스트"""
    print("🚶‍♀️ 점프 방지 및 보행 강화 시스템 테스트")
    
    try:
        from improved_environment import ImprovedGO2Env
        
        print("✅ 개선된 환경 모듈 로드 성공")
        
        # 환경 생성 (렌더링 없이)
        env = ImprovedGO2Env(render_mode=None, use_reference_gait=False)
        obs, info = env.reset()
        
        print(f"관찰 공간: {env.observation_space.shape}")
        print(f"행동 공간: {env.action_space.shape}")
        print(f"최대 에피소드 길이: {env.max_episode_steps}")
        
        # 테스트 메트릭
        episode_rewards = []
        contact_stats = defaultdict(int)  # 접촉 수별 빈도
        vertical_velocities = []
        forward_velocities = []
        heights = []
        jump_count = 0
        walking_steps = 0
        
        test_steps = 500
        print(f"\n🔬 {test_steps}스텝 테스트 시작...")
        
        for step in range(test_steps):
            # 작은 범위의 랜덤 액션 (점프 방지)
            action = np.random.uniform(-0.2, 0.2, env.action_space.shape[0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 메트릭 수집
            episode_rewards.append(reward)
            
            # 접촉 정보
            contacts = env._get_foot_contacts()
            num_contacts = sum(contacts)
            contact_stats[num_contacts] += 1
            
            # 물리 상태
            vertical_vel = env.data.qvel[2]
            forward_vel = env.data.qvel[0]
            height = env.data.qpos[2]
            
            vertical_velocities.append(vertical_vel)
            forward_velocities.append(forward_vel)
            heights.append(height)
            
            # 점프 감지 (상향 속도 > 0.1m/s)
            if vertical_vel > 0.1:
                jump_count += 1
            
            # 보행 감지 (접촉하며 전진)
            if num_contacts >= 1 and forward_vel > 0.02:
                walking_steps += 1
            
            # 주기적 리포트
            if step % 100 == 0 and step > 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_contacts = np.mean([contacts for contacts in contact_stats.keys() 
                                      for _ in range(contact_stats[contacts])][-100:] if contact_stats else [0])
                recent_jumps = sum(1 for v in vertical_velocities[-100:] if v > 0.1)
                recent_walking = sum(1 for i in range(max(0, step-100), step) 
                                   if (sum(env._get_foot_contacts()) >= 1 and 
                                       forward_velocities[i] > 0.02) if i < len(forward_velocities) else False)
                
                print(f"스텝 {step:3d}: 보상 {avg_reward:6.1f}, 평균접촉 {avg_contacts:.1f}, 점프 {recent_jumps:2d}회, 보행 {recent_walking:2d}회")
            
            if terminated or truncated:
                print(f"⚠️ 에피소드 조기 종료: 스텝 {step}")
                obs, info = env.reset()
        
        env.close()
        
        # === 결과 분석 ===
        print(f"\n📊 테스트 결과 분석")
        print(f"{'='*50}")
        
        # 기본 통계
        print(f"평균 보상: {np.mean(episode_rewards):.2f}")
        print(f"보상 표준편차: {np.std(episode_rewards):.2f}")
        print(f"최대 보상: {np.max(episode_rewards):.2f}")
        print(f"최소 보상: {np.min(episode_rewards):.2f}")
        
        # 접촉 분석
        print(f"\n👣 발 접촉 분석:")
        total_steps = sum(contact_stats.values())
        for contacts in sorted(contact_stats.keys()):
            percentage = (contact_stats[contacts] / total_steps) * 100
            print(f"  {contacts}개 발 접촉: {contact_stats[contacts]:4d}회 ({percentage:5.1f}%)")
        
        # 움직임 분석
        print(f"\n🏃‍♀️ 움직임 분석:")
        print(f"점프 횟수 (수직속도>0.1): {jump_count:4d}회 ({jump_count/test_steps*100:5.1f}%)")
        print(f"보행 스텝 (접촉+전진): {walking_steps:4d}회 ({walking_steps/test_steps*100:5.1f}%)")
        print(f"평균 전진 속도: {np.mean(forward_velocities):6.3f} m/s")
        print(f"평균 높이: {np.mean(heights):6.3f} m")
        print(f"수직 속도 최대: {np.max(np.abs(vertical_velocities)):6.3f} m/s")
        
        # 성능 평가
        print(f"\n🎯 성능 평가:")
        jump_rate = jump_count / test_steps * 100
        walk_rate = walking_steps / test_steps * 100
        ground_contact_rate = (sum(contact_stats[i] for i in contact_stats if i > 0) / total_steps) * 100
        
        print(f"점프 비율: {jump_rate:.1f}% ", end="")
        if jump_rate < 5:
            print("✅ 우수 (5% 미만)")
        elif jump_rate < 15:
            print("⚠️ 보통 (15% 미만)")
        else:
            print("❌ 개선 필요 (15% 이상)")
        
        print(f"보행 비율: {walk_rate:.1f}% ", end="")
        if walk_rate > 30:
            print("✅ 우수 (30% 이상)")
        elif walk_rate > 15:
            print("⚠️ 보통 (15% 이상)")
        else:
            print("❌ 개선 필요 (15% 미만)")
        
        print(f"지상 접촉 비율: {ground_contact_rate:.1f}% ", end="")
        if ground_contact_rate > 85:
            print("✅ 우수 (85% 이상)")
        elif ground_contact_rate > 70:
            print("⚠️ 보통 (70% 이상)")
        else:
            print("❌ 개선 필요 (70% 미만)")
        
        # 전체 평가
        improvements = 0
        if jump_rate < 15: improvements += 1
        if walk_rate > 15: improvements += 1
        if ground_contact_rate > 70: improvements += 1
        
        print(f"\n🏆 종합 평가: {improvements}/3 개선사항 달성")
        if improvements >= 2:
            print("✅ 점프 방지 및 보행 강화 시스템 성공!")
        else:
            print("⚠️ 추가 개선이 필요합니다.")
            
        return {
            'jump_rate': jump_rate,
            'walk_rate': walk_rate,
            'ground_contact_rate': ground_contact_rate,
            'avg_reward': np.mean(episode_rewards),
            'improvements': improvements
        }
        
    except ImportError as e:
        print(f"❌ 모듈 로드 실패: {e}")
        print("MuJoCo가 설치되지 않았거나 환경 설정에 문제가 있습니다.")
        return None
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    result = test_walking_behavior()
    if result:
        print(f"\n📋 테스트 완료 - 점프률: {result['jump_rate']:.1f}%, 보행률: {result['walk_rate']:.1f}%")