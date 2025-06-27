#!/usr/bin/env python3
"""
통합 GO2 환경을 위한 훈련 스크립트
참조 레포지터리의 성공적인 방법론을 GO2에 적용
"""

import os
import sys
import numpy as np
import torch
import time
import argparse
from datetime import datetime

# 프로젝트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(rl_dir)

from environments.integrated import IntegratedGO2Env
from agents.ppo_agent import PPOAgent
import gymnasium as gym


class IntegratedTrainer:
    def __init__(self, 
                 total_timesteps=5_000_000,
                 eval_freq=10_000,
                 save_freq=50_000,
                 log_freq=1000,
                 render_training=False):
        
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.render_training = render_training
        
        # 환경 생성 (훈련 중 렌더링 여부에 따라)
        render_mode = "human" if render_training else None
        self.env = IntegratedGO2Env(render_mode=render_mode)
        
        # PPO 에이전트 생성 (참조 레포지터리 하이퍼파라미터 사용)
        self.agent = PPOAgent(
            obs_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            lr=3e-4,           # 참조 방식
            gamma=0.99,        # 참조 방식
            gae_lambda=0.95,   # 참조 방식
            clip_ratio=0.2,    # 참조 방식
            value_coef=0.5,    # 참조 방식
            entropy_coef=0.01, # 참조 방식
            max_grad_norm=0.5, # 참조 방식
            hidden_dim=256     # 참조 방식
        )
        
        # 저장 디렉토리 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(rl_dir, "models", "integrated", f"integrated_go2_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 로그 파일
        self.log_file = os.path.join(self.save_dir, "training_log.txt")
        
        # 통계 추적
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')
        
        print(f"=== 통합 GO2 훈련 시작 ===")
        print(f"총 타임스텝: {total_timesteps:,}")
        print(f"관찰 차원: {self.env.observation_space.shape[0]}")
        print(f"액션 차원: {self.env.action_space.shape[0]}")
        print(f"저장 위치: {self.save_dir}")
        print(f"디바이스: {self.agent.device}")
        print(f"훈련 중 렌더링: {'✅ 활성화' if self.render_training else '❌ 비활성화'}")
        
    def log_message(self, message):
        """로그 메시지 출력 및 파일 저장"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: {message}\n")
    
    def evaluate_agent(self, num_episodes=5):
        """에이전트 평가"""
        eval_rewards = []
        eval_lengths = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, _, _ = self.agent.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        
        return avg_reward, avg_length, eval_rewards, eval_lengths
    
    def train(self):
        """메인 훈련 루프"""
        
        obs, _ = self.env.reset()
        total_steps = 0
        episode_num = 0
        episode_reward = 0
        episode_length = 0
        start_time = time.time()
        
        # 훈련 파라미터 (참조 방식)
        rollout_length = 2048  # 참조 레포지터리 방식
        
        while total_steps < self.total_timesteps:
            
            # Rollout 수집
            for step in range(rollout_length):
                action, log_prob, value = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 훈련 중 렌더링 (최적화됨)
                if self.render_training:
                    # 성능을 위해 매 스텝이 아닌 적절한 간격으로 렌더링
                    should_render = (total_steps % 2 == 0)  # 2스텝마다 렌더링 (25 FPS 정도)
                    
                    if should_render:
                        if total_steps % 100 == 0:  # 100스텝마다 상태 로그
                            print(f"🎬 렌더링 중... (step {total_steps}, episode {episode_num})")
                        
                        try:
                            render_result = self.env.render()
                            
                            # 초기 렌더링 상태 확인
                            if total_steps < 10:
                                print(f"  Step {total_steps}: 렌더링 = {render_result}")
                                if render_result is None:
                                    print(f"    뷰어 상태: {type(self.env.viewer)}")
                                    
                            # 렌더링 실패 시 뷰어 재초기화 시도 (한 번만)
                            if render_result is None and not hasattr(self, '_viewer_reset_attempted'):
                                print("🔧 뷰어 재초기화 시도...")
                                self.env.viewer = None
                                self._viewer_reset_attempted = True
                                
                        except Exception as e:
                            if total_steps < 10:
                                print(f"  ❌ Step {total_steps} 렌더링 실패: {e}")
                            # 치명적이지 않은 렌더링 오류는 무시하고 계속 진행
                
                # 경험 저장
                self.agent.store_transition(obs, action, reward, value, log_prob, terminated)
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # 에피소드 종료
                if terminated or truncated:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # 최고 성능 업데이트
                    if episode_reward > self.best_reward:
                        self.best_reward = episode_reward
                        # 최고 모델 저장
                        best_path = os.path.join(self.save_dir, "best_model.pth")
                        self.agent.save(best_path)
                    
                    # 로그 출력
                    if episode_num % self.log_freq == 0:
                        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
                        avg_reward = np.mean(recent_rewards)
                        elapsed_time = time.time() - start_time
                        
                        message = (f"Episode {episode_num:,} | "
                                 f"Steps: {total_steps:,} | "
                                 f"Reward: {episode_reward:.2f} | "
                                 f"Avg(100): {avg_reward:.2f} | "
                                 f"Best: {self.best_reward:.2f} | "
                                 f"Length: {episode_length} | "
                                 f"Time: {elapsed_time:.1f}s")
                        self.log_message(message)
                        
                        # 상세 정보 출력
                        if 'total' in info:
                            detail_msg = (f"  -> Forward: {info.get('lin_vel_reward', 0):.2f} | "
                                        f"Alive: {info.get('alive_reward', 0):.2f} | "
                                        f"Torque: {info.get('torque_cost', 0):.2f}")
                            self.log_message(detail_msg)
                    
                    # 리셋
                    obs, _ = self.env.reset()
                    episode_num += 1
                    episode_reward = 0
                    episode_length = 0
                
                # 훈련 중단 체크
                if total_steps >= self.total_timesteps:
                    break
            
            # 정책 업데이트 (PPO)
            if len(self.agent.observations) > 0:
                update_info = self.agent.update_policy(n_epochs=10, batch_size=64)
                
                if total_steps % (self.log_freq * 10) == 0:
                    update_msg = (f"Policy Update | "
                                f"Policy Loss: {update_info['policy_loss']:.4f} | "
                                f"Value Loss: {update_info['value_loss']:.4f} | "
                                f"Entropy: {update_info['entropy_loss']:.4f}")
                    self.log_message(update_msg)
            
            # 평가
            if total_steps % self.eval_freq == 0 and total_steps > 0:
                self.log_message("=== 평가 시작 ===")
                avg_reward, avg_length, eval_rewards, eval_lengths = self.evaluate_agent(num_episodes=5)
                
                eval_msg = (f"Evaluation | "
                          f"Avg Reward: {avg_reward:.2f} | "
                          f"Avg Length: {avg_length:.1f} | "
                          f"Rewards: {[f'{r:.1f}' for r in eval_rewards]}")
                self.log_message(eval_msg)
                self.log_message("=== 평가 완료 ===")
            
            # 모델 저장
            if total_steps % self.save_freq == 0 and total_steps > 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{total_steps}.pth")
                self.agent.save(checkpoint_path)
                self.log_message(f"모델 저장: {checkpoint_path}")
        
        # 최종 모델 저장
        final_path = os.path.join(self.save_dir, "final_model.pth")
        self.agent.save(final_path)
        
        # 훈련 완료 메시지
        total_time = time.time() - start_time
        self.log_message("=== 훈련 완료 ===")
        self.log_message(f"총 시간: {total_time/3600:.2f}시간")
        self.log_message(f"총 에피소드: {episode_num}")
        self.log_message(f"최고 보상: {self.best_reward:.2f}")
        self.log_message(f"최종 평균 보상: {np.mean(self.episode_rewards[-100:]):.2f}")
        
        return self.episode_rewards, self.episode_lengths
    
    def test_trained_model(self, model_path, num_episodes=10, render=True):
        """훈련된 모델 테스트"""
        
        # 모델 로드
        self.agent.load(model_path)
        self.log_message(f"모델 로드: {model_path}")
        
        # 렌더링 환경 생성
        if render:
            test_env = IntegratedGO2Env(render_mode="human")
        else:
            test_env = self.env
        
        test_rewards = []
        test_lengths = []
        
        for episode in range(num_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            episode_length = 0
            
            self.log_message(f"=== 테스트 에피소드 {episode + 1} ===")
            
            while True:
                action, _, _ = self.agent.get_action(obs)
                obs, reward, terminated, truncated, info = test_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # 진행 상황 출력
                if episode_length % 100 == 0:
                    pos_x = obs[0] if len(obs) > 0 else 0  # 대략적인 위치
                    forward_vel = info.get('lin_vel_reward', 0)
                    self.log_message(f"  Step {episode_length}: 보상 {reward:.2f}, 전진보상 {forward_vel:.2f}")
                
                if render:
                    test_env.render()
                    time.sleep(0.01)  # 시각적 확인을 위한 지연
                
                if terminated or truncated:
                    break
            
            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
            
            self.log_message(f"에피소드 {episode + 1} 완료: 보상 {episode_reward:.2f}, 길이 {episode_length}")
        
        # 테스트 결과 요약
        avg_reward = np.mean(test_rewards)
        avg_length = np.mean(test_lengths)
        
        self.log_message("=== 테스트 결과 ===")
        self.log_message(f"평균 보상: {avg_reward:.2f}")
        self.log_message(f"평균 길이: {avg_length:.1f}")
        self.log_message(f"모든 보상: {[f'{r:.1f}' for r in test_rewards]}")
        
        if render:
            test_env.close()
        
        return test_rewards, test_lengths


def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="통합 GO2 환경 훈련 스크립트")
    
    # 훈련 관련 옵션
    parser.add_argument("--total_timesteps", type=int, default=3_000_000,
                        help="총 훈련 타임스텝 (기본값: 3,000,000)")
    parser.add_argument("--eval_freq", type=int, default=25_000,
                        help="평가 주기 (기본값: 25,000)")
    parser.add_argument("--save_freq", type=int, default=100_000,
                        help="모델 저장 주기 (기본값: 100,000)")
    parser.add_argument("--log_freq", type=int, default=50,
                        help="로그 출력 주기 (기본값: 50)")
    
    # 렌더링 및 테스트 옵션
    parser.add_argument("--render", action="store_true",
                        help="훈련 중 실시간 렌더링 활성화 (학습 과정 시각화)")
    parser.add_argument("--test_episodes", type=int, default=3,
                        help="테스트 에피소드 수 (기본값: 3)")
    parser.add_argument("--no_test", action="store_true",
                        help="훈련 후 테스트 건너뛰기")
    parser.add_argument("--render_test_only", action="store_true",
                        help="훈련 없이 렌더링 테스트만 즉시 실행")
    parser.add_argument("--quick_render", action="store_true",
                        help="임의 행동으로 즉시 렌더링 테스트 (훈련 없음)")
    parser.add_argument("--render_steps", type=int, default=None,
                        help="짧은 훈련으로 렌더링 테스트 (예: --render_steps 500)")
    
    # 모델 로드 관련
    parser.add_argument("--load_model", type=str, default=None,
                        help="기존 모델 경로에서 훈련 계속하기")
    parser.add_argument("--test_only", action="store_true",
                        help="훈련 없이 테스트만 실행")
    
    return parser.parse_args()


def main():
    """메인 함수"""
    
    # 명령줄 인수 파싱
    args = parse_args()
    
    # 렌더링 테스트를 위한 짧은 훈련 옵션
    if args.render_steps:
        args.total_timesteps = args.render_steps
        args.render = True
        print(f"🎯 짧은 렌더링 테스트 모드: {args.render_steps} 스텝")
    
    # 훈련 시작
    trainer = IntegratedTrainer(
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        render_training=args.render  # 훈련 중 렌더링 옵션 전달
    )
    
    try:
        # 즉시 렌더링 테스트 (훈련 없음)
        if args.quick_render or args.render_test_only:
            print(f"\n=== 즉시 렌더링 테스트 ===")
            test_env = IntegratedGO2Env(render_mode="human")
            obs, _ = test_env.reset()
            
            print("무작위 행동으로 렌더링 테스트 중... (Ctrl+C로 종료)")
            step_count = 0
            
            try:
                while True:
                    # 무작위 행동 (약하게)
                    action = test_env.action_space.sample() * 0.3
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    test_env.render()
                    
                    step_count += 1
                    if step_count % 100 == 0:
                        print(f"Step {step_count}, 보상: {reward:.2f}")
                    
                    if terminated or truncated:
                        print(f"에피소드 종료 (step {step_count}), 리셋...")
                        obs, _ = test_env.reset()
                        step_count = 0
                    
                    time.sleep(0.02)  # 50 FPS
                    
            except KeyboardInterrupt:
                print(f"\n렌더링 테스트 완료 ({step_count} 스텝)")
            finally:
                test_env.close()
            return
        
        # 테스트만 실행하는 경우 (훈련된 모델)
        elif args.test_only:
            if args.load_model and os.path.exists(args.load_model):
                print(f"\n=== 모델 테스트 모드 ===")
                trainer.test_trained_model(args.load_model, num_episodes=args.test_episodes, render=args.render)
            else:
                print("테스트 모드에서는 --load_model 옵션이 필요합니다.")
                return
        else:
            # 기존 모델 로드 (있는 경우)
            if args.load_model and os.path.exists(args.load_model):
                trainer.agent.load(args.load_model)
                print(f"기존 모델 로드: {args.load_model}")
            
            # 훈련 실행
            episode_rewards, episode_lengths = trainer.train()
            
            # 테스트 실행 (건너뛰기 옵션이 없는 경우)
            if not args.no_test:
                best_model_path = os.path.join(trainer.save_dir, "best_model.pth")
                if os.path.exists(best_model_path):
                    print("\n=== 최고 모델 테스트 ===")
                    trainer.test_trained_model(best_model_path, num_episodes=args.test_episodes, render=args.render)
        
    except KeyboardInterrupt:
        print("\n훈련이 사용자에 의해 중단되었습니다.")
        
        # 현재까지의 모델 저장
        interrupt_path = os.path.join(trainer.save_dir, "interrupted_model.pth")
        trainer.agent.save(interrupt_path)
        print(f"중단된 모델 저장: {interrupt_path}")
        
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 환경 정리
        trainer.env.close()
        print("훈련 완료 및 정리")


if __name__ == "__main__":
    main()