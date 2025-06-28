#!/usr/bin/env python3
"""
Stable Baselines3를 사용한 GO2 훈련 - 참조 레포지터리 완전 모방
"""

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# 프로젝트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from go2_mujoco_env import GO2MujocoEnv
from tqdm import tqdm

MODEL_DIR = "models_sb3"
LOG_DIR = "logs_sb3"


def train(args):
    """훈련 함수 - 참조 레포지터리와 동일"""
    # 렌더링을 위한 환경 설정
    env_kwargs = {"ctrl_type": args.ctrl_type}
    
    if args.render_training:
        # 렌더링 모드에서는 단일 환경 사용 (병렬 렌더링 불가)
        env_kwargs["render_mode"] = "human"
        vec_env = make_vec_env(
            GO2MujocoEnv,
            env_kwargs=env_kwargs,
            n_envs=1,  # 렌더링 시 단일 환경
            seed=args.seed,
        )
        print("⚠️  렌더링 모드: 단일 환경으로 자동 전환")
    else:
        vec_env = make_vec_env(
            GO2MujocoEnv,
            env_kwargs=env_kwargs,
            n_envs=args.num_parallel_envs,
            seed=args.seed,
            vec_env_cls=SubprocVecEnv,
        )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name is None:
        run_name = f"{train_time}"
    else:
        run_name = f"{train_time}-{args.run_name}"

    model_path = f"{MODEL_DIR}/{run_name}"
    print(
        f"🚀 {args.num_parallel_envs}개 병렬 환경에서 훈련 시작, 모델 저장: '{model_path}'"
    )

    # 평가 콜백 (참조 레포지터리와 동일)
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=model_path,
        log_path=LOG_DIR,
        eval_freq=args.eval_frequency,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    if args.model_path is not None:
        model = PPO.load(
            path=args.model_path, env=vec_env, verbose=1, tensorboard_log=LOG_DIR
        )
    else:
        # 기본 PPO 설정 (참조 레포지터리와 동일)
        # CPU 사용 권장 (MLP 정책은 GPU 효율이 낮음)
        device = "cpu" if not args.force_gpu else "auto"
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR, device=device)

    print(f"📊 총 타임스텝: {args.total_timesteps:,}")
    print(f"🔄 평가 주기: {args.eval_frequency:,}")
    print(f"🎬 훈련 중 렌더링: {'✅ 활성화' if args.render_training else '❌ 비활성화'}")
    
    if args.render_training:
        print("⚠️  렌더링 모드에서는 훈련 속도가 느려질 수 있습니다")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=eval_callback,
    )
    
    # 최종 모델 저장
    model.save(f"{model_path}/final_model")
    print(f"✅ 훈련 완료! 모델 저장: {model_path}")


def test(args):
    """테스트 함수 - 참조 레포지터리와 동일"""
    model_path = Path(args.model_path)

    if not args.record_test_episodes:
        # 실시간 렌더링
        env = GO2MujocoEnv(
            ctrl_type=args.ctrl_type,
            render_mode="human",
        )
        inter_frame_sleep = 0.016
    else:
        # 에피소드 녹화
        env = GO2MujocoEnv(
            ctrl_type=args.ctrl_type,
            render_mode="rgb_array",
            camera_name="tracking",
            width=1920,
            height=1080,
        )
        env = gym.wrappers.RecordVideo(
            env, video_folder="recordings/", name_prefix=model_path.parent.name
        )
        inter_frame_sleep = 0.0

    model = PPO.load(path=model_path, env=env, verbose=1)

    num_episodes = args.num_test_episodes
    total_reward = 0
    total_length = 0
    
    print(f"🎮 {num_episodes}개 에피소드 테스트 시작...")
    
    for ep in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        env.render()

        ep_len = 0
        ep_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            # 렌더링 속도 조절
            time.sleep(inter_frame_sleep)

            if terminated or truncated:
                print(f"에피소드 {ep+1}: 길이={ep_len}, 보상={ep_reward:.2f}")
                break

        total_length += ep_len
        total_reward += ep_reward

    print(f"📈 평균 에피소드 보상: {total_reward / num_episodes:.2f}")
    print(f"📏 평균 에피소드 길이: {total_length / num_episodes:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GO2 Stable Baselines3 훈련/테스트")
    parser.add_argument("--run", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="실행 이름 (모델 저장시 사용)",
    )
    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        default=12,
        help="병렬 환경 수",
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=5,
        help="테스트 에피소드 수",
    )
    parser.add_argument(
        "--record_test_episodes",
        action="store_true",
        help="테스트 에피소드를 녹화할지 여부",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=5_000_000,
        help="총 훈련 타임스텝",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=10_000,
        help="모델 평가 주기",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="모델 경로 (.zip 파일)",
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["torque", "position"],
        default="torque",
        help="제어 타입",
    )
    parser.add_argument(
        "--render_training",
        action="store_true",
        help="훈련 중 실시간 렌더링 활성화 (속도 저하 있음)",
    )
    parser.add_argument(
        "--force_gpu",
        action="store_true",
        help="GPU 강제 사용 (MLP 정책에서는 비권장)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run == "train":
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        train(args)
    elif args.run == "test":
        if args.model_path is None:
            raise ValueError("테스트를 위해서는 --model_path가 필요합니다")
        test(args)