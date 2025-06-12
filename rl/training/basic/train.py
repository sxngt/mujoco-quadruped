import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from ...environments.basic.environment import GO2ForwardEnv
from ...agents.ppo_agent import PPOAgent
import wandb
import argparse
import os
from datetime import datetime


def train_ppo(args):
    # Initialize environment
    env = GO2ForwardEnv(
        render_mode="human" if args.render else None,
        use_reference_gait=not args.no_reference_gait
    )
    
    # Initialize agent
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio
    )
    
    # Initialize logging
    if args.wandb:
        wandb.init(
            project="go2-forward-locomotion",
            config=vars(args),
            name=f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Training variables
    episode_rewards = []
    episode_lengths = []
    best_reward = -np.inf
    
    print(f"Starting training for {args.total_timesteps} timesteps...")
    print(f"Device: {agent.device}")
    
    timestep = 0
    episode = 0
    
    while timestep < args.total_timesteps:
        obs, _ = env.reset(seed=args.seed + episode if args.seed else None)
        episode_reward = 0
        episode_length = 0
        
        for step in range(args.max_episode_steps):
            # Get action from agent
            action, log_prob, value = agent.get_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(obs, action, reward, value, log_prob, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            timestep += 1
            
            if args.render:
                env.render()
            
            if done or step == args.max_episode_steps - 1:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode += 1
        
        # Update policy every rollout_length timesteps
        if len(agent.observations) >= args.rollout_length:
            losses = agent.update_policy(
                n_epochs=args.ppo_epochs,
                batch_size=args.batch_size
            )
            
            # Log training metrics
            window_size = min(10, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-window_size:]) if episode_rewards else 0.0
            avg_length = np.mean(episode_lengths[-window_size:]) if episode_lengths else 0.0
            
            print(f"Episode {episode:4d} | Timestep {timestep:7d} | "
                  f"Reward: {episode_reward:7.2f} | Avg Reward: {avg_reward:7.2f} | "
                  f"Length: {episode_length:3d}")
            
            if args.wandb:
                log_dict = {
                    "episode": episode,
                    "timestep": timestep,
                    "episode_reward": episode_reward,
                    "avg_reward": avg_reward,
                    "episode_length": episode_length,
                    "avg_length": avg_length,
                }
                # Only add losses if they exist
                if losses:
                    log_dict.update(losses)
                wandb.log(log_dict)
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                if not os.path.exists("models"):
                    os.makedirs("models")
                agent.save(f"models/best_go2_ppo.pth")
                print(f"New best model saved! Avg reward: {best_reward:.2f}")
        
        # Save checkpoint periodically
        if episode % args.save_freq == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            agent.save(f"models/go2_ppo_episode_{episode}.pth")
    
    # Final save
    if not os.path.exists("models"):
        os.makedirs("models")
    agent.save("models/go2_ppo_final.pth")
    
    env.close()
    if args.wandb:
        wandb.finish()
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print(f"Training completed! Best average reward: {best_reward:.2f}")


def evaluate_agent(args):
    # Load environment and agent
    env = GO2ForwardEnv(render_mode="human")
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # Load trained model
    if os.path.exists(args.model_path):
        agent.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Model file {args.model_path} not found!")
        return
    
    # Evaluate for multiple episodes
    total_rewards = []
    
    for episode in range(args.eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(args.max_episode_steps):
            action, _, _ = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            env.render()
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Best Reward: {np.max(total_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for GO2 forward locomotion')
    
    # Training arguments
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], 
                        help='Training or evaluation mode')
    parser.add_argument('--total_timesteps', type=int, default=1000000, 
                        help='Total training timesteps')
    parser.add_argument('--max_episode_steps', type=int, default=1000, 
                        help='Maximum steps per episode')
    parser.add_argument('--rollout_length', type=int, default=2048, 
                        help='Rollout length for PPO updates')
    parser.add_argument('--num_envs', type=int, default=1, 
                        help='Number of parallel environments (for future vectorization)')
    
    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lr_schedule', type=str, default='constant', 
                        choices=['constant', 'linear'], help='Learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='PPO update epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for PPO updates')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm for clipping')
    
    # Logging and saving
    parser.add_argument('--save_freq', type=int, default=100, help='Save frequency (episodes)')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--render', action='store_true', help='Render environment during training')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    # Evaluation arguments
    parser.add_argument('--model_path', type=str, default='models/best_go2_ppo.pth', 
                        help='Path to trained model for evaluation')
    parser.add_argument('--eval_episodes', type=int, default=10, 
                        help='Number of episodes for evaluation')
    
    # Reference gait arguments
    parser.add_argument('--no_reference_gait', action='store_true', 
                        help='Disable reference gait imitation learning')
    
    args = parser.parse_args()
    
    # Set random seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    if args.mode == 'train':
        train_ppo(args)
    elif args.mode == 'eval':
        evaluate_agent(args)


if __name__ == "__main__":
    main()