import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Value head
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        features = self.shared(obs)
        
        # Policy
        mean = self.policy_mean(features)
        std = torch.exp(self.policy_logstd)
        
        # Value
        value = self.value(features)
        
        return mean, std, value
    
    def get_action(self, obs):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob.squeeze(-1), value.squeeze(-1)
    
    def evaluate_actions(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob.squeeze(-1), entropy.squeeze(-1), value.squeeze(-1)


class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, hidden_dim=256):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Storage for rollout data
        self.reset_rollout()
    
    def reset_rollout(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(obs_tensor)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value=0):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        
        advantages = []
        gae = 0
        for i in reversed(range(len(deltas))):
            gae = deltas[i] + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update_policy(self, n_epochs=10, batch_size=64):
        if len(self.observations) < batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
        
        # Compute advantages and returns
        if len(self.observations) > 0:
            last_obs = torch.FloatTensor(self.observations[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, next_value = self.policy(last_obs)
                next_value = next_value.cpu().numpy().item()
        else:
            next_value = 0.0
        
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors with consistent shapes
        obs_array = np.array(self.observations, dtype=np.float32)
        actions_array = np.array(self.actions, dtype=np.float32)
        log_probs_array = np.array(self.log_probs, dtype=np.float32).reshape(-1)
        advantages_array = np.array(advantages, dtype=np.float32).reshape(-1)
        returns_array = np.array(returns, dtype=np.float32).reshape(-1)
        
        obs_tensor = torch.from_numpy(obs_array).to(self.device)
        actions_tensor = torch.from_numpy(actions_array).to(self.device)
        old_log_probs = torch.from_numpy(log_probs_array).to(self.device)
        advantages_tensor = torch.from_numpy(advantages_array).to(self.device)
        returns_tensor = torch.from_numpy(returns_array).to(self.device)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        
        dataset_size = len(self.observations)
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Evaluate current policy
                log_probs, entropy, values = self.policy.evaluate_actions(batch_obs, batch_actions)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1
        
        # Reset rollout buffer
        self.reset_rollout()
        
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy_loss': total_entropy_loss / max(n_updates, 1),
            'approx_kl': 0.0
        }
    
    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])