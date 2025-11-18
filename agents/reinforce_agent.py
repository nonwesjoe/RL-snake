import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReinforceAgent:
    def __init__(self, obs_dim, act_dim, seed=42, hidden_dim=128, lr=1e-3, gamma=0.99):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        
        self.policy = PolicyNet(obs_dim, hidden_dim, act_dim).to(self.device)
        self._init_weights()
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.log_probs = []
        self.rewards = []

    def _init_weights(self):
        for m in self.policy.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy.fc3.weight, gain=0.01)

    def select_action(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
        
        # Store log probability for the update
        dist = torch.distributions.Categorical(logits=self.policy(x))
        log_prob = dist.log_prob(a)
        self.log_probs.append(log_prob)
        
        return int(a.item())

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        if not self.log_probs:
            return 0.0

        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs = []
        self.rewards = []
        
        return loss.item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
