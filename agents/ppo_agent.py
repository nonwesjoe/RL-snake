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

class ValueNet(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)

class PPOAgent:
    def __init__(self, obs_dim, act_dim, seed=42, hidden_dim=128, lr=3e-4, 
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, 
                 ent_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy = PolicyNet(obs_dim, hidden_dim, act_dim).to(self.device)
        self.value = ValueNet(obs_dim, hidden_dim).to(self.device)
        self._init_weights()
        
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)
    
    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.policy.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Policy输出层使用更小的初始化
        nn.init.orthogonal_(self.policy.fc3.weight, gain=0.01)
        
        for m in self.value.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def select_action(self, obs):
        """选择动作并返回log概率"""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
        return int(a.item()), logp.item()
    
    def get_value(self, obs):
        """获取状态价值"""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            val = self.value(x)
        return val.item()
    
    def compute_gae(self, rews, vals, dones, next_val=0.0):
        """计算GAE (Generalized Advantage Estimation)"""
        T = len(rews)
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0
        
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                next_val_t = next_val
            else:
                next_val_t = vals[t + 1]
            
            delta = rews[t] + self.gamma * (1.0 - dones[t]) * next_val_t - vals[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + vals
        return advantages, returns
    
    def update(self, obs_list, act_list, rew_list, done_list, old_logp_list, 
               next_obs=None, epochs=4, batch_size=64):
        """
        PPO更新：使用clip机制防止策略更新过大
        """
        if len(obs_list) == 0:
            return 0.0
        
        # 转换为tensor
        obs = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(act_list, dtype=torch.int64, device=self.device)
        rews = torch.as_tensor(rew_list, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(done_list, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(old_logp_list, dtype=torch.float32, device=self.device)
        
        # 计算next_val
        if next_obs is not None:
            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_val = self.value(next_obs_tensor).item()
        else:
            next_val = 0.0
        
        # 计算value和GAE
        with torch.no_grad():
            vals_old = self.value(obs)
        advantages, returns = self.compute_gae(rews, vals_old, dones, next_val)
        
        # 归一化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备数据索引用于mini-batch训练
        indices = np.arange(len(obs_list))
        total_loss = 0.0
        
        # 多次更新（提高样本效率）
        for epoch in range(epochs):
            np.random.shuffle(indices)
            
            # Mini-batch更新
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_acts = acts[batch_indices]
                batch_old_logp = old_logp[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的log概率
                logits = self.policy(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(batch_acts)
                ent = dist.entropy().mean()
                
                # 计算ratio
                ratio = torch.exp(new_logp - batch_old_logp)
                
                # PPO clip loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * ent
                
                # Value loss
                vals = self.value(batch_obs)
                value_loss = F.mse_loss(vals, batch_returns)
                
                # 更新policy
                self.policy_opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.policy_opt.step()
                
                # 更新value
                self.value_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.max_grad_norm)
                self.value_opt.step()
                
                total_loss += (policy_loss.item() + self.value_coef * value_loss.item())
        
        return total_loss / (epochs * (len(indices) // batch_size + 1))
    
    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "policy_opt": self.policy_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
        }, path)

    def load(self, path, load_optimizers=True):
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        if load_optimizers:
            if "policy_opt" in state:
                self.policy_opt.load_state_dict(state["policy_opt"])
            if "value_opt" in state:
                self.value_opt.load_state_dict(state["value_opt"])
