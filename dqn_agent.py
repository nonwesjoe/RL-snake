import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity,), dtype=torch.int64)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, no, d):
        i = self.ptr % self.capacity
        self.obs[i] = torch.as_tensor(o, dtype=torch.float32)
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.next_obs[i] = torch.as_tensor(no, dtype=torch.float32)
        self.dones[i] = float(d)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.obs[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_obs[idx].to(self.device),
            self.dones[idx].to(self.device),
        )

class DQNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class DQNAgent:
    def __init__(self, obs_dim, act_dim, seed=42, hidden_dim=192, lr=3e-4, gamma=0.99, eps_start=1.0, eps_end=0.1, eps_decay=10000, buffer_size=100000, batch_size=128, target_update=2000, lr_t_max=20000, lr_eta_min=1e-4):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = DQNNet(obs_dim, hidden_dim, act_dim).to(self.device)
        self.target = DQNNet(obs_dim, hidden_dim, act_dim).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=lr_t_max, eta_min=lr_eta_min)
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0
        self.buffer = ReplayBuffer(buffer_size, obs_dim, self.device)
        self.batch_size = batch_size
        self.target_update = target_update
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon(self):
        t = self.steps
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-t / self.eps_decay)

    def select_action(self, obs):
        if torch.rand(()) < self.epsilon():
            return int(torch.randint(0, self.act_dim, (1,)).item())
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            a = int(torch.argmax(q[0]).item())
            return a

    def store(self, o, a, r, no, d):
        self.buffer.add(o, a, r, no, float(d))

    def train_step(self):
        if self.buffer.size < self.batch_size:
            return 0.0
        o, a, r, no, d = self.buffer.sample(self.batch_size)
        q = self.net(o)
        q_sel = q.gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            tq = self.target(no)
            tmax = tq.max(1)[0]
            target = r + (1.0 - d) * self.gamma * tmax
        loss = self.loss_fn(q_sel, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.opt.step()
        self.scheduler.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state)
        self.target.load_state_dict(self.net.state_dict())
