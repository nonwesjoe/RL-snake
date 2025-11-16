import argparse
import os
import time
import numpy as np
import torch
from env import SnakeEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt

def train(episodes, size, seed, save_path, max_steps):
    env = SnakeEnv(size=size, seed=seed)
    obs = env.reset()
    agent = DQNAgent(obs_dim=obs.shape[0], act_dim=3, seed=seed)
    rewards = []
    losses = []
    scores = []
    for ep in range(episodes):
        o = env.reset()
        total = 0.0
        for t in range(max_steps):
            a = agent.select_action(o)
            no, r, d, info = env.step(a)
            agent.store(o, a, r, no, d)
            loss = agent.train_step()
            total += r
            o = no
            if loss:
                losses.append(loss)
            if d:
                break
        rewards.append(total)
        scores.append(info["score"])
        print(f"episode={ep+1} reward={total:.2f} score={info['score']} epsilon={agent.epsilon():.3f}")
    if save_path:
        agent.save(save_path)
        print(f"saved={save_path}")
    if rewards:
        print(f"avg_reward={np.mean(rewards):.3f}")
    if losses:
        print(f"avg_loss={np.mean(losses):.6f}")

    if scores:
        plt.figure(figsize=(12, 6))
        plt.plot(scores)
        plt.title("DQN Training Scores")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig("dqn_scores.png")
        print("Score plot saved to dqn_scores.png")

def evaluate(episodes, size, seed, model_path, max_steps, render=False):
    env = SnakeEnv(size=size, seed=seed)
    obs = env.reset()
    agent = DQNAgent(obs_dim=obs.shape[0], act_dim=3, seed=seed)
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    scores = []
    for ep in range(episodes):
        o = env.reset()
        total = 0.0
        for t in range(max_steps):
            with torch.no_grad():
                qo = agent.net(torch.as_tensor(o, dtype=torch.float32, device=agent.device).unsqueeze(0))
                a = int(torch.argmax(qo[0]).item())
            no, r, d, info = env.step(a)
            if render:
                env.render()
            total += r
            o = no
            if d:
                break
        scores.append(info["score"])
        print(f"eval_episode={ep+1} reward={total:.2f} score={info['score']}")
    if scores:
        print(f"avg_score={np.mean(scores):.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--size", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="dqn_snake_torch.pt")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--eval", type=int, default=3)
    p.add_argument("--render", type=int, default=0)
    args = p.parse_args()
    start = time.time()
    train(args.episodes, args.size, args.seed, args.save, args.max_steps)
    evaluate(args.eval, args.size, args.seed, args.save, args.max_steps, render=bool(args.render))
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"done_in={time.time()-start:.2f}s")

if __name__ == "__main__":
    main()
