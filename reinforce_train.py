import argparse
import os
import time
import numpy as np
import torch
from rl_snake import SnakeEnv
from agents import ReinforceAgent
import matplotlib.pyplot as plt

def train(episodes, size, seed, save_path, max_steps, load_path=None, resume=False):
    env = SnakeEnv(size=size, seed=seed)
    obs = env.reset()
    agent = ReinforceAgent(obs_dim=obs.shape[0], act_dim=3, seed=seed)
    if resume:
        ckpt = load_path if load_path else save_path
        if ckpt and os.path.exists(ckpt):
            try:
                agent.load(ckpt)
                print(f"loaded={ckpt}")
            except Exception as e:
                print(f"load_failed={ckpt} err={e}")
    
    rewards = []
    losses = []
    scores = []
    
    window_size = 100
    recent_rewards = []
    recent_scores = []
    
    total_steps = 0
    
    for episode_count in range(1, episodes + 1):
        o = env.reset()
        episode_reward = 0.0
        
        for t in range(max_steps):
            a = agent.select_action(o)
            no, r, d, info = env.step(a)
            
            agent.store_reward(r)
            
            episode_reward += r
            total_steps += 1
            o = no
            
            if d:
                break
        
        loss = agent.update()
        
        rewards.append(episode_reward)
        scores.append(info['score'])
        if loss != 0.0:
            losses.append(loss)
        
        recent_rewards.append(episode_reward)
        recent_scores.append(info['score'])
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)
            recent_scores.pop(0)
        
        if episode_count % 100 == 0 or episode_count == 1:
            avg_reward = np.mean(recent_rewards)
            avg_score = np.mean(recent_scores)
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"episode={episode_count}/{episodes} | reward={episode_reward:.2f} | score={info['score']} | "
                  f"avg_reward={avg_reward:.2f} | avg_score={avg_score:.2f} | "
                  f"avg_loss={avg_loss:.2f} | steps={total_steps}")
        else:
            print(f"episode={episode_count} reward={episode_reward:.2f} score={info['score']}")

    # ensure save directory exists
    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        agent.save(save_path)
        print(f"saved={save_path}")
    if rewards:
        print(f"final_avg_reward={np.mean(rewards[-window_size:]):.3f}")
        print(f"best_score={max(scores)}")
    if losses:
        print(f"final_avg_loss={np.mean(losses[-window_size:]):.3f}")

    if scores:
        plt.figure(figsize=(12, 6))
        plt.plot(scores)
        plt.title("REINFORCE Training Scores")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig("imgs/reinforce_scores.png")
        print("Score plot saved to imgs/reinforce_scores.png")

def evaluate(episodes, size, seed, model_path, max_steps, render=False):
    env = SnakeEnv(size=size, seed=seed)
    obs = env.reset()
    agent = ReinforceAgent(obs_dim=obs.shape[0], act_dim=3, seed=seed)
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    
    scores = []
    for ep in range(episodes):
        o = env.reset()
        total_reward = 0.0
        for t in range(max_steps):
            with torch.no_grad():
                x = torch.as_tensor(o, dtype=torch.float32, device=agent.device).unsqueeze(0)
                logits = agent.policy(x)
                a = int(torch.argmax(logits[0]).item())
            
            no, r, d, info = env.step(a)
            if render:
                env.render()
            
            total_reward += r
            o = no
            if d:
                break
        
        scores.append(info["score"])
        print(f"eval_episode={ep+1} reward={total_reward:.2f} score={info['score']}")
    
    if scores:
        print(f"avg_score={np.mean(scores):.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--size", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="./models/reinforce_snake_torch.pt")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--eval", type=int, default=3)
    p.add_argument("--render", type=int, default=1)
    p.add_argument("--mode", type=str, default='train')
    p.add_argument("--load", type=str, default="./models/reinforce_snake_torch.pt")
    p.add_argument("--resume", type=int, default=0)

    args = p.parse_args()
    
    start = time.time()
    
    if args.mode == 'train':
        train(args.episodes, args.size, args.seed, args.save, args.max_steps,
              load_path=(args.load if args.load else None), resume=bool(args.resume))
    elif args.mode == 'eval':
        evaluate(args.eval, args.size, args.seed, args.save, args.max_steps, render=bool(args.render))
    
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"done_in={time.time()-start:.2f}s")

if __name__ == "__main__":
    main()
