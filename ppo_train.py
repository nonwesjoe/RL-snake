import argparse
import os
import time
import numpy as np
import torch
from env import SnakeEnv
from ppo_agent import PPOAgent
import matplotlib.pyplot as plt

def train(episodes, size, seed, save_path, max_steps, update_freq=2048, 
          epochs=4, batch_size=64, load_path=None, resume=False):
    """
    PPO训练：收集一定步数后更新
    update_freq: 收集多少步后更新一次
    epochs: 每次更新时使用数据的轮数
    batch_size: mini-batch大小
    """
    env = SnakeEnv(size=size, seed=seed)
    obs = env.reset()
    agent = PPOAgent(obs_dim=obs.shape[0], act_dim=3, seed=seed)
    if resume:
        ckpt = load_path if load_path else save_path
        if ckpt and os.path.exists(ckpt):
            agent.load(ckpt, load_optimizers=True)
    
    rewards = []
    losses = []
    scores = []
    episode_lengths = []
    
    # 用于计算移动平均
    window_size = 100
    recent_rewards = []
    recent_scores = []
    
    # 收集缓冲区
    buffer_obs = []
    buffer_act = []
    buffer_rew = []
    buffer_done = []
    buffer_logp = []
    
    total_steps = 0
    episode_count = 0
    
    while episode_count < episodes:
        o = env.reset()
        total = 0.0
        
        for t in range(max_steps):
            a, logp = agent.select_action(o)
            no, r, d, info = env.step(a)
            
            # 添加到缓冲区
            buffer_obs.append(o)
            buffer_act.append(a)
            buffer_rew.append(r)
            buffer_done.append(1.0 if d else 0.0)
            buffer_logp.append(logp)
            
            total += r
            total_steps += 1
            o = no
            
            if d:
                break
        
        rewards.append(total)
        scores.append(info['score'])
        # 计算当前episode的长度（从上次更新后的步数）
        if len(buffer_obs) > 0:
            # 简单估算：当前episode的步数
            episode_lengths.append(t + 1)
        episode_count += 1
        
        # 更新移动平均
        recent_rewards.append(total)
        recent_scores.append(info['score'])
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)
            recent_scores.pop(0)
        
        # 当收集到足够步数时更新
        should_update = (len(buffer_obs) >= update_freq) or (episode_count == episodes)
        
        if should_update and len(buffer_obs) > 0:
            # 计算最后一个状态的next_val（用于GAE）
            # 如果最后一个状态没有终止，需要获取下一个状态的value
            if buffer_done[-1] == 0.0:
                # Episode未终止，使用当前状态的value作为next_val
                final_next_obs = o  # 使用当前环境状态
            else:
                final_next_obs = None  # Episode终止，next_val=0
            
            # 更新
            loss = agent.update(
                buffer_obs, buffer_act, buffer_rew, buffer_done, buffer_logp,
                next_obs=final_next_obs, epochs=epochs, batch_size=batch_size
            )
            losses.append(loss)
            
            # 清空缓冲区
            buffer_obs = []
            buffer_act = []
            buffer_rew = []
            buffer_done = []
            buffer_logp = []
        
        # 每100个episode打印一次详细统计
        if episode_count % 100 == 0 or episode_count == 1:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_score = np.mean(recent_scores) if recent_scores else 0.0
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0.0
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) > 0 else 0.0
            print(f"episode={episode_count}/{episodes} | reward={total:.2f} | score={info['score']} | "
                  f"avg_reward={avg_reward:.2f} | avg_score={avg_score:.2f} | "
                  f"avg_loss={avg_loss:.6f} | avg_length={avg_length:.1f} | steps={total_steps}")
        else:
            print(f"episode={episode_count} reward={total:.2f} score={info['score']}")
    
    if save_path:
        agent.save(save_path)
        print(f"saved={save_path}")
    if rewards:
        print(f"final_avg_reward={np.mean(rewards[-window_size:]):.3f}")
        print(f"best_score={max(scores)}")
    if losses:
        print(f"final_avg_loss={np.mean(losses[-window_size:]):.6f}")

    if scores:
        plt.figure(figsize=(12, 6))
        plt.plot(scores)
        plt.title("PPO Training Scores")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig("ppo_scores.png")
        print("Score plot saved to ppo_scores.png")

def evaluate(episodes, size, seed, model_path, max_steps, render=False):
    env = SnakeEnv(size=size, seed=seed)
    obs = env.reset()
    agent = PPOAgent(obs_dim=obs.shape[0], act_dim=3, seed=seed)
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    scores = []
    for ep in range(episodes):
        o = env.reset()
        total = 0.0
        for t in range(max_steps):
            with torch.no_grad():
                logits = agent.policy(torch.as_tensor(o, dtype=torch.float32, device=agent.device).unsqueeze(0))
                a = int(torch.argmax(logits[0]).item())
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
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--size", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="ppo_snake_torch.pt")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--update_freq", type=int, default=2048)  # 收集多少步后更新
    p.add_argument("--epochs", type=int, default=4)  # 每次更新的轮数
    p.add_argument("--batch_size", type=int, default=128)  # mini-batch大小
    p.add_argument("--load", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--eval", type=int, default=3)
    p.add_argument("--render", type=int, default=0)
    args = p.parse_args()
    start = time.time()
    train(args.episodes, args.size, args.seed, args.save, args.max_steps, 
          args.update_freq, args.epochs, args.batch_size, 
          load_path=(args.load if args.load else None), resume=bool(args.resume))
    evaluate(args.eval, args.size, args.seed, args.save, args.max_steps, render=bool(args.render))
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"done_in={time.time()-start:.2f}s")

if __name__ == "__main__":
    main()

