import argparse
import numpy as np
from rl_snake import SnakeEnv

def manual_play(size=10, scale=30, fps=10, seed=None):
    import pygame
    env = SnakeEnv(size=size, seed=seed)
    env.reset()
    pygame.init()
    env.render(scale=scale, fps=fps)
    running = True
    key_dir = {
        pygame.K_UP: np.array([-1, 0]),
        pygame.K_RIGHT: np.array([0, 1]),
        pygame.K_DOWN: np.array([1, 0]),
        pygame.K_LEFT: np.array([0, -1]),
    }
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key in key_dir:
                    env.direction = key_dir[e.key]
        _, _, done, info = env.step(1)
        env.render(scale=scale, fps=fps)
        if done:
            print(f"score={info['score']}")
            env.reset()
    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    manual_play(size=args.size, scale=args.scale, fps=args.fps, seed=args.seed)