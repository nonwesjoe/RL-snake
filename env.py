import numpy as np

class SnakeEnv:
    def __init__(self, size=10, seed=None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.direction = np.array([0, 1])
        mid = self.size // 2
        self.snake = [np.array([mid, mid - 1]), np.array([mid, mid])]
        self.head = self.snake[-1].copy()
        self._place_food()
        self.steps = 0
        self.score = 0
        self.no_food_steps = 0
        self.last_dist = self._manhattan(self.head, self.food)
        return self._get_obs()

    def step(self, action):
        self._apply_action(action)
        new_head = self.head + self.direction
        if self._collision(new_head):
            return self._get_obs(), -1.0, True, {"score": self.score}
        ate = np.array_equal(new_head, self.food)
        self.snake.append(new_head)
        self.head = new_head
        if ate:
            self.score += 1
            self._place_food()
            reward = 1.25
            if self.score>=30:
                reward=reward+1.25
            elif self.score>=20:
                reward=reward+1.0
            self.no_food_steps = 0
        else:
            self.snake.pop(0)
            reward = -0.025
            self.no_food_steps += 1
        turn_pen = -0.05 if action != 1 else 0.0
        cur_dist = self._manhattan(self.head, self.food)
        shaping = 0.065 * (self.last_dist - cur_dist)
        # print('last',self.last_dist)
        # print('cur',cur_dist)
        # print('sha',shaping)
        # print('====')
        self.last_dist = cur_dist
        reward = reward + shaping + turn_pen 
        starve_limit =self.size * self.size // 6
        if not ate and self.no_food_steps >= starve_limit:
            return self._get_obs(), reward - 0.03, True, {"score": self.score}
        
        self.steps += 1
        return self._get_obs(), reward, False, {"score": self.score}

    def _apply_action(self, action):
        if action == 0:
            self.direction = self._turn_left(self.direction)
        elif action == 2:
            self.direction = self._turn_right(self.direction)

    def _turn_left(self, d):
        if np.array_equal(d, np.array([0, 1])):
            return np.array([-1, 0])
        if np.array_equal(d, np.array([1, 0])):
            return np.array([0, 1])
        if np.array_equal(d, np.array([0, -1])):
            return np.array([1, 0])
        return np.array([0, -1])

    def _turn_right(self, d):
        if np.array_equal(d, np.array([0, 1])):
            return np.array([1, 0])
        if np.array_equal(d, np.array([1, 0])):
            return np.array([0, -1])
        if np.array_equal(d, np.array([0, -1])):
            return np.array([-1, 0])
        return np.array([0, 1])

    def _place_food(self):
        while True:
            pos = self.rng.integers(0, self.size, size=2)
            occupied = any(np.array_equal(seg, pos) for seg in self.snake)
            if not occupied:
                self.food = pos
                break

    def _collision(self, pos):
        if pos[0] < 0 or pos[0] >= self.size or pos[1] < 0 or pos[1] >= self.size:
            return True
        return any(np.array_equal(seg, pos) for seg in self.snake)

    def _danger_in_dir(self, dvec):
        test = self.head + dvec
        return 1.0 if self._collision(test) else 0.0

    def _free_cells_in_dir(self, dvec):
        cnt = 0
        pos = self.head.copy()
        while True:
            pos = pos + dvec
            if self._collision(pos):
                break
            cnt += 1
        return cnt / float(max(1, self.size - 1))

    def _get_obs(self):
        d_up = 1.0 if np.array_equal(self.direction, np.array([-1, 0])) else 0.0
        d_right = 1.0 if np.array_equal(self.direction, np.array([0, 1])) else 0.0
        d_down = 1.0 if np.array_equal(self.direction, np.array([1, 0])) else 0.0
        d_left = 1.0 if np.array_equal(self.direction, np.array([0, -1])) else 0.0
        ahead = self._free_cells_in_dir(self.direction)
        left = self._free_cells_in_dir(self._turn_left(self.direction))
        right = self._free_cells_in_dir(self._turn_right(self.direction))
        fvec = self.food - self.head
        norm = float(max(1, self.size - 1))
        dx = (fvec @ self.direction) / norm
        dy = (fvec @ self._turn_left(self.direction)) / norm
        length = len(self.snake) / float(self.size * self.size)
        const = 1.0
        occ = self._local_occupancy(radius=2)
        base = np.array([d_up, d_right, d_down, d_left, ahead, left, right, dx, dy, length, const], dtype=np.float32)
        return np.concatenate([base, occ]).astype(np.float32)

    def _local_occupancy(self, radius=2):
        f = self.direction
        l = self._turn_left(self.direction)
        size = 2 * radius + 1
        grid = np.zeros((size, size), dtype=np.float32)
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i == 0 and j == 0:
                    continue
                delta = i * f + j * l
                pos = self.head + delta
                grid[i + radius, j + radius] = 1.0 if self._collision(pos) else 0.0
        return grid.flatten()

    def render(self, scale=30, fps=10):
        try:
            import pygame
        except Exception:
            return
        if not hasattr(self, "_pg_init"):
            pygame.init()
            self._pg_init = True
            self._cell = int(scale)
            self._w = self.size * self._cell
            self._h = self.size * self._cell
            self._screen = pygame.display.set_mode((self._w, self._h))
            self._clock = pygame.time.Clock()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.close()
                return
        self._screen.fill((30, 30, 30))
        for seg in self.snake:
            x = int(seg[1]) * self._cell
            y = int(seg[0]) * self._cell
            pygame.draw.rect(self._screen, (40, 200, 40), (x, y, self._cell, self._cell))
        hx = int(self.head[1]) * self._cell
        hy = int(self.head[0]) * self._cell
        pygame.draw.rect(self._screen, (0, 150, 0), (hx, hy, self._cell, self._cell))
        fx = int(self.food[1]) * self._cell
        fy = int(self.food[0]) * self._cell
        pygame.draw.rect(self._screen, (220, 50, 50), (fx, fy, self._cell, self._cell))
        pygame.display.flip()
        self._clock.tick(int(fps))

    def close(self):
        try:
            import pygame
            if hasattr(self, "_pg_init") and self._pg_init:
                pygame.display.quit()
                pygame.quit()
                self._pg_init = False
        except Exception:
            pass

    def _manhattan(self, a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    manual_play(size=args.size, scale=args.scale, fps=args.fps, seed=args.seed)