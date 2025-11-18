import numpy as np

try:
    from .snake_env_cpp import SnakeEnv as CSnakeEnv
except Exception:
    try:
        from snake_env_cpp import SnakeEnv as CSnakeEnv
    except Exception as e:
        raise ImportError("snake_env_cpp not found. Build the C++ extension first.") from e

class SnakeEnv:
    def __init__(self, size=10, seed=None):
        self.size = int(size)
        self._c = CSnakeEnv(size=int(size), seed=seed)

    def reset(self):
        return self._c.reset()

    def step(self, action):
        return self._c.step(int(action))

    @property
    def direction(self):
        d = self._c.get_direction()
        return np.array([d[0], d[1]])

    @direction.setter
    def direction(self, v):
        a = np.asarray(v, dtype=int)
        self._c.set_direction([int(a[0]), int(a[1])])

    def render(self, scale=30, fps=30):
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
        snake = self._c.get_snake()
        for seg in snake:
            x = int(seg[1]) * self._cell
            y = int(seg[0]) * self._cell
            pygame.draw.rect(self._screen, (40, 200, 40), (x, y, self._cell, self._cell))
        head = self._c.get_head()
        hx = int(head[1]) * self._cell
        hy = int(head[0]) * self._cell
        pygame.draw.rect(self._screen, (0, 150, 0), (hx, hy, self._cell, self._cell))
        food = self._c.get_food()
        fx = int(food[1]) * self._cell
        fy = int(food[0]) * self._cell
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
