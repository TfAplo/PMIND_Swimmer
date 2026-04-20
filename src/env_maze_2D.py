import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.config import MAZE_2D
import math
import matplotlib
import matplotlib.pyplot as plt

class SimpleMaze2D(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, maze_map=MAZE_2D, action_scale=0.3, goal_threshold=1.5, reward_type="dense", continuing_task=False, render_mode=None):
        #print(f"maze_map rows: {len(maze_map)}, action_scale: {action_scale}, goal_threshold: {goal_threshold}")
        super().__init__()
        self.maze_map = maze_map
        self.rows = len(maze_map)
        self.cols = len(maze_map[0])
        self.action_scale = action_scale
        self.goal_threshold = goal_threshold
        self.reward_type = reward_type # "dense" ou "sparse"
        self.continuing_task = continuing_task # True ou False
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.start_pos = self._find('r')
        self.goal_pos = self._find('g')

    def _find(self, target):
        for r, row in enumerate(self.maze_map):
            for c, cell in enumerate(row):
                if cell == target:
                    return np.array([float(c), float(r)], dtype=np.float32)
        raise ValueError(f"'{target}' non trouvé dans la map")

    def _is_wall(self, x, y):
        c, r = int(round(x)), int(round(y))
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.maze_map[r][c] == 1
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_count = getattr(self, '_reset_count', 0) + 1
        #print(f"reset appelé {self._reset_count} fois", flush=True)
        self.pos = self.start_pos.copy()
        return self._get_obs(), {}
    

    def step(self, action):
        self._total_calls = getattr(self, '_total_calls', 0) + 1
        
        # reward dense
        dx = action[0] * self.action_scale
        dy = action[1] * self.action_scale
        
        # 1. On essaie d'abord de bouger uniquement sur l'axe X
        new_x = self.pos[0] + dx
        if not self._is_wall(new_x, self.pos[1]):
            self.pos[0] = new_x  # C'est libre, on valide X
            
        # 2. On essaie ensuite de bouger uniquement sur l'axe Y
        new_y = self.pos[1] + dy
        if not self._is_wall(self.pos[0], new_y):
            self.pos[1] = new_y  # C'est libre, on valide Y

        dist = np.linalg.norm(self.pos - self.goal_pos)
        on_goal = dist <= self.goal_threshold

        # reward
        if self.reward_type == "dense":
            reward = -dist
            if on_goal:
                reward += 100.0 
                
        elif self.reward_type == "sparse":
            if on_goal:
                reward = 100.0
            else:
                reward = -1.0 
        else:
            raise ValueError(f"Type de récompense inconnu: {self.reward_type}")

        # fin d'episode
        if on_goal and not self.continuing_task:
            terminated = True   
        else:
            terminated = False

        truncated = False  # géré par TimeLimit 
        
        return self._get_obs(), reward, terminated, truncated, {}


    def _get_obs(self):
        return np.array([
            self.pos[0], self.pos[1],
            self.goal_pos[0], self.goal_pos[1]
        ], dtype=np.float32)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        
        
        matplotlib.use("Agg")  # pas d'affichage
        
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # murs et goal
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze_map[r][c] == 1:
                    ax.fill([c-0.5, c+0.5, c+0.5, c-0.5],
                            [r-0.5, r-0.5, r+0.5, r+0.5],
                            color='#D3D3D3', alpha=0.9)
                elif self.maze_map[r][c] == 'g':
                    ax.scatter(c, r, color='green', marker='*', s=400, zorder=5)
    
        # position de l'agent
        ax.scatter(self.pos[0], self.pos[1], color='blue', s=200, zorder=6)
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
    
        # convertir en image numpy
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:]  # ARGB -> RGB
        plt.close(fig)
        return img


gym.register(
    id="SimpleMaze2D-v0",
    entry_point="src.env_maze_2D:SimpleMaze2D",
    kwargs={"maze_map": MAZE_2D, "action_scale": 0.5, "goal_threshold": 0.8},
    max_episode_steps=400 
)