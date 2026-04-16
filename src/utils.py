
import torch
import numpy as np
from matplotlib import pyplot
from bbrl.workspace import Workspace
from src.agent import ContinuousDeterministicActor, ContinuousQAgent
from src.wrapper import ActionTimeExtensionWrapper
import os
import re

def make_obs_from_grid(env_type, gx, gy, goal_pos, maze_map):
    """Construit un tensor obs à partir de coordonnées grille (accélération nulle)."""
    if env_type == "pointmaze":
        rows, cols = len(maze_map), len(maze_map[0])
        center_x = (cols - 1) / 2.0
        center_y = (rows - 1) / 2.0
        phys_x =  gx - center_x
        phys_y = -(gy - center_y)
        goal_phys_x =  (goal_pos[0] - 1.0) - center_x
        goal_phys_y = -((goal_pos[1] - 1.0) - center_y)
        # [achieved_goal, desired_goal, pos, vel]
        return torch.tensor([phys_x, phys_y, goal_phys_x, goal_phys_y, phys_x, phys_y, 0.0, 0.0], dtype=torch.float32)

    elif env_type == "maze2D":
        # [pos_x, pos_y, goal_x, goal_y]
        return torch.tensor([gx, gy, goal_pos[0], goal_pos[1]], dtype=torch.float32)

    else:
        raise ValueError(f"env_type inconnu : {env_type}")




def obs_to_grid(env_type, obs, maze_map):
    """Extrait (grid_x, grid_y) depuis un vecteur obs brut"""
    if env_type == "pointmaze":
        rows, cols = len(maze_map), len(maze_map[0])
        center_x = (cols - 1) / 2.0
        center_y = (rows - 1) / 2.0
        return obs[4] + center_x, -obs[5] + center_y

    elif env_type == "maze2D":
        return obs[0], obs[1]

    else:
        raise ValueError(f"env_type inconnu : {env_type}")


def find_pos(env_type, maze_map, target):
    """Trouve la position d'une cellule ('r', 'g') en coordonnées adaptées à l'env."""
    for r, row in enumerate(maze_map):
        for c, cell in enumerate(row):
            if cell == target:
                if env_type == "pointmaze":
                    return [float(c) + 1.0, float(r) + 1.0]
                elif env_type == "maze2D":
                    return [float(c), float(r)]
    raise ValueError(f"'{target}' non trouvé dans la map")