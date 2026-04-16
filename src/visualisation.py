import os

from src.wrapper import ActionTimeExtensionWrapper
from src.utils import make_obs_from_grid, obs_to_grid
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["TQDM_DISABLE"] = "1"
import sys
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gymnasium as gym
from gymnasium.envs.registration import register
import math
import bbrl_gymnasium  # noqa: F401
import torch
import torch.nn as nn
from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import build_mlp, setup_optimizer, soft_update_params
from bbrl_utils.notebook import setup_tensorboard
from bbrl.visu.plot_policies import plot_policy
from omegaconf import OmegaConf
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation
from torch.distributions import Normal
import datetime

import bbrl_utils

bbrl_utils.setup()


def plot_heatmap_and_vector_field(actor, critic, env_name, M, step, save_dir, seed, maze_map, num_vectors_to_show=None, env_type="pointmaze", goal_pos=None):
    """
    Calcule la Heatmap des Q-Valeurs V(s) et superpose le Vector Field des actions.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Limites pour englober la carte proprement
    x_min, x_max = -0.5, len(maze_map[0]) - 0.5
    y_min, y_max = -0.5, len(maze_map) - 0.5

    # On calcule le centre du labyrinthe
    center_x = (len(maze_map[0]) - 1) / 2.0
    center_y = (len(maze_map) - 1) / 2.0

    if num_vectors_to_show is None:
        display_M = M
    else:
        display_M = min(num_vectors_to_show, M)

    plt.figure(figsize=(12, 8))

    # 1. CALCUL DE LA HEATMAP DES Q-VALEURS
    resolution = 0.2
    xs_hm = np.arange(0, len(maze_map[0]), resolution)
    ys_hm = np.arange(0, len(maze_map), resolution)
    X, Y = np.meshgrid(xs_hm, ys_hm)
    V_map = np.zeros_like(X)

    actor.eval()
    critic.eval()

    with torch.no_grad():
        for i in range(len(ys_hm)):
            for j in range(len(xs_hm)):
                x, y = X[i, j], Y[i, j]
                r_idx, c_idx = int(round(y)), int(round(x))
                
                # Si c'est un mur, on met NaN pour que ce soit transparent
                if 0 <= r_idx < len(maze_map) and 0 <= c_idx < len(maze_map[0]) and maze_map[r_idx][c_idx] == 1:
                    V_map[i, j] = np.nan
                    continue

                obs_tensor = make_obs_from_grid(env_type, x, y, goal_pos, maze_map).unsqueeze(0)
                action_tensor = actor.model(obs_tensor)


                # Le Critique évalue l'état + l'action choisie
                obs_act = torch.cat((obs_tensor, action_tensor), dim=1)
                q_value = critic.model(obs_act).squeeze(-1)
                V_map[i, j] = q_value.item()

    # Dessin de la Heatmap
    cmap = cm.viridis
    cmap.set_bad(color='white', alpha=0) # Les NaN (murs) deviennent transparents
    im = plt.pcolormesh(X, Y, V_map, cmap=cmap, shading='nearest', alpha=0.6)
    plt.colorbar(im, label="Valeur estimée $V(s)$")



    # 2. DESSIN DES MURS ET DU VECTOR FIELD
    tailles = {
        1: 0.2,
        2: 0.35,
        3: 0.4,
        4: 0.43,
        5: 0.45
    }

    grid_step = tailles.get(display_M, 1.0)
    xs = np.arange(0, len(maze_map[0]), grid_step)
    ys = np.arange(0, len(maze_map), grid_step)
    
    # Dessin des murs
    for r in range(len(maze_map)):
        for c in range(len(maze_map[0])):
            if maze_map[r][c] == 1:
                plt.fill([c-0.5, c+0.5, c+0.5, c-0.5], [r-0.5, r-0.5, r+0.5, r+0.5], color='#D3D3D3', alpha=0.9)
            elif maze_map[r][c] == 'g':
                plt.scatter(c, r, color='green', marker='*', s=400, zorder=5, label="Goal")
                
    colors = cm.plasma(np.linspace(0, 0.8, M))

    for x in xs:
        for y in ys:
            # On vérifie sur quelle case (r, c) on se trouve
            r_idx, c_idx = int(round(y)), int(round(x))
            
            # Si on est dans la carte et que c'est un MUR, on annule et on passe au point suivant !
            if 0 <= r_idx < len(maze_map) and 0 <= c_idx < len(maze_map[0]):
                if maze_map[r_idx][c_idx] == 1:
                    continue 
            else:
                continue
            
            obs_tensor = make_obs_from_grid(env_type, x, y, goal_pos, maze_map).unsqueeze(0)
            with torch.no_grad():
                action_tensor = actor.model(obs_tensor)
            
            actions = action_tensor.squeeze(0).numpy().reshape(M, -1)
            
            curr_x, curr_y = x, y
            scale = 0.1 #1
            
            for m in range(display_M):
                dx = actions[m, 0] * scale
                dy = -actions[m, 1] * scale
                
                # Des flèches plus fines
                plt.arrow(curr_x, curr_y, dx, dy, 
                          head_width=0.03, head_length=0.04, 
                          fc=colors[m], ec=colors[m], alpha=0.9, linewidth=0.5)
                
                curr_x += dx
                curr_y += dy

    title_str = f"Heatmap V(s) & Vector Field | M={M} | Seed={seed} | Step={step}"
    if display_M < M:
        title_str += f" (Showing first {display_M} action(s))"
    plt.title(title_str, fontsize=15, pad=15)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().invert_yaxis()
    
    if display_M > 1:
        plt.plot([], [], color=colors[0], label="Action 1")
        plt.plot([], [], color=colors[display_M - 1], label=f"Action {display_M}")
        plt.legend(loc='lower right', framealpha=1.0)
    elif display_M == 1:
        plt.plot([], [], color=colors[0], label="Action 1")
        plt.legend(loc='lower right', framealpha=1.0)

    save_path = os.path.join(save_dir, f"vector_field_step{step:06d}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_heatmap_and_real_trajectory(actor, critic, env_name, M, step, save_dir, seed, maze_map, env_type="pointmaze", goal_pos=None):
    """
    Calcule la Heatmap des Q-Valeurs V(s) et trace la trajectoire réelle de l'agent par-dessus en le faisant jouer un épisode complet.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. PRÉPARATION ET CALCUL DE LA HEATMAP
    
    x_min, x_max = -0.5, len(maze_map[0]) - 0.5
    y_min, y_max = -0.5, len(maze_map) - 0.5
    center_x = (len(maze_map[0]) - 1) / 2.0
    center_y = (len(maze_map) - 1) / 2.0

    plt.figure(figsize=(12, 8))

    resolution = 0.2
    xs_hm = np.arange(0, len(maze_map[0]), resolution)
    ys_hm = np.arange(0, len(maze_map), resolution)
    X, Y = np.meshgrid(xs_hm, ys_hm)
    V_map = np.zeros_like(X)

    actor.eval()
    critic.eval()

    with torch.no_grad():
        for i in range(len(ys_hm)):
            for j in range(len(xs_hm)):
                x, y = X[i, j], Y[i, j]
                r_idx, c_idx = int(round(y)), int(round(x))
                
                # Murs = NaN pour la transparence
                if 0 <= r_idx < len(maze_map) and 0 <= c_idx < len(maze_map[0]) and maze_map[r_idx][c_idx] == 1:
                    V_map[i, j] = np.nan
                    continue

                obs_tensor = make_obs_from_grid(env_type, x, y, goal_pos, maze_map).unsqueeze(0)
                action_tensor = actor.model(obs_tensor)

                obs_act = torch.cat((obs_tensor, action_tensor), dim=1)
                q_value = critic.model(obs_act).squeeze(-1)
                V_map[i, j] = q_value.item()

    # Dessin de la Heatmap
    cmap = cm.viridis
    cmap.set_bad(color='white', alpha=0)
    im = plt.pcolormesh(X, Y, V_map, cmap=cmap, shading='nearest', alpha=0.6)
    plt.colorbar(im, label="Valeur estimée $V(s)$")


    # 2. COLLECTE DE LA TRAJECTOIRE RÉELLE
    # On recrée l'environnement avec le wrapper temporel pour lancer la run    
    sim_env = gym.make(env_name)
    sim_env = FlattenObservation(sim_env)
    sim_env = ActionTimeExtensionWrapper(sim_env, M=M)
    
    obs, _ = sim_env.reset(seed=seed)
    
    positions = []
    actions_list = []
    done = False
    step_count = 0
    max_steps = 1000

    while not done and step_count < max_steps:
        img_x, img_y = obs_to_grid(env_type, obs, maze_map)
        positions.append((img_x, img_y))
        
        # Inférence
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action_tensor = actor.model(obs_tensor)
        
        raw_action = action_tensor.squeeze(0).numpy()
        actions_list.append(raw_action.reshape(M, -1))
        
        # Step dans l'environnement physique
        obs, reward, terminated, truncated, _ = sim_env.step(raw_action)
        done = terminated or truncated
        step_count += 1
        
    sim_env.close()


    # 3. DESSIN DE LA TRAJECTOIRE PAR-DESSUS    
    # Murs et objectifs
    for r in range(len(maze_map)):
        for c in range(len(maze_map[0])):
            if maze_map[r][c] == 1:
                plt.fill([c-0.5, c+0.5, c+0.5, c-0.5], [r-0.5, r-0.5, r+0.5, r+0.5], color='#D3D3D3', alpha=0.9)
            elif maze_map[r][c] == 'g':
                plt.scatter(c, r, color='green', marker='*', s=400, zorder=5, label="Goal")
            elif maze_map[r][c] == 'r':
                plt.scatter(c, r, color='red', marker='o', s=200, zorder=5, label="Start")
                
    # Ligne de la trajectoire
    xs_traj = [p[0] for p in positions]
    ys_traj = [p[1] for p in positions]
    plt.plot(xs_traj, ys_traj, color='black', linestyle='--', linewidth=2, alpha=0.5, label="Trajectoire réelle")
    
    # Flèches d'intention (Espacées par le stride)
    colors = cm.plasma(np.linspace(0, 0.8, M))
    scale = 0.3 if env_type=="maze2D" else 0.1 
    #stride = max(1, len(positions) // 30) # Dessine une flèche tous les N steps
    stride = 1 if env_type=="maze2D" else 5

    for i in range(0, len(positions), stride):
        px, py = positions[i]
        acts = actions_list[i]
        
        curr_px, curr_py = px, py
        for m in range(M):
            dx = acts[m, 0] * scale
            dy = acts[m, 1] * scale
            
            plt.arrow(curr_px, curr_py, dx, dy, 
                      head_width=0.03, head_length=0.04, 
                      fc=colors[m], ec=colors[m], alpha=0.9, linewidth=0.8, zorder=6)
            
            curr_px += dx
            curr_py += dy

    plt.title(f"Heatmap V(s) & Real Trajectory | M={M} | Seed={seed} | Step={step} (Ep. Len=1000)", fontsize=15, pad=15)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().invert_yaxis()
    
    if M > 1:
        plt.plot([], [], color=colors[0], label="Action 1")
        plt.plot([], [], color=colors[-1], label=f"Action {M}")
        
    plt.legend(loc='lower right', framealpha=1.0)

    save_path = os.path.join(save_dir, f"real_trajectory_step{step:06d}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()