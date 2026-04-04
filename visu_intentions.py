import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

def plot_policy_vector_field(actor, env_name, M, step, save_dir, seed, maze_map, num_vectors_to_show=None):
    """
    Sonde la politique d'un acteur sur une grille 2D et trace les actions prévues.
    Ne sonde que les espaces libres pour plus de clarté.
    
    Args:
        actor: Le modèle PyTorch de l'acteur (doit avoir une méthode .model(obs)).
        env_name (str): Le nom de l'environnement Gym (ex: "MyMaze-v0").
        M (int): La dimension temporelle de l'action (Action Time Extension).
        step (int): Le pas d'entraînement actuel (pour le nommage du fichier).
        save_dir (str): Le dossier de destination des images.
        seed (int): La graine aléatoire utilisée (pour le nommage).
        maze_map (list): La matrice du labyrinthe (1 = mur, 0 = vide, 'g' = but).
        num_vectors_to_show (int, optional): Nombre d'actions à afficher par séquence. Par défaut, affiche tout.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Récupération d'une observation de base pour l'environnement
    temp_env = gym.make(env_name)
    temp_env = FlattenObservation(temp_env)
    obs_base, _ = temp_env.reset()
    temp_env.close()
    
    # Limites pour englober la carte proprement
    x_min, x_max = -0.5, len(maze_map[0]) - 0.5
    y_min, y_max = -0.5, len(maze_map) - 0.5

    # Détermination du nombre de vecteurs à afficher
    if num_vectors_to_show is None:
        display_M = M
    else:
        display_M = min(num_vectors_to_show, M)

    # Densité de la grille adaptative
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
    
    plt.figure(figsize=(12, 8))
    
    # 2. Dessin des murs et de la cible
    for r in range(len(maze_map)):
        for c in range(len(maze_map[0])):
            if maze_map[r][c] == 1:
                plt.fill([c-0.5, c+0.5, c+0.5, c-0.5], [r-0.5, r-0.5, r+0.5, r+0.5], color='#D3D3D3', alpha=0.9)
            elif maze_map[r][c] == 'g':
                plt.scatter(c, r, color='green', marker='*', s=400, zorder=5, label="Goal")
                
    colors = cm.plasma(np.linspace(0, 0.8, M))

    # 3. Sondage du réseau de neurones
    for x in xs:
        for y in ys:
            r_idx, c_idx = int(round(y)), int(round(x))
            
            # Annulation si on est dans un mur ou hors carte
            if 0 <= r_idx < len(maze_map) and 0 <= c_idx < len(maze_map[0]):
                if maze_map[r_idx][c_idx] == 1:
                    continue 

            center_x = (len(maze_map[0]) - 1) / 2.0  
            center_y = (len(maze_map) - 1) / 2.0     
            
            # Conversion en repère physique (MuJoCo)
            phys_x = x - center_x
            phys_y = -(y - center_y)
            
            obs = obs_base.copy()
            # Position "achieved_goal"
            obs[0], obs[1] = phys_x, phys_y
            # Position physique réelle
            obs[4], obs[5] = phys_x, phys_y
            # Vitesse nulle
            obs[6], obs[7] = 0.0, 0.0
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_tensor = actor.model(obs_tensor)
            
            actions = action_tensor.squeeze(0).numpy().reshape(M, -1)
            
            curr_x, curr_y = x, y
            scale = 0.1
            
            # 4. Tracé des vecteurs
            for m in range(display_M):
                dx = actions[m, 0] * scale
                dy = -actions[m, 1] * scale
                
                plt.arrow(curr_x, curr_y, dx, dy, 
                          head_width=0.03, head_length=0.04, 
                          fc=colors[m], ec=colors[m], alpha=0.9, linewidth=0.5)
                
                curr_x += dx
                curr_y += dy

    # 5. Finitions esthétiques (Titre, Légende, Sauvegarde)
    title_str = f"Intention Vector Field | M={M} | Seed={seed} | Step={step}"
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

    save_path = os.path.join(save_dir, f"vector_field_M{M}_seed{seed}_step{step:06d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()