import copy
import argparse
import datetime
import os
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, Autoreset
import gymnasium_robotics
from omegaconf import OmegaConf
from bbrl_utils.notebook import setup_tensorboard
import torch
import src.env_maze_2D
from src.config import MY_MAZE, params_maze2D, params_PointMaze, checkpoints, MAZE_2D, MAZE_2D_U
from src.wrapper import ActionTimeExtensionWrapper, VelocityControlWrapper
from src.TD3 import TD3, run_td3
gym.register_envs(gymnasium_robotics)


# nohup python main.py --env_type pointmaze --logdir maze2D --nb_seeds 5 --M 5 --plots --save_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Choix de l'environnement
    parser.add_argument("--env_type", required=True, choices=["maze2D", "pointmaze"], help="L'environnement à utiliser")
    
    # Paramètres généraux
    parser.add_argument("--logdir", required=False, default="logs", help="Chemin pour log les résultats")
    parser.add_argument("--nb_seeds", required=False, default=5, type=int, help="Nombre de random seeds")
    parser.add_argument("--seed_start", required=False, default=0, type=int, help="Seed de départ")
    parser.add_argument("--vel_mult", required=False, default=10.0, type=float, help="Multiplicateur de vitesse quand ignore_inertia est actif")
    parser.add_argument("--M", required=False, default=1, type=int, help="Nombre maximum d'époques d'entraînement")
    parser.add_argument("--plots", action="store_true", help="Affiche les heatmaps et trajectoires à la fin de chaque run")
    
    # Options 
    parser.add_argument("--maze_map", required=False, default="wall", choices=["wall", "U"], help="Choix de la map (pour maze2D)")
    parser.add_argument("--save_checkpoint", action="store_true", help="Sauvegarde les checkpoints des modèles")
    parser.add_argument("--visualize_best", action="store_true", help="Enregistre la vidéo de la meilleure run")
    parser.add_argument("--ignore_inertia", action="store_true", help="Ignore l'inertie pour utiliser directement la vitesse au lieu de l'accélération")

    # Methode de multiprocessus
    parser.add_argument("--multiprocess", action="store_true", help="Utilise le multiprocess pour lancer les différentes seeds en parallèle")
    
    args = parser.parse_args()

    if args.env_type == "pointmaze":
        maze_map = MY_MAZE
    elif args.env_type == "maze2D":
        maze_map = MAZE_2D_U if args.maze_map == "U" else MAZE_2D

    if args.multiprocess:
            run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            seed = args.seed_start
            M = args.M

            torch.set_num_threads(2)
            os.environ["OMP_NUM_THREADS"] = "2"
            
            p = copy.deepcopy(params_PointMaze)
            p["M"] = M
            p["base_dir"] = f"{args.logdir}/${{gym_env.env_name}}/${run_time}_td3-S${{algorithm.seed}}_M={M}"
            p["algorithm"]["seed"] = seed
            wrappers = [lambda env: FlattenObservation(env),
                        lambda env, m=M: ActionTimeExtensionWrapper(env, M=m)]
            td3 = TD3(OmegaConf.create(p), env_wrappers=wrappers)
            run_td3(td3, save_checkpoint=args.save_checkpoint,maze_map=maze_map, plots=args.plots)
            td3.visualize_best()

    else:
        for seed in range(args.seed_start, args.seed_start + args.nb_seeds):
            
            for M in range(1, args.M + 1): 
                run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                if args.env_type == "maze2D":
                    current_params = copy.deepcopy(params_maze2D)
                    current_params["M"] = M
                    current_params["algorithm"]["seed"] = seed
                    current_params["gym_env"]["env_args"]["maze_map"] = maze_map
                    current_params["base_dir"] = f"{args.logdir}/${{gym_env.env_name}}/${run_time}_td3-S${{algorithm.seed}}_M={M}"
                    
                    wrappers = [
                        lambda env, m=M: ActionTimeExtensionWrapper(env, M=m),
                        lambda env: Autoreset(env),  
                    ]
                    
                elif args.env_type == "pointmaze":
                    current_params = copy.deepcopy(params_PointMaze)
                    current_params["M"] = M
                    current_params["algorithm"]["seed"] = seed
                    current_params["base_dir"] = f"{args.logdir}/${{gym_env.env_name}}/${run_time}_td3-S${{algorithm.seed}}_M={M}"
                    
                    wrappers = [
                        lambda env: FlattenObservation(env),
                        lambda env, m=M: ActionTimeExtensionWrapper(env, M=m)
                    ]
                    # si on veut ignorer l'intertie, on place ce wrapper entre le flatten l'ActionTimeExtensionWrapper
                    if args.ignore_inertia:
                        wrappers.insert(1, lambda env, vm=args.vel_mult: VelocityControlWrapper(env, velocity_multiplier=vm))

                cfg = OmegaConf.create(current_params)
                
                td3 = TD3(cfg, env_wrappers=wrappers)
                run_td3(td3, save_checkpoint=args.save_checkpoint,maze_map=maze_map, env_type=args.env_type, plots=args.plots)
                
                if args.visualize_best:
                    td3.visualize_best()