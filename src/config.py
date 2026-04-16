import gymnasium as gym
from gymnasium.envs.registration import register

MAZE_2D = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1],
    [1, 'r', 0, 1, 0, 'g', 1],
    [1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

MAZE_2D_U = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 'r', 0, 1, 0, 'g', 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

# Exemple de maze custom (0 = couloir, 1 = mur, R = reset/start, G = goal)
MY_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, "r", 0, 0, 1, 0, 0, "g", 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

params_swimmer = {
    "M":1,
    "save_best": False,
    "base_dir": "${gym_env.env_name}/td3-S${algorithm.seed}_${current_time:}",
    "collect_stats": False,
    # Set to true to have an insight on the learned policy
    # (but slows down the evaluation a lot!)
    "plot_agents": False,
    "algorithm": {
        "policy_delay" : 2,
        'target_policy_noise': 0.2, #0.2
        "target_policy_noise_clip": 0.5, #0.5
        "seed": 6,
        "max_grad_norm": 0.5,
        "n_envs": 1,#1
        "n_steps": 1,#1000
        "nb_evals": 10,#10
        "discount_factor": 0.99999,#0.99999
        "buffer_size": 1e6, #1e6
        "batch_size": 256,
        "tau_target": 0.005,#0.005
        "eval_interval": 5000,#5000
        "max_epochs": 300_000,
        # Minimum number of transitions before learning starts
        "learning_starts": 10000,
        "action_noise": 0.1,#0.1
        "architecture": {
            "actor_hidden_size": [400, 300],
            "critic_hidden_size": [400, 300],
        },
    },
    "gym_env": {
        "env_name": "Swimmer-v5",
    },
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,#1e-3
        "eps": 5e-5,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
        "eps": 5e-5,
    },
}

params_PointMaze = {
    "M":1,
    "save_best": False,
    "base_dir": "${gym_env.env_name}/td3-S${algorithm.seed}_${current_time:}",
    "collect_stats": False,
    # Set to true to have an insight on the learned policy
    # (but slows down the evaluation a lot!)
    "plot_agents": False,
    "algorithm": {
        "policy_delay" : 2,
        'target_policy_noise': 0.2, #0.2
        "target_policy_noise_clip": 0.5, #0.5
        "seed": 6,
        "max_grad_norm": 0.5,
        "n_envs": 1,#1
        "n_steps": 1,#1000
        "nb_evals": 10,#10
        "discount_factor": 0.99999,#0.99999
        "buffer_size": 1e6, #1e6
        "batch_size": 256,
        "tau_target": 0.005,#0.005
        "eval_interval": 5000,#5000
        "max_epochs": 300_000,
        # Minimum number of transitions before learning starts
        "learning_starts": 10000,
        "action_noise": 0.1,#0.1
        "architecture": {
            "actor_hidden_size": [400, 300],
            "critic_hidden_size": [400, 300],
        },
    },
    "gym_env": {
        "env_name": "MyMaze-v0",
    },
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,#1e-3
        "eps": 5e-5,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
        "eps": 5e-5,
    },
}

params_maze2D  = {
    "save_best": False,
    "base_dir": "${gym_env.envname}/td3-S${algorithm.seed}${current_time:}",
    "collect_stats": False,
    # Set to true to have an insight on the learned policy
    # (but slows down the evaluation a lot!)
    "plot_agents": False,
    "algorithm": {
        "policy_delay" : 2,
        'target_policy_noise': 0.2, #0.2
        "target_policy_noise_clip": 0.5, #0.5
        "seed": 4,
        "max_grad_norm": 0.5,
        "n_envs": 1,#1
        "n_steps": 1,#1000
        "nb_evals": 10,#10
        "discount_factor": 0.99,#0.99999
        "buffer_size": 1e6, #1e6
        "batch_size": 256,
        "tau_target": 0.005,
        "eval_interval": 5000,#5000
        "max_epochs": 200_000,
        # Minimum number of transitions before learning starts
        "learning_starts": 10000,
        "action_noise": 0.1,#0.1,
        "architecture": {
            "actor_hidden_size": [400, 300],
            "critic_hidden_size": [400, 300],
        },
    },
    "gym_env": {
        "env_name": "SimpleMaze2D-v0",
        "env_args": {
            "maze_map": MAZE_2D,
            "action_scale": 0.5,
        }
    },
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,#1e-3 3e-4
        "eps": 5e-5,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
        "eps": 5e-5,
    },
}

register(
    id="MyMaze-v0",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": MY_MAZE,
        "render_mode": None,
        "reward_type" : "sparse"
    },
    max_episode_steps=1000,
)

checkpoints = [25_000, 50_000, 100_000, 
               150_000, 200_000, 300_000, 390_000]
