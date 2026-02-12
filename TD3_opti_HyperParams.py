# pour lancer le traitement en fond sur les machines de la fac
# nohup python optimizerTD3.py > optuna.log 2>&1 &


# Prepare the environment

import os
import copy
import numpy as np
import gymnasium as gym
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
import optuna
import bbrl_utils
import joblib
from bbrl.workspace import Workspace
import matplotlib.pyplot as plt
from datetime import datetime
import optuna.visualization as vis
from tqdm import tqdm
import sys
from pathlib import Path



bbrl_utils.setup()

from torch.distributions import Normal



class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        # Get the current state $s_t$ and the chosen action $a_t$
        obs = self.get(("env/env_obs", t))  # shape B x D_{obs}
        action = self.get((f"action", t))  # shape B x D_{action}

        # Compute the Q-value(s_t, a_t)
        obs_act = torch.cat((obs, action), dim=1)  # shape B x (D_{obs} + D_{action})
        # Get the q-value (and remove the last dimension since it is a scalar)
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)

class ContinuousDeterministicActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Tanh() 
        )

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        action = self.model(obs)
        self.set((f"action", t), action)

class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        self.set(("action", t), action)

mse = nn.MSELoss()

def compute_critic_loss(cfg,reward,must_bootstrap,q_values, target_q_values):
    # Compute temporal difference
    q_pred = q_values[0]
    q_t1 = target_q_values[1].detach()

    target = reward[1] + cfg.algorithm.discount_factor * q_t1 * must_bootstrap[1]
    critic_loss = mse(q_pred, target)
    return critic_loss


def compute_actor_loss(q_values):
    return -q_values[0].mean()




class TD3(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic_1/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic_1/")
        
        self.critic_2 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic_2/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix("target-critic_2/")

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
        self.target_actor = copy.deepcopy(self.actor)

        noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)

        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor

        for p in self.target_critic_1.parameters(): p.requires_grad = False
        for p in self.target_critic_2.parameters(): p.requires_grad = False
        for p in self.target_actor.parameters():    p.requires_grad = False
        
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_1, self.critic_2)

def train_td3_step(td3: TD3):
        rb = td3.replay_buffer.get_shuffled(td3.cfg.algorithm.batch_size)
        rb_workspace = rb

        # compute q <- critic(s,a)
        td3.critic_1(rb_workspace, t=0)
        td3.critic_2(rb_workspace, t=0)
        q1 = rb_workspace["critic_1/q_value"]
        q2 = rb_workspace["critic_2/q_value"]
        reward, terminated = rb_workspace["env/reward", "env/terminated"]

        # a' = target_actor(s') + noise
        td3.target_actor(rb_workspace, t=1)
        target_action = rb_workspace["action"][1] 
        noise = torch.randn_like(target_action) * td3.cfg.algorithm.target_policy_noise
        noise = noise.clamp(
            -td3.cfg.algorithm.target_policy_noise_clip, 
            td3.cfg.algorithm.target_policy_noise_clip
        )
        low = torch.tensor(td3.train_env.action_space.low, device=target_action.device)
        high = torch.tensor(td3.train_env.action_space.high, device=target_action.device)
        smoothed_target_action = torch.max(torch.min(target_action + noise, high), low)

        rb_workspace.set("action", 1,smoothed_target_action)

        # compute t_q <- target_critic(s',a')
        with torch.no_grad():
            td3.target_critic_1(rb_workspace, t=1)
            td3.target_critic_2(rb_workspace, t=1)
            target_q1 = rb_workspace["target-critic_1/q_value"]
            target_q2 = rb_workspace["target-critic_2/q_value"]

        # y = r + gamma * (1 - done) * min(q1_t, q2_t)
        min_q = torch.min(target_q1, target_q2)
        must_bootstrap = ~terminated

        loss_q1 = compute_critic_loss(td3.cfg, reward, must_bootstrap, q1, min_q)
        loss_q2 = compute_critic_loss(td3.cfg, reward, must_bootstrap, q2, min_q)
        td3.logger.add_log("critic_1_loss", loss_q1, td3.nb_steps)
        td3.logger.add_log("critic_2_loss", loss_q2, td3.nb_steps)

        critic_loss = loss_q1 + loss_q2

        #Weights update on critics
        td3.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            td3.critic_1.parameters(), td3.cfg.algorithm.max_grad_norm
        )

        torch.nn.utils.clip_grad_norm_(
            td3.critic_2.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_optimizer.step()

        # Si step % policy_delay == 0
        if td3.nb_steps % td3.cfg.algorithm.policy_delay == 0:
            # loss_actor = -critic_1(s, actor(s)).mean()
            td3.actor(rb_workspace, t=0)
            td3.critic_1(rb_workspace, t=0)
            q_values = rb_workspace["critic_1/q_value"]
            loss_actor = compute_actor_loss(q_values)

            #update of Actor's weights by backprop on critic
            td3.actor_optimizer.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(
                td3.actor.parameters(), td3.cfg.algorithm.max_grad_norm
            )
            td3.actor_optimizer.step()

            # soft_updates
            soft_update_params(td3.actor, td3.target_actor, td3.cfg.algorithm.tau_target)
            soft_update_params(td3.critic_1, td3.target_critic_1, td3.cfg.algorithm.tau_target)
            soft_update_params(td3.critic_2, td3.target_critic_2, td3.cfg.algorithm.tau_target)


params = {
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
        "n_envs": 1,
        "n_steps": 1000,
        "nb_evals": 10,#10
        "discount_factor": 0.99999,#0.99999
        "buffer_size": 1e6, #1e6
        "batch_size": 256,
        "tau_target": 0.005,#0.005
        "eval_interval": 5000,#5000
        "max_epochs": 1500,
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
        "lr": 3e-4,#1e-3 3e-4
        "eps": 5e-5,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
        "eps": 5e-5,
    },
}

def evaluate_agent(td3, n_eval=10):
    """fait n épisodes d'évaluation pour avoir une mesure plus robuste"""
    rewards = []

    for i in range(n_eval):
        ws = Workspace()
        td3.eval_agent(ws, t=0, stop_variable="env/done")
        r = ws["env/cumulated_reward"][-1]

        if isinstance(r, torch.Tensor):
            r = r.detach().cpu().mean().item()

        elif isinstance(r, np.ndarray):
            r = float(r.mean())

        rewards.append(float(r))

    return np.mean(rewards)






def runTD3(td3,trial, nb_steps=300000):
    workspace = Workspace()
    td3.train_agent(workspace, t=0, n_steps=1, stochastic=True)
    pbar = tqdm(range(nb_steps), desc=f"Trial Seed {td3.cfg.algorithm.seed}", ascii=True) # barre de progression
    
    for step in pbar:
        td3.train_agent(workspace, t=1, n_steps=1, stochastic=True)
        transition = workspace.get_transitions()
        td3.replay_buffer.put(transition)
    
        td3.nb_steps += transition.batch_size()
        workspace.copy_n_last_steps(1)
    
        if td3.replay_buffer.size() > td3.cfg.algorithm.learning_starts:
            train_td3_step(td3)
    
        # evaluation + logging
        if td3.nb_steps % td3.cfg.algorithm.eval_interval == 0:
            td3.evaluate()
            mean_reward = evaluate_agent(td3, n_eval=1)
            pbar.set_postfix({"reward": f"{mean_reward:.2f}", "step": td3.nb_steps})
            sys.stdout.flush()
            trial.report(mean_reward, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()



# Fonction objective pour Optuna
def make_objective(cfg_base,seeds=(0, 1, 2)):
    def objective(trial):
        rewards_all = []

        for seed in seeds:
            cfg_temp = OmegaConf.create(cfg_base)
            cfg_temp.algorithm.seed = seed
    
            #  hyperparamètres à optimiser
            cfg_temp.algorithm.action_noise = trial.suggest_float("action_noise", 0.0, 0.5)
            cfg_temp.algorithm.target_policy_noise = trial.suggest_float("target_policy_noise", 0.0, 0.5)
            cfg_temp.algorithm.target_policy_noise_clip = trial.suggest_float("target_policy_noise_clip", 0.0, 0.5)
            cfg_temp.algorithm.discount_factor = trial.suggest_float("gamma", 0.98, 0.99999)
            cfg_temp.actor_optimizer.lr = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
            cfg_temp.critic_optimizer.lr = trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True)
            cfg_temp.algorithm.tau_target = trial.suggest_float("tau", 1e-4, 5e-3, log=True)
            cfg_temp.algorithm.learning_starts = trial.suggest_int("learning_starts", 100, 15000)
            cfg_temp.algorithm.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
            td3 = TD3(cfg_temp)
    
            #print(td3.evaluate.__doc__)
            runTD3(td3,trial)
            
            # Évaluer l’agent
            mean_eval_reward = evaluate_agent(td3, n_eval=10)
            rewards_all.append(mean_eval_reward)

           
        # moyenne sur les seed
        mean_reward = np.mean(rewards_all)
        
        return float(mean_reward)


    return objective



if __name__=="__main__":
    
    study_name = "td3_rl_study_v2"
    
    # Créer le study dans SQLite des le départ
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///td3_rl_study_v2.db",
        load_if_exists=True,  # reprend si le study existe déjà,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20000)
    )
    
    # Optimisation
    objective_func = make_objective(params,seeds=(0,))
    study.optimize(objective_func, n_trials=100)
    
    # Résultats
    best_params = study.best_params
    print("Meilleurs hyperparamètres trouvés :", best_params)
    
    # Heatmap
    df = study.trials_dataframe(attrs=("params", "value"))

    plt.figure(figsize=(8, 4))
    plt.scatter(
        df["params_actor_lr"],
        df["params_critic_lr"],
        c=df["value"],
        cmap="RdYlGn_r"
    )
    plt.colorbar(label="Reward")
    plt.xlabel("actor_lr")
    plt.ylabel("critic_lr")
    plt.title("TD3 Hyperparameter search")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # créer un dossier pour le versionning
    path = Path(f"res/{ts}")
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / "lr_scatter.png")
    plt.close()
    

    vis.plot_param_importances(study).write_html(path / "importance.html")
    vis.plot_parallel_coordinate(study).write_html(path / "parallel.html")
    vis.plot_slice(study).write_html(path / "slice.html")