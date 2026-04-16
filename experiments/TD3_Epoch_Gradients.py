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
from torch.distributions import Normal

import bbrl_utils

bbrl_utils.setup()

class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        # Get the current state $s_t$ and the chosen action $a_t$
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))

        # Compute the Q-value(s_t, a_t)
        obs_act = torch.cat((obs, action), dim=1)
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
        self.set(("action", t), action)

class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
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
        self.n_updates = 0

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

def run_td3(td3: TD3):
    for rb in td3.iter_replay_buffers():
        for _ in range(td3.cfg.algorithm.gradient_steps):
            td3.n_updates += 1
            rb_workspace = rb.get_shuffled(td3.cfg.algorithm.batch_size)

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
            # td3.logger.add_log("critic_1_loss", loss_q1, td3.nb_steps)
            # td3.logger.add_log("critic_2_loss", loss_q2, td3.nb_steps)

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
            if td3.n_updates % td3.cfg.algorithm.policy_delay == 0:
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

            if td3.evaluate():
                if td3.cfg.plot_agents:
                    plot_policy(
                        td3.actor,
                        td3.eval_env,
                        td3.best_reward,
                        str(td3.base_dir / "plots"),
                        td3.cfg.gym_env.env_name,
                        stochastic=False,
                    )

params = {
    "save_best": False,
    "base_dir": "${gym_env.env_name}/td3-S${algorithm.seed}_${current_time:}",
    "collect_stats": False,
    # Set to true to have an insight on the learned policy
    # (but slows down the evaluation a lot!)
    "plot_agents": False,
    "algorithm": {
        "gradient_steps" : 1000,
        "policy_delay" : 2,
        'target_policy_noise': 0.2, #0.2
        "target_policy_noise_clip": 0.5, #0.5
        "seed": 6,
        "max_grad_norm": 0.5,
        "n_envs": 1,#1
        "n_steps": 1000,#1000
        "nb_evals": 10,#10
        "discount_factor": 0.99999,#0.99999
        "buffer_size": 1e6, #1e6
        "batch_size": 256,
        "tau_target": 0.005,#0.005
        "eval_interval": 5000,#5000
        "max_epochs": 500,#1500
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