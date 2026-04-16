import copy
import os
from bbrl.agents import Agents
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import setup_optimizer, soft_update_params
import torch
from src.agent import ContinuousQAgent, ContinuousDeterministicActor, AddGaussianNoise
from src.loss import compute_critic_loss, compute_actor_loss
from bbrl.visu.plot_policies import plot_policy

from src.integrated_gradient import integrated_gradients, log_integrated_gradients_attribution
from src.visualisation import plot_heatmap_and_real_trajectory, plot_heatmap_and_vector_field
from src.config import MY_MAZE

class TD3(EpochBasedAlgo):
    def __init__(self, cfg, env_wrappers):
        super().__init__(cfg, env_wrappers)
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
        
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_1, self.critic_2)


def run_td3_clean(td3: TD3):
    for rb in td3.iter_replay_buffers():
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

def run_td3(td3: TD3, save_checkpoint=False, maze_map=MY_MAZE, plots=False):
    last_logged_step = -1
    for rb in td3.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(td3.cfg.algorithm.batch_size)
        replay_actions = rb_workspace["action"][0].clone()  # actions stockées dans le buffer

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
        if td3.nb_steps % 5000 == 0:
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
        

        if (td3.nb_steps % 5000 == 0) and td3.nb_steps != last_logged_step:
            last_logged_step = td3.nb_steps

            if save_checkpoint:
                checkpoint_dir = td3.base_dir / "checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_data = {
                    "actor": td3.actor.state_dict(),
                    "critic_1": td3.critic_1.state_dict(),
                    "step": td3.nb_steps,
                    "M": td3.cfg.M                   
                }

                torch.save(checkpoint_data, checkpoint_dir / f"checkpoint_step{td3.nb_steps:07d}.pt")
            if plots:
                heatmap_vf_dir = os.path.join("outputs", td3.cfg.base_dir, "heatmap_vector_fields")
                heatmap_traj_dir = os.path.join("outputs", td3.cfg.base_dir, "heatmap_real_trajectory")

                # 1. Le Vector Field Statique
                plot_heatmap_and_vector_field(
                    actor=td3.eval_policy, 
                    critic=td3.critic_1, 
                    env_name=td3.cfg.gym_env.env_name, 
                    M=td3.cfg.M, 
                    step=td3.nb_steps, 
                    save_dir=heatmap_vf_dir,
                    seed=td3.cfg.algorithm.seed,
                    maze_map=maze_map,
                    # num_vectors_to_show=1
                )

                # 2. La Trajectoire Réelle Dynamique
                plot_heatmap_and_real_trajectory(
                    actor=td3.eval_policy, 
                    critic=td3.critic_1, 
                    env_name=td3.cfg.gym_env.env_name, 
                    M=td3.cfg.M, 
                    step=td3.nb_steps, 
                    save_dir=heatmap_traj_dir,
                    seed=td3.cfg.algorithm.seed,
                    maze_map=maze_map
                )

                log_integrated_gradients_attribution(
                    critic=td3.critic_1,
                    logger=td3.logger,
                    nb_steps=td3.nb_steps,
                    obs=rb_workspace["env/env_obs"][0],
                    action_seq=replay_actions,
                    integrated_gradients_fn=integrated_gradients,
                    M=td3.cfg.M,
                    critic_optimizer=td3.critic_optimizer,
                    actor_optimizer=td3.actor_optimizer,
                )