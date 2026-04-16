import torch.nn as nn


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