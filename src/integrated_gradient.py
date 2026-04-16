import torch


def log_integrated_gradients_attribution(
    critic,
    logger,
    nb_steps,
    obs,
    action_seq,
    integrated_gradients_fn,
    M,
    critic_optimizer=None,
    actor_optimizer=None,
    eps=1e-8,
):
    """
    Calcule et log les attributions Integrated Gradients sur une séquence d'actions.

    Args:
        critic: réseau critic (ex: td3.critic_1)
        logger: logger possédant add_log()
        nb_steps: step courant pour le logging
        obs: observations [batch_size, obs_dim]
        action_seq: actions [batch_size, M * act_dim]
        integrated_gradients_fn: fonction integrated_gradients(model, obs, action_seq)
        M: nombre de pas/actions dans la séquence
        critic_optimizer: optionnel, pour zero_grad()
        actor_optimizer: optionnel, pour zero_grad()
        eps: petite constante pour éviter division par zéro
    """
    # Désactive les gradients sur le critic
    for p in critic.parameters():
        p.requires_grad_(False)

    try:
        with torch.enable_grad():
            obs = obs.detach()
            action_seq = action_seq.detach()

            act_dim = action_seq.shape[1] // M

            ig = integrated_gradients_fn(critic.model, obs, action_seq)
            ig = ig.view(-1, M, act_dim)

            # Attribution par a_k
            norms = []
            for k in range(M):
                attribution = ig[:, k].norm(dim=-1).mean()
                norms.append(attribution)

                logger.add_log(
                    f"ig_attribution/a{k}",
                    attribution,
                    nb_steps
                )

            # Ratios vs a0
            if M > 1:
                a0_norm = norms[0] + eps

                for k in range(1, M):
                    logger.add_log(
                        f"ig_attribution/ratio_a{k}_vs_a0",
                        norms[k] / a0_norm,
                        nb_steps
                    )

    finally:
        # Réactive toujours les gradients même en cas d'erreur
        for p in critic.parameters():
            p.requires_grad_(True)

        # Nettoyage gradients optimizers
        if critic_optimizer is not None:
            critic_optimizer.zero_grad()

        if actor_optimizer is not None:
            actor_optimizer.zero_grad()

def integrated_gradients(critic_model, obs, action_seq, n_steps=50):
    baseline = torch.zeros_like(action_seq)
    alphas = torch.linspace(0, 1, n_steps, device=action_seq.device)
    
    grads = []
    for alpha in alphas:
        interpolated = (baseline + alpha * (action_seq - baseline)).requires_grad_(True)
        q = critic_model(torch.cat([obs, interpolated], dim=-1))
        grad = torch.autograd.grad(q.sum(), interpolated)[0]
        grads.append(grad.detach())
    
    avg_grads = torch.stack(grads).mean(dim=0)
    ig = (action_seq - baseline) * avg_grads  # (B, M * act_dim)
    return ig