"""Loader for checkpoints produced by ManiSkill's stock PPO baseline (ppo.py).

Replicates the baseline's Agent architecture (3x256 tanh MLP actor-critic)
so saved state dicts load without importing the training script. Dimensions
are inferred from the checkpoint itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PPOAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(obs_dim, 256), nn.Tanh(),
                nn.Linear(256, 256), nn.Tanh(),
                nn.Linear(256, 256), nn.Tanh(),
                nn.Linear(256, out_dim),
            )
        self.critic = mlp(1)
        self.actor_mean = mlp(act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cuda") -> "PPOAgent":
        sd = torch.load(path, map_location="cpu", weights_only=True)
        obs_dim = sd["actor_mean.0.weight"].shape[1]
        act_dim = sd["actor_mean.6.weight"].shape[0]
        agent = cls(obs_dim, act_dim)
        agent.load_state_dict(sd)
        agent.to(device).eval()
        return agent

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean = self.actor_mean(obs)
        if deterministic:
            return mean
        return mean + torch.exp(self.actor_logstd) * torch.randn_like(mean)
