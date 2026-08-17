"""Flow-matching policy with an optional dynamics-prediction head.

Arms (plan Part I §2-3, built per Part II 'Build: training'):
  A      base: rectified-flow action policy only.
  B      treatment: identical network + a two-layer prediction head off the
         trunk, weighted by lambda. The head is DISCARDED at inference, so A
         and B run the same forward pass at deployment.
  Bshuf  shuffled-target control: same head, same gradients, same constraint
         on the trunk; the target batch is permuted (one line), destroying
         only the physics content.
  C      reference: the trained prediction head is KEPT at inference and its
         forecast (from the previous executed action) is concatenated to the
         trunk features before the action head. Changes objective, network
         and inference computation at once - reported, never used for
         attribution.

Prediction targets (all predict the OBJECT's state, never the robot's):
  onestep    object state at t+1
  multistep  object states t+1..t+H_p (H_p in [5,10]; default 8)
  latent     sg[phi(h_{t+1})]: the trunk's own next features, stop-gradient

Structure: history encoder over the last K observations -> trunk phi(h);
rectified-flow velocity head over action chunks; two-layer prediction head
g(phi, a). Loss: ||v(a_tau, tau, phi) - (a1 - a0)||^2 + lambda * pred_loss.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

ARMS = ("A", "B", "Bshuf", "C")
TARGETS = ("onestep", "multistep", "latent")


@dataclass
class PolicyConfig:
    obs_dim: int = 41
    act_dim: int = 8
    obj_dim: int = 13
    K: int = 4
    chunk: int = 8
    pred_h: int = 8
    trunk_dim: int = 256
    hidden: int = 512
    arm: str = "B"
    target: str = "multistep"
    lam: float = 1.0
    time_freqs: int = 16

    def __post_init__(self):
        assert self.arm in ARMS and self.target in TARGETS
        assert 5 <= self.pred_h <= 10 or self.target != "multistep"
        assert self.pred_h <= self.chunk, "pred head conditions on chunk[:pred_h]"


def _mlp(sizes, act=nn.SiLU, out_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(act())
    return nn.Sequential(*layers)


class FlowPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.trunk = _mlp([c.K * c.obs_dim, c.hidden, c.hidden, c.trunk_dim])

        if c.target == "onestep":
            pred_in, self.pred_out = c.act_dim, c.obj_dim
        elif c.target == "multistep":
            pred_in, self.pred_out = c.pred_h * c.act_dim, c.pred_h * c.obj_dim
        else:  # latent
            pred_in, self.pred_out = c.act_dim, c.trunk_dim
        # exactly two layers off the trunk (plan)
        self.pred_head = nn.Sequential(
            nn.Linear(c.trunk_dim + pred_in, 256), nn.SiLU(),
            nn.Linear(256, self.pred_out),
        )

        cond_dim = c.trunk_dim + (self.pred_out if c.arm == "C" else 0)
        chunk_dim = c.chunk * c.act_dim
        self.vel_head = _mlp(
            [chunk_dim + 2 * c.time_freqs + cond_dim, c.hidden, c.hidden, chunk_dim]
        )
        self.register_buffer(
            "time_scales", 2 ** torch.arange(c.time_freqs).float() * torch.pi
        )

    # -- components -----------------------------------------------------
    def features(self, hist: torch.Tensor) -> torch.Tensor:
        """hist: (B, K, obs_dim) normalized -> (B, trunk_dim)."""
        return self.trunk(hist.flatten(1))

    def _time_embed(self, tau: torch.Tensor) -> torch.Tensor:
        x = tau[:, None] * self.time_scales[None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

    def _pred_act_input(self, chunk: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if c.target == "multistep":
            return chunk[:, : c.pred_h].flatten(1)
        return chunk[:, 0]  # onestep / latent condition on the executed action

    def forecast(self, feat: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.pred_head(torch.cat([feat, act], dim=1))

    def _condition(self, feat: torch.Tensor, prev_act: torch.Tensor) -> torch.Tensor:
        """Velocity-head conditioning. Arm C appends the forecast computed
        from the previous executed action (the only action available before
        the new one is generated)."""
        if self.cfg.arm != "C":
            return feat
        if self.cfg.target == "multistep":
            a = prev_act.repeat(1, self.cfg.pred_h)
        else:
            a = prev_act
        return torch.cat([feat, self.pred_head(torch.cat([feat, a], dim=1))], dim=1)

    def velocity(self, a_tau, tau, cond):
        v = self.vel_head(
            torch.cat([a_tau.flatten(1), self._time_embed(tau), cond], dim=1)
        )
        return v.view_as(a_tau)

    # -- losses ----------------------------------------------------------
    def flow_loss(self, cond: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        a1 = chunk
        a0 = torch.randn_like(a1)
        tau = torch.rand(a1.shape[0], device=a1.device)
        a_tau = (1 - tau[:, None, None]) * a0 + tau[:, None, None] * a1
        v = self.velocity(a_tau, tau, cond)
        return ((v - (a1 - a0)) ** 2).mean()

    def pred_loss(
        self,
        feat: torch.Tensor,
        feat_next: torch.Tensor,
        chunk: torch.Tensor,
        future_obj: torch.Tensor,
        shuffle: bool = False,
    ) -> torch.Tensor:
        c = self.cfg
        if c.target == "latent":
            target = feat_next.detach()  # stop-gradient (plan formula)
        elif c.target == "onestep":
            target = future_obj[:, 0]
        else:
            target = future_obj[:, : c.pred_h].flatten(1)
        if shuffle:
            target = target[torch.randperm(target.shape[0], device=target.device)]
        pred = self.forecast(feat, self._pred_act_input(chunk))
        return ((pred - target) ** 2).mean()

    def loss(self, hist_next, chunk, future_obj, prev_act) -> dict:
        """hist_next: (B, K+1, obs_dim); frames [:K] are h_t, [1:] are h_{t+1}."""
        c = self.cfg
        feat = self.features(hist_next[:, : c.K])
        losses = dict(flow=self.flow_loss(self._condition(feat, prev_act), chunk))
        if c.arm != "A":
            feat_next = (
                self.features(hist_next[:, 1:]) if c.target == "latent" else feat
            )
            losses["pred"] = self.pred_loss(
                feat, feat_next, chunk, future_obj, shuffle=(c.arm == "Bshuf")
            )
            losses["total"] = losses["flow"] + c.lam * losses["pred"]
        else:
            losses["total"] = losses["flow"]
        return losses

    # -- inference -------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self, hist: torch.Tensor, prev_act: torch.Tensor, n_steps: int = 10
    ) -> torch.Tensor:
        """Euler integration of the rectified flow. hist: (B, K, obs_dim)
        normalized. Returns (B, chunk, act_dim) in [-1, 1]. Arms A and B run
        the identical computation here; only arm C differs."""
        feat = self.features(hist)
        cond = self._condition(feat, prev_act)
        a = torch.randn(
            hist.shape[0], self.cfg.chunk, self.cfg.act_dim, device=hist.device
        )
        dt = 1.0 / n_steps
        for k in range(n_steps):
            tau = torch.full((hist.shape[0],), k * dt, device=hist.device)
            a = a + dt * self.velocity(a, tau, cond)
        return a.clamp(-1, 1)

    # -- persistence -----------------------------------------------------
    def save(self, path: str, normalizer: dict, extra: dict | None = None):
        torch.save(
            dict(
                config=asdict(self.cfg),
                state_dict=self.state_dict(),
                normalizer={k: np.asarray(v) for k, v in normalizer.items()},
                extra=extra or {},
            ),
            path,
        )

    @classmethod
    def load(cls, path: str, device="cuda"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(PolicyConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return model, ckpt["normalizer"], ckpt.get("extra", {})
