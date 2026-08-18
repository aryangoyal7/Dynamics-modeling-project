"""Instrumented nominal-physics pour: where do the particles actually go?

The T1 gate floored at ~0 success even at nominal c (diagnostic 2026-08-18),
so this traces one batch of fixed pours step by step: TCP position, cup tilt,
and the transfer/retained/spilled particle fractions over time, plus the
final scatter of particle positions relative to the basin.

    python -m dynmod.scripts.t1_pour_probe
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.scripts.premium_gate_t1 import HORIZON, pour_actions


def main():
    n = 64
    env = gym.make("PourT1-v1", num_envs=n, obs_mode="state",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cuda",
                   spawn_grasped=True, randomize_c=False, c_override={})
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    env.reset(seed=6)
    for t in range(HORIZON):
        _, _, _, _, info = env.step(pour_actions(base, act_dim, t, 130))
        if t % 15 == 0 or t == HORIZON - 1:
            tcp = base.agent.tcp.pose.p.mean(0)
            print(f"t={t:3d} tcp=({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}) "
                  f"tilt={info['cup_tilt'].mean():5.2f} "
                  f"transfer={info['transfer_frac'].mean():.3f} "
                  f"retained={info['retained_frac'].mean():.3f} "
                  f"spill={info['spill_frac'].mean():.3f}", flush=True)

    tf = info["transfer_frac"].cpu().numpy()
    print(f"\nfinal transfer_frac: mean {tf.mean():.3f}  "
          f"p10 {np.percentile(tf, 10):.3f}  p50 {np.percentile(tf, 50):.3f}  "
          f"p90 {np.percentile(tf, 90):.3f}  max {tf.max():.3f}")
    print(f"success (>=0.8): {(tf >= 0.8).mean():.3f}   "
          f">=0.5: {(tf >= 0.5).mean():.3f}   >=0.2: {(tf >= 0.2).mean():.3f}")

    # where did the active particles end, relative to the basin center?
    pp = base._particle_positions()  # (P, N, 3)
    act = base.particle_active
    bx, by = base.BASIN_POS
    rel = pp[..., :2] - torch.tensor([bx, by], device=base.device)
    r = rel.norm(dim=-1)[act]
    z = pp[..., 2][act]
    print(f"\nparticle radial dist from basin center: "
          f"p10 {r.quantile(0.1):.3f} p50 {r.quantile(0.5):.3f} "
          f"p90 {r.quantile(0.9):.3f} (basin inner half-width 0.05)")
    print(f"particle heights: p50 {z.quantile(0.5):.3f} "
          f"p90 {z.quantile(0.9):.3f} (basin cutoff {base.basin_wall + base.basin_depth + 0.02:.3f})")
    env.close()


if __name__ == "__main__":
    main()
