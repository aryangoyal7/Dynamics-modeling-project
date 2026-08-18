"""T5 diagnostic: where do tossed blocks actually stop, and how repeatable
is it? First gate floored (~1% success): either the stop distribution
misses the slot (fix geometry/speeds) or within-condition scatter dwarfs
the slot width (chaos - the pour failure mode, would kill the task).

    python -m dynmod.scripts.t5_probe
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.scripts.premium_gate_t5 import HORIZON, make_env, policy

SPEEDS = [0.5, 0.7, 0.9, 1.1, 1.3]


def main():
    for s in SPEEDS:
        env = make_env(64, randomize_c=False, c_override={})
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        speed = torch.full((64,), s, device=base.device)
        env.reset(seed=6)
        launched = torch.zeros(64, dtype=torch.bool, device=base.device)
        for _ in range(HORIZON):
            _, _, _, _, info = env.step(policy(base, act_dim, speed))
            launched |= base.obj.pose.p[:, 2] < base.DECK_T - 0.01
        p = base.obj.pose.p.cpu().numpy()
        v = torch.linalg.norm(base.obj.linear_velocity, dim=1).cpu().numpy()
        x = p[:, 0]
        off = float(launched.float().mean())
        print(f"speed {s}: launched {off:.2f}  final x "
              f"p10 {np.percentile(x,10):+.3f} p50 {np.percentile(x,50):+.3f} "
              f"p90 {np.percentile(x,90):+.3f}  sigma {x.std():.3f}  "
              f"z p50 {np.percentile(p[:,2],50):+.3f}  "
              f"still_moving {(v>0.03).mean():.2f}", flush=True)
        env.close()
    print("slot range (0.24, 0.34), tol +/-0.025; cliff at 0.10")


if __name__ == "__main__":
    main()
