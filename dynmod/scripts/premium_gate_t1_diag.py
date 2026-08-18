"""Diagnose the T1 gate floor effect (2026-08-18).

The premium gate measured blind 2.5% / aware 3.1% under FULL c randomization,
while calibration cells (only fill + pp_friction set, everything else nominal)
had real success. With success floored near zero the gate cannot measure
knowledge. This script runs the same fixed pour (nominal tilt schedule) with
one c axis randomized at a time to find which axis destroys the controller.

    python -m dynmod.scripts.premium_gate_t1_diag
Writes reports/premium_gate_t1_diag.json.
"""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.scripts.premium_gate_t1 import HORIZON, pour_actions

NOMINAL_TILT = 130  # calibration winner for the (25, 0.45) nominal bucket

AXES = dict(
    nominal=None,
    mass_mult=lambda r, k: np.exp(r.uniform(np.log(0.7), np.log(1.4), k)),
    friction=lambda r, k: np.exp(r.uniform(np.log(0.3), np.log(0.7), k)),
    com_frac=lambda r, k: r.uniform(0.0, 0.15, k),
    handle_offset=lambda r, k: r.uniform(-0.02, 0.02, k),
    particle_count=lambda r, k: r.randint(20, 31, k).astype(float),
    pp_friction=lambda r, k: np.exp(r.uniform(np.log(0.2), np.log(0.8), k)),
)


def run_axis(name, sampler, n=128, cycles=4):
    rng = np.random.RandomState(17)
    succ = []
    for cyc in range(cycles):
        co = {} if sampler is None else {name: sampler(rng, n)}
        if name == "com_frac":  # random direction too, else COM is always +x
            co["com_angle"] = rng.uniform(0, 2 * np.pi, n)
        env = gym.make("PourT1-v1", num_envs=n, obs_mode="state",
                       control_mode="pd_ee_delta_pose", sim_backend="physx_cuda",
                       spawn_grasped=True, randomize_c=False, c_override=co)
        base = env.unwrapped
        act_dim = env.action_space.shape[-1]
        env.reset(seed=7000 + cyc)
        seen = torch.zeros(n, dtype=torch.bool, device=base.device)
        for t in range(HORIZON):
            _, _, _, _, info = env.step(
                pour_actions(base, act_dim, t, NOMINAL_TILT))
            seen |= info["success"]
        succ.extend(seen.cpu().numpy().tolist())
        env.close()
    return float(np.mean(succ)), len(succ)


def main():
    out = {}
    for name, sampler in AXES.items():
        s, n = run_axis(name, sampler)
        out[name] = dict(success=s, episodes=n)
        print(f"{name:15s} success {s:.3f}  ({n} eps)", flush=True)
    with open("reports/premium_gate_t1_diag.json", "w") as fp:
        json.dump(out, fp, indent=1)


if __name__ == "__main__":
    main()
