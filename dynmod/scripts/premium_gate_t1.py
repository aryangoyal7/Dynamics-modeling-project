"""Knowledge-premium gate for T1 (pouring), before any teacher investment.

T1 has natural commitment (you cannot un-pour), so the expectation is that
physics knowledge pays here without geometry surgery: how many balls come
out per degree of tilt depends on fill level and content friction, and
overshoot is irreversible.

Controller family: lift the held cup over the basin, tilt about x at fixed
rate to a target angle, hold, then return upright.
  c-blind: one fixed target angle (calibrated on nominal contents).
  c-aware: target angle per (fill, content friction) bucket.

    python -m dynmod.scripts.premium_gate_t1
Writes reports/premium_gate_t1.json.
"""

from __future__ import annotations

import itertools
import json

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401

FILLS = [20, 25, 30]
PPF = [0.25, 0.45, 0.7]
ANGLES = [50, 70, 90, 110, 130]  # tilt-phase steps at fixed rate ~ target angle
HORIZON = 170


def pour_actions(base, act_dim, t, tilt_steps):
    """Phase script: position over basin (0-34), tilt (35..35+tilt_steps),
    hold. tilt_steps may be a per-env tensor."""
    n = base.num_envs
    device = base.device
    bx, by = base.BASIN_POS
    tcp = base.agent.tcp.pose.p
    a = torch.zeros((n, act_dim), device=device)
    if t < 35:
        target = torch.tensor([bx, by + 0.07, 0.24], device=device)
        a[:, :3] = torch.clip((target[None] - tcp) / 0.1, -0.4, 0.4)
    else:
        if not isinstance(tilt_steps, torch.Tensor):
            tilt_steps = torch.full((n,), float(tilt_steps), device=device)
        tilting = (t - 35) < tilt_steps
        a[:, 3] = torch.where(tilting, torch.full((n,), -0.7, device=device),
                              torch.zeros(n, device=device))
    a[:, -1] = -1.0
    return a


def run(c_override, tilt_steps, n=32, seed=6):
    env = gym.make("PourT1-v1", num_envs=n, obs_mode="state",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cuda",
                   spawn_grasped=True, randomize_c=False, c_override=c_override)
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    env.reset(seed=seed)
    seen = torch.zeros(n, dtype=torch.bool, device=base.device)
    for t in range(HORIZON):
        _, _, _, _, info = env.step(pour_actions(base, act_dim, t, tilt_steps))
        seen |= info["success"]
    s = seen.float().mean().item()
    env.close()
    return s


def run_random(mode, table, nominal, n_ep=512):
    succ = []
    env = gym.make("PourT1-v1", num_envs=128, obs_mode="state",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cuda",
                   spawn_grasped=True, reconfiguration_freq=1)
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    fills, ppfs = np.array(FILLS), np.array(PPF)
    for cyc in range(n_ep // 128):
        env.reset(seed=6000 + cyc)
        c = base.get_c()
        if mode == "aware":
            ts = np.array([
                table[(FILLS[np.abs(fills - c["particle_count"][i]).argmin()],
                       PPF[np.abs(ppfs - c["pp_friction"][i]).argmin()])]
                for i in range(128)])
            tilt = torch.tensor(ts, dtype=torch.float32, device=base.device)
        else:
            tilt = nominal
        seen = torch.zeros(128, dtype=torch.bool, device=base.device)
        for t in range(HORIZON):
            _, _, _, _, info = env.step(pour_actions(base, act_dim, t, tilt))
            seen |= info["success"]
        succ.extend(seen.cpu().numpy().tolist())
    env.close()
    return float(np.mean(succ)), len(succ)


def main():
    table = {}
    for fill, ppf in itertools.product(FILLS, PPF):
        co = dict(particle_count=float(fill), pp_friction=ppf)
        best = max((run(co, ang), ang) for ang in ANGLES)
        table[(fill, ppf)] = best[1]
        print(f"fill={fill} ppf={ppf}: best tilt-steps {best[1]} -> {best[0]:.2f}",
              flush=True)
    nominal = table[(25, 0.45)]

    out = dict(calibration={f"{k}": v for k, v in table.items()})
    b, n = run_random("blind", table, nominal)
    a, _ = run_random("aware", table, nominal)
    ci = 2 * (2 * 0.25 / n) ** 0.5
    print(f"blind {b:.3f}  aware {a:.3f}  premium {a - b:+.3f} ± {ci:.3f} (95%)")
    out["result"] = dict(blind=b, aware=a, premium=a - b, ci95=ci)
    with open("reports/premium_gate_t1.json", "w") as fp:
        json.dump(out, fp, indent=1)


if __name__ == "__main__":
    main()
