"""Knowledge-premium gate for T2 (carrying), before any teacher investment.

A task qualifies for the study only if a physics-aware controller beats a
physics-blind one (lesson of the T3 escorting episode). For T2 the suspected
escape hatch is slow carrying: crawl and slosh never happens regardless of
fill or content friction. This gate measures exactly that, under a deadline
that prices out crawling.

Controller family: straight-line carry toward the goal at speed s.
  c-blind: one fixed s (calibrated on nominal contents).
  c-aware: s chosen per (fill level, content friction) bucket from the same
           calibration sweep.

    python -m dynmod.scripts.premium_gate_t2
Writes reports/premium_gate_t2.json.
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
SPEEDS = [0.15, 0.25, 0.35, 0.5, 0.7]
HORIZONS = (120, 80)


GRIP_OFF = {}  # per-env-count cache of the mug's hanging offset from the TCP

def carry_action(base, act_dim, speed):
    """Aim the MUG (which hangs ~9cm below the grip) at the goal, not the
    TCP - the uncorrected aim guarantees a 9cm miss (found by the gate's
    first all-zero run). Settle on arrival so the sway dies before the
    deadline."""
    tcp = base.agent.tcp.pose.p
    goal = base.goal_site.pose.p
    off = GRIP_OFF.get(base.num_envs)
    if off is None:
        off = base.obj.pose.p - tcp
        GRIP_OFF[base.num_envs] = off.clone()
    mug_err = goal - (tcp + off)
    arrived = torch.linalg.norm(base.obj.pose.p - goal, dim=1) < 0.02
    a = torch.zeros((base.num_envs, act_dim), device=base.device)
    if isinstance(speed, torch.Tensor):
        lim = speed[:, None]
        a[:, :3] = torch.clamp(mug_err / 0.1, -lim, lim)
    else:
        a[:, :3] = torch.clip(mug_err / 0.1, -speed, speed)
    a[:, :3] = torch.where(arrived[:, None], torch.zeros_like(a[:, :3]), a[:, :3])
    a[:, 3:] = -1.0
    return a


def run(c_override, speed, horizon, n=32, seed=5):
    env = gym.make("CarryT2-v1", num_envs=n, obs_mode="state",
                   control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                   spawn_grasped=True, randomize_c=False, c_override=c_override)
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    env.reset(seed=seed)
    GRIP_OFF.clear()
    seen = torch.zeros(n, dtype=torch.bool, device=base.device)
    for t in range(horizon):
        _, _, _, _, info = env.step(carry_action(base, act_dim, speed))
        seen |= info["success"]
    s = seen.float().mean().item()
    env.close()
    return s


def run_random(mode, table, nominal_speed, horizon, n_ep=768):
    succ = []
    env = gym.make("CarryT2-v1", num_envs=128, obs_mode="state",
                   control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                   spawn_grasped=True, reconfiguration_freq=1)
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    fills, ppfs = np.array(FILLS), np.array(PPF)
    for cyc in range(n_ep // 128):
        env.reset(seed=4000 + cyc)
        GRIP_OFF.clear()
        c = base.get_c()
        if mode == "aware":
            sp = np.array([
                table[(FILLS[np.abs(fills - c["particle_count"][i]).argmin()],
                       PPF[np.abs(ppfs - c["pp_friction"][i]).argmin()])]
                for i in range(128)])
            sp_t = torch.tensor(sp, dtype=torch.float32, device=base.device)
        seen = torch.zeros(128, dtype=torch.bool, device=base.device)
        for t in range(horizon):
            a = carry_action(base, act_dim, sp_t if mode == "aware" else nominal_speed)
            _, _, _, _, info = env.step(a)
            seen |= info["success"]
        succ.extend(seen.cpu().numpy().tolist())
    env.close()
    return float(np.mean(succ)), len(succ)


def main():
    # calibrate per bucket at the tight deadline (the regime that matters)
    table = {}
    for fill, ppf in itertools.product(FILLS, PPF):
        co = dict(particle_count=float(fill), pp_friction=ppf)
        best = max((run(co, sp, horizon=80), sp) for sp in SPEEDS)
        table[(fill, ppf)] = best[1]
        print(f"fill={fill} ppf={ppf}: best speed {best[1]} -> {best[0]:.2f}")
    nominal = table[(25, 0.45)]

    out = dict(calibration={f"{k}": v for k, v in table.items()})
    for hz in HORIZONS:
        b, n = run_random("blind", table, nominal, hz)
        a, _ = run_random("aware", table, nominal, hz)
        ci = 2 * (2 * 0.25 / n) ** 0.5
        print(f"horizon {hz}: blind {b:.3f}  aware {a:.3f}  "
              f"premium {a - b:+.3f} ± {ci:.3f} (95%)")
        out[f"horizon_{hz}"] = dict(blind=b, aware=a, premium=a - b, ci95=ci)
    with open("reports/premium_gate_t2.json", "w") as fp:
        json.dump(out, fp, indent=1)


if __name__ == "__main__":
    main()
