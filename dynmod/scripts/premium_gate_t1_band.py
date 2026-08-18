"""T1 redesign gate: MEASURED pour (2026-08-18).

The dump-everything gate came back premium -4.9 +/- 6.3 at healthy success
(blind 0.82): once the pour is aimed and decisive, over-tilting costs
nothing, so one fixed tilt works for every c and knowledge cannot pay.
This variant makes over-pouring an irreversible failure: success = ending
with 35-65% of the contents in the basin (pour about half, keep the rest).
How many degrees of tilt release half the cup depends on fill level and
content friction (clumpy contents release late and in bursts) - that is the
physics knowledge being priced. Controller: carry low to the basin edge,
tilt down at fixed rate for `ts` steps, hold, tilt back up, settle.

    python -m dynmod.scripts.premium_gate_t1_band
Writes reports/premium_gate_t1_band.json.
"""

from __future__ import annotations

import itertools
import json

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.scripts.premium_gate_t1_v2 import FILLS, PPF, Y_OFF, Z_POUR, make_env

TILTS = [20, 28, 36, 45, 55, 70]
BAND = (0.35, 0.65)
POS_STEPS = 45
HOLD = 15
RATE = 0.7
HORIZON = 210


def pour_actions(base, act_dim, t, tilt_steps):
    n = base.num_envs
    device = base.device
    bx, by = base.BASIN_POS
    tcp = base.agent.tcp.pose.p
    target = torch.tensor([bx, by + Y_OFF, Z_POUR], device=device)
    a = torch.zeros((n, act_dim), device=device)
    hold = torch.clip((target[None] - tcp) / 0.1, -0.2, 0.2)
    if t < POS_STEPS:
        a[:, :3] = torch.clip((target[None] - tcp) / 0.1, -0.4, 0.4)
    else:
        a[:, :3] = hold
        if not isinstance(tilt_steps, torch.Tensor):
            tilt_steps = torch.full((n,), float(tilt_steps), device=device)
        tt = t - POS_STEPS
        down = tt < tilt_steps
        up = (tt >= tilt_steps + HOLD) & (tt < 2 * tilt_steps + HOLD)
        a[:, 3] = torch.where(down, torch.full((n,), -RATE, device=device),
                  torch.where(up, torch.full((n,), RATE, device=device),
                              torch.zeros(n, device=device)))
    a[:, -1] = -1.0
    return a


def episode(env, base, act_dim, tilt_steps, seed):
    env.reset(seed=seed)
    for t in range(HORIZON):
        _, _, _, _, info = env.step(pour_actions(base, act_dim, t, tilt_steps))
    tf = info["transfer_frac"]
    return (tf >= BAND[0]) & (tf <= BAND[1]), tf


def main():
    table = {}
    for fill, ppf in itertools.product(FILLS, PPF):
        co = dict(particle_count=float(fill), pp_friction=ppf)
        env = make_env(32, randomize_c=False, c_override=co)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        best = (-1.0, TILTS[0])
        for ts in TILTS:
            ok, _ = episode(env, base, act_dim, ts, 6)
            s = float(ok.float().mean())
            if s > best[0]:
                best = (s, ts)
        env.close()
        table[(fill, ppf)] = best[1]
        print(f"fill={fill} ppf={ppf}: best tilt {best[1]} -> {best[0]:.2f}",
              flush=True)
    nominal = table[(25, 0.45)]

    fills, ppfs = np.array(FILLS), np.array(PPF)
    res = {}
    for mode in ("blind", "aware"):
        succ = []
        env = make_env(128, reconfiguration_freq=1)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        for cyc in range(4):
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
            for t in range(HORIZON):
                _, _, _, _, info = env.step(
                    pour_actions(base, act_dim, t, tilt))
            tf = info["transfer_frac"]
            ok = (tf >= BAND[0]) & (tf <= BAND[1])
            succ.extend(ok.cpu().numpy().tolist())
        env.close()
        res[mode] = float(np.mean(succ))
        n_ep = len(succ)
    ci = 2 * (2 * 0.25 / n_ep) ** 0.5
    print(f"blind {res['blind']:.3f}  aware {res['aware']:.3f}  "
          f"premium {res['aware'] - res['blind']:+.3f} ± {ci:.3f}")
    json.dump(dict(band=BAND,
                   calibration={f"{k}": v for k, v in table.items()},
                   result=dict(blind=res["blind"], aware=res["aware"],
                               premium=res["aware"] - res["blind"], ci95=ci)),
              open("reports/premium_gate_t1_band.json", "w"), indent=1)


if __name__ == "__main__":
    main()
