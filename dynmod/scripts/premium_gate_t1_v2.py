"""T1 premium gate, controller v2 (2026-08-18).

v1 poured from 24 cm up while creeping through 135 deg of tilt: particles
dribbled out over the whole sweep and landed with median radius 10 cm from a
5 cm basin (probe t1_pour_probe) - ~0% success even at nominal c, so the
gate measured nothing. v2 pours LOW, AIMED, and DECISIVELY: carry the cup
just past the basin's near edge at low height, then tilt at full rate to a
target angle and hold. The knowledge variable is unchanged - how far to tilt
for a given (fill, content friction) - and both controllers share the same
geometry, so the comparison stays fair.

    python -m dynmod.scripts.premium_gate_t1_v2 --sweep   # geometry search, nominal c
    python -m dynmod.scripts.premium_gate_t1_v2 --gate    # calibrate buckets + premium
Writes reports/t1_v2_sweep.json / reports/premium_gate_t1_v2.json.
"""

from __future__ import annotations

import argparse
import itertools
import json

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401

FILLS = [20, 25, 30]
PPF = [0.25, 0.45, 0.7]
TILTS = [20, 30, 40, 55, 75]  # full-rate tilt steps ~ target angle
HORIZON = 140
POS_STEPS = 45

# geometry (shared by blind and aware; winner of --sweep 2026-08-18:
# nominal success 0.75, transfer 0.62 - vs ~0 for the v1 high dribble)
Y_OFF = 0.03
Z_POUR = 0.11


def pour_actions(base, act_dim, t, tilt_steps, y_off=Y_OFF, z_pour=Z_POUR):
    """Carry low to the basin edge, then tilt at FULL rate for tilt_steps."""
    n = base.num_envs
    device = base.device
    bx, by = base.BASIN_POS
    tcp = base.agent.tcp.pose.p
    target = torch.tensor([bx, by + y_off, z_pour], device=device)
    a = torch.zeros((n, act_dim), device=device)
    if t < POS_STEPS:
        a[:, :3] = torch.clip((target[None] - tcp) / 0.1, -0.4, 0.4)
    else:
        # hold position while pouring; tilt full-rate for tilt_steps then stop
        a[:, :3] = torch.clip((target[None] - tcp) / 0.1, -0.2, 0.2)
        if not isinstance(tilt_steps, torch.Tensor):
            tilt_steps = torch.full((n,), float(tilt_steps), device=device)
        tilting = (t - POS_STEPS) < tilt_steps
        a[:, 3] = torch.where(tilting, torch.full((n,), -1.0, device=device),
                              torch.zeros(n, device=device))
    a[:, -1] = -1.0
    return a


def episode(env, base, act_dim, tilt_steps, seed, y_off=Y_OFF, z_pour=Z_POUR):
    env.reset(seed=seed)
    seen = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for t in range(HORIZON):
        _, _, _, _, info = env.step(
            pour_actions(base, act_dim, t, tilt_steps, y_off, z_pour))
        seen |= info["success"]
    return seen, info


def make_env(n, **kw):
    return gym.make("PourT1-v1", num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pose", sim_backend="physx_cuda",
                    spawn_grasped=True, **kw)


def sweep():
    """Nominal-c geometry search: does ANY (y_off, z, tilt) pour well?"""
    out = []
    for y_off, z in itertools.product([0.03, 0.05, 0.07], [0.11, 0.13, 0.16]):
        env = make_env(32, randomize_c=False, c_override={})
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        for ts in [25, 40, 60]:
            seen, info = episode(env, base, act_dim, ts, 6, y_off, z)
            tf = info["transfer_frac"]
            row = dict(y_off=y_off, z=z, tilt=ts,
                       success=float(seen.float().mean()),
                       transfer=float(tf.mean()))
            out.append(row)
            print(f"y{y_off:.2f} z{z:.2f} t{ts}: succ {row['success']:.2f} "
                  f"transfer {row['transfer']:.2f}", flush=True)
        env.close()
    json.dump(out, open("reports/t1_v2_sweep.json", "w"), indent=1)
    best = max(out, key=lambda r: (r["success"], r["transfer"]))
    print(f"BEST: {best}")


def gate():
    table = {}
    for fill, ppf in itertools.product(FILLS, PPF):
        co = dict(particle_count=float(fill), pp_friction=ppf)
        env = make_env(32, randomize_c=False, c_override=co)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        best = (-1.0, TILTS[0])
        for ang in TILTS:
            seen, _ = episode(env, base, act_dim, ang, 6)
            s = float(seen.float().mean())
            if s > best[0]:
                best = (s, ang)
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
            seen = torch.zeros(128, dtype=torch.bool, device=base.device)
            for t in range(HORIZON):
                _, _, _, _, info = env.step(
                    pour_actions(base, act_dim, t, tilt))
                seen |= info["success"]
            succ.extend(seen.cpu().numpy().tolist())
        env.close()
        res[mode] = float(np.mean(succ))
        n_ep = len(succ)
    ci = 2 * (2 * 0.25 / n_ep) ** 0.5
    print(f"blind {res['blind']:.3f}  aware {res['aware']:.3f}  "
          f"premium {res['aware'] - res['blind']:+.3f} ± {ci:.3f}")
    json.dump(dict(calibration={f"{k}": v for k, v in table.items()},
                   result=dict(blind=res["blind"], aware=res["aware"],
                               premium=res["aware"] - res["blind"], ci95=ci)),
              open("reports/premium_gate_t1_v2.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--gate", action="store_true")
    args = p.parse_args()
    if args.sweep:
        sweep()
    else:
        gate()
