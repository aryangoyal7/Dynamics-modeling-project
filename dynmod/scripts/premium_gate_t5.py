"""Knowledge-premium gate for T5 (cliff toss), before any teacher investment.

Controller family: escort the object along the slick deck to a fixed windup
point, then charge-flick it off the cliff at speed s. After the object
leaves the deck the hand retreats - one committed toss, no recovery.
  c-blind: one fixed s (calibrated on nominal physics).
  c-aware: s per (mass, friction) bucket, from the true c.

Flight range scales with exit speed (mass/friction set it for a given
strike) and the plain table brakes the slide by the object's friction, so
knowing c should price directly into where the object stops.

    python -m dynmod.scripts.premium_gate_t5
Writes reports/premium_gate_t5.json.
"""

from __future__ import annotations

import itertools
import json

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401

MASS_B = [0.75, 1.0, 1.35]
FRICTION_B = [0.33, 0.5, 0.68]
SPEEDS = [0.45, 0.6, 0.75, 0.9, 1.1]
STANDOFF = 0.19  # windup length behind the object (T3 lesson: charge, don't poke)
FLICK_FROM_X = -0.02  # escort the object here before committing the flick
HORIZON = 120


def policy(base, act_dim, flick_speed):
    """Batched controller. flick_speed: per-env (n,) tensor."""
    device = base.device
    n = base.num_envs
    tcp = base.agent.tcp.pose.p
    obj = base.obj.pose.p
    deck_top = base.DECK_T
    push_z = deck_top + 0.015
    cliff = base.DECK_X[1]

    on_deck = obj[:, 2] > deck_top - 0.005  # once it drops, it is committed
    a = torch.zeros((n, act_dim), device=device)
    a[:, 3:] = -1.0

    retreat = torch.zeros_like(tcp)
    retreat[:, 2] = 1.0

    ready = obj[:, 0] > FLICK_FROM_X - 0.02
    standoff = torch.where(ready, torch.full_like(obj[:, 0], STANDOFF),
                           torch.full_like(obj[:, 0], 0.055))
    behind = obj.clone()
    behind[:, 0] -= standoff
    behind[:, 2] = push_z
    to_behind_xy = torch.linalg.norm(tcp[:, :2] - behind[:, :2], dim=1)

    target = behind.clone()
    target[:, 2] = torch.where(to_behind_xy > 0.03,
                               torch.full_like(behind[:, 2], deck_top + 0.10),
                               behind[:, 2])
    goto = torch.clip((target - tcp) / 0.1, -1.0, 1.0)

    speed = torch.where(ready, flick_speed,
                        torch.full_like(flick_speed, 0.35))
    push = torch.zeros_like(tcp)
    push[:, 0] = speed
    push[:, 1] = torch.clip((obj[:, 1] - tcp[:, 1]) * 2.0, -0.2, 0.2)

    at_standoff = (to_behind_xy < 0.02) & (tcp[:, 2] < push_z + 0.015)
    charging = ready & ((obj[:, 0] - tcp[:, 0]) > 0.03) \
        & ((tcp[:, 1] - obj[:, 1]).abs() < 0.025) & (tcp[:, 2] < push_z + 0.02) \
        & (tcp[:, 0] < cliff - 0.01)  # the hand never crosses the cliff
    move = torch.where((at_standoff | charging)[:, None], push, goto)
    a[:, :3] = torch.where(on_deck[:, None], move, retreat)
    return a


def make_env(n, **kw):
    return gym.make("CliffTossT5-v1", num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                    **kw)


def episode(env, base, act_dim, speed, seed):
    if not isinstance(speed, torch.Tensor):
        speed = torch.full((base.num_envs,), float(speed), device=base.device)
    env.reset(seed=seed)
    seen = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for _ in range(HORIZON):
        _, _, _, _, info = env.step(policy(base, act_dim, speed))
        seen |= info["success"]
    return seen, info


def main():
    table = {}
    for m, f in itertools.product(MASS_B, FRICTION_B):
        co = dict(mass_mult=m, friction=f)
        env = make_env(32, randomize_c=False, c_override=co)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        best = (-1.0, SPEEDS[0])
        for s in SPEEDS:
            seen, _ = episode(env, base, act_dim, s, 6)
            v = float(seen.float().mean())
            if v > best[0]:
                best = (v, s)
        env.close()
        table[(m, f)] = best[1]
        print(f"mass={m} friction={f}: best speed {best[1]} -> {best[0]:.2f}",
              flush=True)
    nominal = table[(1.0, 0.5)]

    ms, fs = np.array(MASS_B), np.array(FRICTION_B)
    res = {}
    for mode in ("blind", "aware"):
        succ = []
        env = make_env(128, reconfiguration_freq=1)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        for cyc in range(4):
            env.reset(seed=6000 + cyc)
            c = base.get_c()
            if mode == "aware":
                sp = np.array([
                    table[(MASS_B[np.abs(np.log(ms) - np.log(c["mass_mult"][i])).argmin()],
                           FRICTION_B[np.abs(np.log(fs) - np.log(c["friction"][i])).argmin()])]
                    for i in range(128)])
                speed = torch.tensor(sp, dtype=torch.float32, device=base.device)
            else:
                speed = torch.full((128,), float(nominal), device=base.device)
            seen = torch.zeros(128, dtype=torch.bool, device=base.device)
            for t in range(HORIZON):
                _, _, _, _, info = env.step(policy(base, act_dim, speed))
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
              open("reports/premium_gate_t5.json", "w"), indent=1)


if __name__ == "__main__":
    main()
