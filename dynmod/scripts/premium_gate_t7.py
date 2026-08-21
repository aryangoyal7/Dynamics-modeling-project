"""Knowledge-premium gate for T7 'Router' (route choice by physics).

Controller family: escort the block from the staging area to a channel
mouth (left = slick floor, right = grippy floor), wind up, commit ONE
fixed-strength flick along the channel, retreat. The only knowledge-bearing
decision is DISCRETE: which channel.
  c-blind: one fixed route (whichever calibrates best on the randomized
           mix) - same flick everywhere.
  c-aware: route chosen per (mass, friction) bucket from the true c.

    python -m dynmod.scripts.premium_gate_t7 --probe   # stop-x distributions
    python -m dynmod.scripts.premium_gate_t7           # full gate
Writes reports/premium_gate_t7.json.
"""

from __future__ import annotations

import argparse
import itertools
import json

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401

MASS_B = [0.75, 1.0, 1.35]
FRICTION_B = [0.33, 0.5, 0.68]
FLICK_SPEED = 1.0   # below ~0.85 the block never detaches from the hand
                    # (measured: stop-and-go to a fixed point at every c) -
                    # the same threshold T3's calibration found
FLICK_STANDOFF = 0.19
CROSS_X = -0.20   # side-crossing happens here, behind the walled lanes
MOUTH_X = -0.10   # lane mouth
# v6b: the flick standoff must be REACHABLE. The trace showed the hand
# parking at x=-0.274 while trying to reach obj_x - 0.19 = -0.411: with the
# block still at its spawn (-0.221) the windup position is outside the arm's
# workspace, so `ready` stayed true for 260 steps and no strike ever fired.
# Creep the block forward onto the last of the plain table first; from
# PUSH_TO_X the windup sits at -0.27, which the trace shows the arm holds.
PUSH_TO_X = -0.08
CREEP_SPEED = 0.2  # gentle: the approach runs on high-friction table, so the
                   # block stops quickly and settles in a repeatable place
PUSH_Z = 0.015
LANE_TOL = 0.012    # v6: strike only a block centred this well in its lane
SETTLE_V = 0.01     # v6: ...and this close to at rest
HORIZON = 260       # v6: settling costs steps; 200 cut some episodes off
                    # before they ever launched


def policy(base, act_dim, route):
    """route: (n,) +1 = left/slick channel, -1 = right/grippy channel."""
    device = base.device
    n = base.num_envs
    tcp = base.agent.tcp.pose.p
    obj = base.obj.pose.p
    y_t = route * base.CH_OFF

    # committed only once the block is in a channel AND has outrun the hand
    # (retreating on channel entry alone cut the strike mid-contact)
    launched = (obj[:, 0] > base.CH_X[0] + 0.02) \
        & ((obj[:, 0] - tcp[:, 0]) > 0.05)
    # v6 SETTLE-THEN-STRIKE: probe v5 left 10-20 cm of launch scatter against
    # a ~5 cm slot, because the flick fired while the block was still drifting
    # from the side-push and still off-centre in the lane.  Strike only a
    # block that is (a) centred to 12 mm and (b) at rest, so every launch
    # starts from the same state and the spread that remains is physics.
    at_row = (obj[:, 1] - y_t).abs() < LANE_TOL
    speed = torch.linalg.norm(base.obj.linear_velocity[:, :2], dim=1)
    settled = speed < SETTLE_V
    # v6a: the strike now happens FROM THE STAGING AREA, not from inside the
    # lane. v6's first cut kept v5's slow 0.35 approach push, which drove the
    # block into the slick channel before the flick - so the block coasted a
    # c-dependent distance BEFORE the launch even began, which is exactly the
    # scatter we are trying to remove. Settle the block on its lane row where
    # it spawns, then fire one committed flick through the confined lane.
    advanced = obj[:, 0] > PUSH_TO_X
    ready = at_row & settled & advanced

    a = torch.zeros((n, act_dim), device=device)
    a[:, 3:] = -1.0
    retreat = torch.zeros_like(tcp)
    retreat[:, 2] = 1.0

    # stage 1: side-push the block onto its channel row (push along y)
    y_dir = torch.sign(y_t - obj[:, 1])
    side_stand = obj.clone()
    side_stand[:, 1] -= y_dir * 0.055
    side_stand[:, 2] = PUSH_Z
    side_push = torch.zeros_like(tcp)
    # taper the push as the row is approached: a flat 0.3 overshot the 12 mm
    # window and the block never satisfied `centred`
    side_push[:, 1] = y_dir * torch.clip((y_t - obj[:, 1]).abs() * 6.0, 0.05, 0.3)
    side_push[:, 0] = torch.clip((obj[:, 0] - tcp[:, 0]) * 2.0, -0.2, 0.2)
    to_side = torch.linalg.norm(tcp[:, :2] - side_stand[:, :2], dim=1)
    side_aligned = (to_side < 0.015) & (tcp[:, 2] < PUSH_Z + 0.015)

    # stage 2: T3-style windup + charge along +x at the fixed flick speed
    standoff = torch.where(advanced, torch.full((n,), FLICK_STANDOFF, device=device),
                           torch.full((n,), 0.055, device=device))
    behind = obj.clone()
    behind[:, 0] -= standoff
    behind[:, 2] = PUSH_Z
    to_behind = torch.linalg.norm(tcp[:, :2] - behind[:, :2], dim=1)
    x_speed = torch.where(ready, torch.full((n,), FLICK_SPEED, device=device),
                          torch.full((n,), CREEP_SPEED, device=device))
    x_push = torch.zeros_like(tcp)
    x_push[:, 0] = x_speed
    x_push[:, 1] = torch.clip((obj[:, 1] - tcp[:, 1]) * 2.0, -0.2, 0.2)
    # creep while the block is short of PUSH_TO_X; strike only when ready
    x_aligned = (to_behind < 0.02) & (tcp[:, 2] < PUSH_Z + 0.015) \
        & (ready | (at_row & ~advanced))
    charging = ready & ((obj[:, 0] - tcp[:, 0]) > 0.03) \
        & ((tcp[:, 1] - obj[:, 1]).abs() < 0.025) & (tcp[:, 2] < PUSH_Z + 0.02) \
        & (tcp[:, 0] < 0.02)  # full follow-through (clamping at the channel
        # entrance cut strikes mid-swing -> 15 cm launch scatter, probe v3);
        # the block outruns the hand and `launched` retreats it well before
        # either slot

    def goto(target):
        t = target.clone()
        far = torch.linalg.norm(tcp[:, :2] - target[:, :2], dim=1) > 0.03
        t[:, 2] = torch.where(far, torch.full((n,), 0.10, device=device), target[:, 2])
        return torch.clip((t - tcp) / 0.1, -1.0, 1.0)

    stage2 = at_row
    move = torch.where(
        stage2[:, None],
        torch.where((x_aligned | charging)[:, None], x_push, goto(behind)),
        torch.where(side_aligned[:, None], side_push, goto(side_stand)),
    )
    a[:, :3] = torch.where(launched[:, None], retreat, move)
    return a


def make_env(n, **kw):
    # override the registered 200-step limit: v6 spends steps settling the
    # block before the strike, and a TimeLimit below HORIZON would truncate
    # episodes mid-slide and silently score them as failures
    return gym.make("RouteChoiceT7-v1", num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                    max_episode_steps=HORIZON, **kw)


def probe():
    """Stop-x distributions per channel x (mass, friction) bucket at the
    fixed flick - decides where the two slots go."""
    for side, name in ((1.0, "LEFT/slick"), (-1.0, "RIGHT/grippy")):
        print(f"--- {name} ---", flush=True)
        for m, f in itertools.product(MASS_B, FRICTION_B):
            co = dict(mass_mult=m, friction=f)
            env = make_env(32, randomize_c=False, c_override=co)
            base, act_dim = env.unwrapped, env.action_space.shape[-1]
            route = torch.full((32,), side, device=base.device)
            env.reset(seed=6)
            for _ in range(HORIZON):
                env.step(policy(base, act_dim, route))
            p = base.obj.pose.p.cpu().numpy()
            ch = p[:, 1] * side > base.CH_OFF - base.CH_HALF  # made it into channel
            x = p[ch, 0]
            if len(x):
                print(f"m={m} f={f}: in-channel {ch.mean():.2f}  stop x "
                      f"p25 {np.percentile(x,25):+.3f} p50 {np.percentile(x,50):+.3f} "
                      f"p75 {np.percentile(x,75):+.3f}", flush=True)
            else:
                print(f"m={m} f={f}: in-channel 0.00", flush=True)
            env.close()


def route_table(base, c, aware, table, fixed):
    if not aware:
        return torch.full((base.num_envs,), float(fixed), device=base.device)
    ms, fs = np.array(MASS_B), np.array(FRICTION_B)
    r = np.array([
        table[(MASS_B[np.abs(np.log(ms) - np.log(c["mass_mult"][i])).argmin()],
               FRICTION_B[np.abs(np.log(fs) - np.log(c["friction"][i])).argmin()])]
        for i in range(base.num_envs)])
    return torch.tensor(r, dtype=torch.float32, device=base.device)


def main():
    # per-bucket best route from pinned-c trials
    table = {}
    for m, f in itertools.product(MASS_B, FRICTION_B):
        co = dict(mass_mult=m, friction=f)
        env = make_env(32, randomize_c=False, c_override=co)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        best = (-1.0, 1.0)
        for side in (1.0, -1.0):
            env.reset(seed=6)
            seen = torch.zeros(32, dtype=torch.bool, device=base.device)
            route = torch.full((32,), side, device=base.device)
            for _ in range(HORIZON):
                _, _, _, _, info = env.step(policy(base, act_dim, route))
                seen |= info["success"]
            v = float(seen.float().mean())
            if v > best[0]:
                best = (v, side)
        env.close()
        table[(m, f)] = best[1]
        print(f"mass={m} friction={f}: best route "
              f"{'LEFT' if best[1] > 0 else 'RIGHT'} -> {best[0]:.2f}", flush=True)

    res = {}
    for fixed in (1.0, -1.0):  # calibrate the blind route on the full mix
        env = make_env(128, reconfiguration_freq=1)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        succ = []
        for cyc in range(2):
            env.reset(seed=6000 + cyc)
            route = torch.full((128,), fixed, device=base.device)
            seen = torch.zeros(128, dtype=torch.bool, device=base.device)
            for _ in range(HORIZON):
                _, _, _, _, info = env.step(policy(base, act_dim, route))
                seen |= info["success"]
            succ.extend(seen.cpu().numpy().tolist())
        env.close()
        res[f"blind_{'L' if fixed > 0 else 'R'}"] = float(np.mean(succ))
    blind_route = 1.0 if res["blind_L"] >= res["blind_R"] else -1.0

    final = {}
    for mode in ("blind", "aware"):
        env = make_env(128, reconfiguration_freq=1)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        succ = []
        for cyc in range(4):
            env.reset(seed=7000 + cyc)
            c = base.get_c()
            route = route_table(base, c, mode == "aware", table, blind_route)
            seen = torch.zeros(128, dtype=torch.bool, device=base.device)
            for _ in range(HORIZON):
                _, _, _, _, info = env.step(policy(base, act_dim, route))
                seen |= info["success"]
            succ.extend(seen.cpu().numpy().tolist())
        env.close()
        final[mode] = float(np.mean(succ))
        n_ep = len(succ)
    ci = 2 * (2 * 0.25 / n_ep) ** 0.5
    print(f"blind({'L' if blind_route > 0 else 'R'}) {final['blind']:.3f}  "
          f"aware {final['aware']:.3f}  "
          f"premium {final['aware'] - final['blind']:+.3f} ± {ci:.3f}")
    json.dump(dict(
        route_table={f"{k}": ("L" if v > 0 else "R") for k, v in table.items()},
        blind_route="L" if blind_route > 0 else "R",
        blind_calib=res, result=dict(
            blind=final["blind"], aware=final["aware"],
            premium=final["aware"] - final["blind"], ci95=ci)),
        open("reports/premium_gate_t7.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    probe() if args.probe else main()
