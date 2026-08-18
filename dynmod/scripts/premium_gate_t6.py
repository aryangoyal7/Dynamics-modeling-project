"""Knowledge-premium gate for T6 (hidden-physics Push-T).

Controller family: alternate two primitives until the pose matches or time
runs out. ROTATE - push tangentially at the stem tip (long lever about the
COM) to spin the T toward the goal yaw. TRANSLATE - push along the line
through the ASSUMED center of mass toward the goal position (a push through
the true COM translates without rotating; one through the wrong point
rotates the T and must be corrected, costing steps).

  c-blind: assumes the COM is at the geometric centroid; one fixed speed.
  c-aware: uses the TRUE COM (from c) and a per-(mass,friction) speed.

The premium here is paid in mid-action correction - exactly the property
requested: the pusher acts, observes the response, and corrects, and a
better internal model wastes fewer of the 150 budgeted steps.

    python -m dynmod.scripts.premium_gate_t6 --probe   # nominal shakeout
    python -m dynmod.scripts.premium_gate_t6           # full gate
Writes reports/premium_gate_t6.json.
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
SPEEDS = [0.25, 0.4, 0.55]
CENTROID_X = 0.0257  # geometric centroid of the T in its local frame
STEM_TIP_X = 0.097
PUSH_Z = 0.015
HORIZON = 150
YAW_GO = 0.3  # rotate when |yaw err| above this, else translate


def _rot(yaw, v):
    """Rotate local (n,2) points v by per-env yaw."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] - s * v[:, 1],
                        s * v[:, 0] + c * v[:, 1]], dim=1)


def policy(base, act_dim, com_local, speed):
    """com_local: (n,2) assumed COM in the tee frame; speed: (n,)."""
    device = base.device
    n = base.num_envs
    tcp = base.agent.tcp.pose.p
    obj = base.obj.pose.p
    yaw = base.obj_yaw()
    goal = torch.tensor(base.GOAL_XY, device=device)

    dp = goal[None] - obj[:, :2]
    dyaw = torch.remainder(-yaw + np.pi, 2 * np.pi) - np.pi
    pose_ok = (torch.linalg.norm(dp, dim=1) < base.pos_tol) \
        & (dyaw.abs() < base.yaw_tol)

    com_w = obj[:, :2] + _rot(yaw, com_local)
    rotate = dyaw.abs() > YAW_GO

    # rotate primitive: tangential push at the stem tip
    tip_w = obj[:, :2] + _rot(yaw, torch.tensor(
        [[STEM_TIP_X, 0.0]], device=device).expand(n, 2))
    r = tip_w - com_w
    r_norm = r / torch.linalg.norm(r, dim=1, keepdim=True).clamp(min=1e-6)
    perp = torch.stack([-r_norm[:, 1], r_norm[:, 0]], dim=1)
    rot_dir = torch.sign(dyaw)[:, None] * perp
    rot_contact = tip_w

    # translate primitive: push through the assumed COM toward the goal
    dp_hat = dp / torch.linalg.norm(dp, dim=1, keepdim=True).clamp(min=1e-6)
    tr_dir = dp_hat
    tr_contact = com_w

    push_dir = torch.where(rotate[:, None], rot_dir, tr_dir)
    contact = torch.where(rotate[:, None], rot_contact, tr_contact)
    standoff_len = torch.where(rotate, torch.full((n,), 0.06, device=device),
                               torch.full((n,), 0.10, device=device))
    standoff = contact - push_dir * standoff_len[:, None]

    to_standoff = torch.linalg.norm(tcp[:, :2] - standoff, dim=1)
    # aligned: behind the contact point w.r.t. the push direction and close
    # to the push line
    rel = tcp[:, :2] - contact
    along = (rel * push_dir).sum(1)  # negative when behind the contact
    lateral = (rel - along[:, None] * push_dir).norm(dim=1)
    aligned = (along < -0.01) & (lateral < 0.02) & (tcp[:, 2] < PUSH_Z + 0.02)

    a = torch.zeros((n, act_dim), device=device)
    a[:, 3:] = -1.0
    retreat = torch.zeros_like(tcp)
    retreat[:, 2] = 1.0

    target = torch.zeros_like(tcp)
    target[:, :2] = standoff
    target[:, 2] = torch.where(to_standoff > 0.03,
                               torch.full((n,), 0.10, device=device),
                               torch.full((n,), PUSH_Z, device=device))
    goto = torch.clip((target - tcp) / 0.1, -1.0, 1.0)

    push = torch.zeros_like(tcp)
    push[:, :2] = push_dir * speed[:, None]

    move = torch.where(aligned[:, None], push, goto)
    a[:, :3] = torch.where(pose_ok[:, None], retreat, move)
    return a


def make_env(n, **kw):
    return gym.make("DynPushT6-v1", num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                    **kw)


def com_from_c(base, aware: bool):
    n = base.num_envs
    com = torch.zeros((n, 2), device=base.device)
    com[:, 0] = CENTROID_X
    if aware:
        c = base.get_c()
        com[:, 0] += torch.tensor(c["com_x_frac"] * 0.06, dtype=torch.float32,
                                  device=base.device)
        com[:, 1] += torch.tensor(c["com_y_frac"] * 0.06, dtype=torch.float32,
                                  device=base.device)
    return com


def run_eps(env, base, act_dim, com, speed, seed):
    env.reset(seed=seed)
    seen = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for _ in range(HORIZON):
        _, _, _, _, info = env.step(policy(base, act_dim, com, speed))
        seen |= info["success"]
    return seen, info


def probe():
    env = make_env(64, randomize_c=False, c_override={})
    base, act_dim = env.unwrapped, env.action_space.shape[-1]
    com = com_from_c(base, aware=False)
    for s in SPEEDS:
        speed = torch.full((64,), s, device=base.device)
        seen, info = run_eps(env, base, act_dim, com, speed, 6)
        print(f"speed {s}: success {seen.float().mean():.2f}  "
              f"pos p50 {info['pos_dist'].median():.3f}  "
              f"|yaw| p50 {info['yaw_err'].abs().median():.2f}", flush=True)
    env.close()


def main(cycles=4):
    # aware speed calibration per (mass, friction) bucket, com randomized
    table = {}
    for m, f in itertools.product(MASS_B, FRICTION_B):
        co = dict(mass_mult=m, friction=f)
        env = make_env(32, c_override=co, reconfiguration_freq=1)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        best = (-1.0, SPEEDS[0])
        for s in SPEEDS:
            env.reset(seed=6)
            com = com_from_c(base, aware=True)  # after reset: c is resampled
            speed = torch.full((32,), s, device=base.device)
            seen = torch.zeros(32, dtype=torch.bool, device=base.device)
            for _ in range(HORIZON):
                _, _, _, _, info = env.step(policy(base, act_dim, com, speed))
                seen |= info["success"]
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
        firsts = []  # step of first success, HORIZON+1 if never
        env = make_env(128, reconfiguration_freq=1)
        base, act_dim = env.unwrapped, env.action_space.shape[-1]
        for cyc in range(cycles):
            env.reset(seed=6000 + cyc)
            c = base.get_c()
            com = com_from_c(base, aware=(mode == "aware"))
            if mode == "aware":
                sp = np.array([
                    table[(MASS_B[np.abs(np.log(ms) - np.log(c["mass_mult"][i])).argmin()],
                           FRICTION_B[np.abs(np.log(fs) - np.log(c["friction"][i])).argmin()])]
                    for i in range(128)])
                speed = torch.tensor(sp, dtype=torch.float32, device=base.device)
            else:
                speed = torch.full((128,), float(nominal), device=base.device)
            first = torch.full((128,), HORIZON + 1, device=base.device)
            seen = torch.zeros(128, dtype=torch.bool, device=base.device)
            for t in range(HORIZON):
                _, _, _, _, info = env.step(policy(base, act_dim, com, speed))
                new = info["success"] & ~seen
                first[new] = t
                seen |= info["success"]
            firsts.extend(first.cpu().numpy().tolist())
        env.close()
        f = np.asarray(firsts)
        done = f <= HORIZON
        res[mode] = dict(
            success=float(done.mean()),
            success_at_100=float((f < 100).mean()),
            success_at_75=float((f < 75).mean()),
            median_steps=float(np.median(f[done])) if done.any() else None,
        )
        n_ep = len(firsts)
    ci = 2 * (2 * 0.25 / n_ep) ** 0.5
    for k in ("success", "success_at_100", "success_at_75"):
        print(f"{k}: blind {res['blind'][k]:.3f}  aware {res['aware'][k]:.3f}  "
              f"premium {res['aware'][k] - res['blind'][k]:+.3f} ± {ci:.3f}")
    print(f"median steps-to-success: blind {res['blind']['median_steps']}  "
          f"aware {res['aware']['median_steps']}")
    json.dump(dict(calibration={f"{k}": v for k, v in table.items()},
                   result=res, ci95=ci),
              open("reports/premium_gate_t6.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--cycles", type=int, default=4)
    args = p.parse_args()
    probe() if args.probe else main(args.cycles)
