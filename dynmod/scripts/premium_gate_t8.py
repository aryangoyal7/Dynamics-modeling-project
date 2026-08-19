"""Knowledge-premium gate for T8 'stack-and-carry' (user design 2026-08-19).

Controller family: grasp the cube, place it on the slab at a chosen offset,
re-grasp the slab, carry the stack to the goal at a chosen speed, set it
down. Two knowledge-bearing choices:
  placement offset  c-aware: center the cube's hidden COM over the slab seat
                    c-blind: center the cube's geometry (offset 0)
  carry speed       c-aware: per-(mass, friction) bucket, calibrated
                    c-blind: one fixed speed, calibrated on the full mix

    python -m dynmod.scripts.premium_gate_t8 --probe   # per-bucket speed map
    python -m dynmod.scripts.premium_gate_t8           # full gate
Writes reports/premium_gate_t8.json.
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
SPEEDS = [0.2, 0.35, 0.5, 0.7, 0.9]
HORIZON = 140
HOVER_Z = 0.14
CUBE_GRASP_Z = 0.035
CUBE_PLACE_Z = 0.066
SLAB_GRASP_Z = 0.021
CARRY_Z = 0.06
PLACE_DOWN_Z = 0.023
TRANSIT_V = 0.7


class StackCarryController:
    """Per-env stage machine. speeds: (n,) carry speed. place_comp: (n,2)
    xy compensation added to the cube seat (c-aware: -COM offset)."""

    def __init__(self, base, act_dim, speeds, place_comp):
        self.base, self.act_dim = base, act_dim
        self.n, self.dev = base.num_envs, base.device
        self.speeds = speeds
        self.place_comp = place_comp
        self.stage = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        self.dwell = torch.zeros(self.n, dtype=torch.long, device=self.dev)

    def _adv(self, cond, frm):
        hit = cond & (self.stage == frm)
        self.stage = torch.where(hit, self.stage + 1, self.stage)
        self.dwell = torch.where(hit, torch.zeros_like(self.dwell), self.dwell)

    def act(self):
        base, n = self.base, self.n
        tcp = base.agent.tcp.pose.p
        slab = base.obj.pose.p
        cube = base.top.pose.p
        goal = torch.tensor(base.GOAL_XY, device=self.dev)
        s = self.stage
        self.dwell += 1

        # per-stage TCP targets ------------------------------------------------
        t = torch.zeros((n, 3), device=self.dev)
        above_cube = cube.clone(); above_cube[:, 2] = HOVER_Z
        at_cube = cube.clone(); at_cube[:, 2] = CUBE_GRASP_Z
        seat = slab[:, :2] + torch.tensor(
            [base.TOP_SEAT_DX, 0.0], device=self.dev) + self.place_comp
        above_seat = torch.cat([seat, torch.full((n, 1), HOVER_Z, device=self.dev)], 1)
        at_seat = torch.cat([seat, torch.full((n, 1), CUBE_PLACE_Z, device=self.dev)], 1)
        grip_xy = slab[:, :2] + torch.tensor(
            [base.GRASP_DX, 0.0], device=self.dev)
        above_grip = torch.cat([grip_xy, torch.full((n, 1), 0.12, device=self.dev)], 1)
        at_grip = torch.cat([grip_xy, torch.full((n, 1), SLAB_GRASP_Z, device=self.dev)], 1)
        carry_t = torch.zeros((n, 3), device=self.dev)
        carry_t[:, 0] = goal[0] + base.GRASP_DX
        carry_t[:, 1] = goal[1]
        carry_t[:, 2] = CARRY_Z
        down_t = carry_t.clone(); down_t[:, 2] = PLACE_DOWN_Z
        retreat = tcp.clone(); retreat[:, 2] = 0.16

        targets = [above_cube, at_cube, at_cube, above_cube,  # 0-3
                   above_seat, at_seat, at_seat, above_grip,  # 4-7
                   at_grip, at_grip, carry_t, carry_t,        # 8-11 (10=lift)
                   down_t, down_t, retreat]                   # 12-14
        for k, tgt in enumerate(targets):
            t = torch.where((s == k)[:, None], tgt, t)

        # speed limit: carry stage uses the per-env choice, transit the default
        lim = torch.full((n,), TRANSIT_V, device=self.dev)
        lim = torch.where(s == 11, self.speeds, lim)
        a = torch.zeros((n, self.act_dim), device=self.dev)
        a[:, :3] = torch.clamp((t - tcp) / 0.1, -lim[:, None], lim[:, None])

        # gripper: open on approach/release stages, closed while holding
        holding = ((s >= 2) & (s <= 5)) | ((s >= 9) & (s <= 12))
        a[:, 3:] = torch.where(holding[:, None], -torch.ones((n, 1), device=self.dev),
                               torch.ones((n, 1), device=self.dev))

        # transitions ----------------------------------------------------------
        d_xy = lambda tgt: torch.linalg.norm(tcp[:, :2] - tgt[:, :2], dim=1)
        self._adv((d_xy(above_cube) < 0.012) & (tcp[:, 2] > HOVER_Z - 0.02), 0)
        self._adv((d_xy(at_cube) < 0.012) & (tcp[:, 2] < CUBE_GRASP_Z + 0.012), 1)
        self._adv(self.dwell >= 8, 2)   # close on cube
        self._adv(tcp[:, 2] > HOVER_Z - 0.02, 3)
        self._adv(d_xy(above_seat) < 0.012, 4)
        self._adv(tcp[:, 2] < CUBE_PLACE_Z + 0.012, 5)
        self._adv(self.dwell >= 6, 6)   # open, cube seated
        self._adv((d_xy(above_grip) < 0.012) & (tcp[:, 2] > 0.10), 7)
        self._adv(tcp[:, 2] < SLAB_GRASP_Z + 0.012, 8)
        self._adv(self.dwell >= 8, 9)   # close on slab
        self._adv(tcp[:, 2] > CARRY_Z - 0.01, 10)
        arrived = torch.linalg.norm(slab[:, :2] - goal[None], dim=1) < 0.02
        self._adv(arrived, 11)
        self._adv(tcp[:, 2] < PLACE_DOWN_Z + 0.008, 12)
        self._adv(self.dwell >= 6, 13)  # open, stack delivered
        return a


def make_env(n, **kw):
    return gym.make("StackCarryT8-v1", num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                    **kw)


def episode(env, speed_of, aware, seed):
    """speed_of(base, c) -> (n,) carry speeds; called AFTER the reset so c is
    the c actually simulated (reset reconfigures when reconfiguration_freq=1)."""
    base = env.unwrapped
    env.reset(seed=seed)
    c = base.get_c()
    speeds = speed_of(base, c)
    comp = torch.zeros((base.num_envs, 2), device=base.device)
    if aware:
        comp[:, 0] = -torch.as_tensor(
            c["com_x_frac"] * base.TOP_HALF, dtype=torch.float32, device=base.device)
        comp[:, 1] = -torch.as_tensor(
            c["com_y_frac"] * base.TOP_HALF, dtype=torch.float32, device=base.device)
    ctrl = StackCarryController(
        base, env.action_space.shape[-1], speeds, comp)
    seen = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for _ in range(HORIZON):
        _, _, _, _, info = env.step(ctrl.act())
        seen |= info["success"]
    stk = info["stack_ok"]
    return seen, stk, ctrl.stage


def probe(aware=True):
    """Success per (mass, friction) bucket x carry speed at pinned c."""
    for m, f in itertools.product(MASS_B, FRICTION_B):
        env = make_env(32, randomize_c=False,
                       c_override=dict(mass_mult=m, friction=f, com_frac=0.2))
        base = env.unwrapped
        row = []
        for sp in SPEEDS:
            fixed = lambda b, c, sp=sp: torch.full((b.num_envs,), sp, device=b.device)
            seen, stk, stage = episode(env, fixed, aware, seed=8)
            row.append(f"s={sp}: {seen.float().mean():.2f}"
                       f"/stk{stk.float().mean():.2f}"
                       f"/st{stage.float().mean():.0f}")
        env.close()
        print(f"m={m} f={f}:  " + "  ".join(row), flush=True)


def main():
    # aware speed table per bucket
    table = {}
    for m, f in itertools.product(MASS_B, FRICTION_B):
        env = make_env(64, randomize_c=False,
                       c_override=dict(mass_mult=m, friction=f))
        base = env.unwrapped
        best = (-1.0, SPEEDS[0])
        for sp in SPEEDS:
            fixed = lambda b, c, sp=sp: torch.full((b.num_envs,), sp, device=b.device)
            seen, _, _ = episode(env, fixed, True, seed=8)
            v = float(seen.float().mean())
            if v > best[0]:
                best = (v, sp)
        env.close()
        table[(m, f)] = best[1]
        print(f"aware m={m} f={f}: speed {best[1]} -> {best[0]:.2f}", flush=True)

    # blind: best single fixed speed on the randomized mix
    blind_cal = {}
    for sp in SPEEDS:
        env = make_env(128, reconfiguration_freq=1)
        base = env.unwrapped
        succ = []
        for cyc in range(2):
            fixed = lambda b, c, sp=sp: torch.full((b.num_envs,), sp, device=b.device)
            seen, _, _ = episode(env, fixed, False, seed=8100 + cyc)
            succ.extend(seen.cpu().numpy().tolist())
        env.close()
        blind_cal[sp] = float(np.mean(succ))
        print(f"blind s={sp}: {blind_cal[sp]:.3f}", flush=True)
    blind_speed = max(blind_cal, key=blind_cal.get)

    def aware_speeds(base, c):
        ms, fs = np.array(MASS_B), np.array(FRICTION_B)
        r = np.array([
            table[(MASS_B[np.abs(np.log(ms) - np.log(c["mass_mult"][i])).argmin()],
                   FRICTION_B[np.abs(np.log(fs) - np.log(c["friction"][i])).argmin()])]
            for i in range(base.num_envs)])
        return torch.tensor(r, dtype=torch.float32, device=base.device)

    final = {}
    for mode in ("blind", "aware"):
        env = make_env(128, reconfiguration_freq=1)
        base = env.unwrapped
        succ = []
        speed_of = (aware_speeds if mode == "aware" else
                    lambda b, c: torch.full((b.num_envs,), blind_speed, device=b.device))
        for cyc in range(4):
            seen, _, _ = episode(env, speed_of, mode == "aware", seed=8200 + cyc)
            succ.extend(seen.cpu().numpy().tolist())
        env.close()
        final[mode] = float(np.mean(succ))
        n_ep = len(succ)
    ci = 2 * (2 * 0.25 / n_ep) ** 0.5
    print(f"blind(s={blind_speed}) {final['blind']:.3f}  "
          f"aware {final['aware']:.3f}  "
          f"premium {final['aware'] - final['blind']:+.3f} ± {ci:.3f}")
    json.dump(dict(
        speed_table={str(k): v for k, v in table.items()},
        blind_calib=blind_cal, blind_speed=blind_speed,
        result=dict(blind=final["blind"], aware=final["aware"],
                    premium=final["aware"] - final["blind"], ci95=ci)),
        open("reports/premium_gate_t8.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    probe() if args.probe else main()
