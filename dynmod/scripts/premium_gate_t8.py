"""Knowledge-premium gate for T8 'stack-and-carry' (user design 2026-08-19).

Controller family: grasp the tall block, place it on the narrow beam at a
chosen lateral offset, re-grasp the beam at its free end, carry the stack
along +y to the goal at a chosen speed, set it down, retreat.

  placement offset  THE knowledge-bearing choice. The block's support is the
                    beam's 3.6 cm width, so it stands only while its hidden
                    COM stays inside a narrow band around the beam center
                    line - a band the carry's own acceleration shifts.
                    c-aware: seat the block so its COM lands on a calibrated
                             bias, i.e. offset by (bias - COM)
                    c-blind: one calibrated fixed offset. That IS the best a
                             c-blind controller can do: the COM angle is
                             uniform, so no fixed offset tracks it.
  carry speed       c-aware: per-(mass, friction) bucket; c-blind: one fixed
                    speed calibrated on the full mix. Gate iteration 1 proved
                    this lever alone pays nothing (see the env docstring); it
                    is kept because with an off-center COM the safe speed does
                    depend on c, and the blind arm must still be given its own
                    best speed for the comparison to be honest.

Both arms are calibrated over the SAME grid; only the information each may
use differs. Iteration 2 measured what happens if the aware arm is derived
rather than calibrated - it loses to blind, 0.28 vs 0.47.

    python -m dynmod.scripts.premium_gate_t8 --probe   # placement x speed map
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
SPEEDS = [0.4, 0.55]
# lateral seat bias, metres: where the block's COM should sit relative to the
# beam center line. Measured optimum is around +1.0 to +1.5 cm (the carry's
# opening acceleration throws the block the other way), never 0.
BIASES = [0.0025, 0.005, 0.0075, 0.010, 0.015]
HORIZON = 140
TRANSIT_V = 0.7
# heights (TCP), from the geometry: beam top 0.024, block half-height 0.05
HOVER_Z = 0.17          # above the standing block (top at 0.10)
CLEAR_Z = 0.20          # above the block once seated on the beam (top 0.124)
TOP_GRASP_Z = 0.075     # grasp the standing block 2.5 cm above its center
TOP_PLACE_Z = 0.099     # so the block center lands at 0.074 = seated
BEAM_GRASP_Z = 0.018
CARRY_Z = 0.055
PLACE_DOWN_Z = 0.020


class StackCarryController:
    """Per-env stage machine.
      speeds     (n,)   TCP speed limit while the stack is loaded
      place_comp (n,2)  xy shift added to the block seat (c-aware: -COM)
    """

    LOADED = (11, 13)   # lift / carry / set-down: the chosen speed applies
    HOLD_TOP = (2, 5)   # gripper closed on the block
    HOLD_BEAM = (10, 13)

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
        base, n, dev = self.base, self.n, self.dev
        tcp = base.agent.tcp.pose.p
        beam = base.obj.pose.p
        block = base.top.pose.p
        goal = torch.tensor(base.GOAL_XY, device=dev)
        s = self.stage
        self.dwell += 1
        col = lambda xy, z: torch.cat(
            [xy, torch.full((n, 1), z, device=dev)], dim=1)

        seat = beam[:, :2] + torch.tensor(
            [base.TOP_SEAT_DX, 0.0], device=dev) + self.place_comp
        grip_xy = beam[:, :2] + torch.tensor([base.GRASP_DX, 0.0], device=dev)
        carry_xy = goal[None] + torch.tensor([base.GRASP_DX, 0.0], device=dev)

        above_block = col(block[:, :2], HOVER_Z)
        at_block = col(block[:, :2], TOP_GRASP_Z)
        above_seat = col(seat, CLEAR_Z)
        at_seat = col(seat, TOP_PLACE_Z)
        above_grip = col(grip_xy, CLEAR_Z)
        at_grip = col(grip_xy, BEAM_GRASP_Z)
        lift = col(grip_xy, CARRY_Z)
        carry = col(carry_xy.expand(n, 2), CARRY_Z)
        down = col(carry_xy.expand(n, 2), PLACE_DOWN_Z)
        retreat = col(tcp[:, :2], HOVER_Z)

        targets = [above_block, at_block, at_block, above_block,   # 0-3
                   above_seat, at_seat, at_seat, above_seat,       # 4-7
                   above_grip, at_grip, at_grip, lift,             # 8-11
                   carry, down, down, retreat]                     # 12-15
        t = torch.zeros((n, 3), device=dev)
        for k, tgt in enumerate(targets):
            t = torch.where((s == k)[:, None], tgt, t)

        lo, hi = self.LOADED
        lim = torch.where((s >= lo) & (s <= hi), self.speeds,
                          torch.full((n,), TRANSIT_V, device=dev))
        a = torch.zeros((n, self.act_dim), device=dev)
        a[:, :3] = torch.clamp((t - tcp) / 0.1, -lim[:, None], lim[:, None])

        holding = ((s >= self.HOLD_TOP[0]) & (s <= self.HOLD_TOP[1])) | \
                  ((s >= self.HOLD_BEAM[0]) & (s <= self.HOLD_BEAM[1]))
        a[:, 3:] = torch.where(holding[:, None],
                               -torch.ones((n, 1), device=dev),
                               torch.ones((n, 1), device=dev))

        d_xy = lambda tgt: torch.linalg.norm(tcp[:, :2] - tgt[:, :2], dim=1)
        self._adv((d_xy(above_block) < 0.012) & (tcp[:, 2] > HOVER_Z - 0.02), 0)
        self._adv((d_xy(at_block) < 0.012) & (tcp[:, 2] < TOP_GRASP_Z + 0.012), 1)
        self._adv(self.dwell >= 8, 2)                       # close on block
        self._adv(tcp[:, 2] > HOVER_Z - 0.015, 3)
        self._adv((d_xy(above_seat) < 0.010) & (tcp[:, 2] > CLEAR_Z - 0.03), 4)
        self._adv(tcp[:, 2] < TOP_PLACE_Z + 0.010, 5)
        self._adv(self.dwell >= 8, 6)                       # open: COMMITTED
        self._adv(tcp[:, 2] > CLEAR_Z - 0.02, 7)            # clear the block
        self._adv(d_xy(above_grip) < 0.012, 8)
        self._adv(tcp[:, 2] < BEAM_GRASP_Z + 0.010, 9)
        self._adv(self.dwell >= 8, 10)                      # close on beam
        self._adv(tcp[:, 2] > CARRY_Z - 0.008, 11)
        self._adv(torch.linalg.norm(beam[:, :2] - goal[None], dim=1) < 0.02, 12)
        self._adv(tcp[:, 2] < PLACE_DOWN_Z + 0.008, 13)
        self._adv(self.dwell >= 6, 14)                      # open, delivered
        return a


def make_env(n, **kw):
    return gym.make("StackCarryT8-v1", num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                    **kw)


def com_y(base, c):
    """Hidden COM offset along the beam's narrow axis, in metres."""
    return torch.as_tensor(c["com_y_frac"] * base.TOP_HALF[1],
                           dtype=torch.float32, device=base.device)


def place_blind(bias):
    """Fixed lateral seat offset - all a c-blind controller can do."""
    def f(base, c):
        comp = torch.zeros((base.num_envs, 2), device=base.device)
        comp[:, 1] = bias
        return comp
    return f


def place_aware(bias_of):
    """Seat the block so its hidden COM lands on the calibrated bias.

    The bias is NOT zero and must be calibrated, not derived: a swept
    placement (reports note, gate iteration 2) showed success peaks when the
    COM sits ~+1.0 to +1.5 cm toward the goal, because the carry's opening
    acceleration throws the block the other way. Compensating the COM to the
    beam center line - the 'obvious' aware controller - lands on the falling
    flank and measures WORSE than blind (0.28 vs 0.47). Same lesson as T3:
    the informed arm has to be calibrated too, or the gate reads backwards.
    """
    def f(base, c):
        comp = torch.zeros((base.num_envs, 2), device=base.device)
        comp[:, 1] = bias_of(base, c) - com_y(base, c)
        return comp
    return f


def episode(env, speed_of, place_of, seed, horizon=HORIZON):
    """speed_of(base, c) -> (n,) speeds, place_of(base, c) -> (n,2) seat
    offset; both read c AFTER the reset (with reconfiguration_freq=1 the
    reset resamples the simulated physics).
    Returns (success, stack_ok_at_end, stage, first_success_step)."""
    base = env.unwrapped
    env.reset(seed=seed)
    c = base.get_c()
    speeds = speed_of(base, c)
    comp = place_of(base, c)
    ctrl = StackCarryController(base, env.action_space.shape[-1], speeds, comp)
    seen = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    t_hit = torch.full((base.num_envs,), -1, dtype=torch.long, device=base.device)
    for step in range(horizon):
        _, _, _, _, info = env.step(ctrl.act())
        newly = info["success"] & ~seen
        t_hit = torch.where(newly, torch.full_like(t_hit, step), t_hit)
        seen |= info["success"]
    return seen, info["stack_ok"], ctrl.stage, t_hit


def fixed(sp):
    return lambda b, c: torch.full((b.num_envs,), sp, device=b.device)


def probe():
    """Placement-response curve per bucket: success against where the block's
    COM ends up relative to the beam center line."""
    for m, f in itertools.product(MASS_B, FRICTION_B):
        env = make_env(32, randomize_c=False,
                       c_override=dict(mass_mult=m, friction=f, com_frac=0.35))
        row = []
        for b in BIASES:
            seen, _, _, t = episode(env, fixed(0.4),
                                    place_aware(lambda base, c, b=b: b), seed=8)
            tt = t[t >= 0]
            row.append(f"bias={b * 100:+.1f}cm {seen.float().mean():.2f}"
                       + (f"@{int(tt.float().mean())}" if len(tt) else ""))
        env.close()
        print(f"m={m} f={f}:  " + "  ".join(row), flush=True)


def main():
    # 1. aware: per-bucket (speed, bias), calibrated at pinned physics
    table = {}
    for m, f in itertools.product(MASS_B, FRICTION_B):
        env = make_env(64, randomize_c=False,
                       c_override=dict(mass_mult=m, friction=f))
        best = (-1.0, SPEEDS[0], BIASES[0])
        for sp, b in itertools.product(SPEEDS, BIASES):
            seen, _, _, _ = episode(env, fixed(sp),
                                    place_aware(lambda base, c, b=b: b), seed=8)
            v = float(seen.float().mean())
            if v > best[0]:
                best = (v, sp, b)
        env.close()
        table[(m, f)] = (best[1], best[2])
        print(f"aware m={m} f={f}: speed {best[1]} bias {best[2]*100:+.1f}cm "
              f"-> {best[0]:.2f}", flush=True)

    # 2. blind: one (speed, bias) pair, calibrated on the randomized mix -
    #    the best any c-blind controller can do
    blind_cal = {}
    for sp, b in itertools.product(SPEEDS, BIASES):
        env = make_env(128, reconfiguration_freq=1)
        succ = []
        for cyc in range(2):
            seen, _, _, _ = episode(env, fixed(sp), place_blind(b),
                                    seed=8100 + cyc)
            succ.extend(seen.cpu().numpy().tolist())
        env.close()
        blind_cal[f"{sp}/{b}"] = float(np.mean(succ))
        print(f"blind s={sp} bias={b*100:+.1f}cm: {blind_cal[f'{sp}/{b}']:.3f}",
              flush=True)
    bs, bb = max(blind_cal, key=blind_cal.get).split("/")
    blind_speed, blind_bias = float(bs), float(bb)

    def bucket(base, c, idx):
        ms, fs = np.array(MASS_B), np.array(FRICTION_B)
        return np.array([
            table[(MASS_B[np.abs(np.log(ms) - np.log(c["mass_mult"][i])).argmin()],
                   FRICTION_B[np.abs(np.log(fs) - np.log(c["friction"][i])).argmin()])][idx]
            for i in range(base.num_envs)])

    def aware_speeds(base, c):
        return torch.tensor(bucket(base, c, 0), dtype=torch.float32,
                            device=base.device)

    def aware_bias(base, c):
        return torch.tensor(bucket(base, c, 1), dtype=torch.float32,
                            device=base.device)

    # 3. head to head on the same randomized mix
    final = {}
    for mode in ("blind", "aware"):
        env = make_env(128, reconfiguration_freq=1)
        speed_of = aware_speeds if mode == "aware" else fixed(blind_speed)
        place_of = (place_aware(aware_bias) if mode == "aware"
                    else place_blind(blind_bias))
        succ = []
        for cyc in range(4):
            seen, _, _, _ = episode(env, speed_of, place_of, seed=8200 + cyc)
            succ.extend(seen.cpu().numpy().tolist())
        env.close()
        final[mode] = float(np.mean(succ))
        n_ep = len(succ)
    ci = 2 * (2 * 0.25 / n_ep) ** 0.5
    print(f"blind(s={blind_speed}, bias={blind_bias*100:+.1f}cm) "
          f"{final['blind']:.3f}  aware {final['aware']:.3f}  "
          f"premium {final['aware'] - final['blind']:+.3f} ± {ci:.3f}")
    json.dump(dict(
        aware_table={str(k): v for k, v in table.items()},
        blind_calib=blind_cal, blind_speed=blind_speed, blind_bias=blind_bias,
        episodes=n_ep,
        result=dict(blind=final["blind"], aware=final["aware"],
                    premium=final["aware"] - final["blind"], ci95=ci)),
        open("reports/premium_gate_t8.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    probe() if args.probe else main()
