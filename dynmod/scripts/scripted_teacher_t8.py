"""c-AWARE calibrated scripted teacher for StackCarryT8.

Same paradigm that certified the T3 teacher after nine RL attempts failed:
a scripted controller whose committed choice is looked up from the TRUE c.
Here the committed choice is WHERE to seat the block on the narrow beam -
the block stands only while its hidden centre of mass stays inside a band
around the beam's centre line, and the carry's own acceleration shifts that
band. The teacher seats the block so its COM lands on the calibrated bias
for its (mass, friction) cell; that lookup is the physics knowledge.

Must clear the same two bars as every teacher in this study:
mean success >= 0.55 and worst per-axis marginal spread <= 0.10.

    python -m dynmod.scripts.scripted_teacher_t8 --calibrate --episodes 8 \
        --calib-out reports/t8_seat_calibration_v1.json
    python -m dynmod.scripts.scripted_teacher_t8 --flatness --episodes 32 \
        --calib reports/t8_seat_calibration_v1.json --tag flat_t8_aware_v1
    python -m dynmod.scripts.scripted_teacher_t8 --episodes 1000 --seed 0 \
        --calib reports/t8_seat_calibration_v1.json \
        --out /mnt/scratch/dynamics/data/t8_1e3
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.envs.randomization import C_KEYS, GRANULAR_KEYS
from dynmod.scripts.premium_gate_t8 import (BIASES, SPEEDS,
                                            StackCarryController, com_y)
from mani_skill.utils.wrappers import RecordEpisode

REPORTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                       "reports"))
CALIB = os.path.join(REPORTS, "t8_seat_calibration_v1.json")
ENV_ID = "StackCarryT8-v1"
HORIZON = 140

# calibration grid: same axes and resolution the T3 teacher needed (the 3x3
# table left inter-bucket physics mistuned and failed the spread gate).
# SPEEDS/BIASES are imported so the teacher and the gate search ONE grid.
MASS = np.exp(np.linspace(np.log(0.7), np.log(1.4), 5)).round(3)
FRICTION = np.exp(np.linspace(np.log(0.3), np.log(0.7), 4)).round(3)


def make(n, **kw):
    return gym.make(ENV_ID, num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                    **kw)


def load_table(path=None):
    raw = json.load(open(path or CALIB))
    fr = sorted({float(k.split("_")[0]) for k in raw})
    ma = sorted({float(k.split("_")[1]) for k in raw})
    speed = np.zeros((len(fr), len(ma)))
    bias = np.zeros((len(fr), len(ma)))
    for k, (sp, b) in raw.items():
        f, m = (float(x) for x in k.split("_"))
        speed[fr.index(f), ma.index(m)] = sp
        bias[fr.index(f), ma.index(m)] = b
    return np.array(fr), np.array(ma), speed, bias


def lookup(base, c, calib=None):
    """Per-env carry speed and seat offset from the TRUE c (log-nearest
    bucket, as in T3). The seat offset is bias - COM, so the block's mass -
    not its shape - lands on the calibrated bias."""
    fr, ma, speed, bias = load_table(calib)
    f = np.asarray(c["friction"], dtype=np.float64)
    m = np.asarray(c["mass_mult"], dtype=np.float64)
    fi = np.abs(np.log(f[:, None]) - np.log(fr[None])).argmin(1)
    mi = np.abs(np.log(m[:, None]) - np.log(ma[None])).argmin(1)
    speeds = torch.tensor(speed[fi, mi], dtype=torch.float32,
                          device=base.device)
    comp = torch.zeros((base.num_envs, 2), device=base.device)
    comp[:, 1] = torch.tensor(bias[fi, mi], dtype=torch.float32,
                              device=base.device) - com_y(base, c)
    return speeds, comp


def rollout(env, speeds, comp, seed, horizon=HORIZON):
    base = env.unwrapped
    ctrl = StackCarryController(base, env.action_space.shape[-1], speeds, comp)
    seen = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for _ in range(horizon):
        _, _, _, _, info = env.step(ctrl.act())
        seen |= info["success"]
    return seen


def run_calibrate(args):
    """Best (speed, seat bias) per (mass, friction) cell, all cells and all
    candidates in ONE batched env - the whole grid is 40 x 10 = 400 envs."""
    cells = [(m, f) for m in MASS for f in FRICTION]
    combos = [(sp, b) for sp in SPEEDS for b in BIASES]
    n = len(cells) * len(combos)
    c_override = dict(
        mass_mult=np.array([m for (m, f) in cells for _ in combos]),
        friction=np.array([f for (m, f) in cells for _ in combos]),
        com_frac=np.full(n, 0.2),
    )
    env = make(n, randomize_c=False, c_override=c_override)
    base = env.unwrapped
    speeds = torch.tensor([s for _ in cells for (s, _) in combos],
                          dtype=torch.float32, device=base.device)
    bias = torch.tensor([b for _ in cells for (_, b) in combos],
                        dtype=torch.float32, device=base.device)

    success = np.zeros(n)
    for ep in range(args.episodes):
        env.reset(seed=100 + ep)
        c = base.get_c()
        comp = torch.zeros((n, 2), device=base.device)
        comp[:, 1] = bias - com_y(base, c)
        success += rollout(env, speeds, comp, seed=None).cpu().numpy()
        print(f"calib episode {ep + 1}/{args.episodes}", flush=True)
    env.close()
    success /= args.episodes

    table, detail = {}, {}
    k = len(combos)
    for i, (m, f) in enumerate(cells):
        s = success[i * k:(i + 1) * k]
        best = int(np.argmax(s))
        table[f"{f}_{m}"] = list(combos[best])
        detail[f"{f}_{m}"] = dict(best=list(combos[best]),
                                  best_success=float(s[best]),
                                  all=[[list(cb), float(v)]
                                       for cb, v in zip(combos, s)])
        print(f"mass {m} friction {f}: speed {combos[best][0]} "
              f"bias {combos[best][1] * 100:+.2f}cm -> {s[best]:.3f}", flush=True)
    out = args.calib_out or CALIB
    json.dump(table, open(out, "w"), indent=1)
    json.dump(detail, open(out.replace(".json", "_detail.json"), "w"), indent=1)
    print(f"wrote {out}")


def run_flatness(args):
    COM = np.array([0.0, 0.2, 0.4])
    points = [dict(mass_mult=m, friction=f, com_frac=o)
              for m, f, o in itertools.product(MASS, FRICTION, COM)]
    n = len(points)
    c_override = {k: np.array([p[k] for p in points]) for k in points[0]}
    env = make(n, randomize_c=False, c_override=c_override)
    base = env.unwrapped

    success_once = np.zeros(n)
    for ep in range(args.episodes):
        env.reset(seed=ep)
        speeds, comp = lookup(base, base.get_c(), args.calib)
        success_once += rollout(env, speeds, comp, seed=None).cpu().numpy()
    env.close()
    success_once /= args.episodes

    result = dict(
        ckpt="scripted_teacher_t8 (calibrated c-aware)", episodes=args.episodes,
        mean_success=float(success_once.mean()),
        spread=float(success_once.max() - success_once.min()),
        points=[dict(**p, success_once=float(s))
                for p, s in zip(points, success_once)],
    )
    json.dump(result, open(os.path.join(REPORTS, f"{args.tag}.json"), "w"),
              indent=1)

    worst = 0.0
    for axis in ("mass_mult", "friction", "com_frac"):
        vals = sorted(set(p[axis] for p in result["points"]))
        marg = [np.mean([p["success_once"] for p in result["points"]
                         if p[axis] == v]) for v in vals]
        print(f"{axis} marginals: {[round(float(x), 3) for x in marg]}")
        worst = max(worst, max(marg) - min(marg))
    ok = success_once.mean() >= 0.55 and worst <= 0.10
    print(f"mean {success_once.mean():.3f}  worst marginal spread {worst:.3f}")
    print(f"GATE (mean>=0.55, spread<=0.10): {'PASS' if ok else 'FAIL'}")


def run_dataset(args):
    n = args.num_envs
    cycles = (args.episodes + n - 1) // n
    env = make(n, reconfiguration_freq=1)
    os.makedirs(args.out, exist_ok=True)
    env = RecordEpisode(
        env, output_dir=args.out, save_trajectory=True, save_video=False,
        trajectory_name="trajectory", source_type="scripted-caware",
        source_desc="calibrated stack-and-carry teacher, seat offset from true c",
    )
    base = env.unwrapped
    keys = list(C_KEYS + GRANULAR_KEYS) + ["com_x_frac", "com_y_frac", "mass_kg"]
    c_rows, success_rows = [], []
    for r in range(cycles):
        env.reset(seed=args.seed + r)
        c = base.get_c()
        speeds, comp = lookup(base, c, args.calib)
        seen = rollout(env, speeds, comp, seed=None)
        for i in range(n):
            c_rows.append([float(c[k][i]) for k in keys])
        success_rows.extend(seen.cpu().numpy().tolist())
        if (r + 1) % 10 == 0:
            print(f"cycle {r + 1}/{cycles} "
                  f"running success {np.mean(success_rows):.3f}", flush=True)
    env.close()

    succ = np.asarray(success_rows, dtype=bool)
    np.savez(os.path.join(args.out, "c_metadata.npz"),
             c=np.asarray(c_rows, dtype=np.float32), c_keys=np.array(keys),
             success_once=succ, episodes_per_cycle=n, cycles=cycles,
             seed=args.seed)
    summary = dict(env_id=ENV_ID, episodes=len(succ), num_envs=n,
                   success_once_rate=float(succ.mean()),
                   teacher="scripted-caware-calibrated")
    json.dump(summary, open(os.path.join(args.out, "harness_summary.json"), "w"),
              indent=1)
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--flatness", action="store_true")
    p.add_argument("--calib", default=None)
    p.add_argument("--calib-out", default=None)
    p.add_argument("--tag", default="flat_t8_aware_v1")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--num-envs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    args = p.parse_args()
    if args.calibrate:
        run_calibrate(args)
    elif args.flatness:
        run_flatness(args)
    else:
        assert args.out, "--out required for dataset generation"
        run_dataset(args)


if __name__ == "__main__":
    main()
