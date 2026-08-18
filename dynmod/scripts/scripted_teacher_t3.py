"""c-AWARE calibrated scripted teacher for SlideToSlotT3.

The task-validation controller (2026-08-18: 61.7% vs 45.0% c-blind) rebuilt as
a permanent script. It is the c-blind expert with one change: the committed
flick's speed and run-up length are looked up from the TRUE c via the
calibration table in reports/t3_flick_calibration_v2.json (keys
"<friction>_<mass_mult>" over 3x3 buckets, values [flick_speed, standoff]).
That lookup is the physics knowledge: slippery blocks get a gentler launch,
grippy/heavy ones a faster launch with a longer run-up.

Candidate main-dataset teacher after PPO teachers plateaued below the bar
(v8 grid mean 0.36 vs required 0.55). Must pass the same two gates:
mean success >= 0.55 and per-axis marginal spread <= 0.10.

Flatness gate (same cells + report format as teacher_flatness.py):
    python -m dynmod.scripts.scripted_teacher_t3 --flatness --tag flat_scripted_aware
Dataset generation:
    python -m dynmod.scripts.scripted_teacher_t3 --episodes 1000 --seed 0 \
        --out /mnt/scratch/dynamics/data/t3_1e3
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
from dynmod.scripts.scripted_expert_t3 import policy
from mani_skill.utils.wrappers import RecordEpisode

REPORTS = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
CALIB = os.path.join(REPORTS, "t3_flick_calibration_v2.json")


def load_table(path=None):
    raw = json.load(open(path or CALIB))
    fr = sorted({float(k.split("_")[0]) for k in raw})
    ma = sorted({float(k.split("_")[1]) for k in raw})
    speed = np.zeros((len(fr), len(ma)))
    standoff = np.zeros((len(fr), len(ma)))
    for k, (sp, so) in raw.items():
        f, m = (float(x) for x in k.split("_"))
        speed[fr.index(f), ma.index(m)] = sp
        standoff[fr.index(f), ma.index(m)] = so
    return np.array(fr), np.array(ma), speed, standoff


def lookup(c, device, calib=None):
    """Per-env (flick_speed, flick_standoff) from true friction and mass."""
    fr, ma, speed, standoff = load_table(calib)
    f = np.asarray(c["friction"], dtype=np.float64)
    m = np.asarray(c["mass_mult"], dtype=np.float64)
    fi = np.abs(np.log(f[:, None]) - np.log(fr[None])).argmin(1)
    mi = np.abs(np.log(m[:, None]) - np.log(ma[None])).argmin(1)
    return (torch.tensor(speed[fi, mi], dtype=torch.float32, device=device),
            torch.tensor(standoff[fi, mi], dtype=torch.float32, device=device))


MASS = np.exp(np.linspace(np.log(0.7), np.log(1.4), 5)).round(3)
FRICTION = np.exp(np.linspace(np.log(0.3), np.log(0.7), 4)).round(3)


def run_calibrate(args):
    """Per-grid-cell calibration at full 5x4 resolution.

    The 3x3 v2 table left grid points between bucket centers mistuned
    (flatness 2026-08-18: mean 0.574 PASS, spread 0.125 FAIL, with the weak
    marginals at inter-bucket physics values). Search speed x standoff per
    (mass, friction) grid cell at mid COM and write a finer table.
    """
    SPEEDS = [0.6, 0.7, 0.8, 0.9, 1.0]
    STANDOFFS = [0.13, 0.16, 0.19, 0.22]
    cells = [(m, f) for m in MASS for f in FRICTION]
    combos = [(sp, so) for sp in SPEEDS for so in STANDOFFS]
    n = len(cells) * len(combos)
    c_override = dict(
        mass_mult=np.array([m for (m, f) in cells for _ in combos]),
        friction=np.array([f for (m, f) in cells for _ in combos]),
        com_frac=np.full(n, 0.075),
    )
    env = gym.make("SlideToSlotT3-v1", num_envs=n, obs_mode="state",
                   control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                   randomize_c=False, c_override=c_override)
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    sp = torch.tensor([s for _ in cells for (s, _) in combos],
                      dtype=torch.float32, device=base.device)
    so = torch.tensor([s for _ in cells for (_, s) in combos],
                      dtype=torch.float32, device=base.device)

    success = np.zeros(n)
    for ep in range(args.episodes):
        env.reset(seed=100 + ep)
        seen = torch.zeros(n, dtype=torch.bool, device=base.device)
        for _ in range(args.horizon):
            _, _, _, _, info = env.step(policy(base, act_dim, sp, so))
            seen |= info["success"]
        success += seen.cpu().numpy()
        if (ep + 1) % 16 == 0:
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
                                  all=[[list(c), float(v)]
                                       for c, v in zip(combos, s)])
        print(f"mass {m} friction {f}: speed {combos[best][0]} "
              f"standoff {combos[best][1]} -> {s[best]:.3f}", flush=True)
    out = args.calib_out or os.path.join(REPORTS, "t3_flick_calibration_v3.json")
    json.dump(table, open(out, "w"), indent=1)
    json.dump(detail, open(out.replace(".json", "_detail.json"), "w"), indent=1)
    print(f"wrote {out}")


def run_flatness(args):
    COM = np.array([0.0, 0.075, 0.15])
    points = [dict(mass_mult=m, friction=f, com_frac=o)
              for m, f, o in itertools.product(MASS, FRICTION, COM)]
    n = len(points)
    c_override = {k: np.array([p[k] for p in points]) for k in points[0]}

    env = gym.make("SlideToSlotT3-v1", num_envs=n, obs_mode="state",
                   control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                   randomize_c=False, c_override=c_override)
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    sp, so = lookup(c_override, base.device, args.calib)

    success_once = np.zeros(n)
    for ep in range(args.episodes):
        env.reset(seed=ep)
        seen = torch.zeros(n, dtype=torch.bool, device=base.device)
        for _ in range(args.horizon):
            _, _, _, _, info = env.step(policy(base, act_dim, sp, so))
            seen |= info["success"]
        success_once += seen.cpu().numpy()
    env.close()
    success_once /= args.episodes

    spread = float(success_once.max() - success_once.min())
    result = dict(
        ckpt="scripted_teacher_t3 (calibrated c-aware)", episodes=args.episodes,
        mean_success=float(success_once.mean()), spread=spread,
        points=[dict(**p, success_once=float(s))
                for p, s in zip(points, success_once)],
    )
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
    with open(os.path.join(out_dir, f"{args.tag}.json"), "w") as fp:
        json.dump(result, fp, indent=1)

    # the chain's gate quantities: mean + worst per-axis marginal spread
    worst = 0.0
    for axis in ("mass_mult", "friction", "com_frac"):
        vals = sorted(set(p[axis] for p in result["points"]))
        m = [np.mean([p["success_once"] for p in result["points"] if p[axis] == v])
             for v in vals]
        print(f"{axis} marginals: {[round(float(x), 3) for x in m]}")
        worst = max(worst, max(m) - min(m))
    print(f"mean {success_once.mean():.3f}  worst marginal spread {worst:.3f}")
    print(f"GATE (mean>=0.55, spread<=0.10): "
          f"{'PASS' if success_once.mean() >= 0.55 and worst <= 0.10 else 'FAIL'}")


def run_dataset(args):
    n = args.num_envs
    cycles = (args.episodes + n - 1) // n
    env = gym.make("SlideToSlotT3-v1", num_envs=n, obs_mode="state",
                   control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                   reconfiguration_freq=1)
    os.makedirs(args.out, exist_ok=True)
    env = RecordEpisode(
        env, output_dir=args.out, save_trajectory=True, save_video=False,
        trajectory_name="trajectory", source_type="scripted-caware",
        source_desc="calibrated flick teacher, per-c speed/standoff from true c",
    )
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]

    keys = list(C_KEYS + GRANULAR_KEYS) + ["com_x_frac", "com_y_frac", "mass_kg"]
    c_rows, success_rows = [], []
    for r in range(cycles):
        env.reset(seed=args.seed + r)
        c = base.get_c()
        sp, so = lookup(c, base.device, args.calib)
        seen = torch.zeros(n, dtype=torch.bool, device=base.device)
        for _ in range(args.horizon):
            _, _, _, _, info = env.step(policy(base, act_dim, sp, so))
            seen |= info["success"]
        for i in range(n):
            c_rows.append([float(c[k][i]) for k in keys])
        success_rows.extend(seen.cpu().numpy().tolist())
        if (r + 1) % 10 == 0:
            print(f"cycle {r + 1}/{cycles} "
                  f"running success {np.mean(success_rows):.3f}", flush=True)
    env.close()

    succ = np.asarray(success_rows, dtype=bool)
    summary = dict(env_id="SlideToSlotT3-v1", episodes=len(succ), num_envs=n,
                   success_once_rate=float(succ.mean()),
                   teacher="scripted-caware-calibrated")
    np.savez(os.path.join(args.out, "c_metadata.npz"),
             c=np.asarray(c_rows, dtype=np.float32), c_keys=np.array(keys),
             success_once=succ, episodes_per_cycle=n, cycles=cycles,
             seed=args.seed)
    with open(os.path.join(args.out, "harness_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=1)
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flatness", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calib", default=None,
                        help="calibration table path (default v2 table)")
    parser.add_argument("--calib-out", default=None,
                        help="where --calibrate writes its table")
    parser.add_argument("--tag", default="flat_scripted_aware")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.calibrate:
        run_calibrate(args)
    elif args.flatness:
        run_flatness(args)
    else:
        assert args.out, "--out required for dataset generation"
        run_dataset(args)


if __name__ == "__main__":
    main()
