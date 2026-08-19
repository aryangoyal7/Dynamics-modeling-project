"""Queue the main study arms across GPUs (plan Part II 'Build: training':
'Launch the main arms: A, B across three targets, B-shuffled, C. Ten seeds,
three data scales.').

Builds the full preregistered run matrix and dispatches one training job per
free GPU until the queue drains. Run AFTER the teacher flatness gate passes
and the datasets exist. Safe to re-run: completed runs (final_ckpt.pt
present) are skipped, so a crashed queue resumes where it left off.

    python -m dynmod.scripts.launch_main_arms \
        --data-root /mnt/scratch/dynamics/data --prefix t3 \
        --scales 1e3 1e4 1e5 --seeds 10 --gpus 0 1 2 3 4 5 6 7

    # dry run: print the matrix and exit
    python -m dynmod.scripts.launch_main_arms --data-root ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ARMS = [
    ("A", "multistep"),  # target irrelevant for A (no pred head in loss)
    ("B", "onestep"),
    ("B", "multistep"),
    ("B", "latent"),
    ("Bshuf", "multistep"),
    ("C", "multistep"),
]
OUT_ROOT = "/mnt/scratch/dynamics/policy_runs"


def build_matrix(data_root, prefix, scales, seeds, steps, layout="t3"):
    runs = []
    for scale in scales:
        data = os.path.join(data_root, f"{prefix}_{scale}")
        for arm, target in ARMS:
            for seed in range(seeds):
                name = f"{prefix}-{scale}-{arm}-{target}-s{seed}"
                runs.append(dict(
                    name=name,
                    cmd=[
                        sys.executable, "-m", "dynmod.policy.train",
                        "--data", data, "--arm", arm, "--target", target,
                        "--seed", str(seed), "--steps", str(steps),
                        "--layout", layout, "--out-name", name,
                    ],
                ))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/mnt/scratch/dynamics/data")
    p.add_argument("--prefix", default="t3")
    p.add_argument("--scales", nargs="+", default=["1e3", "1e4", "1e5"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--jobs-per-gpu", type=int, default=3,
                   help="the ~1M-param runs use a few GB each; an 80GB A100 "
                        "packs several")
    p.add_argument("--track", action="store_true",
                   help="pass --track (Weights & Biases) to every run")
    p.add_argument("--layout", default="t3",
                   help="observation layout of the task (see "
                        "dynmod.policy.data.OBS_LAYOUTS)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    runs = build_matrix(args.data_root, args.prefix, args.scales, args.seeds,
                        args.steps, args.layout)
    pending = [r for r in runs
               if not os.path.exists(os.path.join(OUT_ROOT, r["name"], "final_ckpt.pt"))]
    print(f"{len(runs)} runs in matrix, {len(runs) - len(pending)} already "
          f"complete, {len(pending)} to launch on GPUs {args.gpus}")
    if args.dry_run:
        for r in pending:
            print(" ", r["name"])
        return
    for scale in args.scales:
        d = os.path.join(args.data_root, f"{args.prefix}_{scale}", "trajectory.h5")
        assert os.path.exists(d), f"dataset missing: {d} (gate not passed yet?)"

    slots = [(g, s) for g in args.gpus for s in range(args.jobs_per_gpu)]
    active = {}  # (gpu, slot) -> (proc, name)
    queue = list(pending)
    log_dir = os.path.join(OUT_ROOT, "launcher_logs")
    os.makedirs(log_dir, exist_ok=True)
    while queue or active:
        for key in list(active):
            proc, name = active[key]
            if proc.poll() is not None:
                status = "done" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"[gpu {key[0]}] {name}: {status} ({len(queue)} left)", flush=True)
                del active[key]
        for key in slots:
            if key not in active and queue:
                r = queue.pop(0)
                cmd = r["cmd"] + (["--track"] if args.track else [])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(key[0]))
                log = open(os.path.join(log_dir, r["name"] + ".log"), "w")
                proc = subprocess.Popen(cmd, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
                active[key] = (proc, r["name"])
                print(f"[gpu {key[0]}] launched {r['name']}", flush=True)
        time.sleep(20)
    print("queue drained")


if __name__ == "__main__":
    main()
