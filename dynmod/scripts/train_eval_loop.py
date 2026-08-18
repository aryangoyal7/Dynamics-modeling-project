"""Continuous train->evaluate loop (user request 2026-08-18).

Watches the policy-runs directory while an arm matrix trains (chain 3 or
any launcher). Every run matching --prefix that has a final_ckpt.pt and no
evaluation report yet is evaluated on the full delta grid immediately, on
this process's GPU - so evaluation overlaps training instead of waiting
for the whole matrix. When the matrix launcher is gone and nothing is left
to evaluate, it runs the slope comparisons (each B variant vs A, per data
scale) and writes reports/<prefix>_slopes_summary.txt, then exits.

Works for any verified task: --prefix t3 --env-id SlideToSlotT3-v1 today;
point it at another prefix/env when that task passes its gates.

    CUDA_VISIBLE_DEVICES=3 nohup python -m dynmod.scripts.train_eval_loop \
        --prefix t3 > /mnt/scratch/dynamics/t3_eval_loop.log 2>&1 &
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

PY = sys.executable
RUNS = "/mnt/scratch/dynamics/policy_runs"
REPORTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
BACKUP = "/home/azureuser/cloudfiles/code/Users/garyan18/dynamics_modeling/scratch_backup"


def matrix_running(prefix):
    r = subprocess.run(["pgrep", "-f", "launch[_]main_arms|t3[_]chain"],
                       capture_output=True)
    return r.returncode == 0


def pending(prefix, shard=0, shards=1):
    import zlib
    out = []
    for ck in sorted(glob.glob(f"{RUNS}/{prefix}-*/final_ckpt.pt")):
        name = os.path.basename(os.path.dirname(ck))
        if "gate" in name:
            continue
        if zlib.crc32(name.encode()) % shards != shard:
            continue
        if not os.path.exists(f"{REPORTS}/policy_eval_{name}.json"):
            out.append((name, ck))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="t3")
    p.add_argument("--env-id", default="SlideToSlotT3-v1")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--control-mode", default="pd_ee_delta_pos")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()

    quiet_sweeps = 0
    n_done = 0
    while True:
        todo = pending(args.prefix, args.shard, args.shards)
        if not todo:
            # shard 0 owns the final slopes step: it must also wait out the
            # OTHER shards' remaining evaluations
            others = pending(args.prefix, 0, 1) if args.shard == 0 else []
            if not matrix_running(args.prefix) and not others:
                quiet_sweeps += 1
                if quiet_sweeps >= 2:
                    break
            time.sleep(120)
            continue
        quiet_sweeps = 0
        for name, ck in todo:
            print(f"[loop] evaluating {name}", flush=True)
            r = subprocess.run([PY, "-m", "dynmod.policy.evaluate",
                                "--ckpt", ck, "--env-id", args.env_id,
                                "--episodes", str(args.episodes),
                                "--control-mode", args.control_mode,
                                "--name", name])
            if r.returncode != 0:
                print(f"[loop] WARNING: evaluation failed for {name}", flush=True)
            else:
                n_done += 1
                # back up the small eval run artifacts as we go
                subprocess.run(["cp", "-ru", os.path.dirname(ck),
                                os.path.join(BACKUP, "policy_runs")],
                               capture_output=True)
    if args.shard != 0:
        print(f"[loop] shard {args.shard} done ({n_done} evals)", flush=True)
        return
    print(f"[loop] all arms evaluated ({n_done} this session) - slopes", flush=True)

    scales = sorted({os.path.basename(f).split("-")[1]
                     for f in glob.glob(f"{REPORTS}/policy_eval_{args.prefix}-*.json")})
    with open(f"{REPORTS}/{args.prefix}_slopes_summary.txt", "w") as out:
        for scale in scales:
            a = sorted(glob.glob(
                f"{REPORTS}/policy_eval_{args.prefix}-{scale}-A-*.json"))
            for var in ("B-onestep", "B-multistep", "B-latent",
                        "Bshuf-multistep", "C-multistep"):
                arm, target = var.split("-")
                b = sorted(glob.glob(
                    f"{REPORTS}/policy_eval_{args.prefix}-{scale}-{arm}-{target}-*.json"))
                if not a or not b:
                    continue
                r = subprocess.run(
                    [PY, "-m", "dynmod.analysis.slopes", *a, "--vs", *b,
                     "--label-a", "A", "--label-b", var],
                    capture_output=True, text=True)
                out.write(f"=== {scale}: A vs {var} ===\n{r.stdout}\n")
                print(f"[loop] {scale}: A vs {var} done", flush=True)
    print("[loop] COMPLETE - slopes summary written", flush=True)


if __name__ == "__main__":
    main()
