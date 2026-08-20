"""T8 full-fledged program: certify -> datasets -> arm matrix -> evaluations.

Run once the T8 knowledge-premium gate has passed (user instruction
2026-08-19: "if the task passes our test, start with the full-fledged policy
training and evaluation program"). Every stage is a gate: nothing downstream
starts unless the stage above it cleared, and a failure stops the chain with
a reason instead of quietly producing a dataset nobody can trust.

  1. premium gate check  reports/premium_gate_t8.json: aware >= 0.55 healthy
                         success AND premium >= 0.10
  2. teacher flatness    mean >= 0.55 and worst per-axis marginal spread
                         <= 0.10 (the bar every teacher in this study met)
  3. datasets            10^3 and 10^4 demonstrations, each backed up to
                         persistent storage the moment it completes
  4. arm matrix          A, B x 3 targets, B-shuffled, C x 10 seeds x 2 scales
  5. evaluations         4 sharded train_eval_loop workers; shard 0 writes
                         the slope summary when the matrix drains

    nohup python -m dynmod.scripts.t8_chain > /mnt/scratch/dynamics/t8_chain.log &
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS = os.path.join(ROOT, "reports")
DATA = "/mnt/scratch/dynamics/data"
BACKUP = "/home/azureuser/cloudfiles/code/Users/garyan18/dynamics_modeling/scratch_backup"
CALIB = os.path.join(REPORTS, "t8_seat_calibration_v5.json")
ENV_ID = "StackCarryT8-v1"
HORIZON = 140
STEPS = 150000  # 60k left T3 students too weak to have a gap to attribute
SCALES = [("1e3", 1000), ("1e4", 10000)]


def sh(cmd, gpu=None, **kw):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[chain] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, cwd=ROOT, **kw)


def stop(msg):
    print(f"[chain] STOPPED: {msg}", flush=True)
    raise SystemExit(1)


def step1_premium_gate():
    while subprocess.run(["pgrep", "-f", "premium[_]gate_t8"],
                         capture_output=True).returncode == 0:
        print("[chain] waiting for the premium gate to finish...", flush=True)
        time.sleep(60)
    path = os.path.join(REPORTS, "premium_gate_t8.json")
    if not os.path.exists(path):
        stop("premium gate produced no report - reports/premium_gate_t8.json missing")
    r = json.load(open(path))["result"]
    print(f"[chain] premium gate: aware {r['aware']:.3f} blind {r['blind']:.3f} "
          f"premium {r['premium']:+.3f} ± {r['ci95']:.3f} "
          f"({r.get('episodes', '?')} episodes/arm)", flush=True)
    if r["aware"] < 0.55:
        stop(f"aware success {r['aware']:.3f} below the 0.55 healthy bar")
    # the premium must clear its own 95% interval: a task earns a teacher and
    # 120 training runs only if knowing c demonstrably beats not knowing it,
    # not if the point estimate merely looks positive
    if r["premium"] <= r["ci95"]:
        stop(f"premium {r['premium']:+.3f} does not clear its CI ±{r['ci95']:.3f} "
             "- knowledge is not measurably worth anything here, do not invest")
    if r["premium"] < 0.10:
        stop(f"premium {r['premium']:+.3f} below the 0.10 bar - too small to "
             "carry a robustness study, even if statistically nonzero")
    print("[chain] premium gate PASSED", flush=True)


def step2_flatness():
    tag = "flat_t8_aware_v5"
    out = os.path.join(REPORTS, f"{tag}.json")
    if not os.path.exists(out):
        r = sh([PY, "-m", "dynmod.scripts.scripted_teacher_t8", "--flatness",
                "--episodes", "32", "--calib", CALIB, "--tag", tag], gpu=5)
        if r.returncode != 0:
            stop("flatness run failed")
    d = json.load(open(out))
    pts = d["points"]
    worst = 0.0
    for axis in ("mass_mult", "friction", "com_frac"):
        vals = sorted({p[axis] for p in pts})
        marg = [sum(p["success_once"] for p in pts if p[axis] == v)
                / sum(1 for p in pts if p[axis] == v) for v in vals]
        print(f"[chain] {axis} marginals "
              f"{[round(x, 3) for x in marg]}", flush=True)
        worst = max(worst, max(marg) - min(marg))
    mean = d["mean_success"]
    print(f"[chain] teacher mean {mean:.3f} worst marginal spread {worst:.3f}",
          flush=True)
    if mean < 0.55 or worst > 0.10:
        stop(f"teacher NOT certified (mean {mean:.3f} needs >=0.55, spread "
             f"{worst:.3f} needs <=0.10) - recalibrate before generating data")
    print("[chain] teacher CERTIFIED", flush=True)


def step3_datasets():
    for scale, n in SCALES:
        out = os.path.join(DATA, f"t8_{scale}")
        if os.path.exists(os.path.join(out, "c_metadata.npz")):
            print(f"[chain] dataset t8_{scale} already complete", flush=True)
            continue
        r = sh([PY, "-m", "dynmod.scripts.scripted_teacher_t8",
                "--episodes", str(n), "--seed", "0", "--calib", CALIB,
                "--out", out], gpu=5)
        if r.returncode != 0:
            stop(f"dataset generation failed for {scale}")
        # back up immediately: scratch is wiped on every machine restart
        sh(["cp", "-r", out, os.path.join(BACKUP, "data")])
        print(f"[chain] dataset t8_{scale} done and backed up", flush=True)


def step4_arms():
    r = sh([PY, "-m", "dynmod.scripts.launch_main_arms", "--prefix", "t8",
            "--layout", "t8", "--scales", *[s for s, _ in SCALES],
            "--seeds", "10", "--steps", str(STEPS),
            "--gpus", "0", "2", "3", "4", "5", "6", "7",
            "--jobs-per-gpu", "3"])
    if r.returncode != 0:
        stop("arm matrix launcher failed")
    print("[chain] arm matrix drained", flush=True)


def step5_evals():
    procs = []
    for shard in range(4):
        log = open(f"/mnt/scratch/dynamics/t8_eval_shard{shard}.log", "w")
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=str([0, 2, 5, 7][shard]))
        procs.append(subprocess.Popen(
            [PY, "-m", "dynmod.scripts.train_eval_loop", "--prefix", "t8",
             "--env-id", ENV_ID, "--horizon", str(HORIZON),
             "--shard", str(shard), "--shards", "4"],
            env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print("[chain] evaluations complete - see reports/t8_slopes_summary.txt",
          flush=True)


if __name__ == "__main__":
    step1_premium_gate()
    step2_flatness()
    step3_datasets()
    step4_arms()
    step5_evals()
    print("[chain] T8 PROGRAM COMPLETE", flush=True)
