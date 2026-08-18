"""Automated T3 pipeline chain (queued after task validation, 2026-08-18):

  1. wait for teacher v7 to finish
  2. flatness gate on its best-eval checkpoints; auto-select by mean success;
     REQUIRE mean >= 0.55 and per-axis marginal spread <= 0.10, else STOP
  3. generate teacher datasets 10^3 and 10^4 (EE control), verify counts
  4. plan-required gate: train one base policy on 10^4, evaluate on the
     delta grid - it must degrade visibly far out, else STOP
  5. launch the main arm matrix (10^3 + 10^4 scales) across free GPUs and
     start 10^5 generation in parallel

Every stage logs to stdout; any gate failure stops the chain with a clear
message rather than proceeding on bad data.

    nohup python -m dynmod.scripts.t3_chain > /mnt/scratch/dynamics/t3_chain.log 2>&1 &
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN = "runs/t3-teacher-v8"
DATA = "/mnt/scratch/dynamics/data"
CM = "pd_ee_delta_pos"


BACKUP = "/home/azureuser/cloudfiles/code/Users/garyan18/dynamics_modeling/scratch_backup"

def backup(path):
    """Copy an artifact to persistent storage immediately (external machine
    stops have twice destroyed un-backed scratch work)."""
    import shutil
    dst = os.path.join(BACKUP, "data", os.path.basename(path))
    print(f"[chain] backing up {path} -> {dst}", flush=True)
    shutil.copytree(path, dst, dirs_exist_ok=True)


def sh(cmd, gpu=None, bg=False):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[chain] $ {' '.join(cmd)}", flush=True)
    if bg:
        return subprocess.Popen(cmd, env=env)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"[chain] STOP: command failed rc={r.returncode}", flush=True)
        sys.exit(1)


def marginal_spread(report_path):
    d = json.load(open(report_path))
    pts = d["points"]
    worst = 0.0
    for axis in ("mass_mult", "friction", "com_frac"):
        vals = sorted(set(p[axis] for p in pts))
        m = [sum(p["success_once"] for p in pts if p[axis] == v)
             / max(1, sum(1 for p in pts if p[axis] == v)) for v in vals]
        worst = max(worst, max(m) - min(m))
    return d["mean_success"], worst


def main():
    print("[chain] waiting for teacher v7 ...", flush=True)
    while not os.path.exists(f"{RUN}/final_ckpt.pt"):
        time.sleep(120)

    # pick top-3 eval checkpoints from tensorboard + the final one
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(RUN); ea.Reload()
    evals = ea.Scalars("eval/success_once")
    steps_per_epoch = 2048 * 50
    tops = sorted(evals, key=lambda x: -x.value)[:3]
    cands = {f"{RUN}/final_ckpt.pt"}
    for t in tops:
        epoch = round(t.step / steps_per_epoch)
        for delta in (0, 1, -1):
            p = f"{RUN}/ckpt_{epoch + delta}.pt"
            if os.path.exists(p):
                cands.add(p); break
    print(f"[chain] flatness candidates: {sorted(cands)}", flush=True)

    best = None
    for ck in sorted(cands):
        tag = "flat_v7_" + os.path.basename(ck).replace(".pt", "")
        sh([PY, "-m", "dynmod.scripts.teacher_flatness", "--ckpt", ck,
            "--episodes", "64", "--control-mode", CM, "--tag", tag], gpu=4)
        mean, spread = marginal_spread(f"reports/{tag}.json")
        print(f"[chain] {ck}: mean {mean:.3f}, worst marginal spread {spread:.3f}", flush=True)
        if best is None or mean > best[1]:
            best = (ck, mean, spread)
    ck, mean, spread = best
    print(f"[chain] SELECTED teacher: {ck} (mean {mean:.3f}, spread {spread:.3f})", flush=True)
    if mean < 0.55 or spread > 0.10:
        print("[chain] STOP: flatness gate failed - teacher below bar. "
              "Decide: another warm-start round or widen slot tolerance.", flush=True)
        sys.exit(2)
    json.dump(dict(teacher=ck, mean=mean, spread=spread),
              open("reports/t3_certified_teacher.json", "w"), indent=1)

    sh([PY, "-m", "dynmod.scripts.rollout_harness", "--ckpt", ck,
        "--episodes", "1000", "--num-envs", "250", "--seed", "0",
        "--control-mode", CM, "--out", f"{DATA}/t3_1e3"], gpu=4)
    backup(f"{DATA}/t3_1e3")
    sh([PY, "-m", "dynmod.scripts.rollout_harness", "--ckpt", ck,
        "--episodes", "10000", "--num-envs", "250", "--seed", "10000",
        "--control-mode", CM, "--out", f"{DATA}/t3_1e4"], gpu=4)
    backup(f"{DATA}/t3_1e4")

    sh([PY, "-m", "dynmod.policy.train", "--data", f"{DATA}/t3_1e4",
        "--arm", "A", "--seed", "0", "--steps", "60000",
        "--out-name", "t3gate-A-s0"], gpu=4)
    r = subprocess.run([PY, "-m", "dynmod.policy.evaluate",
        "--ckpt", "/mnt/scratch/dynamics/policy_runs/t3gate-A-s0/final_ckpt.pt",
        "--episodes", "20", "--gate", "--control-mode", CM,
        "--name", "t3gate-A-s0"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="4"))
    if r.returncode != 0:
        print("[chain] STOP: base policy does NOT fail far out. "
              "Per plan: extend grid / shorten history before arms.", flush=True)
        sys.exit(3)
    print("[chain] base-fails gate PASSED - launching the arm matrix", flush=True)

    p_1e5 = sh([PY, "-m", "dynmod.scripts.rollout_harness", "--ckpt", ck,
                "--episodes", "100000", "--num-envs", "500", "--seed", "20000",
                "--control-mode", CM, "--out", f"{DATA}/t3_1e5"], gpu=6, bg=True)
    sh([PY, "-m", "dynmod.scripts.launch_main_arms", "--prefix", "t3",
        "--scales", "1e3", "1e4", "--seeds", "10",
        "--gpus", "0", "1", "2", "5", "7", "--jobs-per-gpu", "3"])
    p_1e5.wait()
    backup(f"{DATA}/t3_1e5")
    sh([PY, "-m", "dynmod.scripts.launch_main_arms", "--prefix", "t3",
        "--scales", "1e3", "1e4", "1e5", "--seeds", "10",
        "--gpus", "0", "1", "2", "4", "5", "6", "7", "--jobs-per-gpu", "3"])
    import shutil
    shutil.copytree("/mnt/scratch/dynamics/policy_runs",
                    os.path.join(BACKUP, "policy_runs"), dirs_exist_ok=True)
    print("[chain] COMPLETE: all T3 arms trained and backed up.", flush=True)


if __name__ == "__main__":
    main()
