"""T3 pipeline, chain 2: run with the CERTIFIED scripted c-aware teacher.

The RL teachers plateaued below the bar (v8 grid mean 0.36, v9 unstable);
the calibrated scripted teacher with the fine 5x4 table PASSED both gates on
2026-08-18: mean 0.634 (>= 0.55), worst marginal spread 0.072 (<= 0.10),
report flat_scripted_aware_v4.json. This chain runs the rest of the plan:

  1. generate 10^3 + 10^4 teacher datasets IN PARALLEL (GPUs 4, 5),
     backing each up to persistent storage the moment it finishes
  2. plan-required gate: train a base policy on 10^4, verify it fails
     far out on the delta grid (else STOP)
  3. start 10^5 generation (GPU 6) and the 10^3/10^4 arm matrix together;
     when 10^5 lands, back it up and extend the matrix to all scales
  4. back up all policy runs

    nohup python -m dynmod.scripts.t3_chain2 > \
        /mnt/scratch/dynamics/t3_chain2.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

PY = sys.executable
DATA = "/mnt/scratch/dynamics/data"
CALIB = "reports/t3_flick_calibration_v4.json"
BACKUP = "/home/azureuser/cloudfiles/code/Users/garyan18/dynamics_modeling/scratch_backup"


def backup(path):
    import shutil
    dst = os.path.join(BACKUP, "data", os.path.basename(path))
    print(f"[chain2] backing up {path} -> {dst}", flush=True)
    shutil.copytree(path, dst, dirs_exist_ok=True)


def sh(cmd, gpu=None, bg=False):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[chain2] $ {' '.join(cmd)}", flush=True)
    if bg:
        return subprocess.Popen(cmd, env=env)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"[chain2] STOP: command failed rc={r.returncode}", flush=True)
        sys.exit(1)


def gen_cmd(episodes, seed, out):
    return [PY, "-m", "dynmod.scripts.scripted_teacher_t3",
            "--episodes", str(episodes), "--num-envs", "250",
            "--seed", str(seed), "--calib", CALIB, "--out", out]


def main():
    json.dump(dict(teacher="scripted-caware-calibration-v4",
                   flatness_report="reports/flat_scripted_aware_v4.json",
                   mean=0.634, spread=0.072),
              open("reports/t3_certified_teacher.json", "w"), indent=1)

    # stage 1: 1e3 + 1e4 in parallel
    p3 = sh(gen_cmd(1000, 0, f"{DATA}/t3_1e3"), gpu=4, bg=True)
    p4 = sh(gen_cmd(10000, 10000, f"{DATA}/t3_1e4"), gpu=5, bg=True)
    if p3.wait() != 0:
        print("[chain2] STOP: 1e3 generation failed", flush=True); sys.exit(1)
    backup(f"{DATA}/t3_1e3")
    if p4.wait() != 0:
        print("[chain2] STOP: 1e4 generation failed", flush=True); sys.exit(1)
    backup(f"{DATA}/t3_1e4")

    # stage 2: base policy must fail far out on the delta grid
    sh([PY, "-m", "dynmod.policy.train", "--data", f"{DATA}/t3_1e4",
        "--arm", "A", "--seed", "0", "--steps", "60000",
        "--out-name", "t3gate-A-s0"], gpu=4)
    r = subprocess.run([PY, "-m", "dynmod.policy.evaluate",
        "--ckpt", "/mnt/scratch/dynamics/policy_runs/t3gate-A-s0/final_ckpt.pt",
        "--episodes", "20", "--gate", "--control-mode", "pd_ee_delta_pos",
        "--name", "t3gate-A-s0"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="4"))
    if r.returncode != 0:
        print("[chain2] STOP: base policy does NOT fail far out. "
              "Per plan: extend grid / shorten history before arms.", flush=True)
        sys.exit(3)
    print("[chain2] base-fails gate PASSED - launching the arm matrix", flush=True)

    # stage 3: 1e5 generation alongside the 1e3/1e4 arm matrix
    p5 = sh(gen_cmd(100000, 20000, f"{DATA}/t3_1e5"), gpu=6, bg=True)
    sh([PY, "-m", "dynmod.scripts.launch_main_arms", "--prefix", "t3",
        "--scales", "1e3", "1e4", "--seeds", "10",
        "--gpus", "0", "1", "2", "5", "7", "--jobs-per-gpu", "3"])
    if p5.wait() != 0:
        print("[chain2] STOP: 1e5 generation failed", flush=True); sys.exit(1)
    backup(f"{DATA}/t3_1e5")
    sh([PY, "-m", "dynmod.scripts.launch_main_arms", "--prefix", "t3",
        "--scales", "1e3", "1e4", "1e5", "--seeds", "10",
        "--gpus", "0", "1", "2", "4", "5", "6", "7", "--jobs-per-gpu", "3"])
    import shutil
    shutil.copytree("/mnt/scratch/dynamics/policy_runs",
                    os.path.join(BACKUP, "policy_runs"), dirs_exist_ok=True)
    print("[chain2] COMPLETE: all T3 arms trained and backed up.", flush=True)


if __name__ == "__main__":
    main()
