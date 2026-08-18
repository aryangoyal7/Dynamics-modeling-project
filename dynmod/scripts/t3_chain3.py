"""T3 pipeline, chain 3: longer-trained students.

Chain 2's base-fails gate came back interp 0.225 / far 0.098: the policy
DOES degrade far out (-56% relative) but the 60k-step quick-train leaves it
too weak everywhere for the 20-point absolute bar (teacher: 0.64). The T4
control task passed at 60k because its students train easily; T3's flick
behaviour needs longer. This chain retrains the gate policy at 150k steps,
re-checks the gate, then runs the arm matrix at 150k steps per arm.

    nohup python -m dynmod.scripts.t3_chain3 > \
        /mnt/scratch/dynamics/t3_chain3.log 2>&1 &
"""

from __future__ import annotations

import os
import subprocess
import sys

PY = sys.executable
DATA = "/mnt/scratch/dynamics/data"
STEPS = "150000"
BACKUP = "/home/azureuser/cloudfiles/code/Users/garyan18/dynamics_modeling/scratch_backup"


def backup(path):
    import shutil
    dst = os.path.join(BACKUP, "data", os.path.basename(path))
    print(f"[chain3] backing up {path} -> {dst}", flush=True)
    shutil.copytree(path, dst, dirs_exist_ok=True)


def sh(cmd, gpu=None, bg=False):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[chain3] $ {' '.join(cmd)}", flush=True)
    if bg:
        return subprocess.Popen(cmd, env=env)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"[chain3] STOP: command failed rc={r.returncode}", flush=True)
        sys.exit(1)


def main():
    sh([PY, "-m", "dynmod.policy.train", "--data", f"{DATA}/t3_1e4",
        "--arm", "A", "--seed", "0", "--steps", STEPS,
        "--out-name", "t3gate-A-s0-150k"], gpu=4)
    r = subprocess.run([PY, "-m", "dynmod.policy.evaluate",
        "--ckpt", "/mnt/scratch/dynamics/policy_runs/t3gate-A-s0-150k/final_ckpt.pt",
        "--episodes", "20", "--gate", "--control-mode", "pd_ee_delta_pos",
        "--name", "t3gate-A-s0-150k"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="4"))
    if r.returncode != 0:
        print("[chain3] STOP: base policy still does not fail far out at "
              "150k. Decide: longer training, extended grid, or shorter "
              "history - bring numbers to the user.", flush=True)
        sys.exit(3)
    print("[chain3] base-fails gate PASSED at 150k - launching arms", flush=True)

    p5 = sh([PY, "-m", "dynmod.scripts.scripted_teacher_t3",
             "--episodes", "100000", "--num-envs", "250", "--seed", "20000",
             "--calib", "reports/t3_flick_calibration_v4.json",
             "--out", f"{DATA}/t3_1e5"], gpu=6, bg=True)
    sh([PY, "-m", "dynmod.scripts.launch_main_arms", "--prefix", "t3",
        "--scales", "1e3", "1e4", "--seeds", "10", "--steps", STEPS,
        "--gpus", "0", "1", "2", "5", "7", "--jobs-per-gpu", "3"])
    if p5.wait() != 0:
        print("[chain3] STOP: 1e5 generation failed", flush=True)
        sys.exit(1)
    backup(f"{DATA}/t3_1e5")
    sh([PY, "-m", "dynmod.scripts.launch_main_arms", "--prefix", "t3",
        "--scales", "1e3", "1e4", "1e5", "--seeds", "10", "--steps", STEPS,
        "--gpus", "0", "1", "2", "4", "5", "6", "7", "--jobs-per-gpu", "3"])
    import shutil
    shutil.copytree("/mnt/scratch/dynamics/policy_runs",
                    os.path.join(BACKUP, "policy_runs"), dirs_exist_ok=True)
    print("[chain3] COMPLETE: all T3 arms trained and backed up.", flush=True)


if __name__ == "__main__":
    main()
