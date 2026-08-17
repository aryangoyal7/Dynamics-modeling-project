"""Rollout harness (plan Part II 'Build: data'): sample c, run the privileged
teacher, record observations and actions, store c as metadata only.

The environment is the STUDENT env (expose_c=False), so nothing recorded ever
contains c. The teacher checkpoint was trained on the Teacher twin, whose
observation is exactly [student_obs, c] (c_params is appended last in
_get_obs_extra), so the teacher input is rebuilt each step by concatenating
the env's private c tensor to the recorded observation.

reconfiguration_freq=1 resamples c on every reset, so each rollout cycle of
num_envs parallel episodes carries fresh hidden parameters.

Trajectories go to ManiSkill's HDF5 trajectory format (replayable later into
other obs/control modes, e.g. the vision variant); c goes to a sidecar
c_metadata.npz aligned with trajectory ids, plus a summary json.

Episodes are recorded at full horizon; the teacher was trained with
termination-on-success, so frames after the first success are post-success
drift, not expert behaviour. The per-step `success` flags are stored in the
HDF5 - student dataloaders must truncate each trajectory at first success.

    python -m dynmod.scripts.rollout_harness \
        --ckpt runs/t3-teacher-v1/final_ckpt.pt --episodes 1000 \
        --out /mnt/scratch/dynamics/data/t3_1e3
"""

from __future__ import annotations

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.envs.randomization import C_KEYS, GRANULAR_KEYS
from dynmod.models import PPOAgent
from mani_skill.utils.wrappers import RecordEpisode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--env-id", default="SlideToSlotT3-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample teacher actions instead of the mean")
    args = parser.parse_args()

    n = args.num_envs
    cycles = (args.episodes + n - 1) // n
    os.makedirs(args.out, exist_ok=True)

    env = gym.make(
        args.env_id, num_envs=n, obs_mode="state",
        control_mode=args.control_mode, sim_backend="physx_cuda",
        reconfiguration_freq=1,
    )
    env = RecordEpisode(
        env, output_dir=args.out, save_trajectory=True, save_video=False,
        trajectory_name="trajectory", source_type="ppo-teacher",
        source_desc=f"privileged teacher {os.path.basename(args.ckpt)}",
    )
    base = env.unwrapped
    device = base.device
    agent = PPOAgent.load(args.ckpt, device=device)
    teacher_obs_dim = agent.actor_mean[0].in_features

    c_rows, success_rows = [], []
    keys = list(C_KEYS + GRANULAR_KEYS) + ["com_x_frac", "com_y_frac", "mass_kg"]
    for r in range(cycles):
        obs, _ = env.reset(seed=args.seed + r)
        assert obs.shape[-1] + base.c_tensor.shape[-1] == teacher_obs_dim, (
            f"teacher expects {teacher_obs_dim} dims, student obs "
            f"{obs.shape[-1]} + c {base.c_tensor.shape[-1]} must match"
        )
        c = base.get_c()
        seen = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(args.horizon):
            act = agent.act(torch.cat([obs, base.c_tensor], dim=-1),
                            deterministic=not args.stochastic)
            obs, _, _, _, info = env.step(act)
            seen |= info["success"]
        for i in range(n):
            c_rows.append([float(c[k][i]) for k in keys])
        success_rows.extend(seen.cpu().numpy().tolist())
    env.close()  # flushes the final cycle's trajectories

    c_arr = np.asarray(c_rows, dtype=np.float32)
    succ = np.asarray(success_rows, dtype=bool)
    np.savez(
        os.path.join(args.out, "c_metadata.npz"),
        c=c_arr, c_keys=np.array(keys), success_once=succ,
        episodes_per_cycle=n, cycles=cycles, seed=args.seed,
    )
    summary = dict(
        ckpt=args.ckpt, env_id=args.env_id, episodes=len(succ),
        num_envs=n, cycles=cycles, horizon=args.horizon,
        success_once_rate=float(succ.mean()), stochastic=bool(args.stochastic),
    )
    with open(os.path.join(args.out, "harness_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=1)
    print(json.dumps(summary, indent=1))

    # cross-check: the recorder's episode count must match our metadata rows
    traj_json = os.path.join(args.out, "trajectory.json")
    if os.path.exists(traj_json):
        with open(traj_json) as fp:
            meta = json.load(fp)
        n_rec = len(meta.get("episodes", []))
        print(f"recorded {n_rec} trajectories; c metadata rows {len(succ)}")
        if n_rec != len(succ):
            print("WARNING: trajectory/metadata count mismatch - inspect before use")


if __name__ == "__main__":
    main()
