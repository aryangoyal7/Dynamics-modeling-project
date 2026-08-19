"""Dataset over ManiSkill HDF5 trajectories for the flow-policy arms.

Provides per-sample:
  hist_next  (K+1, obs_dim)  observation window ending at t+1; the model uses
                             frames [:K] as h_t and [1:] as h_{t+1} (latent target)
  chunk      (H_a, act_dim)  action chunk from t (flow-matching target)
  future_obj (H_p, obj_dim)  the OBJECT's states t+1..t+H_p (prediction targets;
                             never the robot's joint configuration)
  prev_act   (act_dim,)      executed action at t-1 (arm C conditioning)

Trajectories are truncated at first success + a small settle margin: the
teacher was trained with termination-on-success, so later frames are drift,
not expert behaviour. Actions are clamped to [-1, 1] (what the env executed).

Same dataset serves every arm; target type and shuffling live in the loss.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# verified against SlideToSlotT3-v1 state obs (see dynmod/README.md):
# qpos[0:9] qvel[9:18] tcp_pose[18:25] obj_pose[25:32] obj_lin_vel[32:35]
# obj_ang_vel[35:38] slot_pos[38:41]
OBS_LAYOUT_T3 = dict(obs_dim=41, obj_slice=(25, 38))

# StackCarryT8-v1 carries two objects, so the layout differs: the first 38
# dims are as above with the BEAM as obj, then the block's 13-d state
# (pose[38:45] lin_vel[45:48] ang_vel[48:51]) and goal_pos[51:54]. The
# prediction head is pointed at the BLOCK - the beam is rigidly gripped
# during the carry, so its motion is the arm's, not the physics'.
OBS_LAYOUT_T8 = dict(obs_dim=54, obj_slice=(38, 51))

# T4 (PickPlaceT4-v1) matches T3's dimensions: 38 + goal_pos(3).
OBS_LAYOUTS = dict(t3=OBS_LAYOUT_T3, t4=OBS_LAYOUT_T3, t8=OBS_LAYOUT_T8)
SETTLE_MARGIN = 5


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        K: int = 4,
        chunk: int = 8,
        pred_h: int = 8,
        obs_dim: int = 41,
        obj_slice: tuple = (25, 38),
        data_fraction: float = 1.0,
        max_trajectories: int | None = None,
        val: bool = False,
        val_fraction: float = 0.05,
        seed: int = 0,
    ):
        self.K, self.chunk, self.pred_h = K, chunk, pred_h
        self.obj_slice = obj_slice
        path = os.path.join(data_dir, "trajectory.h5")
        obs_list, act_list = [], []
        with h5py.File(path, "r") as f:
            names = sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(names))
            n_val = max(1, int(len(names) * val_fraction))
            keep = order[:n_val] if val else order[n_val:]
            n_keep = max(1, int(round(len(keep) * data_fraction)))
            keep = set(keep[:n_keep].tolist())
            for idx, name in enumerate(names):
                if idx not in keep:
                    continue
                t = f[name]
                obs = np.asarray(t["obs"], dtype=np.float32)
                act = np.clip(np.asarray(t["actions"], dtype=np.float32), -1, 1)
                assert obs.shape[1] == obs_dim, (
                    f"obs dim {obs.shape[1]} != expected {obs_dim}; the obs "
                    "layout constants no longer match the environment"
                )
                succ = np.asarray(t["success"]).astype(bool)
                if succ.any():
                    t_end = min(int(np.argmax(succ)) + SETTLE_MARGIN, len(act))
                else:
                    t_end = len(act)
                obs_list.append(obs[: t_end + 1])
                act_list.append(act[:t_end])
        self.obs = obs_list
        self.act = act_list
        self.index = [
            (i, t) for i, a in enumerate(self.act) for t in range(len(a))
        ]
        cat = np.concatenate([o for o in obs_list], axis=0)
        self.obs_mean = cat.mean(0)
        self.obs_std = cat.std(0) + 1e-6

    def normalizer(self) -> dict:
        return dict(obs_mean=self.obs_mean, obs_std=self.obs_std)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k: int):
        i, t = self.index[k]
        obs, act = self.obs[i], self.act[i]
        T = len(act)
        K, H_a, H_p = self.K, self.chunk, self.pred_h

        # window obs[t-K+1 .. t+1], left-padded by repeating the first frame
        idx = np.clip(np.arange(t - K + 1, t + 2), 0, len(obs) - 1)
        hist_next = (obs[idx] - self.obs_mean) / self.obs_std

        # action chunk from t, right-padded by repeating the last action
        idx = np.clip(np.arange(t, t + H_a), 0, T - 1)
        chunk = act[idx]

        # object states t+1 .. t+H_p, right-padded by repeating the last frame
        lo, hi = self.obj_slice
        idx = np.clip(np.arange(t + 1, t + H_p + 1), 0, len(obs) - 1)
        future_obj = (obs[idx, lo:hi] - self.obs_mean[lo:hi]) / self.obs_std[lo:hi]

        prev_act = act[t - 1] if t > 0 else np.zeros_like(act[0])
        return (
            torch.from_numpy(np.ascontiguousarray(hist_next)),
            torch.from_numpy(np.ascontiguousarray(chunk)),
            torch.from_numpy(np.ascontiguousarray(future_obj)),
            torch.from_numpy(np.ascontiguousarray(prev_act)),
        )


def load_c_metadata(data_dir: str) -> dict:
    m = np.load(os.path.join(data_dir, "c_metadata.npz"))
    return {k: m[k] for k in m.files}


def summarize(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "harness_summary.json")) as fp:
        return json.load(fp)
