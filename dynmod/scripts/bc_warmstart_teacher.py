"""Behavior-clone the scripted expert into a PPO Agent for warm-starting.

PPO exploration plateaus (~18%) on the tunnel task because the charge-flick
technique is a narrow behavior to discover by noise; the scripted expert
proves it works (43%). This trains ppo.py's Agent architecture (3x256 tanh)
on the SUCCESSFUL scripted episodes so PPO can start from working flicks and
learn the c-conditional modulation on top (its input includes c).

Input obs = [student obs (41), c (4)] = the Teacher env's observation layout
(c_params is appended last). Actions are pd_ee_delta_pos - the PPO fine-tune
must use --control_mode pd_ee_delta_pos.

    python -m dynmod.scripts.bc_warmstart_teacher \
        --data /mnt/scratch/dynamics/data/t3_scripted_warmstart \
        --out /mnt/scratch/dynamics/policy_runs/bc_warmstart/agent.pt
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch

from dynmod.models import PPOAgent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=4096)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = np.load(os.path.join(args.data, "c_metadata.npz"))
    c_all = meta["c"]
    keys = list(meta["c_keys"])
    idx = [keys.index(k) for k in ("mass_mult", "friction", "com_x_frac", "com_y_frac")]
    obs_list, act_list = [], []
    with h5py.File(os.path.join(args.data, "trajectory.h5")) as f:
        names = sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))
        assert len(names) == len(c_all), "episode/metadata mismatch"
        for i, name in enumerate(names):
            t = f[name]
            succ = np.asarray(t["success"]).astype(bool)
            if not succ.any():
                continue
            t_end = min(int(np.argmax(succ)) + 3, len(t["actions"]))
            obs = np.asarray(t["obs"][:t_end], dtype=np.float32)
            c_vec = np.tile(c_all[i][idx].astype(np.float32), (t_end, 1))
            obs_list.append(np.concatenate([obs, c_vec], axis=1))
            act_list.append(np.clip(np.asarray(t["actions"][:t_end], np.float32), -1, 1))
    X = torch.tensor(np.concatenate(obs_list), device=device)
    Y = torch.tensor(np.concatenate(act_list), device=device)
    print(f"BC dataset: {len(X)} transitions from successful episodes "
          f"(obs {X.shape[1]}, act {Y.shape[1]})")

    agent = PPOAgent(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    for s in range(args.steps):
        i = torch.randint(0, len(X), (args.batch,), device=device)
        loss = ((agent.actor_mean(X[i]) - Y[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0:
            print(f"step {s}: bc loss {loss.item():.4f}")
    # PPO fine-tuning needs a sane starting exploration noise
    with torch.no_grad():
        agent.actor_logstd.fill_(-1.0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(agent.state_dict(), args.out)
    print(f"final bc loss {loss.item():.4f} -> {args.out}")


if __name__ == "__main__":
    main()
