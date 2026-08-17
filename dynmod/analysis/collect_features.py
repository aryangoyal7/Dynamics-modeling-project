"""Collect trunk features from trained policies and probe them for c
(plan Part II 'Build: evaluation': extract activations from every frozen arm
on held-out episodes paired with their true c; fit ridge and MLP probes with
untrained-network and raw-input baselines).

    python -m dynmod.analysis.collect_features \
        --ckpts /mnt/scratch/dynamics/policy_runs/t4-1e4-A-multistep-s0/final_ckpt.pt ... \
        --env-id PickPlaceT4-v1 --horizon 80 --tag t4_probes
Writes reports/<tag>.json with per-checkpoint probe R^2 per c component.
"""

from __future__ import annotations

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.analysis.probes import probe_report
from dynmod.policy.model import FlowPolicy, PolicyConfig

C_NAMES = ("mass_mult", "friction", "com_x_frac", "com_y_frac")


def collect(model, normalizer, env_id, horizon, control_mode, episodes=3,
            num_envs=128, seed=9):
    env = gym.make(env_id, num_envs=num_envs, obs_mode="state",
                   control_mode=control_mode, sim_backend="physx_cuda",
                   reconfiguration_freq=1)
    base = env.unwrapped
    device = base.device
    model.to(device)
    mean = torch.as_tensor(normalizer["obs_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(normalizer["obs_std"], device=device, dtype=torch.float32)
    cfg = model.cfg
    feats, cs, raws = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        c = base.get_c()
        c_mat = np.stack([c[k] for k in C_NAMES], axis=1)
        obs_n = (obs - mean) / std
        hist = obs_n[:, None, :].repeat(1, cfg.K, 1)
        prev = torch.zeros((num_envs, cfg.act_dim), device=device)
        for t in range(horizon):
            if t % 4 == 0 and t >= cfg.K:  # skip warm-up frames
                with torch.no_grad():
                    f = model.features(hist)
                feats.append(f.cpu().numpy())
                cs.append(c_mat)
                raws.append(hist.flatten(1).cpu().numpy())
            if t % 4 == 0:
                chunk = model.sample_actions(hist, prev, n_steps=8)
            a = chunk[:, t % 4]
            obs, _, _, _, _ = env.step(a)
            obs_n = (obs - mean) / std
            hist = torch.cat([hist[:, 1:], obs_n[:, None, :]], dim=1)
            prev = a
    env.close()
    return (np.concatenate(feats), np.concatenate(cs), np.concatenate(raws))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--env-id", default="PickPlaceT4-v1")
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--control-mode", default="pd_joint_delta_pos")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    results = {}
    for ck in args.ckpts:
        name = os.path.basename(os.path.dirname(os.path.abspath(ck)))
        model, norm, _ = FlowPolicy.load(ck)
        F, C, R = collect(model, norm, args.env_id, args.horizon,
                          args.control_mode)
        untrained = FlowPolicy(PolicyConfig(**{**model.cfg.__dict__})).to("cuda")
        with torch.no_grad():
            F0 = untrained.features(
                torch.tensor(R.reshape(len(R), model.cfg.K, -1),
                             dtype=torch.float32, device="cuda")).cpu().numpy()
        c_named = {k: C[:, i] for i, k in enumerate(C_NAMES)}
        rep = probe_report(dict(trained=F, untrained_baseline=F0, raw_input=R),
                           c_named)
        results[name] = rep
        tr = {k: rep["trained"][k]["ridge_r2"] for k in C_NAMES}
        print(f"{name}: trained ridge R2 {tr}")

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                       "reports", f"{args.tag}.json"))
    with open(out, "w") as fp:
        json.dump(results, fp, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
