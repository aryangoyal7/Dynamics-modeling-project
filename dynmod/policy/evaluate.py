"""Roll out a trained flow policy across the delta grid (plan Part II
'Build: training' gate + 'Build: evaluation' measurements).

One parallel env is pinned to each grid point via c_override; the policy runs
receding-horizon (sample a chunk, execute the first E actions). Per point it
reports success_once AND the continuous final distance to the slot (success
saturates at far grid points, which flattens slopes artificially).

Gate mode (--gate) implements 'train the base policy alone and check it fails
on the far delta grid points': compares interpolation-tagged points against
the far tail (delta >= --far-delta) and passes iff performance visibly drops.

    python -m dynmod.policy.evaluate --ckpt <run>/final_ckpt.pt --episodes 20
    python -m dynmod.policy.evaluate --ckpt <run>/final_ckpt.pt --gate

Writes reports/policy_eval_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.envs.delta_grid import (components_out_of_range, delta_of,
                                    make_grid, tag_of)
from dynmod.policy.model import FlowPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env-id", default="SlideToSlotT3-v1")
    p.add_argument("--episodes", type=int, default=20, help="per grid point")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--exec-horizon", type=int, default=4,
                   help="actions executed per sampled chunk")
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate", action="store_true")
    p.add_argument("--far-delta", type=float, default=4.0)
    p.add_argument("--name", default=None)
    p.add_argument("--control-mode", default="pd_joint_delta_pos",
                   help="must match the control mode the dataset was recorded in")
    args = p.parse_args()

    model, normalizer, extra = FlowPolicy.load(args.ckpt)
    cfg = model.cfg
    grid = make_grid()
    n = len(grid)
    c_override = {
        k: np.array([g[k] for g in grid])
        for k in ("mass_mult", "friction", "com_frac")
    }
    env = gym.make(
        args.env_id, num_envs=n, obs_mode="state",
        control_mode=args.control_mode, sim_backend="physx_cuda",
        randomize_c=False, c_override=c_override,
    )
    # Re-tag against the env's OWN training spec. The grid points themselves
    # are fixed constants, but which of them count as in-range depends on the
    # task: T8 trains over COM offsets up to 0.4, so grid points the default
    # spec (0.15) calls extrapolation are interpolation there. Tagging with
    # the wrong spec would silently mislabel the per-tag slope comparisons.
    spec = env.unwrapped.c_spec
    for g in grid:
        g["out_of_range"] = components_out_of_range(g, spec)
        g["tag"] = tag_of(g, spec)
        g["delta"] = round(delta_of(g, spec), 4)

    device = env.unwrapped.device
    model.to(device)
    mean = torch.as_tensor(normalizer["obs_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(normalizer["obs_std"], device=device, dtype=torch.float32)

    success = np.zeros(n)
    final_dist = np.zeros(n)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed * 100003 + ep)
        obs_n = (obs - mean) / std
        hist = obs_n[:, None, :].repeat(1, cfg.K, 1)
        prev_act = torch.zeros((n, cfg.act_dim), device=device)
        seen = torch.zeros(n, dtype=torch.bool, device=device)
        t = 0
        while t < args.horizon:
            chunk = model.sample_actions(hist, prev_act, n_steps=args.flow_steps)
            for j in range(min(args.exec_horizon, args.horizon - t)):
                a = chunk[:, j]
                obs, _, _, _, info = env.step(a)
                obs_n = (obs - mean) / std
                hist = torch.cat([hist[:, 1:], obs_n[:, None, :]], dim=1)
                prev_act = a
                seen |= info["success"]
                t += 1
        success += seen.cpu().numpy()
        dist_key = "obj_to_slot_dist" if "obj_to_slot_dist" in info else "obj_to_goal_dist"
        final_dist += info[dist_key].cpu().numpy()
    env.close()
    success /= args.episodes
    final_dist /= args.episodes

    points = [
        dict(**g, success=float(s), mean_final_dist=float(d))
        for g, s, d in zip(grid, success, final_dist)
    ]
    interp = [p_ for p_ in points if p_["tag"] == "interpolation"]
    far = [p_ for p_ in points if p_["delta"] >= args.far_delta]
    s_interp = float(np.mean([p_["success"] for p_ in interp]))
    s_far = float(np.mean([p_["success"] for p_ in far]))

    name = args.name or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    result = dict(
        ckpt=args.ckpt, arm=extra.get("args", {}).get("arm"),
        episodes=args.episodes, seed=args.seed,
        success_interpolation=s_interp, success_far=s_far,
        far_delta=args.far_delta, points=points,
    )
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"policy_eval_{name}.json")
    with open(path, "w") as fp:
        json.dump(result, fp, indent=1)

    print(f"success: interpolation {s_interp:.3f}  far(delta>={args.far_delta}) "
          f"{s_far:.3f}  -> {path}")
    if args.gate:
        degrades = s_far < s_interp - 0.2
        print("GATE (base policy fails on far grid): "
              + ("PASS - a gap exists to attribute"
                 if degrades else
                 "FAIL - widen ranges, push held-out values further, or "
                 "shorten the history"))
        raise SystemExit(0 if degrades else 1)


if __name__ == "__main__":
    main()
