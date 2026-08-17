"""GATE (plan Part II 'Build: data'): teacher success must be flat across the
TRAINING range of c.

A teacher that handles one corner of c-space worse produces systematically
worse demonstrations there, and every policy trained on them degrades there
for reasons unrelated to prediction. This script evaluates a PPO teacher
checkpoint on a grid of c values inside the training ranges (one parallel env
pinned to each grid point via c_override), reports the success surface, and
plots it.

    python -m dynmod.scripts.teacher_flatness --ckpt runs/t3-teacher-v1/final_ckpt.pt

Writes reports/teacher_flatness.json and reports/teacher_flatness.png.
Advisory gate: max-min success spread <= 0.15 across in-range grid points.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.models import PPOAgent

MASS = np.exp(np.linspace(np.log(0.7), np.log(1.4), 5)).round(3)
FRICTION = np.exp(np.linspace(np.log(0.3), np.log(0.7), 4)).round(3)
COM = np.array([0.0, 0.075, 0.15])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--env-id", default="SlideToSlotT3Teacher-v1")
    parser.add_argument("--episodes", type=int, default=64,
                        help="episodes per grid point")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--tag", default="teacher_flatness")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    args = parser.parse_args()

    points = [dict(mass_mult=m, friction=f, com_frac=o)
              for m, f, o in itertools.product(MASS, FRICTION, COM)]
    n = len(points)
    c_override = {k: np.array([p[k] for p in points]) for k in points[0]}

    env = gym.make(
        args.env_id, num_envs=n, obs_mode="state",
        control_mode=args.control_mode, sim_backend="physx_cuda",
        randomize_c=False, c_override=c_override,
    )
    device = env.unwrapped.device
    agent = PPOAgent.load(args.ckpt, device=device)

    success_once = np.zeros(n)
    final_dist = np.zeros(n)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        seen = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(args.horizon):
            obs, _, _, _, info = env.step(agent.act(obs))
            seen |= info["success"]
        success_once += seen.cpu().numpy()
        dist_key = "obj_to_slot_dist" if "obj_to_slot_dist" in info else "obj_to_goal_dist"
        final_dist += info[dist_key].cpu().numpy()
    env.close()
    success_once /= args.episodes
    final_dist /= args.episodes

    spread = float(success_once.max() - success_once.min())
    result = dict(
        ckpt=args.ckpt, env_id=args.env_id, episodes=args.episodes,
        mean_success=float(success_once.mean()), spread=spread,
        gate_passed_advisory=bool(spread <= 0.15),
        points=[
            dict(**p, success_once=float(s), mean_final_dist=float(d))
            for p, s, d in zip(points, success_once, final_dist)
        ],
    )
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.tag}.json"), "w") as fp:
        json.dump(result, fp, indent=1)

    # heatmaps: success over mass x friction, one panel per COM level
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = success_once.reshape(len(MASS), len(FRICTION), len(COM))
    fig, axes = plt.subplots(1, len(COM), figsize=(4 * len(COM), 3.6), squeeze=False)
    for k, ax in enumerate(axes[0]):
        im = ax.imshow(grid[:, :, k], vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xticks(range(len(FRICTION)), [f"{v:g}" for v in FRICTION])
        ax.set_yticks(range(len(MASS)), [f"{v:g}" for v in MASS])
        ax.set_xlabel("friction"); ax.set_ylabel("mass mult")
        ax.set_title(f"COM {COM[k]:g}")
        for (i, j), v in np.ndenumerate(grid[:, :, k]):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="w" if v < 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    fig.suptitle(
        f"teacher success across training-range c "
        f"(mean {success_once.mean():.2f}, spread {spread:.2f})"
    )
    png = os.path.join(out_dir, f"{args.tag}.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")

    print(f"mean success {success_once.mean():.3f}  "
          f"min {success_once.min():.3f}  max {success_once.max():.3f}  "
          f"spread {spread:.3f}")
    print(f"advisory gate (spread<=0.15): "
          f"{'PASS' if spread <= 0.15 else 'FAIL - retrain or narrow the range'}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
