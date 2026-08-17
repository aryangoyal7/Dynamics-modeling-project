"""Held-out evaluation manifest (plan Part II 'Build: data'): episodes across
the full delta grid, 100 per grid point per seed.

Student policies are evaluated by rolling out live (not from recorded data),
so what must be pinned down is the evaluation *configuration*: for every
(grid point, evaluation seed) pair, a deterministic block of episode seeds
that every arm faces identically. Episode seeds are contiguous blocks, so the
manifest stays small; the evaluator derives episode k of block (g, s) as
episode_seed_start + k.

    python -m dynmod.scripts.make_eval_manifest
Writes configs/eval_manifest.json.
"""

from __future__ import annotations

import json
import os

from dynmod.envs.delta_grid import make_grid

N_SEEDS = 10
EPISODES_PER_POINT_PER_SEED = 100


def main():
    grid = make_grid()
    blocks = []
    for p in grid:
        for s in range(N_SEEDS):
            blocks.append(dict(
                grid_id=p["id"], seed=s,
                c=dict(mass_mult=p["mass_mult"], friction=p["friction"],
                       com_frac=p["com_frac"]),
                tag=p["tag"], delta=p["delta"],
                episode_seed_start=(p["id"] * N_SEEDS + s)
                * EPISODES_PER_POINT_PER_SEED,
                episodes=EPISODES_PER_POINT_PER_SEED,
            ))
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "eval_manifest.json")
    with open(path, "w") as fp:
        json.dump(dict(
            n_grid_points=len(grid), n_seeds=N_SEEDS,
            episodes_per_point_per_seed=EPISODES_PER_POINT_PER_SEED,
            total_episodes=len(blocks) * EPISODES_PER_POINT_PER_SEED,
            blocks=blocks,
        ), fp, indent=1)
    print(f"wrote {len(blocks)} blocks "
          f"({len(blocks) * EPISODES_PER_POINT_PER_SEED} episodes) -> {path}")


if __name__ == "__main__":
    main()
