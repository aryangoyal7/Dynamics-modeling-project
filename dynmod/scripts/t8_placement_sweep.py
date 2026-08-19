"""T8 mechanism experiment: success against WHERE the block's mass sits.

The block is teleported onto the beam at a chosen lateral offset (no
pick-and-place, so placement noise cannot confound), then only the carry
runs. Sweeping the offset at two pinned hidden COMs answers the question the
gate depends on: does the hidden COM move the optimal placement, and by how
much? Both curves should be the same curve, displaced by the COM offset -
that displacement IS the knowledge premium's mechanism.

    python -m dynmod.scripts.t8_placement_sweep
Writes reports/t8_placement_sweep.json.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from mani_skill.utils.structs import Pose

from dynmod.scripts.premium_gate_t8 import (
    StackCarryController, fixed, make_env)

N = 64
OFFSETS_CM = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
COMS = [(0.0, 0.0, "com_0"), (0.35, np.pi / 2, "com_plus_1.05cm")]
SPEED = 0.4
CARRY_STEPS = 110


def seat_block(base, d_m):
    """Teleport the block onto the beam, offset d_m along the beam's narrow
    axis. The GPU flush is required: pose writes outside _initialize_episode
    never reach the simulation buffer without it (measured - the sweep
    silently ran with the block still at its spawn)."""
    beam = base.obj.pose.p
    xyz = beam.clone()
    xyz[:, 0] += base.TOP_SEAT_DX
    xyz[:, 1] += d_m
    xyz[:, 2] = 0.074  # beam top 0.024 + block half-height 0.05
    q = torch.zeros((base.num_envs, 4), device=base.device)
    q[:, 0] = 1.0
    base.top.set_pose(Pose.create_from_pq(p=xyz, q=q))
    base.top.set_linear_velocity(torch.zeros((base.num_envs, 3), device=base.device))
    base.top.set_angular_velocity(torch.zeros((base.num_envs, 3), device=base.device))
    base.scene._gpu_apply_all()
    base.scene._gpu_fetch_all()


def main():
    out = {}
    for com_frac, angle, tag in COMS:
        env = make_env(N, randomize_c=False, c_override=dict(
            mass_mult=1.0, friction=0.5, com_frac=com_frac, com_angle=angle))
        base = env.unwrapped
        curve = {}
        for d_cm in OFFSETS_CM:
            env.reset(seed=8)
            c = base.get_c()
            seat_block(base, d_cm / 100.0)
            ctrl = StackCarryController(
                base, env.action_space.shape[-1], fixed(SPEED)(base, c),
                torch.zeros((N, 2), device=base.device))
            ctrl.stage[:] = 8  # skip the pick-and-place: carry only
            seen = torch.zeros(N, dtype=torch.bool, device=base.device)
            for _ in range(CARRY_STEPS):
                _, _, _, _, info = env.step(ctrl.act())
                seen |= info["success"]
            curve[f"{d_cm}"] = float(seen.float().mean())
            print(f"{tag} offset {d_cm:+.1f} cm -> {curve[f'{d_cm}']:.2f}",
                  flush=True)
        env.close()
        out[tag] = dict(com_frac=com_frac, com_angle=angle,
                        com_y_cm=float(com_frac * 3.0 * np.sin(angle)),
                        curve=curve)
    out["meta"] = dict(episodes=N, speed=SPEED, offsets_cm=OFFSETS_CM,
                       beam_half_width_cm=1.8, block_half_width_cm=3.0)
    json.dump(out, open("reports/t8_placement_sweep.json", "w"), indent=1)
    print("wrote reports/t8_placement_sweep.json")


if __name__ == "__main__":
    main()
