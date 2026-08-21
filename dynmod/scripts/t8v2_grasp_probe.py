"""T8-v2 probe: does the BEST BEAM END flip with the hidden centre of mass?

This is the one measurement that decides whether T8 comes back.  T8-v1 was
descoped because its continuous placement compensation and its teacher's
evenness were anti-correlated - there was no budget at which knowledge paid
AND the task was teachable.  v2 borrows T7's structure: make the knowledge
DISCRETE (which end of the beam to grasp) and block the safe middle grasp
with the block itself.

The design only works if, at a pinned hidden COM, the two ends give clearly
different success AND the winner reverses with the sign of the COM.  If the
same end always wins, there is a c-independent best choice and design law 3
says the task can never pay - descope stands, at the cost of one probe.

    python -m dynmod.scripts.t8v2_grasp_probe
Writes reports/t8v2_grasp_probe.json.
"""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.scripts.premium_gate_t8 import StackCarryController

N_ENV = 32
HORIZON = 140
CARRY_SPEED = 0.3          # the calm budget where v1's teacher certified
COM_FRACS = [-0.6, -0.3, 0.0, 0.3, 0.6]   # signed offset along the beam
MASS_MID, FRIC_MID = 0.9, 0.5


def make_env(n, **kw):
    return gym.make("StackGraspT8-v2", num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
                    **kw)


def run(env, grasp_dx, seed):
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    env.reset(seed=seed)
    n = base.num_envs
    ctrl = StackCarryController(
        base, act_dim,
        speeds=torch.full((n,), CARRY_SPEED, device=base.device),
        place_comp=torch.zeros((n, 2), device=base.device),
        grasp_dx=torch.full((n,), grasp_dx, device=base.device))
    seen = torch.zeros(n, dtype=torch.bool, device=base.device)
    for _ in range(HORIZON):
        _, _, _, _, info = env.step(ctrl.act())
        seen |= info["success"]
    return float(seen.float().mean()), ctrl.stage.float().mean().item()


def main():
    rows = []
    for cf in COM_FRACS:
        # com_angle 0 puts the COM at +x along the beam, pi puts it at -x;
        # the env zeroes com_y, so only this sign matters
        co = dict(mass_mult=MASS_MID, friction=FRIC_MID,
                  com_frac=abs(cf), com_angle=0.0 if cf >= 0 else np.pi)
        env = make_env(N_ENV, randomize_c=False, c_override=co)
        row = dict(com_x=cf)
        for name, dx in (("near_minus", -0.045), ("near_plus", +0.045)):
            s, stg = run(env, dx, seed=11)
            row[name] = s
            row[name + "_stage"] = round(stg, 1)
        env.close()
        row["best"] = "minus" if row["near_minus"] > row["near_plus"] else "plus"
        row["gap"] = round(abs(row["near_minus"] - row["near_plus"]), 3)
        rows.append(row)
        print(f"com_x {cf:+.2f}:  grasp -x {row['near_minus']:.3f} "
              f"(stage {row['near_minus_stage']})   "
              f"grasp +x {row['near_plus']:.3f} "
              f"(stage {row['near_plus_stage']})   "
              f"best {row['best']}  gap {row['gap']:.3f}", flush=True)

    neg = [r for r in rows if r["com_x"] < 0]
    pos = [r for r in rows if r["com_x"] > 0]
    flips = (all(r["best"] == neg[0]["best"] for r in neg)
             and all(r["best"] == pos[0]["best"] for r in pos)
             and neg[0]["best"] != pos[0]["best"])
    best_gap = max(r["gap"] for r in rows)
    print(f"\nwinner reverses with the COM sign: {flips}   "
          f"largest gap {best_gap:.3f}")
    if flips and best_gap >= 0.15:
        print("PROBE PASSES -> build the calibrated gate")
    else:
        print("PROBE FAILS -> no c-dependent end choice; T8 stays descoped")
    json.dump(dict(rows=rows, flips=bool(flips), best_gap=best_gap,
                   carry_speed=CARRY_SPEED, horizon=HORIZON),
              open("reports/t8v2_grasp_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
