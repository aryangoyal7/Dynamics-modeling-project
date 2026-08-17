"""GATE (plan Part II 'Build: environments'): verify the randomisation reaches
the physics.

Replays one identical open-loop action sequence in parallel SlideToSlotT3
environments that differ only in c, and confirms the object ends up in
different places along the channel. Includes a duplicated-c pair as a
determinism reference and an opposed COM-offset pair, which must deflect or
rotate the object differently. If outcomes do not separate, everything
downstream is noise - fix the environment first.

Usage:
    python -m dynmod.scripts.verify_randomization [--env-id SlideToSlotT3-v1]
Writes reports/verify_randomization.json; exit code 0 iff the gate passes.
"""

from __future__ import annotations

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401  (registers envs)

# one env per row: name, mass_mult, friction, com_frac, com_angle
CONDITIONS = [
    ("nominal", 1.0, 0.5, 0.0, 0.0),
    ("nominal-duplicate", 1.0, 0.5, 0.0, 0.0),  # determinism reference
    ("light-slippery", 0.3, 0.1, 0.0, 0.0),
    ("heavy-rough", 3.0, 1.2, 0.0, 0.0),
    ("heavy-only", 3.0, 0.5, 0.0, 0.0),
    ("slippery-only", 1.0, 0.1, 0.0, 0.0),
    ("com+y", 1.0, 0.5, 0.4, np.pi / 2),
    ("com-y", 1.0, 0.5, 0.4, -np.pi / 2),
]

OBJ_START = np.array([-0.06, 0.0, 0.02])


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SlideToSlotT3-v1")
    parser.add_argument("--sim-backend", default="gpu")
    args = parser.parse_args()

    n = len(CONDITIONS)
    c_override = dict(
        mass_mult=np.array([c[1] for c in CONDITIONS]),
        friction=np.array([c[2] for c in CONDITIONS]),
        com_frac=np.array([c[3] for c in CONDITIONS]),
        com_angle=np.array([c[4] for c in CONDITIONS]),
    )
    env = gym.make(
        args.env_id,
        num_envs=n,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        sim_backend=args.sim_backend,
        robot_init_qpos_noise=0.0,
        deterministic_spawn=True,
        randomize_c=False,
        c_override=c_override,
    )
    env.reset(seed=0)
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    device = base.device

    def step_all(a3, gripper=-1.0):
        a = torch.zeros((n, act_dim), device=device)
        a[:, :3] = torch.as_tensor(a3, dtype=torch.float32, device=device)
        a[:, 3:] = gripper
        env.step(a)

    def goto(waypoint, steps):
        """Proportional open-loop guidance computed from env 0's TCP only, so
        every env receives the identical action sequence."""
        for _ in range(steps):
            tcp0 = base.agent.tcp.pose.p[0].cpu().numpy()
            delta = np.clip((waypoint - tcp0) / 0.1, -1.0, 1.0)
            step_all(delta)

    # above/behind the object, drop into the channel, shove hard in +x, wait
    goto(OBJ_START + np.array([-0.12, 0.0, 0.10]), 15)
    goto(OBJ_START + np.array([-0.10, 0.0, 0.005]), 12)
    for _ in range(12):
        step_all(np.array([1.0, 0.0, 0.0]))
    for _ in range(60):
        step_all(np.array([0.0, 0.0, 0.0]))

    pos = base.obj.pose.p.cpu().numpy()
    quat = base.obj.pose.q.cpu().numpy()
    finals = {
        CONDITIONS[i][0]: dict(
            xy=[float(pos[i, 0]), float(pos[i, 1])], yaw=yaw_of(quat[i])
        )
        for i in range(n)
    }

    def d(i, j):
        return float(np.linalg.norm(pos[i, :2] - pos[j, :2]))

    dyaw_com = abs(yaw_of(quat[6]) - yaw_of(quat[7]))
    com_effect = max(d(6, 7), dyaw_com * 0.05)  # rotation counts too
    displacement = float(np.linalg.norm(pos[0, :2] - OBJ_START[:2]))
    checks = {
        "push_moved_nominal_obj_>5cm": (displacement, displacement > 0.05, True),
        "mass_changes_outcome_>1cm": (d(0, 4), d(0, 4) > 0.01, True),
        "friction_changes_outcome_>1cm": (d(0, 5), d(0, 5) > 0.01, True),
        "combined_extremes_differ_>2cm": (d(2, 3), d(2, 3) > 0.02, True),
        "com_offset_pair_differs": (com_effect, com_effect > 0.003, True),
        "identical_c_matches_<2mm": (d(0, 1), d(0, 1) < 0.002, False),  # warn only
    }

    report = dict(env_id=args.env_id, obj_start=OBJ_START.tolist(), finals=finals,
                  checks={})
    hard_fail = False
    print(f"\n=== randomisation-reaches-physics gate: {args.env_id} ===")
    print(f"nominal object displacement: {displacement * 100:.1f} cm")
    for name, (value, ok, required) in checks.items():
        status = "PASS" if ok else ("FAIL" if required else "WARN")
        hard_fail |= required and not ok
        print(f"  [{status}] {name}: {value:.4f}")
        report["checks"][name] = dict(value=value, ok=bool(ok), required=required)
    for cname, f in finals.items():
        print(f"    {cname:20s} xy=({f['xy'][0]:+.3f}, {f['xy'][1]:+.3f}) yaw={f['yaw']:+.3f}")
    report["gate_passed"] = not hard_fail

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "verify_randomization.json")
    with open(path, "w") as fp:
        json.dump(report, fp, indent=1)
    print(f"\ngate {'PASSED' if not hard_fail else 'FAILED'} -> {path}")
    env.close()
    raise SystemExit(0 if not hard_fail else 1)


if __name__ == "__main__":
    main()
