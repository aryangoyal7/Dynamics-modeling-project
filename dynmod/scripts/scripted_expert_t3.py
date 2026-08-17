"""c-blind scripted expert for SlideToSlotT3 (plan Part II 'Build: data'):
the secondary dataset where the demonstrator has no obligation to know c.

ManiSkill ships motion-planning experts only for its built-in tasks, so this
is the equivalent for T3: closed-loop Cartesian guarded moves that use
observations (object/slot poses) but never c. It creeps the object toward the
slot with a fixed conservative push regardless of mass or friction, re-
approaching from the other side if the object ends up past the slot. Because
it never has to anticipate the slide, its actions carry no information about
c beyond what the observations already show.

Records in pd_ee_delta_pos; ManiSkill's replay tool can convert control modes
later if the student trains in joint space.

    python -m dynmod.scripts.scripted_expert_t3 --episodes 1000 \
        --out /mnt/scratch/dynamics/data/t3_cblind_1e3
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
from mani_skill.utils.wrappers import RecordEpisode

PUSH_SPEED = 0.35  # creep speed for escort/recovery phases
FLICK_SPEED = 1.0  # FIXED charge-flick: identical for every c (calibrated once
                   # on nominal physics 2026-08-17: 41% nominal success, 6% tunnel-dead)
BEHIND = 0.055  # standoff behind the object along the push direction
HIGH_Z = 0.10
PUSH_Z = 0.015
FLICK_STANDOFF = 0.13  # run-up length for the charge
FLICK_FROM_X = -0.01  # escort the object here, then commit the fixed flick.
# The windup matters: launching from the tunnel mouth fails at every speed
# (the roof blocks the hand's follow-through, so the object enters at creep
# speed and dies underneath). ~9cm of runway lets the EE build real momentum.


def policy(base, act_dim: int) -> torch.Tensor:
    """One batched action from privileged-free quantities (poses only).

    Tunnel-era strategy: escort the object to just before the tunnel mouth,
    then commit a FIXED-strength flick through it. The flick is identical
    regardless of the true physics - the defining property of the c-blind
    condition. Overshoots are recovered by creep-pushing back in the open
    region past the slot; deep undershoots stop under the tunnel roof and
    are honestly unrecoverable.
    """
    device = base.device
    n = base.num_envs
    tcp = base.agent.tcp.pose.p
    obj = base.obj.pose.p
    slot_x = base.slot_marker.pose.p[:, 0]
    dx = slot_x - obj[:, 0]
    in_slot = dx.abs() < base.slot_half_width * 0.8  # aim for the slot center
    tun0, tun1 = base.TUNNEL_X
    t20, t21 = base.TUNNEL2_X

    before_tunnel = obj[:, 0] < tun0 - 0.005
    beyond_roof2 = obj[:, 0] > t21 + 0.005
    in_window = (obj[:, 0] > tun1 + 0.005) & (obj[:, 0] < t20 - 0.005)
    # direction: forward flick from before roof 1; REVERSE flick from beyond
    # roof 2 (a committed slide back through it); in-window nudges follow dx
    direction = torch.where(before_tunnel, torch.ones_like(dx),
                torch.where(beyond_roof2, -torch.ones_like(dx), torch.sign(dx)))
    flicking = (before_tunnel & (obj[:, 0] > FLICK_FROM_X - 0.02)) | beyond_roof2

    # the flick is a CHARGE: retreat to a far standoff, then drive at full
    # speed so the strike happens with the EE already moving (striking from
    # a near standstill enters the tunnel at ~0.4 m/s and dies underneath)
    standoff = torch.where(flicking, torch.full_like(dx, FLICK_STANDOFF),
                           torch.full_like(dx, BEHIND))
    behind = obj.clone()
    behind[:, 0] -= direction * standoff
    behind[:, 0] = behind[:, 0].clamp(max=0.55)  # stay clear of the end wall
    behind[:, 2] = PUSH_Z

    to_behind_xy = torch.linalg.norm(tcp[:, :2] - behind[:, :2], dim=1)
    a = torch.zeros((n, act_dim), device=device)
    a[:, 3:] = -1.0  # gripper closed throughout

    # mode 1: done -> retreat straight up (breaks contact, satisfies release)
    retreat = torch.zeros_like(tcp)
    retreat[:, 2] = 1.0
    # mode 2: reposition -> travel high toward the standoff point, then drop.
    # never path through the tunnel block: route the high leg above HIGH_Z
    target = behind.clone()
    target[:, 2] = torch.where(to_behind_xy > 0.03,
                               torch.full_like(behind[:, 2], HIGH_Z),
                               behind[:, 2])
    goto = torch.clip((target - tcp) / 0.1, -1.0, 1.0)
    # mode 3: at the standoff -> push. Speed is the FIXED flick when
    # launching through the tunnel, else the slow creep.
    speed = torch.where(flicking, torch.full_like(dx, FLICK_SPEED),
                        torch.full_like(dx, PUSH_SPEED))
    push = torch.zeros_like(tcp)
    push[:, 0] = direction * speed
    push[:, 1] = torch.clip((obj[:, 1] - tcp[:, 1]) * 2.0, -0.2, 0.2)

    at_standoff = (to_behind_xy < 0.02) & (tcp[:, 2] < PUSH_Z + 0.015)
    # once low and behind (w.r.t. the push direction) in the corridor during
    # the flick phase, charge - stateless: holds through run-up and strike
    charging = flicking & (direction * (obj[:, 0] - tcp[:, 0]) > 0.03) \
        & ((tcp[:, 1] - obj[:, 1]).abs() < 0.025) & (tcp[:, 2] < PUSH_Z + 0.02)
    move = torch.where((at_standoff | charging)[:, None], push, goto)
    # objects resting under either roof are unreachable: retreat rather
    # than ram the tunnels
    dead = (((obj[:, 0] > tun0) & (obj[:, 0] < tun1))
            | ((obj[:, 0] > t20) & (obj[:, 0] < t21))) & ~in_slot
    a[:, :3] = torch.where((in_slot | dead)[:, None], retreat, move)
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-record", action="store_true",
                        help="dry run: report success rate only")
    args = parser.parse_args()

    n = args.num_envs
    cycles = (args.episodes + n - 1) // n
    env = gym.make(
        "SlideToSlotT3-v1", num_envs=n, obs_mode="state",
        control_mode="pd_ee_delta_pos", sim_backend="physx_cuda",
        reconfiguration_freq=1,
    )
    if not args.no_record:
        os.makedirs(args.out, exist_ok=True)
        env = RecordEpisode(
            env, output_dir=args.out, save_trajectory=True, save_video=False,
            trajectory_name="trajectory", source_type="scripted-cblind",
            source_desc="guarded-push scripted expert, no access to c",
        )
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]

    keys = list(C_KEYS + GRANULAR_KEYS) + ["com_x_frac", "com_y_frac", "mass_kg"]
    c_rows, success_rows, attempts = [], [], []
    for r in range(cycles):
        env.reset(seed=args.seed + r)
        c = base.get_c()
        seen = torch.zeros(n, dtype=torch.bool, device=base.device)
        for _ in range(args.horizon):
            _, _, _, _, info = env.step(policy(base, act_dim))
            seen |= info["success"]
        for i in range(n):
            c_rows.append([float(c[k][i]) for k in keys])
        success_rows.extend(seen.cpu().numpy().tolist())
        attempts.extend(info["attempt_count"].cpu().numpy().tolist())
    env.close()

    succ = np.asarray(success_rows, dtype=bool)
    summary = dict(
        env_id="SlideToSlotT3-v1", episodes=len(succ), num_envs=n,
        success_once_rate=float(succ.mean()),
        mean_attempts=float(np.mean(attempts)),
    )
    if not args.no_record:
        np.savez(
            os.path.join(args.out, "c_metadata.npz"),
            c=np.asarray(c_rows, dtype=np.float32), c_keys=np.array(keys),
            success_once=succ, episodes_per_cycle=n, cycles=cycles,
            seed=args.seed,
        )
        with open(os.path.join(args.out, "harness_summary.json"), "w") as fp:
            json.dump(summary, fp, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
