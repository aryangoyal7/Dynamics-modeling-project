"""Record short demonstration videos of each task for human viewing.

Each video is a 2x2 grid: four parallel environments running the SAME
controller under CONTRASTING hidden physics c, so the effect of the hidden
parameters is directly visible. Controllers used:

  t3  c-blind scripted expert (guarded nudging into the slot)
  t4  trained PPO teacher (pick-and-place control task)
  t2  scripted carry: move the held, ball-filled mug toward the goal
  t1  scripted pour: lift the held cup over the basin and tilt

T1/T2 use scripted motions because their teachers are not trained yet; the
point is to show what the task looks like, not to solve it well.

    python -m dynmod.scripts.record_demos --tasks t3 t4 t2 t1

Videos land in Dynamics-modeling-project/demos/<task>/ with a README.txt
naming the physics in each quadrant (tiles are row-major: 0 1 / 2 3).
"""

from __future__ import annotations

import argparse
import glob
import os

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401
from dynmod.models import PPOAgent
from dynmod.scripts.scripted_expert_t3 import policy as t3_policy
from mani_skill.utils.wrappers import RecordEpisode

DEMO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "demos")
)

# quadrants: 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right
RIGID_C = dict(
    mass_mult=np.array([1.0, 0.3, 3.0, 1.0]),
    friction=np.array([0.5, 0.1, 1.2, 0.5]),
    com_frac=np.array([0.0, 0.0, 0.0, 0.4]),
)
RIGID_LABELS = [
    "nominal physics",
    "light + slippery (overshoots)",
    "heavy + rough (stops short)",
    "off-centre weight (veers/rotates)",
]
GRAN_C = dict(
    mass_mult=np.array([1.0, 1.0, 1.0, 1.0]),
    friction=np.array([0.5, 0.5, 0.5, 0.5]),
    com_frac=np.array([0.0, 0.0, 0.0, 0.0]),
    particle_count=np.array([20.0, 30.0, 25.0, 30.0]),
    pp_friction=np.array([0.4, 0.2, 0.8, 0.6]),
    handle_offset=np.array([0.0, 0.02, -0.02, 0.0]),
)
GRAN_LABELS = [
    "20 balls, medium content friction",
    "30 balls, slippery contents",
    "25 balls, sticky contents",
    "30 balls, shifted handle",
]


def make(env_id, out, c_override, episodes, **kwargs):
    env = gym.make(
        env_id, num_envs=4, obs_mode="state", render_mode="rgb_array",
        sim_backend="physx_cuda", randomize_c=False, c_override=c_override,
        **kwargs,
    )
    env = RecordEpisode(
        env, output_dir=out, save_trajectory=False, save_video=True,
        max_steps_per_video=10_000, video_fps=30,
    )
    return env


def write_readme(out, title, labels, note=""):
    with open(os.path.join(out, "README.txt"), "w") as fp:
        fp.write(f"{title}\nQuadrants (row-major):\n")
        for i, lab in enumerate(labels):
            fp.write(f"  {i}: {lab}\n")
        if note:
            fp.write(note + "\n")


def demo_t3(episodes):
    out = os.path.join(DEMO_ROOT, "t3_slide_to_slot")
    env = make("SlideToSlotT3-v1", out, RIGID_C, episodes,
               control_mode="pd_ee_delta_pos")
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    for ep in range(episodes):
        env.reset(seed=10 + ep)
        for _ in range(200):
            env.step(t3_policy(base, act_dim))
    env.close()
    write_readme(out, "T3 slide-to-slot: nudge the blue block so it rests on "
                 "the green slot inside the channel.", RIGID_LABELS,
                 "Controller: c-blind scripted expert (slow guarded pushes).")


def demo_t4(episodes):
    out = os.path.join(DEMO_ROOT, "t4_pick_place")
    ckpts = sorted(glob.glob("runs/t4-teacher-v1/ckpt_*.pt"),
                   key=os.path.getmtime)
    agent = None
    if ckpts:
        agent = PPOAgent.load(ckpts[-1], device="cuda")
        print(f"t4: using teacher checkpoint {ckpts[-1]}")
    env = make("PickPlaceT4Teacher-v1", out, RIGID_C, episodes,
               control_mode="pd_joint_delta_pos")
    for ep in range(episodes):
        obs, _ = env.reset(seed=20 + ep)
        for _ in range(80):
            a = agent.act(obs) if agent else torch.zeros_like(
                torch.as_tensor(env.action_space.sample()))
            obs, *_ = env.step(a)
    env.close()
    write_readme(out, "T4 pick-and-place (control task): grasp the blue cube "
                 "and hold it at the goal point.", RIGID_LABELS,
                 "Controller: trained privileged PPO teacher. Once grasped, "
                 "hidden physics stop mattering - that is the point of this task.")


def demo_t2(episodes):
    out = os.path.join(DEMO_ROOT, "t2_carry_mug")
    env = make("CarryT2-v1", out, GRAN_C, episodes,
               control_mode="pd_ee_delta_pos", spawn_grasped=True)
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    for ep in range(episodes):
        env.reset(seed=30 + ep)
        for _ in range(120):
            tcp = base.agent.tcp.pose.p
            goal = base.goal_site.pose.p
            a = torch.zeros((4, act_dim), device=base.device)
            a[:, :3] = torch.clip((goal - tcp) / 0.1, -0.35, 0.35)  # brisk carry
            a[:, 3:] = -1.0
            env.step(a)
    env.close()
    write_readme(out, "T2 carrying: move the ball-filled mug (held by its "
                 "handle) to the green goal without spilling.", GRAN_LABELS,
                 "Controller: scripted straight-line carry - the acceleration "
                 "itself excites the sloshing it must then survive.")


def demo_t1(episodes):
    out = os.path.join(DEMO_ROOT, "t1_pour")
    env = make("PourT1-v1", out, GRAN_C, episodes,
               control_mode="pd_ee_delta_pose", spawn_grasped=True)
    base = env.unwrapped
    act_dim = env.action_space.shape[-1]
    bx, by = base.BASIN_POS
    for ep in range(episodes):
        env.reset(seed=40 + ep)
        for t in range(170):
            tcp = base.agent.tcp.pose.p
            a = torch.zeros((4, act_dim), device=base.device)
            if t < 35:  # move above the basin, offset so the cup body clears it
                target = torch.tensor([bx, by + 0.07, 0.24], device=base.device)
                a[:, :3] = torch.clip((target[None] - tcp) / 0.1, -0.4, 0.4)
            elif t < 150:  # tilt: rotate about x so the cup tips toward the basin
                a[:, 3] = -0.7
            a[:, -1] = -1.0
            env.step(a)
    env.close()
    write_readme(out, "T1 pouring: tilt the held cup so its balls fall into "
                 "the grey basin.", GRAN_LABELS,
                 "Controller: scripted lift-and-tilt. Outflow lags the tilt; "
                 "how much comes out per degree depends on fill level and "
                 "content friction.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["t3", "t4", "t2", "t1"])
    p.add_argument("--episodes", type=int, default=2)
    args = p.parse_args()
    os.makedirs(DEMO_ROOT, exist_ok=True)
    for t in args.tasks:
        print(f"=== recording {t} ===")
        dict(t1=demo_t1, t2=demo_t2, t3=demo_t3, t4=demo_t4)[t](args.episodes)
    print(f"videos in {DEMO_ROOT}")
