"""Smoke test: build every registered env, roll random actions, check the c
plumbing (teacher obs is wider than student obs by exactly the c dimension;
c stays inside training ranges; c is retrievable as metadata; granular tasks
activate the right number of particles).

    python -m dynmod.scripts.smoke_test [--quick]
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch

import dynmod.envs  # noqa: F401

RIGID_ENVS = ["SlideToSlotT3-v1", "PickPlaceT4-v1"]  # c dim 4
GRANULAR_ENVS = ["PourT1-v1", "CarryT2-v1"]  # c dim 7
NUM_ENVS = 16
NUM_ENVS_GRANULAR = 4  # particle builds are per-env-per-particle: keep small
STEPS = 30


def obs_dim(env):
    obs, _ = env.reset(seed=0)
    assert torch.isfinite(obs).all(), "non-finite observations at reset"
    return obs.shape[-1]


def check_c_ranges(c: dict) -> None:
    assert (c["mass_mult"] >= 0.7 - 1e-9).all() and (c["mass_mult"] <= 1.4 + 1e-9).all()
    assert (c["friction"] >= 0.3 - 1e-9).all() and (c["friction"] <= 0.7 + 1e-9).all()
    assert (c["com_frac"] >= 0).all() and (c["com_frac"] <= 0.15 + 1e-9).all()
    assert (c["particle_count"] >= 20).all() and (c["particle_count"] <= 30).all()
    assert (c["pp_friction"] >= 0.2 - 1e-9).all() and (c["pp_friction"] <= 0.8 + 1e-9).all()
    assert len(np.unique(c["mass_mult"])) > 1, "c not randomized across envs"
    assert (c["mass_kg"] > 0).all()


def run(env_id: str, n: int, c_dim: int) -> None:
    student = gym.make(env_id, num_envs=n, obs_mode="state", sim_backend="gpu")
    teacher_id = env_id.replace("-v1", "Teacher-v1")
    teacher = gym.make(teacher_id, num_envs=n, obs_mode="state", sim_backend="gpu")

    ds, dt_ = obs_dim(student), obs_dim(teacher)
    assert dt_ == ds + c_dim, (
        f"{teacher_id}: expected +{c_dim} obs dims for c, got {ds} -> {dt_}"
    )

    c = student.unwrapped.get_c()
    check_c_ranges(c)

    extra = ""
    if env_id in GRANULAR_ENVS:
        base = student.unwrapped
        act = base.particle_active.sum(0).cpu().numpy()
        assert np.array_equal(act, c["particle_count"].astype(int)), (
            "active particle slots disagree with hidden particle_count"
        )
        info = base.evaluate()
        retained = info["retained_frac"].cpu().numpy()
        extra = f", retained@t0 {retained.min():.2f}..{retained.max():.2f}"

    for env in (student, teacher):
        for _ in range(STEPS):
            a = torch.rand(
                n, *env.action_space.shape[1:], device=env.unwrapped.device
            ) * 2 - 1
            obs, rew, term, trunc, info = env.step(a)
            assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        assert "success" in info, f"{env_id}: evaluate() must report success"
        if env_id == "SlideToSlotT3-v1":
            assert "attempt_count" in info and "first_recovery_correct" in info

    # c must never leak into the student observation, and overrides must land
    fixed = gym.make(
        env_id, num_envs=min(n, 4), obs_mode="state", sim_backend="gpu",
        randomize_c=False, c_override=dict(mass_mult=2.0),
    )
    cf = fixed.unwrapped.get_c()
    assert np.allclose(cf["mass_mult"], 2.0), "c_override ignored"
    assert np.allclose(cf["friction"], 0.5), "randomize_c=False should pin friction"
    assert obs_dim(fixed) == ds, "c_override must not change student obs dim"

    student.close(); teacher.close(); fixed.close()
    print(f"[PASS] {env_id} (student {ds} dims, teacher {dt_} dims, "
          f"mass_kg {c['mass_kg'].min():.4f}..{c['mass_kg'].max():.4f}{extra})")


def hold_check(env_id: str, n: int = 4) -> None:
    """Empirical check of the spawn-grasped path: with a permanently closing
    gripper and zero arm motion, does the container stay held and retain its
    contents for 40 steps?"""
    env = gym.make(
        env_id, num_envs=n, obs_mode="state", sim_backend="gpu",
        control_mode="pd_ee_delta_pos", spawn_grasped=True,
    )
    env.reset(seed=0)
    base = env.unwrapped
    z0 = base.obj.pose.p[:, 2].clone()
    act_dim = env.action_space.shape[-1]
    a = torch.zeros((n, act_dim), device=base.device)
    a[:, 3:] = -1.0  # keep gripper closed
    for _ in range(40):
        env.step(a)
    z1 = base.obj.pose.p[:, 2]
    held = (z1 > z0 - 0.03).cpu().numpy()
    retained = base.evaluate()["retained_frac"].cpu().numpy()
    env.close()
    print(f"[{'PASS' if held.all() else 'WARN'}] {env_id} spawn_grasped hold: "
          f"held {held.sum()}/{n}, retained {retained.min():.2f}..{retained.max():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="rigid envs only")
    args = parser.parse_args()
    for env_id in RIGID_ENVS:
        run(env_id, NUM_ENVS, c_dim=4)
    if not args.quick:
        for env_id in GRANULAR_ENVS:
            run(env_id, NUM_ENVS_GRANULAR, c_dim=7)
        hold_check("CarryT2-v1")
    print("smoke test PASSED")
