"""Hidden physics parameters c: sampling and per-parallel-env application.

Plan (Part II, Build: environments):
  - Per-episode randomisation: build each parallel environment's objects as
    separate actors (ManiSkill "build separate" pattern), merge into one Actor,
    and set density, friction material, and centre-of-mass offset on each
    environment's rigid body component.
  - c sampler: log-uniform mass multiplier over [0.7, 1.4], log-uniform
    surface friction over [0.3, 0.7], COM offset uniform up to 15% of the
    object half-width with a random in-plane direction. For T1 (pouring) and
    T2 (carrying), add particle count and inter-particle friction; T2 also
    hides the handle offset.

c never enters observations unless the env is constructed with expose_c=True
(the privileged PPO teacher). Everything here is numpy / CPU because it runs
inside _load_scene, before the GPU simulation is initialised - which is also
why c is resampled per *reconfigure* rather than per reset: physical
properties are baked into the PhysX scene at initialisation. For training,
many parallel envs each carry their own c; to resample within one env slot,
pass reconfiguration_freq>=1 to the env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sapien
import sapien.physx as physx
from sapien.physx import PhysxRigidBodyComponent

from mani_skill.utils.structs import Actor

C_KEYS = ("mass_mult", "friction", "com_frac", "com_angle")
GRANULAR_KEYS = ("particle_count", "pp_friction", "handle_offset")
NOMINAL_C = dict(
    mass_mult=1.0, friction=0.5, com_frac=0.0, com_angle=0.0,
    particle_count=25.0, pp_friction=0.4, handle_offset=0.0,
)
OBJ_COLOR = np.array([12, 42, 160, 255]) / 255
PARTICLE_COLOR = np.array([230, 160, 20, 255]) / 255


@dataclass
class CTrainSpec:
    """Training-time ranges of the hidden parameters."""

    mass_mult_range: tuple = (0.7, 1.4)  # log-uniform
    friction_range: tuple = (0.3, 0.7)  # log-uniform, static == dynamic
    com_frac_max: float = 0.15  # uniform [0, max], fraction of half-width
    # granular extension (T1 pouring / T2 carrying)
    particle_count_range: tuple = (20, 30)  # uniform integer, inclusive
    pp_friction_range: tuple = (0.2, 0.8)  # log-uniform, inter-particle
    handle_offset_max: float = 0.02  # uniform [-max, max], metres (T2 only)

    def sample_single(self, rng: np.random.RandomState) -> dict:
        lo, hi = self.mass_mult_range
        mass_mult = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        lo, hi = self.friction_range
        friction = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        lo, hi = self.pp_friction_range
        pp_friction = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        n_lo, n_hi = self.particle_count_range
        return dict(
            mass_mult=mass_mult,
            friction=friction,
            com_frac=float(rng.uniform(0.0, self.com_frac_max)),
            com_angle=float(rng.uniform(0.0, 2 * np.pi)),
            particle_count=float(rng.randint(n_lo, n_hi + 1)),
            pp_friction=pp_friction,
            handle_offset=float(
                rng.uniform(-self.handle_offset_max, self.handle_offset_max)
            ),
        )


def resolve_c(
    num_envs: int,
    batched_rng,
    spec: CTrainSpec,
    override: dict | None = None,
    randomize: bool = True,
) -> dict:
    """Produce per-env c arrays (all keys, core + granular; tasks use what
    they need). override maps any key to a scalar (broadcast) or a
    length-num_envs array. Components absent from override are sampled from
    spec when randomize=True, else set to their nominal value. com_angle is a
    nuisance direction: when a com_frac override meets randomize=False it is
    still drawn reproducibly rather than pinned."""
    override = dict(override or {})
    keys = C_KEYS + GRANULAR_KEYS
    out = {k: np.empty(num_envs, dtype=np.float64) for k in keys}
    for i in range(num_envs):
        rng = batched_rng[i]
        base = spec.sample_single(rng) if randomize else dict(NOMINAL_C)
        if not randomize and "com_frac" in override and "com_angle" not in override:
            base["com_angle"] = float(rng.uniform(0.0, 2 * np.pi))
        for k in keys:
            if k in override:
                v = np.asarray(override[k], dtype=np.float64)
                out[k][i] = float(v) if v.ndim == 0 else float(v[i])
            else:
                out[k][i] = base[k]
    out["com_x_frac"] = out["com_frac"] * np.cos(out["com_angle"])
    out["com_y_frac"] = out["com_frac"] * np.sin(out["com_angle"])
    return out


def c_matrix(c: dict, granular: bool = False) -> np.ndarray:
    """The reported c vector per env. Core: (mass_mult, friction, com_x_frac,
    com_y_frac). Granular tasks append (particle_count, pp_friction,
    handle_offset)."""
    cols = [c["mass_mult"], c["friction"], c["com_x_frac"], c["com_y_frac"]]
    if granular:
        cols += [c["particle_count"], c["pp_friction"], c["handle_offset"]]
    return np.stack(cols, axis=1).astype(np.float32)


def _apply_com_offset(merged: Actor, c: dict, half_width: float) -> None:
    """COM offset on each parallel environment's rigid body component, as a
    fraction of the object half-width; records the realised mass in c."""
    masses = np.empty(len(merged._objs), dtype=np.float64)
    for i, ent in enumerate(merged._objs):
        comp: PhysxRigidBodyComponent = ent.find_component_by_type(
            PhysxRigidBodyComponent
        )
        cm = comp.cmass_local_pose
        comp.cmass_local_pose = sapien.Pose(
            p=[
                float(cm.p[0] + c["com_x_frac"][i] * half_width),
                float(cm.p[1] + c["com_y_frac"][i] * half_width),
                float(cm.p[2]),
            ],
            q=cm.q,
        )
        masses[i] = comp.mass
    c["mass_kg"] = masses


def build_randomized_box(
    env, half_size, c: dict, name: str = "obj", base_density: float = 1000.0
) -> Actor:
    """One box per parallel env with per-env physics, merged into one Actor.

    Density carries the mass multiplier (mass, inertia and default COM stay
    mutually consistent), a fresh PhysxMaterial per env carries surface
    friction (fresh objects avoid SAPIEN's shared-material optimisation), and
    the COM offset is applied afterwards through cmass_local_pose. Runs before
    GPU init, so every property reaches the solver.
    """
    if np.isscalar(half_size):
        half_size = [half_size] * 3
    scene = env.scene
    boxes = []
    for i in range(env.num_envs):
        builder = scene.create_actor_builder()
        mat = physx.PhysxMaterial(
            static_friction=float(c["friction"][i]),
            dynamic_friction=float(c["friction"][i]),
            restitution=0.0,
        )
        builder.add_box_collision(
            half_size=half_size,
            material=mat,
            density=float(base_density * c["mass_mult"][i]),
        )
        builder.add_box_visual(
            half_size=half_size,
            material=sapien.render.RenderMaterial(base_color=OBJ_COLOR),
        )
        builder.set_scene_idxs([i])
        builder.initial_pose = sapien.Pose(p=[0, 0, half_size[2]])
        box = builder.build(name=f"{name}_{i}")
        env.remove_from_state_dict_registry(box)
        boxes.append(box)
    merged = Actor.merge(boxes, name=name)
    env.add_to_state_dict_registry(merged)
    _apply_com_offset(merged, c, half_width=half_size[0])
    return merged


def build_randomized_cup(
    env,
    c: dict,
    name: str = "cup",
    inner_half: float = 0.025,
    wall_t: float = 0.005,
    depth: float = 0.06,
    base_density: float = 400.0,
    with_handle: bool = False,
    handle_len: float = 0.05,
    handle_half_t: float = 0.006,
) -> Actor:
    """Open-top square cup as a rigid compound (bottom + 4 walls), one per
    env, with per-env density (container inertia via mass_mult) and COM
    offset. with_handle adds a horizontal grip bar above the +x wall whose
    x-position is shifted by the hidden handle_offset (T2). Cup frame origin:
    bottom-outer center.
    """
    scene = env.scene
    outer = inner_half + wall_t
    zc = wall_t + depth / 2
    shapes = [
        ([outer, outer, wall_t / 2], [0, 0, wall_t / 2]),
        ([wall_t / 2, outer, depth / 2], [inner_half + wall_t / 2, 0, zc]),
        ([wall_t / 2, outer, depth / 2], [-inner_half - wall_t / 2, 0, zc]),
        ([inner_half, wall_t / 2, depth / 2], [0, inner_half + wall_t / 2, zc]),
        ([inner_half, wall_t / 2, depth / 2], [0, -inner_half - wall_t / 2, zc]),
    ]
    cups = []
    for i in range(env.num_envs):
        builder = scene.create_actor_builder()
        mat = physx.PhysxMaterial(
            static_friction=float(c["friction"][i]),
            dynamic_friction=float(c["friction"][i]),
            restitution=0.0,
        )
        density = float(base_density * c["mass_mult"][i])
        env_shapes = list(shapes)
        if with_handle:
            hx = inner_half + wall_t / 2 + float(c["handle_offset"][i])
            top = wall_t + depth
            # post rising from the +x wall, then the grip bar along x above it
            env_shapes.append(([wall_t / 2, handle_half_t, 0.01], [hx, 0, top + 0.01]))
            env_shapes.append(
                ([handle_len / 2, handle_half_t, handle_half_t], [hx, 0, top + 0.02 + handle_half_t])
            )
        for half, pos in env_shapes:
            builder.add_box_collision(
                half_size=half, material=mat, density=density,
                pose=sapien.Pose(p=pos),
            )
            builder.add_box_visual(
                half_size=half, pose=sapien.Pose(p=pos),
                material=sapien.render.RenderMaterial(base_color=OBJ_COLOR),
            )
        builder.set_scene_idxs([i])
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.0])
        cup = builder.build(name=f"{name}_{i}")
        env.remove_from_state_dict_registry(cup)
        cups.append(cup)
    merged = Actor.merge(cups, name=name)
    env.add_to_state_dict_registry(merged)
    _apply_com_offset(merged, c, half_width=inner_half)
    return merged


def build_particles(
    env,
    c: dict,
    max_count: int,
    radius: float = 0.005,
    density: float = 1000.0,
    name: str = "particle",
) -> list:
    """Granular contents (plan build note: rigid spheres, not fluid).

    Returns max_count merged Actors, each spanning all parallel envs, with the
    per-env inter-particle friction material. Which particles are *active* in
    env i is c['particle_count'][i]; inactive ones must be parked in the
    graveyard (see graveyard_pose) by the task's episode initialisation.

    Cost note: this creates max_count * num_envs actor builds per
    reconfigure, so granular tasks should train with fewer parallel envs than
    the rigid tasks.
    """
    scene = env.scene
    particles = []
    for j in range(max_count):
        per_env = []
        for i in range(env.num_envs):
            builder = scene.create_actor_builder()
            mat = physx.PhysxMaterial(
                static_friction=float(c["pp_friction"][i]),
                dynamic_friction=float(c["pp_friction"][i]),
                restitution=0.0,
            )
            builder.add_sphere_collision(radius=radius, material=mat, density=density)
            builder.add_sphere_visual(
                radius=radius,
                material=sapien.render.RenderMaterial(base_color=PARTICLE_COLOR),
            )
            builder.set_scene_idxs([i])
            builder.initial_pose = graveyard_pose(j)
            p = builder.build(name=f"{name}_{j}_{i}")
            env.remove_from_state_dict_registry(p)
            per_env.append(p)
        pj = Actor.merge(per_env, name=f"{name}_{j}")
        env.add_to_state_dict_registry(pj)
        particles.append(pj)
    return particles


def graveyard_pose(j: int) -> sapien.Pose:
    """Resting spot on the floor, far from the table, for inactive particles."""
    return sapien.Pose(p=[2.0 + 0.03 * (j % 10), 2.0 + 0.03 * (j // 10), 0.006])
