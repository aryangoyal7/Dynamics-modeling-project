"""Task environments (updated plan, Part I 'The tasks' / Part II 'Build: environments').

  SlideToSlotT3-v1   PRIMARY, built first. Slide an object along a channel into
                     a slot; recover when it stops short or overshoots. The
                     recovery *direction* flips with friction. Logs attempt
                     count and whether the first recovery push went the correct
                     way (isolates inference from execution).
  PourT1-v1          Tilt a container to transfer granular contents (rigid
                     spheres, not fluid) into a target vessel. Outflow lags the
                     tilt. Harder tier: added once T3 works.
  CarryT2-v1         Move a mug held by its handle without spilling or
                     dropping; the robot's own acceleration excites the
                     disturbance it must damp. Harder tier.
  PickPlaceT4-v1     Specificity control: once grasped the object is rigid to
                     the gripper and its hidden physics stop mattering.

Each has a *Teacher-v1 twin whose only difference is expose_c=True: the hidden
physics vector c enters the observation. Recorded student data must come from
expose_c=False observations; c is retrievable as metadata via
env.unwrapped.get_c(). All arms share the Panda and emit actions directly.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from dynmod.envs.randomization import (
    CTrainSpec,
    build_particles,
    build_randomized_box,
    build_randomized_cup,
    build_randomized_tee,
    c_matrix,
    graveyard_pose,
    resolve_c,
)


def _quat_zz(q: torch.Tensor) -> torch.Tensor:
    """(3,3) element R_zz of the rotation matrix: cos(tilt from upright)."""
    return 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)


# =========================================================================== #
# Shared base: table + hidden-parameter pipeline.                             #
# =========================================================================== #
class DynBaseEnv(BaseEnv):
    """Extra constructor args over BaseEnv:
      expose_c            put c into the observation (privileged teacher only)
      randomize_c         sample c per env (False -> nominal physics)
      c_override          dict overriding c components (scalar or per-env
                          array); pins envs to delta-grid points at evaluation
      deterministic_spawn fixed spawns/goals (used by the physics gate)
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda
    GRANULAR = False  # granular tasks report the extended c vector

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        expose_c: bool = False,
        randomize_c: bool = True,
        c_override: dict | None = None,
        deterministic_spawn: bool = False,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.expose_c = expose_c
        self.randomize_c = randomize_c
        self.c_override = c_override
        self.deterministic_spawn = deterministic_spawn
        self.c_spec = CTrainSpec(**getattr(self, "C_SPEC_KW", {}))
        self._c = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128,
                         fov=np.pi / 2, near=0.01, far=100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._c = resolve_c(
            self.num_envs,
            self._batched_episode_rng,
            self.c_spec,
            override=self.c_override,
            randomize=self.randomize_c,
        )
        self.c_tensor = common.to_tensor(
            c_matrix(self._c, granular=self.GRANULAR), device=self.device
        )
        self._build_task(options)

    def _build_task(self, options: dict):
        raise NotImplementedError

    def get_c(self) -> dict:
        """Per-env hidden parameters as numpy arrays. Metadata only: never
        write this into recorded observations."""
        return {k: np.array(v) for k, v in self._c.items()}

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._init_task(env_idx)

    def _init_task(self, env_idx: torch.Tensor):
        raise NotImplementedError

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                obj_lin_vel=self.obj.linear_velocity,
                obj_ang_vel=self.obj.angular_velocity,
            )
            obs.update(self._task_state_obs())
        if self.expose_c:
            obs["c_params"] = self.c_tensor
        return obs

    def _task_state_obs(self) -> Dict[str, torch.Tensor]:
        return {}

    def _tcp_to_obj_dist(self) -> torch.Tensor:
        return torch.linalg.norm(self.agent.tcp.pose.p - self.obj.pose.p, dim=1)


# =========================================================================== #
# T3 (PRIMARY): slide an object along a channel into a slot, with recovery.   #
# =========================================================================== #
@register_env("SlideToSlotT3-v1", max_episode_steps=200)
class SlideToSlotT3Env(DynBaseEnv):
    """A walled channel along +x confines the object laterally; a marked slot
    sits at slot_x inside it. Friction sets the stopping distance: stop short
    and the correct recovery is another push in +x, overshoot and it is a push
    back in -x. Success permits multiple attempts (long horizon) and requires
    the object at rest in the slot with the end-effector clear.

    Logged per plan: attempt_count (distinct slides of the object) and
    first_recovery_correct: after the first slide ends at rest outside the
    slot, did the *next* motion of the object start in the direction of the
    slot? (-1 = no recovery yet, 0 = wrong way, 1 = correct way.)
    """

    obj_half = 0.02
    slot_half_width = 0.025  # success tolerance along x
    release_dist = 0.06
    static_vel = 0.03
    slot_x_range = (0.15, 0.21)
    spawn_x_range = (-0.10, -0.04)

    CHANNEL_INNER_HALF_Y = 0.06
    CHANNEL_X = (-0.20, 0.58)
    # Tunnel: a low plate over the channel that the object slides under but
    # the gripper cannot enter. Added 2026-08-17 after the base-policy gate
    # exposed shepherding (TCP within 6cm of the object for ~80% of the final
    # approach): with the slot fully reachable the policy escorts the object
    # and never commits, making success physics-insensitive. Placed directly
    # BEFORE the slot so the final approach is always a committed ballistic
    # pass: the policy may escort up to the tunnel mouth (keeps the task
    # learnable), but the last stretch is free flight whose stopping point c
    # decides. Undershoots stop in open channel and can be re-flicked;
    # overshoots are reachable from above - both recovery directions survive.
    TUNNEL_X = (0.08, 0.13)
    # Second roof segment past the slot window (added 2026-08-18): without
    # it, deliberate overshoot was fully recoverable by contact (bounce off
    # the end wall, escort back), a c-free winning strategy - measured: a
    # c-AWARE scripted flick beat a c-blind one by NOTHING, and the c-seeing
    # RL teacher converged exactly to the c-blind script's success. With two
    # roofs bracketing an ~11 cm open window, undershoot and overshoot both
    # strand the object under a roof; landing in the window requires a
    # correctly c-calibrated flick, small in-window nudges keep the
    # recovery-direction measurement, and far overshoots (past roof 2) can
    # only be recovered by a committed REVERSE flick back through roof 2.
    TUNNEL2_X = (0.245, 0.32)
    TUNNEL_CLEAR_Z = 0.055  # underside height: object (0.04) passes, hand does not
    # Slick channel floor (added 2026-08-18): per-bucket calibration showed
    # the strike is command-saturated - even max-strength flicks undershoot
    # for high-friction cubes (19-22% ceiling), so no controller can express
    # c-knowledge. Effective sliding friction is the cube-floor average; a
    # low-friction floor drops required launch speeds ~40% into the arm's
    # authority while preserving the full relative spread of cube friction.
    FLOOR_FRICTION = 0.05
    FLOOR_T = 0.002  # plate thickness; object rest height = obj_half + FLOOR_T

    @property
    def _default_sensor_configs(self):
        # frame the channel (spawn ~-0.1 to past the slot ~0.2), not the
        # generic tabletop center; vision-variant observations come from here
        pose = sapien_utils.look_at(eye=[0.42, -0.42, 0.45], target=[0.1, 0.0, 0.0])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128,
                         fov=np.pi / 2, near=0.01, far=100)
        ]

    def _build_task(self, options: dict):
        self.obj = build_randomized_box(
            self, self.obj_half, self._c, name="slide_obj"
        )
        # static channel: two rails and an end stop, shared across all envs
        x0, x1 = self.CHANNEL_X
        cx, hx = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
        wall_h, wall_t = 0.02, 0.01
        yc = self.CHANNEL_INNER_HALF_Y + wall_t
        builder = self.scene.create_actor_builder()
        gray = sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 1])
        plate_t = 0.008
        shapes = [
            ([hx, wall_t, wall_h], [cx, yc, wall_h]),
            ([hx, wall_t, wall_h], [cx, -yc, wall_h]),
            ([wall_t, yc + wall_t, wall_h], [x1 + wall_t, 0, wall_h]),
        ]
        # slick floor plate spanning the whole channel (no mid-flight ledges)
        import sapien.physx as _physx
        floor_builder = self.scene.create_actor_builder()
        floor_builder.add_box_collision(
            half_size=[hx, self.CHANNEL_INNER_HALF_Y, self.FLOOR_T / 2],
            pose=sapien.Pose(p=[cx, 0, self.FLOOR_T / 2]),
            material=_physx.PhysxMaterial(self.FLOOR_FRICTION, self.FLOOR_FRICTION, 0.0),
        )
        floor_builder.add_box_visual(
            half_size=[hx, self.CHANNEL_INNER_HALF_Y, self.FLOOR_T / 2],
            pose=sapien.Pose(p=[cx, 0, self.FLOOR_T / 2]),
            material=sapien.render.RenderMaterial(base_color=[0.75, 0.85, 0.95, 1]),
        )
        floor_builder.initial_pose = sapien.Pose()
        self.channel_floor = floor_builder.build_static(name="channel_floor")
        for tx0, tx1 in (self.TUNNEL_X, self.TUNNEL2_X):
            tcx, thx = 0.5 * (tx0 + tx1), 0.5 * (tx1 - tx0)
            shapes += [
                # roof spanning the channel
                ([thx, yc + wall_t, plate_t], [tcx, 0, self.TUNNEL_CLEAR_Z + plate_t]),
                # side pillars filling the gap between rail top and roof
                ([thx, wall_t, self.TUNNEL_CLEAR_Z / 2 + plate_t],
                 [tcx, yc, self.TUNNEL_CLEAR_Z / 2 + plate_t]),
                ([thx, wall_t, self.TUNNEL_CLEAR_Z / 2 + plate_t],
                 [tcx, -yc, self.TUNNEL_CLEAR_Z / 2 + plate_t]),
            ]
        for half, pos in shapes:
            builder.add_box_collision(half_size=half, pose=sapien.Pose(p=pos))
            builder.add_box_visual(half_size=half, pose=sapien.Pose(p=pos), material=gray)
        builder.initial_pose = sapien.Pose()
        self.channel = builder.build_static(name="channel")
        # visual-only slot marker (kinematic so its pose can move per episode)
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[self.slot_half_width, self.CHANNEL_INNER_HALF_Y, 6e-4],
            material=sapien.render.RenderMaterial(base_color=[0.1, 0.7, 0.1, 1]),
        )
        builder.initial_pose = sapien.Pose(p=[0.15, 0, 1e-3])
        self.slot_marker = builder.build_kinematic(name="slot_marker")
        # recovery-tracking buffers
        n = self.num_envs
        self._phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self._attempts = torch.zeros(n, device=self.device)
        self._needed_dir = torch.zeros(n, device=self.device)
        self._rec_correct = -torch.ones(n, device=self.device)
        self._prev_moving = torch.zeros(n, dtype=torch.bool, device=self.device)

    def _init_task(self, env_idx: torch.Tensor):
        b = len(env_idx)
        xyz = torch.zeros((b, 3), device=self.device)
        lo, hi = self.spawn_x_range
        slo, shi = self.slot_x_range
        if self.deterministic_spawn:
            xyz[:, 0] = -0.06
            slot_x = torch.full((b,), 0.5 * (slo + shi), device=self.device)
        else:
            xyz[:, 0] = torch.rand(b, device=self.device) * (hi - lo) + lo
            xyz[:, 1] = (torch.rand(b, device=self.device) * 2 - 1) * 0.02
            slot_x = torch.rand(b, device=self.device) * (shi - slo) + slo
        xyz[:, 2] = self.obj_half + self.FLOOR_T
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))
        self.obj.set_linear_velocity(torch.zeros((b, 3), device=self.device))
        self.obj.set_angular_velocity(torch.zeros((b, 3), device=self.device))
        marker = torch.zeros((b, 3), device=self.device)
        marker[:, 0] = slot_x
        marker[:, 2] = 1e-3
        self.slot_marker.set_pose(Pose.create_from_pq(p=marker))
        self._phase[env_idx] = 0
        self._attempts[env_idx] = 0
        self._needed_dir[env_idx] = 0
        self._rec_correct[env_idx] = -1
        self._prev_moving[env_idx] = False

    def evaluate(self):
        obj_x = self.obj.pose.p[:, 0]
        slot_x = self.slot_marker.pose.p[:, 0]
        dx = slot_x - obj_x  # signed: >0 means slot is further along +x
        in_slot = dx.abs() < self.slot_half_width
        speed = torch.linalg.norm(self.obj.linear_velocity, dim=1)
        moving = speed > self.static_vel
        static = ~moving
        tcp_dist = self._tcp_to_obj_dist()
        released = tcp_dist > self.release_dist

        # attempt / recovery bookkeeping (batched state machine)
        edge = moving & ~self._prev_moving
        self._attempts += edge.float()
        # phase 0 -> 1: first slide begins
        self._phase = torch.where(edge & (self._phase == 0),
                                  torch.ones_like(self._phase), self._phase)
        # phase 1 -> 2: first slide ends at rest outside the slot
        rest_out = (self._phase == 1) & static & ~in_slot
        self._needed_dir = torch.where(rest_out, torch.sign(dx), self._needed_dir)
        self._phase = torch.where(rest_out, 2 * torch.ones_like(self._phase), self._phase)
        # phase 2 -> 3: object moves again; record the direction it went
        rec = (self._phase == 2) & edge
        rec_dir = torch.sign(self.obj.linear_velocity[:, 0])
        self._rec_correct = torch.where(
            rec, (rec_dir == self._needed_dir).float(), self._rec_correct
        )
        self._phase = torch.where(rec, 3 * torch.ones_like(self._phase), self._phase)
        self._prev_moving = moving

        return dict(
            success=in_slot & static & released,
            in_slot=in_slot,
            obj_static=static,
            released=released,
            slot_dx=dx,
            obj_to_slot_dist=dx.abs(),
            tcp_to_obj_dist=tcp_dist,
            obj_speed=speed,
            attempt_count=self._attempts.clone(),
            first_recovery_correct=self._rec_correct.clone(),
        )

    def _task_state_obs(self):
        return dict(slot_pos=self.slot_marker.pose.p)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reaching = 1 - torch.tanh(
            5.0 * (info["tcp_to_obj_dist"] - self.obj_half).clamp(min=0.0)
        )
        progress = 1 - torch.tanh(4.0 * info["obj_to_slot_dist"])
        settle = info["in_slot"].float() * (1 - torch.tanh(3.0 * info["obj_speed"]))
        release = (info["in_slot"] & info["obj_static"]).float() * info["released"].float()
        # launch incentive: escorting is physically blocked by the tunnel, so
        # exploration must discover flicking; reward object velocity toward
        # the slot while it is before the tunnel exit (teacher-side shaping
        # only - students imitate demonstrations, never see rewards)
        before_exit = self.obj.pose.p[:, 0] < (self.TUNNEL_X[1] + 0.02)
        vx = self.obj.linear_velocity[:, 0]
        launch = before_exit.float() * torch.tanh(2.0 * vx.clamp(min=0.0))
        reward = 0.5 * reaching + 2.0 * progress + settle + release + 0.8 * launch
        reward[info["success"]] = 6.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.0


# =========================================================================== #
# Granular base for T1/T2: cup + rigid-sphere contents.                       #
# =========================================================================== #
class GranularEnv(DynBaseEnv):
    """Contents are 20-30 rigid spheres (plan build note: granular rather than
    fluid). One merged Actor per particle slot; slots beyond the env's hidden
    particle_count are parked in a graveyard far from the table. Particle
    poses are never observed: fill level and content friction act only
    through the physics.

    spawn_grasped=True places the container in the closed gripper after reset
    (full resets only). The grip axis is derived from the TCP orientation at
    run time.
    """

    GRANULAR = True
    particle_radius = 0.005
    cup_inner = 0.025
    cup_wall = 0.005
    cup_depth = 0.06
    WITH_HANDLE = False

    def __init__(self, *args, spawn_grasped: bool | None = None, **kwargs):
        if spawn_grasped is not None:
            self.spawn_grasped = spawn_grasped
        super().__init__(*args, **kwargs)

    def _build_task(self, options: dict):
        self.obj = build_randomized_cup(
            self,
            self._c,
            name="cup",
            inner_half=self.cup_inner,
            wall_t=self.cup_wall,
            depth=self.cup_depth,
            with_handle=self.WITH_HANDLE,
        )
        self.max_particles = int(self.c_spec.particle_count_range[1])
        self.particles = build_particles(
            self, self._c, max_count=self.max_particles, radius=self.particle_radius
        )
        counts = common.to_tensor(self._c["particle_count"], device=self.device)
        slots = torch.arange(self.max_particles, device=self.device)
        self.particle_active = slots[:, None] < counts[None, :]  # (P, N)
        # particle fill offsets in the cup frame: 3x3 grid, stacked layers
        s = 2.2 * self.particle_radius
        offs = []
        for j in range(self.max_particles):
            layer, k = divmod(j, 9)
            row, col = divmod(k, 3)
            offs.append([(col - 1) * s, (row - 1) * s,
                         self.cup_wall + self.particle_radius + layer * s])
        self._fill_offsets = torch.tensor(offs, dtype=torch.float32, device=self.device)
        self._graveyard = torch.stack(
            [common.to_tensor(np.array(graveyard_pose(j).p, dtype=np.float32),
                              device=self.device)
             for j in range(self.max_particles)]
        )
        self._build_granular_task(options)

    def _build_granular_task(self, options: dict):
        raise NotImplementedError

    # -- particle helpers --
    def _place_particles(self, env_idx: torch.Tensor, cup_pos: torch.Tensor):
        """Fill active particle slots inside the cup (cup assumed upright);
        park inactive slots in the graveyard."""
        b = len(env_idx)
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        zero = torch.zeros((b, 3), device=self.device)
        for j, pj in enumerate(self.particles):
            active = self.particle_active[j][env_idx]
            pos = torch.where(
                active[:, None],
                cup_pos + self._fill_offsets[j][None, :],
                self._graveyard[j][None, :].expand(b, 3),
            )
            pj.set_pose(Pose.create_from_pq(p=pos, q=q))
            pj.set_linear_velocity(zero)
            pj.set_angular_velocity(zero)

    def _particle_positions(self) -> torch.Tensor:
        return torch.stack([p.pose.p for p in self.particles], dim=0)  # (P, N, 3)

    def _in_cup_frac(self) -> torch.Tensor:
        """Fraction of active particles still inside the cup. Cylindrical
        approximation around the cup axis: accurate near upright, adequate as
        a retention metric."""
        pp = self._particle_positions()
        cup = self.obj.pose.p[None, :, :]
        horiz = torch.linalg.norm(pp[..., :2] - cup[..., :2], dim=-1)
        dz = pp[..., 2] - cup[..., 2]
        inside = (horiz < self.cup_inner * 1.3) & (dz > 0) & (
            dz < self.cup_depth + 0.03
        )
        act = self.particle_active
        return (inside & act).sum(0).float() / act.sum(0).clamp(min=1).float()

    # -- grasp spawning --
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        partial = isinstance(options, dict) and "env_idx" in options
        if getattr(self, "spawn_grasped", False) and not partial:
            self._snap_to_grasp()
            if self.device.type == "cuda":
                self.scene._gpu_apply_all()
                self.scene._gpu_fetch_all()
            obs = self.get_obs()
        return obs, info

    def _grip_local(self) -> torch.Tensor:
        """(N, 3) grip point in the cup frame."""
        raise NotImplementedError

    def _snap_to_grasp(self):
        n = self.num_envs
        tcp = self.agent.tcp.pose
        # bar/wall axis: TCP x-axis (perpendicular to the finger-closing axis)
        # projected to the horizontal plane -> the cup yaw that fits the grip
        qw, qx, qy, qz = tcp.q[:, 0], tcp.q[:, 1], tcp.q[:, 2], tcp.q[:, 3]
        x_ax = torch.stack(
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy + qw * qz)], dim=1
        )  # world xy of tcp x-axis
        yaw = torch.atan2(x_ax[:, 1], x_ax[:, 0])
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        grip = self._grip_local()
        grip_w = torch.stack(
            [cy * grip[:, 0] - sy * grip[:, 1],
             sy * grip[:, 0] + cy * grip[:, 1],
             grip[:, 2]], dim=1,
        )
        cup_pos = tcp.p - grip_w
        cup_q = torch.stack(
            [torch.cos(yaw / 2), torch.zeros(n, device=self.device),
             torch.zeros(n, device=self.device), torch.sin(yaw / 2)], dim=1,
        )
        self.obj.set_pose(Pose.create_from_pq(p=cup_pos, q=cup_q))
        zero = torch.zeros((n, 3), device=self.device)
        self.obj.set_linear_velocity(zero)
        self.obj.set_angular_velocity(zero)
        qpos = self.agent.robot.get_qpos()
        qpos[:, -2:] = self._grip_half_gap()
        self.agent.robot.set_qpos(qpos)
        # refill contents relative to the new cup pose (ignore yaw: the fill
        # grid is symmetric enough under rotation about the cup axis)
        self._place_particles(torch.arange(n, device=self.device), cup_pos)

    def _grip_half_gap(self) -> float:
        raise NotImplementedError

    def _cup_tilt(self) -> torch.Tensor:
        return torch.arccos(_quat_zz(self.obj.pose.q).clamp(-1, 1))


# =========================================================================== #
# T1: pouring. Tilt the container to transfer contents into a target vessel. #
# =========================================================================== #
@register_env("PourT1-v1", max_episode_steps=150)
class PourT1Env(GranularEnv):
    """Target vessel (static basin) on the table; success when >= 80% of the
    active particles end up inside it. spawn_grasped defaults False: the cup
    starts on the table and grasping is part of the task (flip it on if the
    teacher cannot learn the grasp; the pour dynamics are the point).
    Continuous measures: transfer_frac, spill_frac.
    """

    spawn_grasped = False
    BASIN_POS = (0.05, 0.15)
    basin_inner = 0.05
    basin_wall = 0.005
    basin_depth = 0.03
    CUP_SPAWN = (0.0, -0.12)
    transfer_target = 0.8

    def _build_granular_task(self, options: dict):
        bx, by = self.BASIN_POS
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=[0.7, 0.7, 0.75, 1])
        o, t, d = self.basin_inner + self.basin_wall, self.basin_wall, self.basin_depth
        zc = t + d / 2
        for half, pos in [
            ([o, o, t / 2], [bx, by, t / 2]),
            ([t / 2, o, d / 2], [bx + self.basin_inner + t / 2, by, zc]),
            ([t / 2, o, d / 2], [bx - self.basin_inner - t / 2, by, zc]),
            ([self.basin_inner, t / 2, d / 2], [bx, by + self.basin_inner + t / 2, zc]),
            ([self.basin_inner, t / 2, d / 2], [bx, by - self.basin_inner - t / 2, zc]),
        ]:
            builder.add_box_collision(half_size=half, pose=sapien.Pose(p=pos))
            builder.add_box_visual(half_size=half, pose=sapien.Pose(p=pos), material=mat)
        builder.initial_pose = sapien.Pose()
        self.basin = builder.build_static(name="basin")
        self._basin_center = torch.tensor(
            [bx, by], dtype=torch.float32, device=self.device
        )

    def _init_task(self, env_idx: torch.Tensor):
        b = len(env_idx)
        pos = torch.zeros((b, 3), device=self.device)
        pos[:, 0] = self.CUP_SPAWN[0]
        pos[:, 1] = self.CUP_SPAWN[1]
        if not self.deterministic_spawn:
            pos[:, :2] += (torch.rand((b, 2), device=self.device) * 2 - 1) * 0.02
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        self.obj.set_pose(Pose.create_from_pq(p=pos, q=q))
        zero = torch.zeros((b, 3), device=self.device)
        self.obj.set_linear_velocity(zero)
        self.obj.set_angular_velocity(zero)
        self._place_particles(env_idx, pos)

    def _grip_local(self) -> torch.Tensor:
        g = torch.zeros((self.num_envs, 3), device=self.device)
        g[:, 1] = self.cup_inner + self.cup_wall / 2  # +y wall
        g[:, 2] = self.cup_wall + self.cup_depth - 0.012  # just below the rim
        return g

    def _grip_half_gap(self) -> float:
        return self.cup_wall / 2 + 0.001

    def evaluate(self):
        pp = self._particle_positions()
        act = self.particle_active
        in_basin = (
            ((pp[..., :2] - self._basin_center[None, None, :]).abs()
             < self.basin_inner).all(-1)
            & (pp[..., 2] < self.basin_wall + self.basin_depth + 0.02)
        )
        counts = act.sum(0).clamp(min=1).float()
        transfer_frac = (in_basin & act).sum(0).float() / counts
        in_cup = self._in_cup_frac()
        spill_frac = (1 - transfer_frac - in_cup).clamp(min=0.0)
        return dict(
            success=transfer_frac >= self.transfer_target,
            transfer_frac=transfer_frac,
            retained_frac=in_cup,
            spill_frac=spill_frac,
            tcp_to_obj_dist=self._tcp_to_obj_dist(),
            cup_tilt=self._cup_tilt(),
        )

    def _task_state_obs(self):
        basin = torch.zeros((self.num_envs, 3), device=self.device)
        basin[:, :2] = self._basin_center
        basin[:, 2] = self.basin_wall + self.basin_depth
        return dict(basin_pos=basin)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reaching = 1 - torch.tanh(5.0 * info["tcp_to_obj_dist"])
        cup_xy = self.obj.pose.p[:, :2]
        over = 1 - torch.tanh(5.0 * torch.linalg.norm(
            cup_xy - self._basin_center[None, :], dim=1))
        reward = 0.5 * reaching + over + 2.0 * info["transfer_frac"] - info["spill_frac"]
        reward[info["success"]] = 4.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0


# =========================================================================== #
# T2: carrying. Move a mug held by its handle without spilling or dropping.   #
# =========================================================================== #
@register_env("CarryT2-v1", max_episode_steps=120)
class CarryT2Env(GranularEnv):
    """Starts held (spawn_grasped default True): the grip bar above the rim
    sits between the closed fingers; the hidden handle_offset shifts the bar
    along the rim and with it the hang torque. Success: mug near the goal,
    near-upright, slow, and >= 90% of contents retained. Continuous measure:
    peak_content_disp, the running max of content displacement from the mug
    axis (measures the self-induced disturbance directly).
    """

    spawn_grasped = True
    WITH_HANDLE = True
    goal_tol = 0.03
    tilt_tol = 0.35
    retain_target = 0.9
    static_vel = 0.08
    handle_half_t = 0.006

    def _build_granular_task(self, options: dict):
        self.goal_site = actors.build_sphere(
            self.scene, radius=0.01, color=[0, 1, 0, 1], name="goal_site",
            body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[0.05, 0, 0.2]),
        )
        self._hidden_objects.append(self.goal_site)
        self._peak_disp = torch.zeros(self.num_envs, device=self.device)

    def _init_task(self, env_idx: torch.Tensor):
        b = len(env_idx)
        # placeholder on-table pose; _snap_to_grasp overrides it post-reset
        pos = torch.zeros((b, 3), device=self.device)
        pos[:, 1] = -0.1
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        self.obj.set_pose(Pose.create_from_pq(p=pos, q=q))
        zero = torch.zeros((b, 3), device=self.device)
        self.obj.set_linear_velocity(zero)
        self.obj.set_angular_velocity(zero)
        self._place_particles(env_idx, pos)
        goal = torch.zeros((b, 3), device=self.device)
        if self.deterministic_spawn:
            goal[:, 0], goal[:, 2] = 0.08, 0.25
        else:
            goal[:, 0] = torch.rand(b, device=self.device) * 0.15 - 0.05
            goal[:, 1] = torch.rand(b, device=self.device) * 0.3 - 0.15
            goal[:, 2] = torch.rand(b, device=self.device) * 0.15 + 0.15
        self.goal_site.set_pose(Pose.create_from_pq(p=goal))
        self._peak_disp[env_idx] = 0

    def _grip_local(self) -> torch.Tensor:
        g = torch.zeros((self.num_envs, 3), device=self.device)
        g[:, 0] = self.cup_inner + self.cup_wall / 2 + common.to_tensor(
            self._c["handle_offset"], device=self.device
        )
        g[:, 2] = self.cup_wall + self.cup_depth + 0.02 + self.handle_half_t
        return g

    def _grip_half_gap(self) -> float:
        return self.handle_half_t + 0.001

    def evaluate(self):
        retained = self._in_cup_frac()
        tilt = self._cup_tilt()
        mug_to_goal = torch.linalg.norm(
            self.obj.pose.p - self.goal_site.pose.p, dim=1
        )
        speed = torch.linalg.norm(self.obj.linear_velocity, dim=1)
        pp = self._particle_positions()
        act = self.particle_active
        disp = torch.linalg.norm(
            pp[..., :2] - self.obj.pose.p[None, :, :2], dim=-1
        )
        disp = torch.where(act, disp, torch.zeros_like(disp)).max(0).values
        self._peak_disp = torch.maximum(self._peak_disp, disp)
        return dict(
            success=(mug_to_goal < self.goal_tol) & (tilt < self.tilt_tol)
            & (retained >= self.retain_target) & (speed < self.static_vel),
            mug_to_goal_dist=mug_to_goal,
            retained_frac=retained,
            cup_tilt=tilt,
            mug_speed=speed,
            peak_content_disp=self._peak_disp.clone(),
            tcp_to_obj_dist=self._tcp_to_obj_dist(),
        )

    def _task_state_obs(self):
        return dict(goal_pos=self.goal_site.pose.p)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        approach = 1 - torch.tanh(5.0 * info["mug_to_goal_dist"])
        upright = 1 - torch.tanh(3.0 * info["cup_tilt"])
        reward = approach + 0.5 * upright + info["retained_frac"]
        reward[info["success"]] = 4.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0


# =========================================================================== #
# T4 (control): pick-and-place. Hidden physics stop mattering once grasped.   #
# =========================================================================== #
@register_env("PickPlaceT4-v1", max_episode_steps=80)
class PickPlaceT4Env(DynBaseEnv):
    obj_half = 0.02
    goal_thresh = 0.025

    def _build_task(self, options: dict):
        self.obj = build_randomized_box(self, self.obj_half, self._c, name="cube")
        self.goal_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1],
            name="goal_site", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
        self._hidden_objects.append(self.goal_site)

    def _init_task(self, env_idx: torch.Tensor):
        b = len(env_idx)
        xyz = torch.zeros((b, 3), device=self.device)
        if not self.deterministic_spawn:
            xyz[:, 0] = torch.rand(b, device=self.device) * 0.16 - 0.10
            xyz[:, 1] = torch.rand(b, device=self.device) * 0.16 - 0.08
        else:
            xyz[:, 0] = -0.06
        xyz[:, 2] = self.obj_half
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))
        zero = torch.zeros((b, 3), device=self.device)
        self.obj.set_linear_velocity(zero)
        self.obj.set_angular_velocity(zero)
        goal = torch.zeros((b, 3), device=self.device)
        if self.deterministic_spawn:
            goal[:, 2] = 0.2
        else:
            goal[:, 0] = torch.rand(b, device=self.device) * 0.16 - 0.08
            goal[:, 1] = torch.rand(b, device=self.device) * 0.16 - 0.08
            goal[:, 2] = torch.rand(b, device=self.device) * 0.20 + 0.10
        self.goal_site.set_pose(Pose.create_from_pq(p=goal))

    def evaluate(self):
        obj_to_goal = torch.linalg.norm(
            self.goal_site.pose.p - self.obj.pose.p, dim=1
        )
        is_placed = obj_to_goal <= self.goal_thresh
        return dict(
            success=is_placed & self.agent.is_static(0.2),
            obj_to_goal_dist=obj_to_goal,
            is_placed=is_placed,
            is_grasped=self.agent.is_grasping(self.obj),
            tcp_to_obj_dist=self._tcp_to_obj_dist(),
        )

    def _task_state_obs(self):
        return dict(goal_pos=self.goal_site.pose.p)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reaching = 1 - torch.tanh(5.0 * info["tcp_to_obj_dist"])
        grasped = info["is_grasped"].float()
        place = (1 - torch.tanh(5.0 * info["obj_to_goal_dist"])) * grasped
        qvel = self.agent.robot.get_qvel()[..., :-2]
        static = (1 - torch.tanh(5.0 * torch.linalg.norm(qvel, dim=1))) * info[
            "is_placed"
        ].float()
        reward = reaching + grasped + place + static
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0


@register_env("CliffTossT5-v1", max_episode_steps=200)
class CliffTossT5Env(SlideToSlotT3Env):
    """Committed toss off a raised slick deck (added 2026-08-18 after the
    granular T1/T2 both measured zero physics premium and were descoped).

    The object starts on a slick deck 10 cm above the table. The controller
    charge-flicks it off the cliff edge; from that moment physics owns the
    outcome twice over: flight range scales with launch speed (which mass and
    friction set, given a fixed strike), and after landing the plain table's
    friction brakes the slide. Success = object at rest on the slot marker
    with the hand released. There are no recovery regions and no roofs: this
    task assumes a SCRIPTED c-aware teacher (the T3 finding - RL teachers
    plateau below the quality bar while the calibrated script clears it), so
    there is no reward-hacker to fence out with geometry. Undershoot lands
    short, overshoot slides long - both end at rest off the slot, and the
    controller never touches the object after the cliff.
    """

    DECK_T = 0.10           # deck top height - the cliff drop
    DECK_X = (-0.20, 0.10)  # slick launch deck along x
    slot_x_range = (0.24, 0.34)
    spawn_x_range = (-0.10, -0.04)

    def _build_task(self, options: dict):
        self.obj = build_randomized_box(
            self, self.obj_half, self._c, name="slide_obj"
        )
        import sapien.physx as _physx
        x0, x1 = self.CHANNEL_X
        d0, d1 = self.DECK_X
        wall_t = 0.01
        yc = self.CHANNEL_INNER_HALF_Y + wall_t
        gray = sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 1])

        # slick deck: one tall static box, low-friction all over
        dcx, dhx = 0.5 * (d0 + d1), 0.5 * (d1 - d0)
        fb = self.scene.create_actor_builder()
        fb.add_box_collision(
            half_size=[dhx, self.CHANNEL_INNER_HALF_Y, self.DECK_T / 2],
            pose=sapien.Pose(p=[dcx, 0, self.DECK_T / 2]),
            material=_physx.PhysxMaterial(self.FLOOR_FRICTION, self.FLOOR_FRICTION, 0.0),
        )
        fb.add_box_visual(
            half_size=[dhx, self.CHANNEL_INNER_HALF_Y, self.DECK_T / 2],
            pose=sapien.Pose(p=[dcx, 0, self.DECK_T / 2]),
            material=sapien.render.RenderMaterial(base_color=[0.75, 0.85, 0.95, 1]),
        )
        fb.initial_pose = sapien.Pose()
        self.deck = fb.build_static(name="deck")

        # rails along the whole run: tops only 3 cm above the deck so the
        # wrist can dip between them (28 cm-tall rails made the push pose
        # unreachable - probe 2026-08-18: 92% of episodes never touched the
        # block); plus a taller end stop on the table
        rail_hz = self.DECK_T / 2 + 0.015
        cx, hx = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
        builder = self.scene.create_actor_builder()
        for half, pos in [
            ([hx, wall_t, rail_hz], [cx, yc, rail_hz]),
            ([hx, wall_t, rail_hz], [cx, -yc, rail_hz]),
            ([wall_t, yc + wall_t, 0.08], [x1 + wall_t, 0, 0.08]),
        ]:
            builder.add_box_collision(half_size=half, pose=sapien.Pose(p=pos))
            builder.add_box_visual(half_size=half, pose=sapien.Pose(p=pos), material=gray)
        builder.initial_pose = sapien.Pose()
        self.channel = builder.build_static(name="channel")

        # visual-only slot marker on the table
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[self.slot_half_width, self.CHANNEL_INNER_HALF_Y, 6e-4],
            material=sapien.render.RenderMaterial(base_color=[0.1, 0.7, 0.1, 1]),
        )
        builder.initial_pose = sapien.Pose(p=[0.29, 0, 1e-3])
        self.slot_marker = builder.build_kinematic(name="slot_marker")

        n = self.num_envs
        self._phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self._attempts = torch.zeros(n, device=self.device)
        self._needed_dir = torch.zeros(n, device=self.device)
        self._rec_correct = -torch.ones(n, device=self.device)
        self._prev_moving = torch.zeros(n, dtype=torch.bool, device=self.device)

    def _init_task(self, env_idx: torch.Tensor):
        b = len(env_idx)
        xyz = torch.zeros((b, 3), device=self.device)
        lo, hi = self.spawn_x_range
        slo, shi = self.slot_x_range
        if self.deterministic_spawn:
            xyz[:, 0] = -0.06
            slot_x = torch.full((b,), 0.5 * (slo + shi), device=self.device)
        else:
            xyz[:, 0] = torch.rand(b, device=self.device) * (hi - lo) + lo
            xyz[:, 1] = (torch.rand(b, device=self.device) * 2 - 1) * 0.02
            slot_x = torch.rand(b, device=self.device) * (shi - slo) + slo
        xyz[:, 2] = self.obj_half + self.DECK_T
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))
        self.obj.set_linear_velocity(torch.zeros((b, 3), device=self.device))
        self.obj.set_angular_velocity(torch.zeros((b, 3), device=self.device))
        marker = torch.zeros((b, 3), device=self.device)
        marker[:, 0] = slot_x
        marker[:, 2] = 1e-3
        self.slot_marker.set_pose(Pose.create_from_pq(p=marker))
        self._phase[env_idx] = 0
        self._attempts[env_idx] = 0
        self._needed_dir[env_idx] = 0
        self._rec_correct[env_idx] = -1
        self._prev_moving[env_idx] = False


@register_env("DynPushT6-v1", max_episode_steps=150)
class DynPushT6Env(DynBaseEnv):
    """Hidden-physics Push-T (added 2026-08-18 on user direction: the task
    must reward understanding dynamics DURING the action, with time to
    observe and correct - not a one-shot calibrated force).

    Modeled on the established Push-T benchmark (diffusion-policy /
    ManiSkill PushT-v1): push a flat T-shaped block into a target pose
    (position AND orientation) using many small contacts. Physics runs all
    the way through: every push splits into translation and rotation
    according to the T's mass, surface friction, and center of mass, so a
    wrong internal model shows up as over/under-rotation the pusher must
    spend time correcting - and the episode budget (150 steps) is what makes
    that time expensive. No commitment geometry is needed: the outcome is a
    POSE, which cannot be escorted, only shaped through contact dynamics.
    """

    # precision regime (v3, 2026-08-18): at pos 0.025 / yaw 0.25 the blind
    # feedback controller matched the aware one on success AND speed - a
    # +/-2.4 cm COM error causes ~0.2-0.4 rad of surprise rotation per
    # translate push, inside the old tolerance, so mistakes were free.
    # Tightened so a wrong-COM push near the goal overshoots the tolerance
    # and costs a full correction cycle.
    pos_tol = 0.015
    yaw_tol = 0.12
    static_vel = 0.05
    release_dist = 0.05
    GOAL_XY = (0.05, 0.0)  # goal pose: fixed position, yaw = 0
    spawn_r = (0.10, 0.16)  # spawn: ring around the goal, random yaw
    # wider hidden-COM range than the rigid default (0.15): on a 15 cm tee a
    # +/-0.9 cm COM shift was too subtle - first gate (2026-08-18) measured
    # zero premium because blind FEEDBACK corrects small rotation surprises
    # for free. At 0.4 the COM moves up to +/-2.4 cm, enough to make wrong
    # push lines cost real correction time against the step budget.
    C_SPEC_KW = dict(com_frac_max=0.4)

    def _build_task(self, options: dict):
        self.obj = build_randomized_tee(self, self._c, name="tee")
        # flat green goal outline: same T shape, visual only
        builder = self.scene.create_actor_builder()
        green = sapien.render.RenderMaterial(base_color=[0.1, 0.7, 0.1, 1])
        builder.add_box_visual(half_size=[0.015, 0.06, 6e-4], material=green)
        builder.add_box_visual(half_size=[0.045, 0.015, 6e-4],
                               pose=sapien.Pose(p=[0.06, 0, 0]), material=green)
        builder.initial_pose = sapien.Pose(p=[self.GOAL_XY[0], self.GOAL_XY[1], 1e-3])
        self.goal_marker = builder.build_kinematic(name="goal_marker")

    def _init_task(self, env_idx: torch.Tensor):
        b = len(env_idx)
        if self.deterministic_spawn:
            r = torch.full((b,), 0.13, device=self.device)
            ang = torch.full((b,), np.pi / 2, device=self.device)
            yaw = torch.full((b,), np.pi / 2, device=self.device)
        else:
            lo, hi = self.spawn_r
            r = torch.rand(b, device=self.device) * (hi - lo) + lo
            # keep spawns on the reachable side (away from the robot base -x)
            ang = (torch.rand(b, device=self.device) * 1.5 - 0.75) * np.pi
            yaw = (torch.rand(b, device=self.device) * 2 - 1) * np.pi
        xyz = torch.zeros((b, 3), device=self.device)
        xyz[:, 0] = self.GOAL_XY[0] + r * torch.cos(ang)
        xyz[:, 1] = self.GOAL_XY[1] + r * torch.sin(ang)
        xyz[:, 2] = 0.02
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = torch.cos(yaw / 2)
        q[:, 3] = torch.sin(yaw / 2)
        self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))
        self.obj.set_linear_velocity(torch.zeros((b, 3), device=self.device))
        self.obj.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def obj_yaw(self) -> torch.Tensor:
        q = self.obj.pose.q  # (n, 4) wxyz
        return torch.atan2(
            2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
        )

    def evaluate(self):
        p = self.obj.pose.p
        goal = torch.tensor(self.GOAL_XY, device=self.device)
        pos_dist = torch.linalg.norm(p[:, :2] - goal[None], dim=1)
        yaw = self.obj_yaw()
        yaw_err = torch.remainder(yaw + np.pi, 2 * np.pi) - np.pi
        speed = torch.linalg.norm(self.obj.linear_velocity, dim=1)
        static = speed < self.static_vel
        tcp_dist = self._tcp_to_obj_dist()
        released = tcp_dist > self.release_dist
        pose_ok = (pos_dist < self.pos_tol) & (yaw_err.abs() < self.yaw_tol)
        return dict(
            success=pose_ok & static & released,
            pose_ok=pose_ok,
            pos_dist=pos_dist,
            yaw_err=yaw_err,
            obj_to_goal_dist=pos_dist,
            tcp_to_obj_dist=tcp_dist,
            obj_speed=speed,
        )

    def _task_state_obs(self):
        return dict(goal_pos=self.goal_marker.pose.p)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reaching = 1 - torch.tanh(5.0 * (info["tcp_to_obj_dist"] - 0.02).clamp(min=0))
        pos = 1 - torch.tanh(4.0 * info["pos_dist"])
        rot = 1 - torch.tanh(2.0 * info["yaw_err"].abs())
        reward = 0.5 * reaching + 2.0 * pos + 2.0 * rot
        reward[info["success"]] = 6.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.0


@register_env("RouteChoiceT7-v1", max_episode_steps=200)
class RouteChoiceT7Env(SlideToSlotT3Env):
    """T7 'Router' core (single block; the LIBERO-Long-shaped multi-block
    version builds on this only if the premium gate passes).

    Two parallel channels lead away from an open staging area. The LEFT
    channel floor is slick (long slides); the RIGHT channel is grippy
    (short slides). Each channel has its own slot, placed (from probe
    measurements) so that ONE fixed-strength flick lands a block on the
    left slot only for one region of physics-space and on the right slot
    only for the complementary region. The controller's knowledge is a
    DISCRETE ROUTE CHOICE: escort the block to a channel mouth, commit one
    flick, retreat. Wrong route = wrong stopping point, at rest off-slot.
    """

    CH_OFF = 0.095          # channel center offsets: +y = slick, -y = grippy
    CH_HALF = 0.06          # channel inner half-width (hand must fit: T3 uses 0.06)
    CH_X = (-0.05, 0.50)    # floored channel span
    DIV_X = (-0.16, 0.50)   # divider extends back: walled approach lanes, so
                            # the windup happens confined (v2 probe: open-area
                            # windups scatter launches over a 15-20 cm IQR)
    STAGE_X = -0.26
    # v1 probe: 0.9 stopped every block within 3 cm of the mouth regardless
    # of c - a universal safe route. 0.35 keeps the right channel braking
    # harder than the slick left while letting c decide the stop point.
    GRIPPY_FRICTION = 0.35
    # per-channel slot centers, set from the stopping-distribution probe
    SLOT_L_X = 0.11
    SLOT_R_X = 0.05
    spawn_x_range = (-0.24, -0.20)  # behind the lanes, in the crossing zone

    def _build_task(self, options: dict):
        self.obj = build_randomized_box(
            self, self.obj_half, self._c, name="slide_obj"
        )
        import sapien.physx as _physx
        x0, x1 = self.CH_X
        wall_t, wall_h = 0.01, 0.02
        cx, hx = 0.5 * (self.STAGE_X + x1), 0.5 * (x1 - self.STAGE_X)
        ccx, chx = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
        y_out = self.CH_OFF + self.CH_HALF + wall_t
        gray = sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 1])

        builder = self.scene.create_actor_builder()
        shapes = [
            # outer rails along everything, end wall across both channels
            ([hx, wall_t, wall_h], [cx, y_out, wall_h]),
            ([hx, wall_t, wall_h], [cx, -y_out, wall_h]),
            ([wall_t, y_out + wall_t, wall_h], [x1 + wall_t, 0, wall_h]),
            # center divider along channels + approach lanes (crossing zone
            # stays open behind DIV_X[0])
            ([0.5 * (self.DIV_X[1] - self.DIV_X[0]),
              self.CH_OFF - self.CH_HALF, wall_h],
             [0.5 * (self.DIV_X[0] + self.DIV_X[1]), 0, wall_h]),
        ]
        for half, pos in shapes:
            builder.add_box_collision(half_size=half, pose=sapien.Pose(p=pos))
            builder.add_box_visual(half_size=half, pose=sapien.Pose(p=pos), material=gray)
        builder.initial_pose = sapien.Pose()
        self.channel = builder.build_static(name="channel")

        # floors: slick left, grippy right (staging area stays plain table)
        for y_c, fric, color, name in (
            (self.CH_OFF, self.FLOOR_FRICTION, [0.75, 0.85, 0.95, 1], "floor_slick"),
            (-self.CH_OFF, self.GRIPPY_FRICTION, [0.95, 0.85, 0.75, 1], "floor_grippy"),
        ):
            fb = self.scene.create_actor_builder()
            fb.add_box_collision(
                half_size=[chx, self.CH_HALF, self.FLOOR_T / 2],
                pose=sapien.Pose(p=[ccx, y_c, self.FLOOR_T / 2]),
                material=_physx.PhysxMaterial(fric, fric, 0.0),
            )
            fb.add_box_visual(
                half_size=[chx, self.CH_HALF, self.FLOOR_T / 2],
                pose=sapien.Pose(p=[ccx, y_c, self.FLOOR_T / 2]),
                material=sapien.render.RenderMaterial(base_color=color),
            )
            fb.initial_pose = sapien.Pose()
            setattr(self, name, fb.build_static(name=name))

        # one visual slot marker per channel
        self.slot_markers = []
        for slot_x, y_c in ((self.SLOT_L_X, self.CH_OFF), (self.SLOT_R_X, -self.CH_OFF)):
            b = self.scene.create_actor_builder()
            b.add_box_visual(
                half_size=[self.slot_half_width, self.CH_HALF, 6e-4],
                material=sapien.render.RenderMaterial(base_color=[0.1, 0.7, 0.1, 1]),
            )
            b.initial_pose = sapien.Pose(p=[slot_x, y_c, self.FLOOR_T + 1e-3])
            self.slot_markers.append(b.build_kinematic(name=f"slot_{len(self.slot_markers)}"))
        self.slot_marker = self.slot_markers[0]  # parent-class compatibility

        n = self.num_envs
        self._phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self._attempts = torch.zeros(n, device=self.device)
        self._needed_dir = torch.zeros(n, device=self.device)
        self._rec_correct = -torch.ones(n, device=self.device)
        self._prev_moving = torch.zeros(n, dtype=torch.bool, device=self.device)

    def _init_task(self, env_idx: torch.Tensor):
        b = len(env_idx)
        xyz = torch.zeros((b, 3), device=self.device)
        lo, hi = self.spawn_x_range
        if self.deterministic_spawn:
            xyz[:, 0] = -0.22
        else:
            xyz[:, 0] = torch.rand(b, device=self.device) * (hi - lo) + lo
            xyz[:, 1] = (torch.rand(b, device=self.device) * 2 - 1) * 0.03
        xyz[:, 2] = self.obj_half + self.FLOOR_T
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))
        self.obj.set_linear_velocity(torch.zeros((b, 3), device=self.device))
        self.obj.set_angular_velocity(torch.zeros((b, 3), device=self.device))
        self._phase[env_idx] = 0
        self._attempts[env_idx] = 0
        self._needed_dir[env_idx] = 0
        self._rec_correct[env_idx] = -1
        self._prev_moving[env_idx] = False

    def evaluate(self):
        p = self.obj.pose.p
        in_left = p[:, 1] > self.CH_OFF - self.CH_HALF
        in_right = p[:, 1] < -(self.CH_OFF - self.CH_HALF)
        d_left = (p[:, 0] - self.SLOT_L_X).abs()
        d_right = (p[:, 0] - self.SLOT_R_X).abs()
        on_slot = (in_left & (d_left < self.slot_half_width)) | \
                  (in_right & (d_right < self.slot_half_width))
        dist = torch.where(in_right, d_right, d_left)
        speed = torch.linalg.norm(self.obj.linear_velocity, dim=1)
        static = speed < self.static_vel
        tcp_dist = self._tcp_to_obj_dist()
        released = tcp_dist > self.release_dist
        self._prev_moving = speed > self.static_vel
        return dict(
            success=on_slot & static & released,
            in_slot=on_slot,
            obj_static=static,
            released=released,
            obj_to_slot_dist=dist,
            tcp_to_obj_dist=tcp_dist,
            obj_speed=speed,
            attempt_count=self._attempts.clone(),
            first_recovery_correct=self._rec_correct.clone(),
        )

    def _task_state_obs(self):
        return dict(slot_pos=torch.cat(
            [m.pose.p[:, :2] for m in self.slot_markers], dim=1))


# =========================================================================== #
# T8: stack two different shapes, then carry the stack to a goal zone.        #
# =========================================================================== #
@register_env("StackCarryT8-v1", max_episode_steps=140)
class StackCarryT8Env(DynBaseEnv):
    """Stack-and-carry (user design 2026-08-19), in the BALANCE regime.

    Two objects of different shape: a flat beam (12 x 3.6 x 2.4 cm) and a
    tall block (6 x 6 x 10 cm) whose center of mass is hidden and offset by
    up to 40% of its half-width (+-1.2 cm). Place the block on the narrow
    beam, grasp the beam at its free end, and carry the stack sideways to a
    goal zone inside a step budget.

    Why THIS geometry (measured, gate iteration 1). The first version made
    the carry SPEED the knowledge-bearing choice: the block rides on a wide
    slab and slips off when the carry exceeds the friction cone. The probe
    showed the premium is structurally zero there, for a reason worth
    recording: the slip threshold s_safe(c) is c-dependent but the deadline
    threshold s_min is NOT, and slower is always safer, so the blind
    controller simply parks at s_min - the same corner an omniscient
    controller would choose - and succeeds on exactly the same episodes.
    A monotone choice with a c-independent floor can never pay.

    The balance regime has no safe corner. The block's support is the beam's
    3.6 cm width, so it stands only while its hidden COM stays within +-1.8 cm
    of the beam center line; the carry runs along y, and the acceleration and
    braking phases tilt the effective gravity so an off-center COM topples the
    block. The correct placement therefore differs in DIRECTION with c (the
    COM angle is uniform), which is the structure that paid in T3: no single
    fixed placement is right, over- and under-shooting both fail, and the
    choice is committed the moment the gripper opens.
    """

    BASE_HALF = [0.06, 0.018, 0.012]  # beam: long x, NARROW y = the support
    TOP_HALF = [0.03, 0.03, 0.05]  # tall block: tips before it slides
    TOP_SEAT_DX = 0.02  # block seat, +x of the beam center
    GRASP_DX = -0.04  # beam grasped at its free -x end
    BASE_SPAWN = (-0.10, -0.16)
    TOP_SPAWN = (0.06, -0.16)
    GOAL_XY = (-0.10, 0.18)  # carry runs along +y: braking stresses the stack
    goal_tol = 0.04
    static_vel = 0.05
    release_dist = 0.06
    # com_frac_max: T6 lesson, 0.15 is sub-measurable on a small object.
    # mass/friction narrowed from the study defaults (0.7-1.4, 0.3-0.7). The
    # heavy end is excluded for a MEASURED mechanical reason: the stack hangs
    # off the beam's free end, so its weight torques the beam inside the
    # gripper, and the beam's tilt during the carry rises from 2.7 deg median
    # at mass 0.7 to 8.6 deg at mass 1.2 - tilting the block's own support
    # surface by more than any placement can answer. That corner is not
    # teachable by any calibration (certification failed there and only
    # there), so it is out of the TRAINING range. Two alternatives were tried
    # and measured worse: lightening the block fixed the tilt but erased the
    # premium (aware 1.00 vs blind 1.00 - the T2 trap), and narrowing the beam
    # restored difficulty without restoring the advantage. The delta grid
    # still probes far outside these ranges, and evaluate.py re-tags
    # interpolation/extrapolation from this spec.
    C_SPEC_KW = dict(com_frac_max=0.6, mass_mult_range=(0.7, 1.1),
                     friction_range=(0.35, 0.7))

    def _build_task(self, options: dict):
        # beam: per-env friction/density but NO hidden COM (the block owns it)
        base_c = dict(self._c)
        base_c["com_x_frac"] = np.zeros(self.num_envs)
        base_c["com_y_frac"] = np.zeros(self.num_envs)
        self.obj = build_randomized_box(
            self, self.BASE_HALF, base_c, name="beam")
        self.top = build_randomized_box(
            self, self.TOP_HALF, self._c, name="block", base_density=1000.0)
        builder = self.scene.create_actor_builder()
        green = sapien.render.RenderMaterial(base_color=[0.1, 0.7, 0.1, 1])
        builder.add_box_visual(half_size=[0.05, 0.05, 6e-4], material=green)
        builder.initial_pose = sapien.Pose(
            p=[self.GOAL_XY[0], self.GOAL_XY[1], 1e-3])
        self.goal_marker = builder.build_kinematic(name="goal_marker")

    def _init_task(self, env_idx: torch.Tensor):
        b = len(env_idx)
        jitter = (
            torch.zeros((b, 4), device=self.device)
            if self.deterministic_spawn
            else (torch.rand((b, 4), device=self.device) * 2 - 1) * 0.02
        )
        for actor, spawn, z, j in (
            (self.obj, self.BASE_SPAWN, self.BASE_HALF[2], jitter[:, :2]),
            (self.top, self.TOP_SPAWN, self.TOP_HALF[2], jitter[:, 2:]),
        ):
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = spawn[0] + j[:, 0]
            xyz[:, 1] = spawn[1] + j[:, 1]
            xyz[:, 2] = z
            q = torch.zeros((b, 4), device=self.device)
            q[:, 0] = 1.0
            actor.set_pose(Pose.create_from_pq(p=xyz, q=q))
            actor.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            actor.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def evaluate(self):
        base_p = self.obj.pose.p
        top_p = self.top.pose.p
        goal = torch.tensor(self.GOAL_XY, device=self.device)
        goal_dist = torch.linalg.norm(base_p[:, :2] - goal[None], dim=1)
        at_goal = goal_dist < self.goal_tol
        # the block must still stand on the beam: high enough to be ON it
        # (0.062 when seated, 0.042 if it has toppled onto it, 0.038 if it
        # fell beside it), still over the beam in xy, and upright
        riding = (
            (torch.linalg.norm(top_p[:, :2] - base_p[:, :2], dim=1) < 0.045)
            & ((top_p[:, 2] - base_p[:, 2]) > 0.05)
        )
        upright = _quat_zz(self.top.pose.q) > 0.9
        placed = base_p[:, 2] < 0.02
        speed = torch.linalg.norm(self.obj.linear_velocity, dim=1)
        static = speed < self.static_vel
        tcp_dist = self._tcp_to_obj_dist()
        released = tcp_dist > self.release_dist
        stack_ok = riding & upright
        return dict(
            success=at_goal & stack_ok & placed & static & released,
            stack_ok=stack_ok,
            at_goal=at_goal,
            obj_to_goal_dist=goal_dist,
            tcp_to_obj_dist=tcp_dist,
            obj_speed=speed,
        )

    def _task_state_obs(self):
        # the BLOCK's full 13-d state is kept contiguous and in the same
        # pose/lin/ang order the base class uses for the manipulated object:
        # it is this task's physics-bearing object (the beam is rigidly held,
        # so its motion is the arm's, like T4's grasped cube), and arm B's
        # prediction head is pointed at it. See OBS_LAYOUTS["t8"].
        return dict(
            top_pose=self.top.pose.raw_pose,
            top_lin_vel=self.top.linear_velocity,
            top_ang_vel=self.top.angular_velocity,
            goal_pos=self.goal_marker.pose.p,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reach = 1 - torch.tanh(5.0 * (info["tcp_to_obj_dist"] - 0.02).clamp(min=0))
        goal = 1 - torch.tanh(4.0 * info["obj_to_goal_dist"])
        reward = 0.5 * reach + 2.0 * goal + 1.5 * info["stack_ok"].float()
        reward[info["success"]] = 6.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.0


# =========================================================================== #
# Teacher twins: identical tasks with c exposed in the observation.           #
# =========================================================================== #
def _teacher(cls):
    class TeacherEnv(cls):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("expose_c", True)
            super().__init__(*args, **kwargs)

    TeacherEnv.__name__ = cls.__name__.replace("Env", "TeacherEnv")
    return TeacherEnv


SlideToSlotT3TeacherEnv = register_env("SlideToSlotT3Teacher-v1", max_episode_steps=200)(
    _teacher(SlideToSlotT3Env)
)
PourT1TeacherEnv = register_env("PourT1Teacher-v1", max_episode_steps=150)(
    _teacher(PourT1Env)
)
CarryT2TeacherEnv = register_env("CarryT2Teacher-v1", max_episode_steps=120)(
    _teacher(CarryT2Env)
)
PickPlaceT4TeacherEnv = register_env("PickPlaceT4Teacher-v1", max_episode_steps=80)(
    _teacher(PickPlaceT4Env)
)
CliffTossT5TeacherEnv = register_env("CliffTossT5Teacher-v1", max_episode_steps=200)(
    _teacher(CliffTossT5Env)
)
DynPushT6TeacherEnv = register_env("DynPushT6Teacher-v1", max_episode_steps=150)(
    _teacher(DynPushT6Env)
)
StackCarryT8TeacherEnv = register_env("StackCarryT8Teacher-v1", max_episode_steps=140)(
    _teacher(StackCarryT8Env)
)
RouteChoiceT7TeacherEnv = register_env("RouteChoiceT7Teacher-v1", max_episode_steps=200)(
    _teacher(RouteChoiceT7Env)
)
