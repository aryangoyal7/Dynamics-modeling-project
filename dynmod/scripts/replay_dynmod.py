"""Run ManiSkill's trajectory replay tool with dynmod envs registered.

Used for the vision variant (plan Part II 'Build: data'): replay recorded
state trajectories into image observations without regenerating anything.
Replay drives the sim by setting recorded env states each step, so the
rendered frames are exact regardless of the per-episode hidden physics.

    python -m dynmod.scripts.replay_dynmod \
        --traj-path /mnt/scratch/dynamics/data/t3_1e4/trajectory.h5 \
        --obs-mode rgb --save-traj --use-env-states -b physx_cuda

All arguments after the module name are passed to
mani_skill.trajectory.replay_trajectory unchanged.
"""

import runpy
import sys

import dynmod.envs  # noqa: F401  (registers all dynmod envs)

if __name__ == "__main__":
    sys.argv = ["replay_trajectory"] + sys.argv[1:]
    runpy.run_module("mani_skill.trajectory.replay_trajectory", run_name="__main__")
