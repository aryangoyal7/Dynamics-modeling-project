"""Run ManiSkill's stock PPO baseline on the dynmod environments.

The plan requires borrowing the baseline unmodified ('change only the
observation spec'); this wrapper only registers our envs before handing over
to the untouched script, so any env id from dynmod works, e.g.:

    python -m dynmod.scripts.ppo_dynmod --env_id=SlideToSlotT3Teacher-v1 \
        --num_envs=1024 --total_timesteps=2000000 --no-capture_video

Set PPO_SCRIPT to point at a different checkout if needed.
"""

import os
import runpy
import sys

import dynmod.envs  # noqa: F401  (registers all dynmod envs)

PPO_SCRIPT = os.environ.get(
    "PPO_SCRIPT",
    "/mnt/scratch/dynamics/ManiSkill/examples/baselines/ppo/ppo.py",
)

if __name__ == "__main__":
    sys.argv = [PPO_SCRIPT] + sys.argv[1:]
    runpy.run_path(PPO_SCRIPT, run_name="__main__")
