# dynmod — environments for the dynamics-prediction robustness study

Code for Part II, "Build: environments" of `project_full (1).tex` (updated
plan: T3 slide-to-slot primary; T1 pouring / T2 carrying granular tier; T4
pick-and-place control).

## Setup on this machine

- Python env: `/mnt/scratch/dynamics/ms3env` (torch 2.5.1+cu121, ManiSkill 3.0.1
  editable from `/mnt/scratch/dynamics/ManiSkill`)
- Run everything from this directory (`Dynamics-modeling-project/`) so `dynmod`
  is importable:
  `PY=/mnt/scratch/dynamics/ms3env/bin/python`

## Checklist → code map

| Plan item | Where | Verify with |
|---|---|---|
| ManiSkill 3 + stock PPO trains end to end | `/mnt/scratch/dynamics/ManiSkill`, run logs in `.../examples/baselines/ppo/runs/pushcube-baseline-gate` | eval_success_once_mean=1.0 (done) |
| T3 environment first, multi-attempt success, attempt count + first-recovery direction logged | `envs/tasks.py` → `SlideToSlotT3-v1` (`attempt_count`, `first_recovery_correct` in info) | `$PY -m dynmod.scripts.smoke_test --quick` |
| Per-episode randomisation via build-separate + per-env body components | `envs/randomization.py` → `build_randomized_box/cup`, `build_particles` | gate below |
| c sampler (log-uniform mass [0.7,1.4], friction [0.3,0.7], COM ≤15% half-width; + particle count, inter-particle friction for T1/T2) | `envs/randomization.py` → `CTrainSpec`, `resolve_c` | smoke test asserts ranges |
| δ grid config with interpolation/extrapolation/composition tags | `envs/delta_grid.py` → `configs/delta_grid.json` | `$PY -m dynmod.envs.delta_grid` |
| c exposed to teacher only | `expose_c` flag; `*Teacher-v1` env ids; metadata via `env.unwrapped.get_c()` | smoke test asserts +c dims |
| **Gate: randomisation reaches the physics** | `scripts/verify_randomization.py` → `reports/verify_randomization.json` | `$PY -m dynmod.scripts.verify_randomization` |
| T1/T2 (granular spheres, not fluid) + T4 control | `envs/tasks.py` → `PourT1-v1`, `CarryT2-v1`, `PickPlaceT4-v1` | `$PY -m dynmod.scripts.smoke_test` |
| Calibration tier: scalar + piecewise-affine with hand-written LQR expert (incl. dummy c component for ablation validation) | `calibration/lqr.py` | `$PY -m dynmod.calibration.lqr` |

## Env construction knobs (all envs)

- `expose_c=True` — privileged teacher observation (or use the `*Teacher-v1` id)
- `randomize_c=False` — nominal physics
- `c_override=dict(mass_mult=…, friction=…, com_frac=…, …)` — pin envs to a
  δ-grid point (scalars broadcast; per-env arrays allowed)
- `deterministic_spawn=True` — fixed spawns/goals (used by the gate)
- `reconfiguration_freq=k` — resample c every k resets (c is otherwise fixed
  per parallel env slot, varying across the batch)
- granular tasks: `spawn_grasped=True/False` (T2 defaults True, T1 False)

## Training the teacher (next section of the plan)

```
$PY -m dynmod.scripts.ppo_dynmod --env_id=SlideToSlotT3Teacher-v1 \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=10000000 --num-steps=50 --no-capture_video
```

## Section 10 (Build: data) tooling

| Plan item | Where | Notes |
|---|---|---|
| Privileged PPO teacher | `scripts/ppo_dynmod.py` on `SlideToSlotT3Teacher-v1` (γ=0.97 for the 200-step horizon) | run: `runs/t3-teacher-v1`, log `/mnt/scratch/dynamics/t3-teacher-v1.log` |
| **Gate: teacher success flat across c** | `scripts/teacher_flatness.py` → `reports/teacher_flatness.{json,png}` | 60 in-range c points, success heatmaps; advisory spread ≤ 0.15 |
| Rollout harness (c as metadata only) | `scripts/rollout_harness.py` | student env recorded (obs has no c); teacher fed `[obs, c]`; HDF5 + `c_metadata.npz` sidecar; `reconfiguration_freq=1` resamples c per episode |
| δ-grid eval episodes (100/point/seed) | `scripts/make_eval_manifest.py` → `configs/eval_manifest.json` | fixed episode-seed blocks so every arm faces identical conditions |
| c-blind secondary set | `scripts/scripted_expert_t3.py` | guarded-push Cartesian expert, no access to c; records in `pd_ee_delta_pos` (replayable to joint space) |
| Datasets 10³/10⁴/10⁵ + vision variant | run harness after the flatness gate passes | store under `/mnt/scratch/dynamics/data/` |

## Section 11 (Build: training) implementation

| Plan item | Where |
|---|---|
| Base flow policy (rectified flow, K-history encoder, action chunking, velocity head) | `policy/model.py` (`FlowPolicy`, arm `A`) |
| Prediction head: 2 layers off the trunk, predicts the OBJECT's state (obs dims [25:38], never robot joints) | `policy/model.py` (`pred_head`) |
| Three targets: onestep / multistep (H=8 ∈ [5,10]) / latent + stop-grad | `--target` flag |
| Shuffled-target control (one line: batch permutation) | arm `Bshuf` |
| Arm C: head kept at inference, forecast concatenated to trunk features | arm `C` (forecast from previous executed action) |
| Dataset over ManiSkill HDF5 (+ success-truncation, normalization) | `policy/data.py` |
| Training entry with substitution knobs (weight decay / Jacobian penalty / data fraction / extended schedule) | `policy/train.py` |
| Log-spaced checkpoints throughout training | `policy/train.py` (`log_spaced_checkpoints`) |
| δ-grid rollout evaluator + "base fails far" gate (`--gate`) | `policy/evaluate.py` |
| Preregistered Test D budget + falsification rule (fixed before any run) | `configs/preregistration.json` |
| Implementation smoke test (all arms × targets, parity, descent, round-trip) | `scripts/policy_smoke_test.py` |

Arm launches (main arms, λ sweep, substitutions) are deliberately NOT run yet;
commands are one-liners on `policy/train.py` once teacher data exists.

## Known v1 caveats

- Cube/table friction combine: PhysX averages the two materials' friction, so
  the *effective* object-table friction is compressed toward the table's
  value; the c → outcome mapping stays monotone (the gate measures it
  empirically).
- Granular reconfigure cost scales with `max_particles × num_envs` actor
  builds; train T1/T2 with fewer parallel envs than the rigid tasks.
- `spawn_grasped` places the container into the gripper after reset using the
  TCP pose; it applies on full resets only (fine for the PPO baseline and the
  eval harness, which reset all envs together).
- T1 defaults to the cup starting on the table (grasping is part of the
  task); flip `spawn_grasped=True` if the teacher cannot learn the grasp.
