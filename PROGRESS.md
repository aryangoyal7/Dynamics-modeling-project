# Project Progress

## ✅ T3 TEACHER CERTIFIED — pipeline running (2026-08-18, evening)

After two calibration rounds, the scripted physics-aware teacher **passed
both quality gates**: average success **0.634** (bar ≥ 0.55) and worst
per-axis unevenness **0.072** (bar ≤ 0.10) — report
`flat_scripted_aware_v4.json`. The winning fix was recalibrating the flick
table at full 5×4 physics-grid resolution with a wider search (5 speeds ×
4 run-up lengths, 128 episodes per cell). The RL teachers never got close
(v8 grid mean 0.36; v9 touched 0.578 once mid-training but oscillated back
to ~0.40 — both checkpoint sets kept in `runs/`). Chain 2
(`t3_chain2.py`) now runs the rest of the plan: 10^3/10^4 datasets in
parallel (each backed up on completion), the base-must-fail-far-out check,
then the 10-seed arm matrix plus 10^5 generation.

## ❌ T1 dump-pour verdict: no physics premium — redesigned as MEASURED pour

With the pour fixed (low, aimed, decisive — nominal success 0.75 after the
geometry sweep), the gate came back healthy but null: blind **82.0%**,
aware **77.1%**, premium **−4.9 ± 6.3** (statistically zero). The reason is
structural: success only requires ≥80% of contents in the basin, so
over-tilting costs nothing and one fixed "dump it all" tilt works for every
physics. Knowledge can't pay when overshoot is free — the same lesson as
T2, discovered honestly by the gate.

**New test (from these lessons): the measured pour.** Success = ending with
**35–65% of contents transferred** (pour half, keep half). Over-pour is now
irreversible failure, and how much comes out per degree of tilt depends on
fill and content friction — that dependence is the knowledge being priced.

**Measured-pour verdict (2026-08-18, 20:20): also zero — T1 DESCOPED.**
Blind 14.8%, aware 17.6%, premium **+2.7 ± 6.2**. The decisive detail: even
with exact knowledge and per-bucket tuning under FIXED physics, the
calibrated controller hits the band only 19–47% of the time — the amount
poured is dominated by granular chaos, not knowable physics. Three
independent granular measurements (T2 carry, T1 dump, T1 measured) all
found no premium: in this simulator, granular outcomes are too noisy for
physics knowledge to price into. The study drops granular tasks entirely.

## 🆕 T6 "hidden-physics Push-T" (2026-08-18, night) — the in-action task

**T5 was dropped by user decision**: a one-shot calibrated toss tests
"learning a distribution of force," not understanding dynamics — the robot
should have time to observe and correct DURING the action. Per the same
direction, the replacement is modeled on an established benchmark rather
than invented: **Push-T** (diffusion-policy / ManiSkill), pushing a
T-shaped block into a target position AND orientation through many small
contacts. Our version (`DynPushT6-v1`) hides physics in the T: mass,
friction, and center of mass — and for a T, the COM decides how every push
splits into translation vs rotation, so a wrong internal model shows up as
over/under-rotation that costs correction time against a 150-step budget.
This finally makes the COM axis decisive (it was flat in T3 and
undecodable in T4). No anti-cheat geometry needed: a POSE cannot be
escorted, only shaped through contact dynamics.

First probe passed immediately (unique among all tasks): blind controller
50% at nominal physics, errors right at tolerance. Premium gate running:
blind assumes the COM is at the centroid; aware pushes through the TRUE
COM with per-bucket speeds.

**Gate iterations (2026-08-18, night).** v1 (loose tolerance): premium
zero — per-step feedback lets the blind pusher identify the COM by
watching, and ±0.25 rad of yaw tolerance forgives its mistakes. v2 (COM
range widened to ±2.4 cm, time-to-success measured): still zero on every
axis, even speed. This is itself a finding: with corrections available and
loose goals, feedback fully substitutes for knowledge — the reason the
plan's original tasks were commitment-based. v3 (precision regime: 1.5 cm
/ 0.12 rad tolerance, so a wrong-COM push near the goal overshoots the
tolerance and costs a full correction cycle): **first directionally
positive result — all four metrics favor aware** (success +2.9,
success@100 +4.5, success@75 +2.3, median steps 80.5 vs 93) but not yet
significant at ±6.2. A 4×-power rerun (CI ±3.1) is running to settle it.

## 🚧 T5 "cliff toss" — DESCOPED (user decision; kept for reference)

Replacement second task, rigid-body like the tasks that worked. The object
starts on a slick deck 10 cm above the table; the controller charge-flicks
it off the cliff; flight range (launch speed vs mass/friction) and the
table's friction braking decide where it stops; success = at rest on the
slot, hand away, no recovery allowed after the edge. Design note: the
elaborate anti-cheat roofs T3 needed existed to stop RL teachers from
reward-hacking — with the now-standard scripted teacher there is no hacker
to fence out, so T5's geometry is clean. Env `CliffTossT5-v1` + gate
`premium_gate_t5.py` (running). Bars as always: healthy success, premium
clearly positive.

## 🧭 Designed, not built: T6 "Router" — the long-horizon dynamics task

Requested: a long-horizon many-object task where dynamics change the
outcome completely (a "make coffee" analog). Direct kitchen simulation
fails our own evidence twice (granular/liquid = chaos kills the premium;
recoverable steps = controllers correct instead of commit). The design that
keeps the spirit measurable: **three blocks, each with hidden physics, and
two routes to the goal** — a launch ramp only light/slippery blocks can
survive, and a friction corridor where light blocks overshoot into a dead
zone but heavy ones stop correctly. The right ROUTE per block depends on
its hidden c; a mis-routed block clogs that route for the rest, so one
wrong physics judgment changes the whole episode's outcome. Knowledge here
is discrete (plan choice), not a tuned magnitude — a qualitatively new test
of whether prediction heads help strategy selection. Build candidate after
the T5 verdict and T3 arms; every component reuses validated geometry.

## Fourth machine wipe (2026-08-18 ~19:55)

Scratch wiped again; zero loss (all verdicts/reports on persistent storage,
datasets backed up as generated). The rebuild recipe now lives permanently
at `dynamics_modeling/rebuild_env.sh` (~6 min). **The Azure auto-stop /
restart schedule on this machine is still active — please disable it.**

## Third machine wipe + both verdicts landed (2026-08-18, midday)

The machine was externally restarted a THIRD time (~11:37), wiping
/mnt/scratch again. Nothing important was lost: teacher checkpoints live on
persistent storage (`runs/`), all datasets were backed up the moment they
were generated (3.6 GB in `scratch_backup/`), and both pending verdicts had
already been written to `reports/` before the crash. The environment was
rebuilt in 6 minutes from a saved recipe. **Please disable the Azure
auto-stop schedule on this machine — each wipe costs about an hour.**

**T3 teacher verdict: the RL teacher FAILED the quality bar.** Teacher v8
finished all 40M steps (09:28) and the automated chain evaluated its best
four checkpoints across the physics range: best mean success **0.36**
(bar: ≥ 0.55), spread 0.11 (bar: ≤ 0.10). The eval curve had plateaued at
0.39–0.42. The chain stopped itself instead of generating datasets from a
weak teacher — exactly what it is designed to do. Note the task itself is
fine (validated +16.7); it is the RL teacher that can't yet do it well.

**Plan (running in parallel now):**
1. **Scripted calibrated teacher (new, favored).** The physics-aware scripted
   controller that measured **61.7%** during validation is above the 0.55
   bar. It is rebuilt as a permanent script
   (`dynmod/scripts/scripted_teacher_t3.py`: same expert, but flick speed +
   run-up looked up from the true physics via the saved calibration table)
   and is being run through the SAME flatness gate as any RL teacher
   (GPU 4). If it passes both bars it becomes the main-dataset teacher —
   its actions depend on the true physics, which is all the experiment
   requires of a teacher.
2. **PPO v9 (insurance).** Another 40M-step fine-tune from v8's final
   checkpoint (GPU 2, ~11 h). Killed if option 1 certifies first.

**T1 verdict: gate result unusable — floor effect.** The pouring gate
finished before the crash: blind 2.5%, aware 3.1%, premium +0.6 ± 6.3. Both
controllers succeed almost NEVER under full physics randomization (they had
real success in calibration, where only fill and content friction varied).
When success is ~3% the gate can't measure knowledge — the pour script is
too fragile, not necessarily the task. A per-axis diagnostic is running
(GPU 3): same fixed pour, one physics axis randomized at a time, to find
which axis breaks it. Then: one fair fix for both controllers and a gate
rerun — or descope with numbers if the premium is still zero at healthy
success rates.

## Task-set decision (2026-08-18): T2 removed

The validity gate measured T2's physics-knowledge premium at **+1.2 ± 5.1
points — statistically zero**: a hanging mug is a natural shock absorber
(the pendulum low-pass-filters the arm's accelerations), so contents barely
feel the motion at any fill level or friction. Rather than redesign a task
with no measured physics signal, T2 is **descoped by user decision** (made
before any T2 arms existed). The study proceeds with T3 (primary, validated
+16.7), T4 (control, complete), and T1 (gate running; irreversible pouring
gives it built-in commitment). T2's environment code remains in the repo,
marked unused.


## 🎉 First preregistered result (2026-08-18): the control task passes

All 60 T4 study arms trained (A, B×3 targets, B-shuffled, C × 10 seeds; zero
failures) and evaluated across the δ grid. **A-vs-B slopes are statistically
identical on every tag** (e.g. composition +0.0113±0.0009 vs +0.0115±0.0014,
CIs overlap) — on the task where physics can't matter, the prediction head
changes nothing. The specificity control is certified, and with it the whole
train→evaluate→slope pipeline at 10-seed power.

## ✅ T3 VALIDATED (2026-08-18): knowledge premium +16.7 points

After the slick-floor fix restored launch authority, the high-powered gate
(3,000+ episodes, clean per-bucket calibration) shows a physics-aware
controller beats a physics-blind one **45.0% → 61.7% (±3.6)** — identical
under time pressure, so the premium is in the launch calibration itself.
The calibration table is cleanly physical (slippery wants softer flicks,
grippy demands maximum with a long run-up). Teacher v7 (BC warm-started on
final-geometry demos, 43% source) is training; c-blind sets regenerating.

## Bonus mechanism finding (2026-08-18): probes on the T4 arms

B's trunk decodes **2× more mass** than A's (ridge R² 0.53-0.59 vs
0.27-0.31) while the shuffled control sits at A's level — the prediction
*content* injects knowledge. Only mass moves (friction/COM stay
undecodable — physically correct for a grasped object). Together with the
null slopes: **knowledge without effect**, exactly the decoupling the
control task predicts. The full measurement triad (slopes, probes,
controls) now coheres on real policies.

## Historical: how T3 earned its validation

New standing gate learned from the escorting episode: a task qualifies only
if a physics-aware controller beats a physics-blind one. Evidence so far on
T3: the c-seeing RL teacher converged to exactly the c-blind script's success
(~46-48%), and a naive hand-scaled c-aware flick *loses* to the fixed flick.
A second roof segment now brackets the slot (both failure directions are
committed; recovery is a committed reverse flick) — physics gate re-passed.
The decisive measurement is running: per-physics-bucket calibrated flicks
(the true "knows c" ceiling) plus within-physics chaos scatter, which will
show whether knowledge pays or simulator noise caps the premium.


**The experiment in one line:** train two robot policies that are identical
except one must also *predict how the object will move* — then test whether
that one survives physics changes better, and if so, why.

Plan: `project_full (1).tex` · Code: `dynmod/` (its README maps every plan
item to a file) · Big files (datasets, model runs): `/mnt/scratch/dynamics/`

**Glossary (used everywhere below)**
- **c** — the hidden physics of an episode (mass, friction, off-center
  weight; fill level for the granular tasks). The robot *feels* c but is
  never told it.
- **Teacher** — an RL agent trained *with* c visible (cheat mode). It
  generates expert demonstrations; student policies learn from them blind.
- **δ grid** — 120 physics settings beyond the training range, used to
  measure how fast a policy breaks as physics drift.
- **Gate** — a go/no-go check the plan requires before continuing. Skipping
  one silently invalidates everything after it.

---

## Phase 1 — Environments ✅ complete

| What | Meaning | Result |
|---|---|---|
| ManiSkill 3 installed (PhysX physics engine under it) | The simulator; runs thousands of scene copies in parallel on one GPU | Sanity-proven: built-in task trains to 100% |
| Four tasks written | T3 slide-into-slot (primary: wrong friction → stop short / overshoot, and the correct *recovery direction* reveals physics understanding); T1 pouring, T2 mug-carrying (balls as "liquid"); T4 pick-and-place (control: physics can't matter once gripped) | All build, run, and log the plan's metrics (attempt count, first-recovery direction, spill fractions) |
| Hidden physics pipeline | Each parallel scene gets its own c, baked into the physics engine; a teacher-only flag exposes c in the observation | Teacher obs is exactly +4 numbers wider (rigid) / +7 (granular); students provably never see c |
| **Gate: randomization reaches the physics** | Replay identical robot motion under different c — outcomes must differ | **PASSED**: mass alone moves the endpoint 8.8 cm, friction 4.4 cm, extremes 16 cm, off-center weight flips rotation |
| δ grid + tags | Every off-range physics point tagged interpolation / extrapolation / composition | 120 points written to `configs/delta_grid.json` |
| Calibration tier | Tiny math-worlds where the true answer is known — used later to prove our measurement tools aren't lying | Self-test passed, incl. a decoy c-component the expert provably ignores |
| Demo videos | 2×2 grids: same controller, four different physics | `demos/` — one folder per task |

## Phase 2 — Data (teachers + demonstrations) 🟡 nearly done

| What | Meaning | Result |
|---|---|---|
| T4 teacher trained | RL with c visible, stock PPO baseline, only the observation changed | Eval success 1.0 |
| **Gate: T4 teacher flat across c** | If the teacher is worse in some corner of physics-space, its demos poison that corner | **PASSED**: 0.996 success, spread 0.03 |
| T3 teacher trained | Same recipe; two real bugs found and fixed on the way (see "Lessons") | ✅ **Certified: `runs/t3-teacher-v2b/ckpt_41.pt`**, success 0.924 |
| **Gate: T3 teacher flat across c** | If the teacher is worse in some region, its demos poison that region | ✅ **PASSED on the proper statistic**: raw cell-spread (0.28) is what binomial noise predicts for a flat 0.90 teacher at 64 episodes/cell; per-axis marginals (where noise averages out) show ≤3.5-point effects on every axis — no systematic slope. Naive spread threshold documented as revised |
| T3 datasets 10³ / 10⁴ | Demonstrations from the certified teacher, c stored as metadata only | ✅ 1,000 eps (92.5%) and 10,000 eps (92.7%) done and verified |
| T3 dataset 10⁵ | The big training set | ⏸ stopped mid-run for machine shutdown; regenerates exactly from its fixed seed (~2-3 GPU-hours) |
| Demonstration recorder | Records what the *student* sees (no c inside), stores c separately as metadata; ManiSkill HDF5 format so the same data can be replayed later as camera images | Validated end-to-end; counts always cross-checked |
| c-blind expert + datasets | A scripted slow-nudge controller that succeeds ~100% *without knowing c* — the "no reason to learn physics" comparison condition | 1,000-episode and 10,000-episode sets done (99.98% success) |
| T4 dataset (10⁴ episodes) | Demonstrations from the certified T4 teacher | ✅ 10,000 episodes, 99.7% success, metadata verified |
| Vision variant pipeline | Replays recorded states into 128×128 camera images (object visible only as pixels) | Verified; camera repointed to frame the task properly |
| Held-out eval manifest | Fixed episode seeds so every policy faces identical test conditions | 120k episodes defined in `configs/eval_manifest.json` |

## Phase 3 — Training machinery ✅ built, main arms not launched

| What | Meaning | Result |
|---|---|---|
| Flow policy (arm A) | Modern imitation learner (rectified flow, like Diffusion Policy): sees last K observations, outputs short action sequences | Built; smoke-tested |
| Arm B | Same network + a 2-layer head that must also predict the *object's* future motion (never the robot's own joints); head deleted at test time | Verified: A and B are byte-identical at deployment — the whole comparison rests on this |
| Controls | B-shuffled (same head, garbage targets — separates "physics" from "any second task") and arm C (keeps the predictor at test time, like published systems) | Built; all 12 arm×target combos pass checks |
| Substitution knobs | Weight decay / smoothness penalty / more data / longer training — rival explanations to test in the mechanism phase | Flags implemented |
| **Pre-registration** | Pass/fail rules and search budgets frozen in `configs/preregistration.json` *before* any run, so results can't be reinterpreted later | Written, dated |
| δ-grid evaluator | Rolls any trained policy over all 120 physics points; includes the "base policy must fail far out" gate | Validated |
| Pilot arms A & B | First real training runs, on the c-blind dataset (both a stack-validation and a planned study condition) | ✅ trained; **first data point**: the policy imitating the c-blind expert stays robust far out of range (97.5% → 80.8%) — guarded slow nudging doesn't need physics knowledge, exactly as the plan predicted for this condition. The decisive "base policy must fail far out" gate runs on the teacher-data policy next |

## Analysis machinery (Phase 4, in progress)

| What | Meaning | Result |
|---|---|---|
| Slope fitting (`dynmod/analysis/slopes.py`) | The headline test: fit "how fast does performance decay as physics drift" per seed, compare arms by confidence-interval overlap (preregistered rule) | ✅ built, verified on pilot results |
| Probe library (`dynmod/analysis/probes.py`) | Ridge + MLP readouts measuring how much physics knowledge a policy's internals contain; sample-complexity metric; ablation with random-direction control | ✅ built |
| **Gate: probes validated on calibration tier** | Prove the measurement tools aren't lying, on a system where ground truth is known | ✅ **PASSED** (2026-08-17) after five instructive failures — see "Methodology findings". Probes/baselines/specificity certified; ablation erasure power measured and its interpretation rule preregistered |

## ✅ Resolved (2026-08-17): why the base policy never failed — task redesign

The gate kept failing (10⁴-trained base policy: 0.92 in-range → 0.80 at
extreme physics, even after extending the grid to 4.5× mass and halving the
horizon). Instrumentation found the real cause: **the policy was escorting
the object** — its gripper stayed within 6cm of the object for ~80% of the
final approach, nudging continuously instead of committing to a slide.
Continuous contact = continuous correction = physics can't matter. No
evaluation tweak fixes a task that permits escorting.

**Fix: a tunnel.** A low roof over the channel that the object slides
through but the hand cannot enter. The launch is now a physical commitment;
the zone past the tunnel stays reachable from above, so recovery pushes in
both directions (the task's key scientific structure) survive. The physics
gate on the new env shows the identical launch now spreads outcomes across
**17 cm** — heavy/rough objects die at the tunnel mouth, nominal ones sail
to the slot.

Cost, accepted deliberately: the T3 teacher and all T3 datasets were for the
tunnel-less task and are **invalidated** (T4 and the calibration tier are
unaffected; the c-blind expert needs a new flick-based script since creeping
cannot cross a tunnel). Teacher retraining on the new task started
immediately. This is precisely what the plan's precondition gate exists to
catch before 180 arm runs get spent on a saturated metric.

## Methodology findings from the calibration tier (2026-08-17)

The probe-validation gate failed five times in a row — each failure a real
discovery about measurement, found on a toy system instead of on the real
experiment:

1. **No persistent excitation** → the expert settles, physics become
   unobservable in the data. (Fixed: process noise.)
2. **Shortcut learning** → the imitator can copy actions via a ratio trick
   without ever representing the physics. (Fixed: query-state design.)
3. **Under-identified system** → the expert's optimal gain barely varied
   with physics, so there was nothing to decode. (Fixed: cheap-control
   parametrization; gain now spans 9-40.)
4. **Query-entangled features** → probe directions didn't transfer.
   (Fixed: query-free trunk, mirroring the real policy architecture.)
5. **Redundant encoding defeats linear ablation** → even 16-dimension
   iterative erasure (INLP) left the quantity decodable, and hurt no more
   than random erasure — while the decoy control stayed exactly null.
   **Consequence, written into the preregistration before any arm runs:
   ablation results count as evidence only when POSITIVE (ablation ≫ random
   control); null ablations are uninformative.**

Probes, baselines, sample-complexity and specificity all validate; the
ablation's limits are now measured rather than assumed.

## Task iteration: tunnel position (2026-08-17)

Tunnel v1 (early in the channel) made the full 24cm flick so precision-
critical (~5% launch-speed tolerance) that PPO exploration never found it
(4% success). Tunnel v2 sits directly BEFORE the slot: escorting is allowed
up to the tunnel mouth (learnable — the escort task was solved at 92%), but
the final stretch is always a committed ballistic pass whose stopping point
the hidden physics decide. Teacher v5 training on this geometry now.

## Lessons so far (bugs caught by the gates)

1. **Hovering teacher**: ending the episode on success made *not finishing*
   more valuable than finishing (reward stream stops at success). Fix: don't
   end the episode; occupying the success state keeps paying. Teacher then
   learned to finish *and hold*.
2. **Last checkpoint ≠ best checkpoint**: late RL training degraded the T3
   teacher (0.91 → 0.84); we keep checkpoints throughout and pick by
   evaluation.
3. **GPU physics isn't perfectly deterministic**: identical scenes can differ
   by ~4 cm after a violent shove — statistics must average over episodes,
   which the plan already requires.

## Machine shutdown state (2026-08-16)

Everything valuable is on persistent storage; the compute instance is safe
to stop. `/mnt/scratch` (local disk, wiped on stop) held the Python env, the
ManiSkill clone, and working copies of the data — all either backed up or
reproducible.

- Backed up to `dynamics_modeling/scratch_backup/`: all completed datasets
  (t3 10³/10⁴, c-blind 10³/10⁴, t4 10⁴, pilot set) + the three pilot policy
  checkpoints. Verified readable.
- Already persistent: all code (`dynmod/`), teacher checkpoints (`runs/`),
  reports, configs, demos.
- Discarded: the half-finished 10⁵ set (regenerates exactly from seed 20000).

**Resume checklist for the next session**
1. Recreate the env (~5 min):
   `python -m venv /mnt/scratch/dynamics/ms3env && pip install torch --index-url https://download.pytorch.org/whl/cu121 && git clone https://github.com/haosulab/ManiSkill /mnt/scratch/dynamics/ManiSkill && pip install -e /mnt/scratch/dynamics/ManiSkill tyro tensorboard h5py matplotlib "imageio[pyav]"`
2. Copy `scratch_backup/` contents back to `/mnt/scratch/dynamics/` (or point
   scripts at the backup paths directly).
3. Regenerate the 10⁵ set: `rollout_harness --ckpt runs/t3-teacher-v2b/ckpt_41.pt --episodes 100000 --num-envs 500 --seed 20000 --out .../t3_1e5`.
4. Resolve the two open items below, then fire `launch_main_arms.py`.

## Next up

1. Decide the far-grid fix (open finding above): re-test with a 10⁴-trained
   base policy, then extend the δ grid and/or shorten the eval horizon.
2. Finish the probe-calibration gate (open item above).
3. Regenerate the 10⁵ dataset + the vision variant (10⁴, replay pipeline
   ready).
4. Launch the main arms (`launch_main_arms.py`, 180 runs) and the λ sweep,
   then fit degradation slopes.
