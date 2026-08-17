# T3 "Slide to Slot" — final task design (video: 0.mp4)

**Watch:** a 2x2 grid — four copies of the scene running the same controller
under different hidden physics (mass, friction, weight offset). The block ends
up in different places purely because the physics differ.

## The task
The robot must send the blue block down a walled channel so it comes to rest
on the green slot. The slot sits in an **open window between two grey roofs**.
The block slides under the roofs; the robot's hand cannot enter them. So:

- The robot may guide the block up to the first roof, but the final approach
  is always a **committed launch** — once released, only physics decide where
  the block stops.
- Launch too soft → the block strands under roof 1 (unrecoverable).
- Launch too hard → it flies past the window under roof 2; the only recovery
  is an equally committed **reverse launch** back through roof 2.
- Land in the window → small nudges are allowed to finish the job (and
  *which direction* the robot corrects reveals whether it understood what
  went wrong — a key measurement).

The floor of the channel is **slippery on purpose** (light blue strip), so
the arm has enough authority to launch even heavy, grippy blocks.

## Why the design looks like this (each feature fixes a measured failure)
1. **Roof 1** — without it, the robot cheated by keeping its gripper on the
   block the whole way ("escorting"), making physics irrelevant. Measured:
   gripper within 6 cm of the block for ~80% of the approach.
2. **Roof 2** — without it, the robot could deliberately overshoot, bounce
   off the end wall, and escort the block backwards into the slot — another
   physics-free strategy.
3. **Slick floor** — without it, even maximum-strength launches undershot
   for heavy/grippy blocks (19-22% ceiling), so no controller could express
   physics knowledge at all.

## Proof the design works (measured 2026-08-18)
A controller that is TOLD the true physics succeeds **61.7%**; the same
controller with one fixed launch for everything succeeds **45.0%**
(3,000+ episodes, 95% CI ±3.6). Knowing the physics is worth ~17 points —
that gap is what the main experiment measures: can a policy *learn* to close
it from demonstrations, and does adding a prediction objective help?
