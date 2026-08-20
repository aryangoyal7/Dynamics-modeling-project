"""Does a tighter deadline restore T8's knowledge premium?

The full gate on the final configuration measured +0.006 +/- 0.062 - zero -
because the c-blind controller escaped by carrying SLOWLY: given a 140-step
budget it picked the grid's slowest speed (0.25) and scored 0.975, and a calm
carry makes placement irrelevant. That is T8's own design law (i): a monotone
choice against a c-independent floor lets the blind arm park at the safe
corner. The earlier +12.7 came from a family whose slowest option was 0.4 -
a restriction imposed by grid bounds, not by the task.

So the question is whether the TASK can forbid the slow carry. Measured
finishing times: ~114 steps at speed 0.4, ~129 at 0.25, ~138 at 0.2. A budget
near 120 admits a fast carry and rules out a slow one. This sweeps the budget
and, at each one, gives BOTH arms their best (speed, bias) over the same grid.

    python -m dynmod.scripts.t8_deadline_probe
Writes reports/t8_deadline_probe.json.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import torch

from dynmod.scripts.premium_gate_t8 import (BIASES, SPEEDS, episode, fixed,
                                            make_env, place_aware, place_blind)

BUDGETS = [100, 110, 120, 130]
N = 128
CYCLES = 2


def main():
    out = {}
    for budget in BUDGETS:
        env = make_env(N, reconfiguration_freq=1)
        best = {"aware": (-1.0, None), "blind": (-1.0, None)}
        for sp, b in itertools.product(SPEEDS, BIASES):
            for mode in ("aware", "blind"):
                place = (place_aware(lambda base, c, b=b: b) if mode == "aware"
                         else place_blind(b))
                succ = []
                for cyc in range(CYCLES):
                    seen, _, _, _ = episode(env, fixed(sp), place,
                                            seed=9000 + cyc, horizon=budget)
                    succ.extend(seen.cpu().numpy().tolist())
                v = float(np.mean(succ))
                if v > best[mode][0]:
                    best[mode] = (v, (sp, b))
        env.close()
        n_ep = N * CYCLES
        ci = 2 * (2 * 0.25 / n_ep) ** 0.5
        prem = best["aware"][0] - best["blind"][0]
        out[str(budget)] = dict(
            aware=best["aware"][0], aware_cfg=best["aware"][1],
            blind=best["blind"][0], blind_cfg=best["blind"][1],
            premium=prem, ci95=ci, episodes=n_ep)
        print(f"budget {budget}: aware {best['aware'][0]:.3f} "
              f"{best['aware'][1]}  blind {best['blind'][0]:.3f} "
              f"{best['blind'][1]}  premium {prem:+.3f} ± {ci:.3f}", flush=True)
    json.dump(out, open("reports/t8_deadline_probe.json", "w"), indent=1)
    print("wrote reports/t8_deadline_probe.json")


if __name__ == "__main__":
    main()
