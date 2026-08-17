"""Degradation-slope fitting and arm comparison (plan §5 and Part II
'Build: evaluation' / 'Read the results').

Per the preregistration (configs/preregistration.json):
  - regret R = 1 - success_once (and separately mean final distance)
  - fit R(delta) per seed by ordinary least squares over grid points
  - the slope distributions of two arms are compared by 95% t-intervals over
    seeds; overlapping CIs on the A-vs-B comparison = the effect is absent,
    and that is the reported result
  - interpolation / extrapolation / composition are fitted and reported
    separately, never averaged

Input: the JSON files written by dynmod.policy.evaluate (one per seed).

    python -m dynmod.analysis.slopes reports/policy_eval_A-s*.json \
        --label-a A --vs reports/policy_eval_B-s*.json --label-b B
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

TAGS = ("interpolation", "extrapolation", "composition", "all")


def fit_slope(points: list, tag: str, measure: str) -> float:
    """OLS slope of regret against delta over the selected grid points."""
    rows = [p for p in points if tag == "all" or p["tag"] == tag]
    x = np.array([p["delta"] for p in rows])
    if measure == "success":
        y = 1.0 - np.array([p["success"] for p in rows])  # regret
    else:
        y = np.array([p["mean_final_dist"] for p in rows])
    if len(x) < 2 or np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def seed_slopes(files: list, measure: str) -> dict:
    """tag -> array of per-seed slopes."""
    out = {t: [] for t in TAGS}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        for t in TAGS:
            out[t].append(fit_slope(data["points"], t, measure))
    return {t: np.array(v) for t, v in out.items()}


def t_ci(x: np.ndarray, conf: float = 0.95):
    """Mean and 95% t-interval. Falls back to +-inf half-width for n<2."""
    x = x[~np.isnan(x)]
    n = len(x)
    m = float(x.mean()) if n else float("nan")
    if n < 2:
        return m, float("inf")
    # two-sided t critical values for common small n (df = n-1)
    tcrit = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 19: 2.093}
    df = n - 1
    t = tcrit.get(df, 2.0 + 2.0 / df)
    return m, float(t * x.std(ddof=1) / math.sqrt(n))


def compare(files_a, files_b, label_a="A", label_b="B", measure="success"):
    sa, sb = seed_slopes(files_a, measure), seed_slopes(files_b, measure)
    print(f"== degradation slopes ({measure} regret vs delta), "
          f"{len(files_a)} vs {len(files_b)} seeds ==")
    result = {}
    for t in TAGS:
        ma, ha = t_ci(sa[t])
        mb, hb = t_ci(sb[t])
        overlap = (ma - ha <= mb + hb) and (mb - hb <= ma + ha)
        verdict = ("CIs OVERLAP -> effect absent under the preregistered rule"
                   if overlap else "CIs SEPARATED")
        print(f"  {t:14s} {label_a}: {ma:+.4f} ± {ha:.4f}   "
              f"{label_b}: {mb:+.4f} ± {hb:.4f}   {verdict}")
        result[t] = dict(a_mean=ma, a_hw=ha, b_mean=mb, b_hw=hb,
                         overlap=bool(overlap))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files_a", nargs="+")
    p.add_argument("--vs", nargs="+", default=None)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--measure", choices=["success", "dist"], default="success")
    args = p.parse_args()
    if args.vs:
        compare(args.files_a, args.vs, args.label_a, args.label_b, args.measure)
    else:
        s = seed_slopes(args.files_a, args.measure)
        for t in TAGS:
            m, h = t_ci(s[t])
            print(f"  {t:14s} slope {m:+.4f} ± {h:.4f} "
                  f"(n={np.sum(~np.isnan(s[t]))})")


if __name__ == "__main__":
    main()
