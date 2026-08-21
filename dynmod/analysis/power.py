"""Proper two-sample statistics on the per-seed degradation slopes.

The preregistered rule ("overlapping 95% CIs = effect absent") is
deliberately CONSERVATIVE: two 95% intervals can overlap while a direct
two-sample test still rejects.  Reporting only the overlap rule would let
us claim a null we never actually tested, so this module runs the tests
the null claim really needs:

  1. Welch's t-test (unpaired) and a paired t-test over seed index -- at a
     given scale, arm A seed s and arm B seed s were trained on the SAME
     dataset, so pairing is legitimate and strictly more powerful.
  2. TOST equivalence against a margin the study can defend, answering
     "is the difference small enough to call it nothing?"  A null is only
     meaningful with this number attached.
  3. The minimum difference 10 seeds could have detected at 80% power --
     i.e. how big an effect we would have missed.

    python -m dynmod.analysis.power
"""

from __future__ import annotations

import glob
import json
import math

import numpy as np

from dynmod.analysis.slopes import seed_slopes

ARMS = ["B-onestep", "B-multistep", "B-latent", "Bshuf-multistep", "C-multistep"]
BASE = "A-multistep"
TAGS = ("extrapolation", "composition", "all")
SCALES = ("1e3", "1e4", "1e5")


def _t_sf(t, df):
    """Two-sided p from |t| via the incomplete beta (no scipy dependency)."""
    t, df = abs(float(t)), float(df)
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b), continued fraction (NR 6.4)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def welch(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df, _t_sf(t, df)


def paired(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan"), float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    return t, n - 1, _t_sf(t, n - 1)


def tost(a, b, margin):
    """Two one-sided tests: is |mean difference| provably below margin?"""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    se = math.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    if se == 0:
        return float("nan")
    df = na + nb - 2
    diff = a.mean() - b.mean()
    p_lo = _t_sf((diff + margin) / se, df) / 2 if diff + margin > 0 else \
        1 - _t_sf((diff + margin) / se, df) / 2
    p_hi = _t_sf((diff - margin) / se, df) / 2 if diff - margin < 0 else \
        1 - _t_sf((diff - margin) / se, df) / 2
    return max(p_lo, p_hi)


def mde(a, b, n=10):
    """Smallest true difference detectable at 80% power, alpha=.05, n/arm."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 2.8 * sd * math.sqrt(2.0 / n)   # (1.96+0.84) * se_of_difference


def load(scale, arm):
    files = sorted(glob.glob(f"reports/policy_eval_t3-{scale}-{arm}-s*.json"))
    return files


def main():
    print("=" * 78)
    print("T3 degradation slopes: the tests a null result actually needs")
    print("(slope = regret gained per unit physics shift; LOWER = more robust)")
    print("=" * 78)
    for scale in SCALES:
        base_files = load(scale, BASE)
        if len(base_files) < 2:
            continue
        base_s = seed_slopes(base_files, "success")
        print(f"\n################  scale {scale}  "
              f"({len(base_files)} seeds/arm)  ################")
        for arm in ARMS:
            files = load(scale, arm)
            if len(files) < 2:
                continue
            arm_s = seed_slopes(files, "success")
            print(f"\n--- {BASE} (A) vs {arm} ---")
            for tag in TAGS:
                a, b = base_s[tag], arm_s[tag]
                diff = np.nanmean(a) - np.nanmean(b)
                _, _, pw = welch(a, b)
                _, _, pp = paired(a, b)
                m = mde(a, b, n=min(len(a), len(b)))
                # margin: 20% of A's own slope = a difference small enough
                # that it could never change a robustness conclusion
                marg = 0.2 * abs(np.nanmean(a))
                pe = tost(a, b, marg)
                verdict = ("A WORSE (less robust)" if diff > 0 else
                           "A better") if min(pw, pp) < 0.05 else "no difference"
                print(f"  {tag:<14} diff {diff:+.5f}  "
                      f"Welch p={pw:.3f}  paired p={pp:.3f}  "
                      f"| detectable>={m:.5f}  TOST(±{marg:.4f}) p={pe:.3f}"
                      f"  -> {verdict}")


if __name__ == "__main__":
    main()
