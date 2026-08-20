"""Publication figures from the persistent reports/ JSONs.

Regenerate everything with:
    python -m dynmod.analysis.make_figs        # or python3 dynmod/analysis/make_figs.py
Writes PNGs into figs/. Safe to re-run as more evaluations land - the T3
panels pick up whatever policy_eval_t3-*.json files exist.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS = os.path.join(ROOT, "reports")
FIGS = os.path.join(ROOT, "figs")
TAGS = ("interpolation", "extrapolation", "composition", "all")
TCRIT = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

ARM_COLORS = {"A-multistep": "#444444", "B-onestep": "#7fb3d5", "B-multistep": "#1f77b4",
              "B-latent": "#9467bd", "Bshuf-multistep": "#e0a03c",
              "C-multistep": "#2ca02c"}


def load(prefix, scale, arm, target):
    files = sorted(glob.glob(
        f"{REPORTS}/policy_eval_{prefix}-{scale}-{arm}-{target}-s*.json"))
    return [json.load(open(f)) for f in files]


def slope(points, tag):
    rows = [p for p in points if tag == "all" or p["tag"] == tag]
    x = np.array([p["delta"] for p in rows])
    y = 1.0 - np.array([p["success"] for p in rows])
    return float(np.polyfit(x, y, 1)[0]) if len(x) > 1 and np.ptp(x) else np.nan


def ci(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), np.inf
    t = TCRIT.get(len(v) - 1, 2.0 + 2.0 / (len(v) - 1))
    return float(v.mean()), float(t * v.std(ddof=1) / np.sqrt(len(v)))


def fig_t3_degradation(scale="1e3"):
    """Success vs shift magnitude + per-tag slope CIs, all arms with >=2 seeds."""
    arms = [("A", "multistep"), ("B", "onestep"), ("B", "multistep"),
            ("B", "latent"), ("Bshuf", "multistep"), ("C", "multistep")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    plotted = []
    for arm, target in arms:
        runs = load("t3", scale, arm, target)
        if len(runs) < 2:
            continue
        name = f"{arm}-{target}"
        plotted.append(name)
        # panel 1: binned mean success vs delta (mean over seeds & grid points)
        deltas = np.array([p["delta"] for p in runs[0]["points"]])
        succ = np.array([[p["success"] for p in r["points"]] for r in runs])
        m = succ.mean(axis=0)
        bins = np.quantile(deltas, np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(deltas, bins) - 1, 0, 7)
        bx = [deltas[idx == b].mean() for b in range(8)]
        by = [m[idx == b].mean() for b in range(8)]
        be = [m[idx == b].std() / np.sqrt(max((idx == b).sum(), 1))
              for b in range(8)]
        ax1.errorbar(bx, by, yerr=be, marker="o", ms=4, capsize=2,
                     color=ARM_COLORS[name], label=f"{name} (n={len(runs)})")
        # panel 2: per-tag slope mean +- 95% CI (interpolation omitted: too
        # few grid points for a stable per-seed OLS, CI ~ +-0.2 swamps the plot)
        panel_tags = ("extrapolation", "composition", "all")
        for i, tag in enumerate(panel_tags):
            mval, h = ci([slope(r["points"], tag) for r in runs])
            x = i + (plotted.index(name) - 2.5) * 0.09
            ax2.errorbar([x], [mval], yerr=[h if np.isfinite(h) else 0],
                         marker="o", ms=5, capsize=3, color=ARM_COLORS[name])
            if tag == "all":
                print(f"  {scale} {name} all-tag slope {mval:+.4f} ± {h:.4f}")
    if not plotted:
        plt.close(fig)
        print(f"t3_{scale}: no arms with >=2 seeds evaluated yet - skipped")
        return
    ax1.set_xlabel("shift magnitude $\\delta$ (normalized)")
    ax1.set_ylabel("success rate")
    ax1.set_title(f"T3 at {scale}: robustness degradation")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(("extrapolation", "composition", "all"), fontsize=9)
    ax2.set_ylabel("degradation slope (regret / $\\delta$)")
    ax2.set_title("per-seed slopes, mean $\\pm$ 95% t-CI\n(overlap = no effect, preregistered)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/t3_{scale}_result.png", dpi=150)
    plt.close(fig)
    print(f"t3_{scale}_result.png  arms: {plotted}")


def fig_t8_mechanism():
    """T8's premium mechanism: the hidden COM displaces the optimal placement
    by exactly its own size. Left: success against where the block is SEATED,
    one curve per hidden COM. Right: the same points against where its MASS
    ends up - the two curves collapse, which is what makes the knowledge
    worth having."""
    path = f"{REPORTS}/t8_placement_sweep.json"
    if not os.path.exists(path):
        print("t8_mechanism: sweep not run yet - skipped")
        return
    d = json.load(open(path))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"com_0": "#1f77b4", "com_plus_1.05cm": "#d62728"}
    for tag, rec in d.items():
        if tag == "meta":
            continue
        x = np.array([float(k) for k in rec["curve"]])
        y = np.array(list(rec["curve"].values()))
        lab = (f"hidden COM {rec['com_y_cm']:+.2f} cm"
               if rec["com_y_cm"] else "hidden COM centred")
        ax1.plot(x, y, "o-", color=colors.get(tag, "#444"), label=lab)
        ax2.plot(x + rec["com_y_cm"], y, "o-", color=colors.get(tag, "#444"),
                 label=lab)
    half = d["meta"]["beam_half_width_cm"]
    for ax, xl in ((ax1, "where the block is SEATED (cm from beam centre)"),
                   (ax2, "where its MASS ends up (cm from beam centre)")):
        ax.axvspan(-half, half, color="#999", alpha=0.12,
                   label="beam width" if ax is ax1 else None)
        ax.set_xlabel(xl)
        ax.set_ylabel("carry success")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax1.set_title("The hidden mass shifts the best placement...")
    ax2.set_title("...by exactly its own offset (curves collapse)")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/t8_mechanism.png", dpi=150)
    plt.close(fig)
    print("t8_mechanism.png")


def fig_gate_premiums():
    """Knowledge-premium gate verdict for every task designed so far."""
    # T8's final verdict is the gate at the budget where its teacher can be
    # certified; at the tighter budget the premium exists but the teacher
    # cannot (see reports/t8_deadline_probe.json)
    t8 = (0.6, 6.2)
    f8 = f"{REPORTS}/premium_gate_t8_com04_budget140.json"
    if os.path.exists(f8):
        r = json.load(open(f8))["result"]
        t8 = (100 * r["premium"], 100 * r["ci95"])
    rows = [  # (label, premium %, ci %, passed)
        ("T8 stack-and-carry\n(teachable budget)", t8[0], t8[1], False),
        ("T3 slide-to-slot\n(commit at launch)", 16.7, 3.6, True),
        ("T1 dump-pour", -4.9, 6.3, False),
        ("T1 measured-pour", 2.7, 6.2, False),
        ("T2 carry-rails", 1.2, 5.1, False),
        ("T5 cliff-toss", 0.6, 6.3, False),
        ("T6 push-T\n(free correction)", -0.2, 3.1, False),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(rows))[::-1]
    for yi, (label, p, c, ok) in zip(y, rows):
        ax.barh(yi, p, xerr=c, capsize=4,
                color="#2ca02c" if ok else "#bbbbbb",
                edgecolor="#333333", height=0.6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("knowledge premium: c-aware $-$ c-blind success (points, $\\pm$95% CI)")
    ax.set_title("Validity gate across task designs: where physics knowledge pays")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/gate_premiums.png", dpi=150)
    plt.close(fig)
    print("gate_premiums.png")


def fig_t4_control():
    """T4 control: identical slopes across all 6 arms + probe R^2 showing the
    prediction head injects physics knowledge anyway."""
    arms = [("A", "multistep"), ("B", "onestep"), ("B", "multistep"),
            ("B", "latent"), ("Bshuf", "multistep"), ("C", "multistep")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = []
    for i, (arm, target) in enumerate(arms):
        runs = load("t4", "1e4", arm, target)
        if not runs:
            continue
        name = f"{arm}-{target}"
        labels.append(name)
        m, h = ci([slope(r["points"], "all") for r in runs])
        ax1.errorbar([i], [m], yerr=[h], marker="o", ms=6, capsize=4,
                     color=ARM_COLORS[name])
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, fontsize=8, ha="right")
    ax1.set_ylabel("degradation slope (all tags)")
    ax1.set_title("T4 control: slopes identical across arms (10 seeds)")
    ax1.grid(alpha=0.3)

    probes = json.load(open(f"{REPORTS}/t4_probes.json"))
    per_arm = {}
    for run, d in probes.items():
        arm = "-".join(run.split("-")[2:4])
        per_arm.setdefault(arm, {"trained": [], "untrained": [], "raw": []})
        per_arm[arm]["trained"].append(d["trained"]["mass_mult"]["mlp_r2"])
        per_arm[arm]["untrained"].append(
            d["untrained_baseline"]["mass_mult"]["mlp_r2"])
        per_arm[arm]["raw"].append(d["raw_input"]["mass_mult"]["mlp_r2"])
    order = [f"{a}-{t}" for a, t in arms if f"{a}-{t}" in per_arm]
    x = np.arange(len(order))
    tr = [np.mean(per_arm[a]["trained"]) for a in order]
    te = [np.std(per_arm[a]["trained"]) / np.sqrt(len(per_arm[a]["trained"]))
          for a in order]
    ax2.bar(x, tr, yerr=te, capsize=3,
            color=[ARM_COLORS[a] for a in order], edgecolor="#333")
    un = np.mean([v for a in order for v in per_arm[a]["untrained"]])
    ax2.axhline(un, color="k", ls="--", lw=1, label="untrained-net baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=20, fontsize=8, ha="right")
    ax2.set_ylabel("mass decodable from trunk (probe $R^2$)")
    ax2.set_title("...but B trunks carry more physics\n(knowledge without effect)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/t4_control.png", dpi=150)
    plt.close(fig)
    print("t4_control.png")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_t3_degradation("1e3")
    fig_t3_degradation("1e4")
    fig_gate_premiums()
    fig_t4_control()
    fig_t8_mechanism()
