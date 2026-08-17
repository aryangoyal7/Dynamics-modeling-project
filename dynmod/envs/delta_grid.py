"""The delta evaluation grid over hidden physics c.

Plan (Part II, Build: environments):
  mass multiplier: {0.3, 0.5, 1.0, 1.6, 2.2, 3.0}
  friction:        {0.1, 0.2, 0.5, 0.9, 1.2}
  COM offset:      up to 40% of half-width -> {0.0, 0.10, 0.25, 0.40}
  Tag each grid point as interpolation, extrapolation, or composition.

Tagging: a component is in-range if it lies inside the training range
(mass [0.7, 1.4], friction [0.3, 0.7], COM offset [0, 0.15]). 0 components
out -> interpolation, exactly 1 out -> extrapolation, 2+ out -> composition.

delta = || normalized displacement of c from the training-distribution mean ||.
Mass and friction are measured in log space (they are sampled log-uniformly),
the COM offset linearly; each component is normalized by the half-width of its
training range so the components are commensurate. delta <= sqrt(3) means every
component is inside its training range.

Run `python -m dynmod.envs.delta_grid` to write configs/delta_grid.json and
print a summary table.
"""

from __future__ import annotations

import itertools
import json
import math
import os

from dynmod.envs.randomization import CTrainSpec

# extended 2026-08-17 per the plan's base-policy gate ("push the held-out
# values further"): the base policy degraded only 0.92 -> 0.80 on the
# original grid, leaving no gap to attribute. Extension happened BEFORE any
# study arm was launched. Original axes were mass {0.3..3.0}, friction
# {0.1..1.2}, com {0..0.40}.
GRID_MASS_MULT = [0.3, 0.5, 1.0, 1.6, 2.2, 3.0, 4.5]
GRID_FRICTION = [0.1, 0.2, 0.5, 0.9, 1.2, 1.8]
GRID_COM_FRAC = [0.0, 0.10, 0.25, 0.40, 0.60]

_EPS = 1e-9


def _axes(spec: CTrainSpec):
    """(center, half-width, transform) per component, in the space delta is measured in."""
    m_lo, m_hi = spec.mass_mult_range
    f_lo, f_hi = spec.friction_range
    return {
        "mass_mult": (
            0.5 * (math.log(m_lo) + math.log(m_hi)),
            0.5 * (math.log(m_hi) - math.log(m_lo)),
            math.log,
        ),
        "friction": (
            0.5 * (math.log(f_lo) + math.log(f_hi)),
            0.5 * (math.log(f_hi) - math.log(f_lo)),
            math.log,
        ),
        "com_frac": (
            0.5 * spec.com_frac_max,
            0.5 * spec.com_frac_max,
            lambda x: x,
        ),
    }


def _normalized_disp(point: dict, spec: CTrainSpec) -> dict:
    axes = _axes(spec)
    return {
        k: (tf(point[k]) - center) / halfw if halfw > 0 else 0.0
        for k, (center, halfw, tf) in axes.items()
    }


def delta_of(point: dict, spec: CTrainSpec | None = None) -> float:
    spec = spec or CTrainSpec()
    d = _normalized_disp(point, spec)
    return math.sqrt(sum(v * v for v in d.values()))


def components_out_of_range(point: dict, spec: CTrainSpec | None = None) -> list:
    spec = spec or CTrainSpec()
    out = []
    if not (
        spec.mass_mult_range[0] - _EPS
        <= point["mass_mult"]
        <= spec.mass_mult_range[1] + _EPS
    ):
        out.append("mass_mult")
    if not (
        spec.friction_range[0] - _EPS
        <= point["friction"]
        <= spec.friction_range[1] + _EPS
    ):
        out.append("friction")
    if not (0.0 <= point["com_frac"] <= spec.com_frac_max + _EPS):
        out.append("com_frac")
    return out


def tag_of(point: dict, spec: CTrainSpec | None = None) -> str:
    n_out = len(components_out_of_range(point, spec))
    if n_out == 0:
        return "interpolation"
    if n_out == 1:
        return "extrapolation"
    return "composition"


def make_grid(spec: CTrainSpec | None = None) -> list:
    """All grid points, each a dict with the c components, tag, and delta."""
    spec = spec or CTrainSpec()
    grid = []
    for i, (m, f, o) in enumerate(
        itertools.product(GRID_MASS_MULT, GRID_FRICTION, GRID_COM_FRAC)
    ):
        point = dict(mass_mult=m, friction=f, com_frac=o)
        grid.append(
            dict(
                id=i,
                **point,
                out_of_range=components_out_of_range(point, spec),
                tag=tag_of(point, spec),
                delta=round(delta_of(point, spec), 4),
            )
        )
    return grid


def main():
    grid = make_grid()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "configs")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "delta_grid.json")
    with open(path, "w") as fp:
        json.dump(
            dict(
                train_spec=vars(CTrainSpec()),
                axes=dict(
                    mass_mult=GRID_MASS_MULT,
                    friction=GRID_FRICTION,
                    com_frac=GRID_COM_FRAC,
                ),
                points=grid,
            ),
            fp,
            indent=1,
            default=str,
        )
    tags = {}
    for p in grid:
        tags[p["tag"]] = tags.get(p["tag"], 0) + 1
    print(f"wrote {len(grid)} grid points -> {path}")
    print("tags:", tags)
    deltas = sorted(p["delta"] for p in grid)
    print(f"delta range: {deltas[0]:.3f} .. {deltas[-1]:.3f}")


if __name__ == "__main__":
    main()
