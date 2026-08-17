"""Probing and ablation library (plan Part II 'Build: evaluation').

  - ridge and 2-layer MLP probes decoding hidden-parameter components from
    frozen features, with the two required baselines (untrained network of
    identical architecture, raw input)
  - probe sample complexity: labelled examples needed to reach a fixed R^2
  - ablation: project out the probe direction and re-measure, against a
    random direction of matched effect (report the difference, not the raw
    drop)

Validated on the calibration tier by dynmod.scripts.validate_probes_calibration
before being trusted on real policies (plan gate).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _split(n, frac=0.8, seed=0):
    idx = np.random.RandomState(seed).permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]


def fit_ridge(F: np.ndarray, y: np.ndarray, lam: float = 1e-2, seed: int = 0):
    """Closed-form ridge probe. Returns (test R^2, weight vector w)."""
    tr, te = _split(len(F), seed=seed)
    Fm, Fs = F[tr].mean(0), F[tr].std(0) + 1e-8
    ym, ys = y[tr].mean(), y[tr].std() + 1e-8
    A = (F[tr] - Fm) / Fs
    b = (y[tr] - ym) / ys
    w = np.linalg.solve(A.T @ A + lam * len(tr) * np.eye(A.shape[1]), A.T @ b)
    pred = ((F[te] - Fm) / Fs) @ w
    yt = (y[te] - ym) / ys
    r2 = 1.0 - np.mean((pred - yt) ** 2) / (np.mean(yt**2) + 1e-12)
    w_raw = w / Fs  # direction in raw feature space
    return float(r2), w_raw


def fit_mlp(F: np.ndarray, y: np.ndarray, hidden: int = 128, epochs: int = 60,
            seed: int = 0, device: str = "cpu"):
    """2-layer MLP probe (matched depth to the prediction head)."""
    torch.manual_seed(seed)
    tr, te = _split(len(F), seed=seed)
    Fm, Fs = F[tr].mean(0), F[tr].std(0) + 1e-8
    ym, ys = y[tr].mean(), y[tr].std() + 1e-8
    Xtr = torch.tensor((F[tr] - Fm) / Fs, dtype=torch.float32, device=device)
    ytr = torch.tensor((y[tr] - ym) / ys, dtype=torch.float32, device=device)
    Xte = torch.tensor((F[te] - Fm) / Fs, dtype=torch.float32, device=device)
    yte = torch.tensor((y[te] - ym) / ys, dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(F.shape[1], hidden), nn.SiLU(),
                        nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(Xtr).squeeze(-1) - ytr) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xte).squeeze(-1)
        r2 = 1.0 - ((pred - yte) ** 2).mean() / (yte**2).mean().clamp(min=1e-12)
    return float(r2)


def sample_complexity(F: np.ndarray, y: np.ndarray, r2_target: float = 0.5,
                      sizes=(32, 64, 128, 256, 512, 1024, 2048, 4096)):
    """Smallest number of labelled examples for a ridge probe to reach the
    target test R^2 ('the certification number nobody reports'). Returns -1
    if never reached."""
    for n in sizes:
        if n > int(0.8 * len(F)):
            break
        idx = np.random.RandomState(1).permutation(len(F))[: int(n / 0.8)]
        r2, _ = fit_ridge(F[idx], y[idx])
        if r2 >= r2_target:
            return int(n)
    return -1


def project_out(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Remove the component of every feature vector along w."""
    u = w / (np.linalg.norm(w) + 1e-12)
    return F - np.outer(F @ u, u)


def random_direction_like(w: np.ndarray, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*w.shape)


def inlp_subspace(F: np.ndarray, y: np.ndarray, max_dims: int = 16,
                  r2_stop: float = 0.05, lam: float = 1e-2) -> np.ndarray:
    """Iterative nullspace projection: repeatedly fit a ridge probe and
    remove its direction until the target stops being decodable. Returns an
    orthonormal basis (d, D) of the erased subspace.

    Motivated by the calibration gate: with redundant encodings, removing a
    SINGLE probe direction leaves the information readable from the residual
    subspace and the ablation silently under-reports (a false negative).
    Ablations must erase the whole decodable subspace, with an
    equal-dimension random-subspace control.
    """
    Fc = F.copy()
    basis = []
    for _ in range(max_dims):
        r2, w = fit_ridge(Fc, y, lam=lam)
        if r2 < r2_stop:
            break
        u = w / (np.linalg.norm(w) + 1e-12)
        for b in basis:  # keep the basis orthonormal
            u = u - (u @ b) * b
        n = np.linalg.norm(u)
        if n < 1e-8:
            break
        u = u / n
        basis.append(u)
        Fc = Fc - np.outer(Fc @ u, u)
    return np.stack(basis) if basis else np.zeros((0, F.shape[1]))


def random_subspace_like(W: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random orthonormal basis with the same shape as W (the control)."""
    d, D = max(W.shape[0], 1), W.shape[1]
    q, _ = np.linalg.qr(np.random.RandomState(seed).randn(D, d))
    return q.T[: W.shape[0]] if W.shape[0] else q.T[:1]


def project_out_subspace(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    for u in W:
        F = F - np.outer(F @ u, u)
    return F


def probe_report(feats: dict, c: dict, r2_target: float = 0.5) -> dict:
    """feats: name -> (N, D) feature sets (e.g. trained / untrained / raw
    input); c: component name -> (N,) true values. Returns nested results."""
    out = {}
    for fname, F in feats.items():
        out[fname] = {}
        for cname, y in c.items():
            r2_ridge, w = fit_ridge(F, y)
            r2_mlp = fit_mlp(F, y)
            out[fname][cname] = dict(
                ridge_r2=round(r2_ridge, 4),
                mlp_r2=round(r2_mlp, 4),
                n_for_r2=sample_complexity(F, y, r2_target),
            )
    return out
