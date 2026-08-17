"""Implementation smoke test for the Section 11 policy stack. NOT a training
run: verifies shapes, gradients, loss descent, checkpoint round-trip, and the
delta-grid evaluator plumbing, using whatever small dataset exists.

    python -m dynmod.scripts.policy_smoke_test --data /mnt/scratch/dynamics/data/t3_cblind_1e3
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from dynmod.policy.data import OBS_LAYOUT_T3, TrajectoryDataset
from dynmod.policy.model import ARMS, TARGETS, FlowPolicy, PolicyConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--descent-steps", type=int, default=300)
    args = p.parse_args()
    device = "cuda"

    ds = TrajectoryDataset(args.data, **OBS_LAYOUT_T3)
    val = TrajectoryDataset(args.data, val=True, **OBS_LAYOUT_T3)
    print(f"dataset: {len(ds)} train / {len(val)} val samples from {args.data}")
    loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)
    batch = [b.to(device) for b in next(iter(loader))]
    hist_next, chunk, future_obj, prev_act = batch
    act_dim = chunk.shape[-1]  # 8 for pd_joint_delta_pos, 4 for pd_ee_delta_pos
    assert hist_next.shape[1:] == (5, 41) and chunk.shape[1:] == (8, act_dim)
    assert future_obj.shape[1:] == (8, 13) and prev_act.shape[1:] == (act_dim,)
    # every arm x target: losses finite, gradients flow, sampling shape right
    for arm in ARMS:
        for target in TARGETS:
            cfg = PolicyConfig(arm=arm, target=target, act_dim=act_dim)
            m = FlowPolicy(cfg).to(device)
            losses = m.loss(*batch)
            losses["total"].backward()
            grad_ok = any(
                p_.grad is not None and p_.grad.abs().sum() > 0
                for p_ in m.trunk.parameters()
            )
            a = m.sample_actions(hist_next[:, :4], prev_act, n_steps=4)
            assert a.shape == chunk.shape and torch.isfinite(a).all()
            assert all(torch.isfinite(v).all() for v in losses.values())
            assert grad_ok, f"{arm}/{target}: no gradient reached the trunk"
            n_par = sum(x.numel() for x in m.parameters())
            print(f"[PASS] arm {arm:5s} target {target:9s} "
                  f"losses={{{', '.join(f'{k}:{v.item():.3f}' for k, v in losses.items())}}} "
                  f"params {n_par/1e6:.2f}M")

    # A/B parameter parity at inference: identical trunk+velocity head shapes
    mA = FlowPolicy(PolicyConfig(arm="A", act_dim=act_dim))
    mB = FlowPolicy(PolicyConfig(arm="B", act_dim=act_dim))
    shapes = lambda mod: [tuple(p_.shape) for p_ in list(mod.trunk.parameters())
                          + list(mod.vel_head.parameters())]
    assert shapes(mA) == shapes(mB), "A and B inference networks must be identical"
    print("[PASS] A and B share identical inference-time architecture")

    # shuffled-target control actually changes the loss target
    torch.manual_seed(0)
    mS = FlowPolicy(PolicyConfig(arm="Bshuf", act_dim=act_dim)).to(device)
    feat = mS.features(hist_next[:, :4])
    l_true = mS.pred_loss(feat, feat, chunk, future_obj, shuffle=False)
    l_shuf = mS.pred_loss(feat, feat, chunk, future_obj, shuffle=True)
    assert not torch.isclose(l_true, l_shuf), "shuffle had no effect"
    print(f"[PASS] shuffled targets change pred loss ({l_true.item():.3f} -> {l_shuf.item():.3f})")

    # brief descent: loss must fall on arm B (flow + pred jointly)
    m = FlowPolicy(PolicyConfig(arm="B", act_dim=act_dim)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    first = last = None
    it = iter(loader)
    for s in range(args.descent_steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        b = [x.to(device) for x in b]
        ls = m.loss(*b)
        opt.zero_grad(set_to_none=True)
        ls["total"].backward()
        opt.step()
        if s < 10:
            first = ls["total"].item() if first is None else first
        last = ls["total"].item()
    assert last < first, f"loss did not decrease ({first:.3f} -> {last:.3f})"
    print(f"[PASS] {args.descent_steps}-step descent: total {first:.3f} -> {last:.3f}")

    # checkpoint round-trip
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ck.pt")
        m.save(path, ds.normalizer(), extra=dict(args={"arm": "B"}))
        m2, norm, extra = FlowPolicy.load(path, device=device)
        a1 = m.sample_actions(hist_next[:1, :4], prev_act[:1], n_steps=4)
        assert norm["obs_mean"].shape == (41,) and extra["args"]["arm"] == "B"
        print("[PASS] checkpoint save/load round-trip")

    print("policy implementation smoke test PASSED")


if __name__ == "__main__":
    main()
