"""Train one arm of the study (plan Part II 'Build: training').

Arms and targets are constructor flags on one shared implementation, so every
comparison differs by exactly the intended term:

    python -m dynmod.policy.train --data /mnt/scratch/dynamics/data/t3_1e4 \
        --arm B --target multistep --lam 1.0 --seed 0 --out-name b-multistep-s0

Substitution arms (Test D) reuse the base arm with one intervention each:
    --weight-decay W       (M2 regularisation)
    --jacobian-penalty B   (M5 representation smoothing: ||d phi / d h||^2,
                            random-direction finite-difference estimate)
    --data-fraction F      (M3 gradient density: paired with a larger source
                            dataset at launch time)
    --schedule-mult M      (M4 optimisation: M x steps, cosine re-stretched)
The search budget for these is preregistered in configs/preregistration.json
and must not be revised after the first run.

Checkpoints are saved at logarithmically spaced steps (the emergence
measurement needs them; retraining to recover them is avoidable waste).
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dynmod.policy.data import OBS_LAYOUT_T3, TrajectoryDataset
from dynmod.policy.model import ARMS, TARGETS, FlowPolicy, PolicyConfig

DEFAULT_OUT_ROOT = "/mnt/scratch/dynamics/policy_runs"


def log_spaced_checkpoints(total_steps: int, n: int = 15) -> list:
    pts = np.unique(
        np.round(np.logspace(np.log10(200), np.log10(total_steps), n)).astype(int)
    )
    return sorted(set(pts.tolist()) | {total_steps})


def jacobian_penalty(model: FlowPolicy, hist: torch.Tensor, sigma: float = 1e-3):
    """Random-direction finite-difference estimate of ||d phi / d h||^2."""
    eps = torch.randn_like(hist)
    eps = eps / (eps.flatten(1).norm(dim=1)[:, None, None] + 1e-8)
    d = model.features(hist + sigma * eps) - model.features(hist)
    return (d / sigma).pow(2).sum(dim=1).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--arm", choices=ARMS, default="A")
    p.add_argument("--target", choices=TARGETS, default="multistep")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--pred-h", type=int, default=8)
    # substitution knobs (Test D) - preregistered budget, see configs/
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--jacobian-penalty", type=float, default=0.0)
    p.add_argument("--data-fraction", type=float, default=1.0)
    p.add_argument("--schedule-mult", type=float, default=1.0)
    p.add_argument("--out-name", default=None)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--track", action="store_true",
                   help="mirror the tensorboard scalars to Weights & Biases")
    p.add_argument("--wandb-project", default="dynamics-modeling")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_steps = int(args.steps * args.schedule_mult)

    name = args.out_name or (
        f"{args.arm}-{args.target}-lam{args.lam:g}-s{args.seed}"
    )
    out = os.path.join(args.out_root, name)
    os.makedirs(out, exist_ok=True)
    if args.track:
        # sync_tensorboard mirrors every SummaryWriter scalar to W&B, so the
        # logging below needs no changes. A failed init (no login, no
        # network) must never kill an overnight queue - warn and continue.
        try:
            import wandb

            wandb.init(project=args.wandb_project, name=name,
                       config=vars(args), sync_tensorboard=True,
                       settings=wandb.Settings(silent=True))
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: wandb disabled ({e}); continuing with tensorboard only")
    writer = SummaryWriter(out)
    with open(os.path.join(out, "args.json"), "w") as fp:
        json.dump(vars(args), fp, indent=1)

    common = dict(
        K=args.K, chunk=args.chunk, pred_h=args.pred_h,
        obs_dim=OBS_LAYOUT_T3["obs_dim"], obj_slice=OBS_LAYOUT_T3["obj_slice"],
        seed=args.seed,
    )
    train_ds = TrajectoryDataset(args.data, data_fraction=args.data_fraction, **common)
    val_ds = TrajectoryDataset(args.data, val=True, **common)
    print(f"train samples {len(train_ds)}, val samples {len(val_ds)}")
    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=0)

    cfg = PolicyConfig(
        obs_dim=OBS_LAYOUT_T3["obs_dim"],
        obj_dim=OBS_LAYOUT_T3["obj_slice"][1] - OBS_LAYOUT_T3["obj_slice"][0],
        act_dim=train_ds.act[0].shape[-1],
        K=args.K, chunk=args.chunk, pred_h=args.pred_h,
        arm=args.arm, target=args.target, lam=args.lam,
    )
    model = FlowPolicy(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"arm {args.arm}/{args.target} params {n_params/1e6:.2f}M -> {out}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    ckpt_steps = set(log_spaced_checkpoints(total_steps))
    normalizer = train_ds.normalizer()

    def run_val():
        model.eval()
        agg, n = {}, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                ls = model.loss(*batch)
                for k, v in ls.items():
                    agg[k] = agg.get(k, 0.0) + v.item() * batch[0].shape[0]
                n += batch[0].shape[0]
        model.train()
        return {k: v / n for k, v in agg.items()}

    step, data_iter = 0, iter(loader)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = [b.to(device, non_blocking=True) for b in batch]
        losses = model.loss(*batch)
        loss = losses["total"]
        if args.jacobian_penalty > 0:
            jp = jacobian_penalty(model, batch[0][:, : cfg.K])
            loss = loss + args.jacobian_penalty * jp
            if step % 500 == 0:
                writer.add_scalar("train/jacobian_penalty", jp.item(), step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % 500 == 0:
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", v.item(), step)
        if step in ckpt_steps:
            for k, v in run_val().items():
                writer.add_scalar(f"val/{k}", v, step)
            model.save(
                os.path.join(out, f"ckpt_{step:07d}.pt"), normalizer,
                extra=dict(step=step, args=vars(args)),
            )
            print(f"step {step}/{total_steps} ckpt saved "
                  f"(flow {losses['flow'].item():.4f})")

    model.save(os.path.join(out, "final_ckpt.pt"), normalizer,
               extra=dict(step=step, args=vars(args), val=run_val()))
    writer.close()
    print(f"done -> {out}/final_ckpt.pt")


if __name__ == "__main__":
    main()
