#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_nn_sim.models import ResMLP
from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.one_particle_data import collect_episodes_1p, episodes_to_XY_residual_1p
from particle_nn_sim.train import apply_standardizer


def resolve_device(flag: str) -> str:
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Residual scatter: pred vs GT (split collision/non-collision)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_1p_resmlp.pt")
    p.add_argument("--episodes", type=int, default=300, help="Fresh eval episodes to sample")
    p.add_argument("--steps", type=int, default=None, help="Rollout steps per episode (default: ckpt config)")
    p.add_argument("--max-points", type=int, default=120000, help="Max points to plot after subsampling")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", type=str, default="", help="Output PNG path")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = ResMLP(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    meta = ckpt["meta"]
    cfg = ckpt.get("config", {})

    steps = int(args.steps if args.steps is not None else cfg.get("steps", 500))
    speed_max = float(cfg.get("speed_max", 0.7))

    sim = ParticleSim2D(
        W=float(meta["W"]),
        H=float(meta["H"]),
        radii=np.asarray(meta["radii"], dtype=float),
        masses=np.asarray(meta["masses"], dtype=float),
        restitution=float(meta["restitution"]),
        seed=args.seed + 1,
    )

    pos_all, vel_all, coll_all, _ = collect_episodes_1p(
        sim=sim,
        E=int(args.episodes),
        steps=steps,
        dt=float(meta["dt"]),
        speed_max=speed_max,
        seed=args.seed + 2,
    )

    idx = np.arange(pos_all.shape[0])
    X, Y_gt, C = episodes_to_XY_residual_1p(pos_all, vel_all, coll_all, meta, idx)
    C = C.astype(bool)

    Xn = apply_standardizer(X, x_mean, x_std)
    with torch.no_grad():
        Yn_pred = model(torch.from_numpy(Xn).float().to(device)).cpu().numpy()
    Y_pred = Yn_pred * y_std + y_mean  # de-normalize

    n = len(Y_gt)
    if n > args.max_points:
        keep = rng.choice(n, size=args.max_points, replace=False)
        Y_gt = Y_gt[keep]
        Y_pred = Y_pred[keep]
        C = C[keep]

    names = [r"$\\Delta x$", r"$\\Delta y$", r"$\\Delta v_x$", r"$\\Delta v_y$"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for j, ax in enumerate(axes.flat):
        g = Y_gt[:, j]
        p = Y_pred[:, j]

        ax.scatter(g[~C], p[~C], s=3, alpha=0.15, label="non-collision", color="#1f77b4")
        ax.scatter(g[C], p[C], s=4, alpha=0.25, label="collision", color="#d62728")

        lo = np.percentile(np.r_[g, p], 0.5)
        hi = np.percentile(np.r_[g, p], 99.5)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)

        ax.set_xlabel(f"{names[j]} GT")
        ax.set_ylabel(f"{names[j]} Pred")
        ax.set_title(names[j])
        ax.grid(alpha=0.25)

    axes[0, 0].legend(frameon=False)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else ckpt_path.parent / "residual_scatter_split.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
