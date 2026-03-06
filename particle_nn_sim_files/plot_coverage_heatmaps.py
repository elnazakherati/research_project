#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from particle_nn_sim.one_particle_data import collect_episodes_1p
from particle_nn_sim.simulator import ParticleSim2D


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate position coverage heatmaps for one-particle data."
    )
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=60)
    p.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints/data_coverage_e1000_s500",
        help="Directory where PNGs are saved.",
    )
    return p.parse_args()


def save_heatmap(H, out_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(
        H.T,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(label="count")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep tiny radius epsilon for stable wall-contact handling.
    sim = ParticleSim2D(
        W=1.0,
        H=1.0,
        radii=[1e-6],
        masses=[1.0],
        restitution=1.0,
        seed=args.seed,
    )

    pos_all, _, coll_all, _ = collect_episodes_1p(
        sim,
        E=int(args.episodes),
        steps=int(args.steps),
        dt=float(args.dt),
        speed_max=float(args.speed_max),
        seed=int(args.seed),
    )

    # All visited positions over all episodes/time.
    xy_all = pos_all[:, :, 0, :].reshape(-1, 2)
    H_all, _, _ = np.histogram2d(
        xy_all[:, 0],
        xy_all[:, 1],
        bins=int(args.bins),
        range=[[0.0, 1.0], [0.0, 1.0]],
    )

    # Collision positions at frame t+1 where collision flag at t is true.
    xy_col = pos_all[:, 1:, 0, :][coll_all.astype(bool)]
    H_col, _, _ = np.histogram2d(
        xy_col[:, 0],
        xy_col[:, 1],
        bins=int(args.bins),
        range=[[0.0, 1.0], [0.0, 1.0]],
    )

    blank_pct_all = 100.0 * float((H_all == 0).mean())
    blank_pct_col = 100.0 * float((H_col == 0).mean())

    out_all = out_dir / "occupancy_all.png"
    out_col = out_dir / "occupancy_collision.png"

    save_heatmap(H_all, out_all, "occupancy_all")
    save_heatmap(H_col, out_col, "occupancy_collision")

    print("Saved:")
    print(" -", out_all)
    print(" -", out_col)
    print(f"Blank cells (all): {blank_pct_all:.2f}%")
    print(f"Blank cells (collision): {blank_pct_col:.2f}%")


if __name__ == "__main__":
    main()
