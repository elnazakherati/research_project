#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from particle_nn_sim.models import ResMLP
from particle_nn_sim.one_particle_rollout import nn_rollout_residual_1p
from particle_nn_sim.simulator import ParticleSim2D


def resolve_device(flag: str) -> str:
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def eval_final_pos_err(
    model,
    meta,
    x_mean,
    x_std,
    y_mean,
    y_std,
    device,
    pos0,
    vel0,
    steps,
    seed,
):
    sim = ParticleSim2D(
        W=float(meta["W"]),
        H=float(meta["H"]),
        radii=np.asarray(meta["radii"], dtype=float),
        masses=np.asarray(meta["masses"], dtype=float),
        restitution=float(meta["restitution"]),
        seed=int(seed),
    )

    pos0_arr = np.asarray(pos0, dtype=np.float32).reshape(1, 2)
    vel0_arr = np.asarray(vel0, dtype=np.float32).reshape(1, 2)

    sim.reset(pos0_arr, vel0_arr)
    pos_true, _ = sim.rollout(dt=float(meta["dt"]), steps=int(steps))

    pos_pred, _ = nn_rollout_residual_1p(
        model=model,
        pos0=pos0_arr,
        vel0=vel0_arr,
        radius=float(np.asarray(meta["radii"], dtype=np.float32)[0]),
        mass=float(np.asarray(meta["masses"], dtype=np.float32)[0]),
        steps=int(steps),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        dt=float(meta["dt"]),
    )

    n = min(len(pos_true), len(pos_pred))
    if n == 0:
        return np.nan
    final_err = np.linalg.norm(pos_true[n - 1, 0, :] - pos_pred[n - 1, 0, :])
    return float(final_err)


def parse_args():
    p = argparse.ArgumentParser(description="Scan ICs and plot 500-step final position-error heatmaps")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_1p_resmlp.pt")
    p.add_argument("--out-dir", type=str, required=True, help="Directory to save heatmaps")
    p.add_argument("--rollout-steps", type=int, default=500, help="Rollout steps for final error")
    p.add_argument("--grid-n", type=int, default=25, help="Position grid resolution")
    p.add_argument("--n-angles", type=int, default=36, help="Velocity-angle bins")
    p.add_argument("--n-speeds", type=int, default=20, help="Velocity-speed bins")
    p.add_argument("--fixed-speed", type=float, default=0.5, help="Fixed speed for position scan")
    p.add_argument("--fixed-angle", type=float, default=0.8, help="Fixed angle (rad) for position scan")
    p.add_argument("--fixed-x", type=float, default=0.5, help="Fixed x for velocity scan")
    p.add_argument("--fixed-y", type=float, default=0.5, help="Fixed y for velocity scan")
    p.add_argument("--speed-max", type=float, default=-1.0, help="Max speed for velocity scan; -1 uses ckpt config speed_max or 0.7")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ResMLP(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = ckpt["meta"]
    cfg = ckpt.get("config", {})

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)

    W = float(meta["W"])
    H = float(meta["H"])
    radius = float(np.asarray(meta["radii"], dtype=np.float32)[0])

    # 1) Fix velocity, sweep initial positions
    vel = np.array(
        [args.fixed_speed * np.cos(args.fixed_angle), args.fixed_speed * np.sin(args.fixed_angle)],
        dtype=np.float32,
    )

    xs = np.linspace(radius, W - radius, int(args.grid_n), dtype=np.float32)
    ys = np.linspace(radius, H - radius, int(args.grid_n), dtype=np.float32)
    err_pos = np.zeros((len(ys), len(xs)), dtype=np.float32)

    ctr = 0
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            err_pos[iy, ix] = eval_final_pos_err(
                model=model,
                meta=meta,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                device=device,
                pos0=np.array([x0, y0], dtype=np.float32),
                vel0=vel,
                steps=int(args.rollout_steps),
                seed=args.seed + ctr,
            )
            ctr += 1

    plt.figure(figsize=(6, 5))
    im1 = plt.imshow(
        err_pos,
        origin="lower",
        extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
        aspect="auto",
    )
    plt.colorbar(im1, label="final position error")
    plt.xlabel("x0")
    plt.ylabel("y0")
    plt.title("Final error over initial position (fixed velocity)")
    plt.tight_layout()
    out1 = out_dir / "heatmap_pos_fixed_vel.png"
    plt.savefig(out1, dpi=180)
    plt.close()

    # 2) Fix initial position, sweep velocity angle/speed
    speed_max = float(args.speed_max)
    if speed_max < 0.0:
        speed_max = float(cfg.get("speed_max", 0.7))

    fx = float(np.clip(args.fixed_x, radius, W - radius))
    fy = float(np.clip(args.fixed_y, radius, H - radius))
    pos_fixed = np.array([fx, fy], dtype=np.float32)

    angles = np.linspace(0.0, 2.0 * np.pi, int(args.n_angles), endpoint=False, dtype=np.float32)
    speeds = np.linspace(0.0, speed_max, int(args.n_speeds), dtype=np.float32)
    err_vel = np.zeros((len(speeds), len(angles)), dtype=np.float32)

    ctr = 1000000
    for ispd, s in enumerate(speeds):
        for ia, a in enumerate(angles):
            vel0 = np.array([s * np.cos(a), s * np.sin(a)], dtype=np.float32)
            err_vel[ispd, ia] = eval_final_pos_err(
                model=model,
                meta=meta,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                device=device,
                pos0=pos_fixed,
                vel0=vel0,
                steps=int(args.rollout_steps),
                seed=args.seed + ctr,
            )
            ctr += 1

    plt.figure(figsize=(7, 5))
    im2 = plt.imshow(
        err_vel,
        origin="lower",
        extent=[0.0, 2.0 * np.pi, float(speeds[0]), float(speeds[-1])],
        aspect="auto",
    )
    plt.colorbar(im2, label="final position error")
    plt.xlabel("velocity angle (rad)")
    plt.ylabel("speed")
    plt.title("Final error over velocity (fixed position)")
    plt.tight_layout()
    out2 = out_dir / "heatmap_vel_fixed_pos.png"
    plt.savefig(out2, dpi=180)
    plt.close()

    print("Saved:")
    print(" -", out1)
    print(" -", out2)


if __name__ == "__main__":
    main()
