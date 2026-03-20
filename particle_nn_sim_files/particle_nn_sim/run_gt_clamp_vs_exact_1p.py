import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_nn_sim.one_particle_data import sample_init_1p
from particle_nn_sim.one_particle_rollout import animate_side_by_side_1p, save_animation_mp4
from particle_nn_sim.simulator import ParticleSim2D


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare GT clamp-vs-exact wall collision modes for one particle."
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--radius", type=float, default=0.0)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--frame-stride", type=int, default=1, help="Use every k-th frame in MP4.")
    p.add_argument("--fixed-x", type=float, default=None, help="Optional fixed initial x.")
    p.add_argument("--fixed-y", type=float, default=None, help="Optional fixed initial y.")
    p.add_argument("--fixed-vx", type=float, default=None, help="Optional fixed initial vx.")
    p.add_argument("--fixed-vy", type=float, default=None, help="Optional fixed initial vy.")
    p.add_argument("--out-dir", type=str, default="checkpoints/gt_clamp_vs_exact_1p")
    return p.parse_args()


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if (args.fixed_x is None) != (args.fixed_y is None):
        raise ValueError("Set both --fixed-x and --fixed-y, or neither.")
    if (args.fixed_vx is None) != (args.fixed_vy is None):
        raise ValueError("Set both --fixed-vx and --fixed-vy, or neither.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W = 1.0
    H = 1.0
    radius_eff = args.radius if args.radius > 0.0 else 1e-6

    # Shared initial condition for both modes.
    if args.fixed_x is not None and args.fixed_vx is not None:
        pos0 = np.array([[float(args.fixed_x), float(args.fixed_y)]], dtype=np.float32)
        vel0 = np.array([[float(args.fixed_vx), float(args.fixed_vy)]], dtype=np.float32)
    elif args.fixed_x is not None:
        pos0 = np.array([[float(args.fixed_x), float(args.fixed_y)]], dtype=np.float32)
        _, vel0 = sample_init_1p(W, H, radius_eff, speed_max=args.speed_max, seed=args.seed)
    elif args.fixed_vx is not None:
        pos0, _ = sample_init_1p(W, H, radius_eff, speed_max=args.speed_max, seed=args.seed)
        vel0 = np.array([[float(args.fixed_vx), float(args.fixed_vy)]], dtype=np.float32)
    else:
        pos0, vel0 = sample_init_1p(W, H, radius_eff, speed_max=args.speed_max, seed=args.seed)
    pos0 = pos0.astype(np.float32)
    vel0 = vel0.astype(np.float32)
    pos0[0, 0] = np.clip(pos0[0, 0], radius_eff, W - radius_eff)
    pos0[0, 1] = np.clip(pos0[0, 1], radius_eff, H - radius_eff)

    sim_clamp = ParticleSim2D(
        W=W,
        H=H,
        radii=[radius_eff],
        masses=[args.mass],
        restitution=1.0,
        seed=args.seed,
        wall_mode="clamp",
    )
    sim_exact = ParticleSim2D(
        W=W,
        H=H,
        radii=[radius_eff],
        masses=[args.mass],
        restitution=1.0,
        seed=args.seed,
        wall_mode="exact",
    )

    sim_clamp.reset(pos0, vel0)
    sim_exact.reset(pos0, vel0)

    pos_clamp, vel_clamp = sim_clamp.rollout(dt=args.dt, steps=args.steps)
    pos_exact, vel_exact = sim_exact.rollout(dt=args.dt, steps=args.steps)
    pos_clamp = pos_clamp.astype(np.float32)
    pos_exact = pos_exact.astype(np.float32)

    # Error curve between modes.
    pos_err = np.linalg.norm(pos_clamp[:, 0, :] - pos_exact[:, 0, :], axis=1)
    err_plot = out_dir / "clamp_vs_exact_error_vs_time_1p.png"
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(pos_err)), pos_err, lw=2, color="tab:red")
    plt.xlabel("step")
    plt.ylabel("position error ||x_clamp - x_exact||")
    plt.title("Clamp vs Exact GT Trajectory Difference (1P)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(err_plot, dpi=150)
    plt.close()

    pos_clamp_v = pos_clamp[:: args.frame_stride]
    pos_exact_v = pos_exact[:: args.frame_stride]
    dt_v = float(args.dt) * float(args.frame_stride)

    anim = animate_side_by_side_1p(
        pos_true=pos_clamp_v,
        pos_pred=pos_exact_v,
        radius=radius_eff,
        W=W,
        H=H,
        dt=dt_v,
        title_left="GT clamp mode",
        title_right="GT exact mode",
    )
    mp4_path = out_dir / "gt_clamp_vs_exact_side_by_side_1p.mp4"
    save_animation_mp4(anim, str(mp4_path), fps=args.fps)

    summary = {
        "steps": int(args.steps),
        "dt": float(args.dt),
        "seed": int(args.seed),
        "radius": float(radius_eff),
        "mass": float(args.mass),
        "speed_max": float(args.speed_max),
        "frame_stride": int(args.frame_stride),
        "init_pos": pos0[0].tolist(),
        "init_vel": vel0[0].tolist(),
        "mean_position_error": float(np.mean(pos_err)),
        "max_position_error": float(np.max(pos_err)),
        "final_position_error": float(pos_err[-1]),
        "video": mp4_path.name,
        "error_plot": err_plot.name,
    }
    with open(out_dir / "clamp_vs_exact_summary_1p.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Run complete.")
    print("Artifacts:")
    print(" -", mp4_path)
    print(" -", err_plot)
    print(" -", out_dir / "clamp_vs_exact_summary_1p.json")


if __name__ == "__main__":
    main()
