import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_nn_sim.one_particle_data import sample_init_1p
from particle_nn_sim.one_particle_rollout import animate_overlay_gt_perturbed_1p, save_animation_mp4
from particle_nn_sim.one_particle_rollout import animate_side_by_side_1p
from particle_nn_sim.simulator import ParticleSim2D


def parse_args():
    p = argparse.ArgumentParser(
        description="Run GT vs perturbed-GT one-particle rollout and save overlay MP4 + error plot."
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--sigma-pos", type=float, default=0.005, help="Std-dev Gaussian noise for initial position.")
    p.add_argument("--sigma-vel", type=float, default=0.0, help="Std-dev Gaussian noise for initial velocity.")
    p.add_argument("--radius", type=float, default=0.0)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturb-seed", type=int, default=123)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--frame-stride", type=int, default=1, help="Use every k-th frame when rendering videos.")
    p.add_argument(
        "--view-mode",
        type=str,
        default="side_by_side",
        choices=["side_by_side", "overlay"],
        help="Render style for GT vs perturbed visualization.",
    )
    p.add_argument("--out-dir", type=str, default="checkpoints/one_particle_gt_perturbation")
    return p.parse_args()


def main():
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W = 1.0
    H = 1.0
    radius_eff = args.radius if args.radius > 0.0 else 1e-6

    pos0, vel0 = sample_init_1p(W, H, radius_eff, speed_max=args.speed_max, seed=args.seed)
    pos0 = pos0.astype(np.float32)
    vel0 = vel0.astype(np.float32)

    rng = np.random.default_rng(args.perturb_seed)
    pos0_pert = pos0.copy()
    vel0_pert = vel0.copy()
    if float(args.sigma_pos) > 0.0:
        pos0_pert[0] += rng.normal(0.0, float(args.sigma_pos), size=(2,)).astype(np.float32)
    if float(args.sigma_vel) > 0.0:
        vel0_pert[0] += rng.normal(0.0, float(args.sigma_vel), size=(2,)).astype(np.float32)

    # Keep perturbed particle inside the box.
    pos0_pert[0, 0] = np.clip(pos0_pert[0, 0], radius_eff, W - radius_eff)
    pos0_pert[0, 1] = np.clip(pos0_pert[0, 1], radius_eff, H - radius_eff)

    sim_ref = ParticleSim2D(
        W=W,
        H=H,
        radii=[radius_eff],
        masses=[args.mass],
        restitution=1.0,
        seed=args.seed,
    )
    sim_pert = ParticleSim2D(
        W=W,
        H=H,
        radii=[radius_eff],
        masses=[args.mass],
        restitution=1.0,
        seed=args.seed,
    )

    sim_ref.reset(pos0, vel0)
    sim_pert.reset(pos0_pert, vel0_pert)

    pos_ref, vel_ref = sim_ref.rollout(dt=args.dt, steps=args.steps)
    pos_pert, vel_pert = sim_pert.rollout(dt=args.dt, steps=args.steps)
    pos_ref = pos_ref.astype(np.float32)
    pos_pert = pos_pert.astype(np.float32)

    # Error curve
    pos_err = np.linalg.norm(pos_ref[:, 0, :] - pos_pert[:, 0, :], axis=1)
    t = np.arange(len(pos_err), dtype=np.float32) * float(args.dt)

    err_plot = out_dir / "gt_vs_gt_perturbed_error_vs_time_1p.png"
    plt.figure(figsize=(7, 4))
    plt.plot(t, pos_err, lw=2, color="tab:red")
    plt.xlabel("time (s)")
    plt.ylabel("position error ||x_pert - x_ref||")
    plt.title("GT Sensitivity to Initial Gaussian Perturbation (1P)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(err_plot, dpi=150)
    plt.close()

    pos_ref_v = pos_ref[:: args.frame_stride]
    pos_pert_v = pos_pert[:: args.frame_stride]
    dt_v = float(args.dt) * float(args.frame_stride)

    if args.view_mode == "overlay":
        anim = animate_overlay_gt_perturbed_1p(
            pos_ref=pos_ref_v,
            pos_pert=pos_pert_v,
            radius=radius_eff,
            W=W,
            H=H,
            dt=dt_v,
            title="Ground Truth vs Perturbed Ground Truth (1P)",
        )
        mp4_path = out_dir / "gt_vs_gt_perturbed_overlay_1p.mp4"
    else:
        anim = animate_side_by_side_1p(
            pos_true=pos_ref_v,
            pos_pred=pos_pert_v,
            radius=radius_eff,
            W=W,
            H=H,
            dt=dt_v,
            title_left="Ground Truth",
            title_right="Perturbed Ground Truth",
        )
        mp4_path = out_dir / "gt_vs_gt_perturbed_side_by_side_1p.mp4"
    save_animation_mp4(anim, str(mp4_path), fps=args.fps)

    summary = {
        "mean_position_error": float(np.mean(pos_err)),
        "max_position_error": float(np.max(pos_err)),
        "final_position_error": float(pos_err[-1]),
        "sigma_pos": float(args.sigma_pos),
        "sigma_vel": float(args.sigma_vel),
        "steps": int(args.steps),
        "dt": float(args.dt),
        "seed": int(args.seed),
        "perturb_seed": int(args.perturb_seed),
        "view_mode": args.view_mode,
        "frame_stride": int(args.frame_stride),
    }
    with open(out_dir / "gt_vs_gt_perturbed_summary_1p.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Run complete.")
    print("Artifacts:")
    print(" -", mp4_path)
    print(" -", err_plot)
    print(" -", out_dir / "gt_vs_gt_perturbed_summary_1p.json")


if __name__ == "__main__":
    main()
