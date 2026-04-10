import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_nn_sim.models import ResMLP
from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.one_particle_rollout import (
    animate_overlay_gt_perturbed_1p,
    animate_side_by_side_1p,
    nn_rollout_residual_1p,
    save_animation_mp4,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate one-particle checkpoint on new rollouts only.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_1p_resmlp.pt")
    p.add_argument("--num-rollouts", type=int, default=5, help="How many episodes from split to evaluate")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--start-idx", type=int, default=0, help="Start offset inside chosen split")
    p.add_argument("--rollout-steps", type=int, default=1000)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument(
        "--fixed-speed",
        type=float,
        default=None,
        help="If set, sample all eval initial velocities with this fixed speed magnitude.",
    )
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--fixed-x", type=float, default=None, help="Fixed initial x position for all eval rollouts.")
    p.add_argument("--fixed-y", type=float, default=None, help="Fixed initial y position for all eval rollouts.")
    p.add_argument("--fixed-vx", type=float, default=None, help="Fixed initial vx for all eval rollouts.")
    p.add_argument("--fixed-vy", type=float, default=None, help="Fixed initial vy for all eval rollouts.")
    p.add_argument("--frame-stride", type=int, default=1, help="Use every k-th frame when rendering videos.")
    p.add_argument("--save-overlay", type=str2bool, default=True, help="Also save single-axis GT/NN overlay video.")
    p.add_argument("--no-render", type=str2bool, default=False, help="Skip MP4/PNG rendering and compute metrics only.")
    p.add_argument(
        "--divergence-threshold",
        type=float,
        default=0.3,
        help="First step where position error exceeds this value is divergence step (TTF).",
    )
    p.add_argument("--out-dir", type=str, default="checkpoints/one_particle_eval_only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--wall-collision-mode",
        type=str,
        default="auto",
        choices=["auto", "clamp", "exact"],
        help="Ground-truth simulator wall mode. auto uses checkpoint/meta mode when available, else clamp.",
    )
    return p.parse_args()


def resolve_device(flag):
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.divergence_threshold < 0.0:
        raise ValueError("--divergence-threshold must be >= 0")
    if args.fixed_speed is not None and args.fixed_speed < 0.0:
        raise ValueError("--fixed-speed must be >= 0 when provided.")
    fixed_ic_vals = (args.fixed_x, args.fixed_y, args.fixed_vx, args.fixed_vy)
    use_fixed_ic = any(v is not None for v in fixed_ic_vals)
    if use_fixed_ic and not all(v is not None for v in fixed_ic_vals):
        raise ValueError("If any of --fixed-x/--fixed-y/--fixed-vx/--fixed-vy is set, all four must be set.")
    if args.start_idx < 0:
        raise ValueError("--start-idx must be >= 0")

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    # weights_only=False is required because checkpoint stores numpy arrays/config dicts too.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_kwargs = ckpt["model_kwargs"]
    model = ResMLP(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    meta = ckpt["meta"]
    split_indices = ckpt.get("split_indices", None)
    episode_init = ckpt.get("episode_init", None)
    if split_indices is None or episode_init is None:
        raise RuntimeError(
            "Checkpoint is missing split_indices/episode_init. "
            "Use a checkpoint produced by run_one_particle_pipeline.py with split metadata."
        )

    W = float(meta["W"])
    H = float(meta["H"])
    dt = float(meta["dt"])
    radius = float(np.asarray(meta["radii"], dtype=np.float32)[0])
    mass = float(np.asarray(meta["masses"], dtype=np.float32)[0])
    restitution = float(meta["restitution"])
    if use_fixed_ic:
        fx = float(args.fixed_x)
        fy = float(args.fixed_y)
        if not (radius <= fx <= W - radius):
            raise ValueError(f"--fixed-x={fx} outside valid range [{radius}, {W-radius}]")
        if not (radius <= fy <= H - radius):
            raise ValueError(f"--fixed-y={fy} outside valid range [{radius}, {H-radius}]")
    if args.wall_collision_mode == "auto":
        wall_mode = str(meta.get("wall_mode", "clamp"))
    else:
        wall_mode = args.wall_collision_mode

    train_eps = np.asarray(split_indices["train_eps"], dtype=np.int64)
    test_eps = np.asarray(split_indices["test_eps"], dtype=np.int64)
    pos0_all = np.asarray(episode_init["pos0"], dtype=np.float32)
    vel0_all = np.asarray(episode_init["vel0"], dtype=np.float32)
    split_eps = train_eps if args.split == "train" else test_eps
    if len(split_eps) == 0:
        raise RuntimeError(f"No episodes available in split '{args.split}'.")
    end_idx = min(args.start_idx + int(args.num_rollouts), len(split_eps))
    eval_eps = split_eps[args.start_idx:end_idx]
    if len(eval_eps) == 0:
        raise RuntimeError(
            f"Requested empty episode range: split={args.split}, start={args.start_idx}, num={args.num_rollouts}"
        )

    rollout_rows = []

    for i, e in enumerate(eval_eps):
        e = int(e)
        seed_i = 10_000 + e
        pos0 = pos0_all[e].astype(np.float32)
        vel0 = vel0_all[e].astype(np.float32)

        sim_true = ParticleSim2D(
            W=W,
            H=H,
            radii=[radius],
            masses=[mass],
            restitution=restitution,
            seed=seed_i + 1,
            wall_mode=wall_mode,
        )
        sim_true.reset(pos0, vel0)
        pos_true, vel_true = sim_true.rollout(dt=dt, steps=args.rollout_steps)
        pos_true = pos_true.astype(np.float32)
        vel_true = vel_true.astype(np.float32)

        pos_pred, vel_pred = nn_rollout_residual_1p(
            model=model,
            pos0=pos0.astype(np.float32),
            vel0=vel0.astype(np.float32),
            radius=radius,
            mass=mass,
            steps=args.rollout_steps,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            dt=dt,
        )

        pos_err = np.linalg.norm(pos_true[:, 0, :] - pos_pred[:, 0, :], axis=1)
        div_idx = np.where(pos_err > float(args.divergence_threshold))[0]
        diverged = bool(len(div_idx) > 0)
        divergence_step = int(div_idx[0]) if diverged else int(args.rollout_steps)

        row = {
            "rollout_idx": int(i),
            "episode_idx_global": int(e),
            "sample_seed": int(seed_i),
            "mean_err": float(np.mean(pos_err)),
            "max_err": float(np.max(pos_err)),
            "final_err": float(pos_err[-1]),
            "diverged": diverged,
            "divergence_step": int(divergence_step),
        }

        if not args.no_render:
            pos_true_v = pos_true[:: args.frame_stride]
            pos_pred_v = pos_pred[:: args.frame_stride]
            dt_v = dt * float(args.frame_stride)

            anim = animate_side_by_side_1p(
                pos_true=pos_true_v,
                pos_pred=pos_pred_v,
                radius=radius,
                W=W,
                H=H,
                dt=dt_v,
            )
            out_mp4 = out_dir / f"rollout_{i:03d}_gt_vs_pred_1p.mp4"
            save_animation_mp4(anim, str(out_mp4), fps=args.fps)
            row["video"] = out_mp4.name

            overlay_name = ""
            if args.save_overlay:
                overlay_anim = animate_overlay_gt_perturbed_1p(
                    pos_ref=pos_true_v,
                    pos_pert=pos_pred_v,
                    radius=radius,
                    W=W,
                    H=H,
                    dt=dt_v,
                    title="GT vs NN rollout (same init)",
                    label_ref="GT",
                    label_pert="NN rollout",
                )
                overlay_mp4 = out_dir / f"rollout_{i:03d}_gt_vs_pred_overlay_1p.mp4"
                save_animation_mp4(overlay_anim, str(overlay_mp4), fps=args.fps)
                overlay_name = overlay_mp4.name
                row["overlay_video"] = overlay_name

            plot_path = out_dir / f"rollout_{i:03d}_error_vs_step.png"
            plt.figure(figsize=(7, 4))
            plt.plot(np.arange(len(pos_err)), pos_err, lw=2, color="tab:red")
            plt.xlabel("step")
            plt.ylabel("position error ||x_pred - x_true||")
            plt.title(f"Rollout {i:03d} Error vs Time Step")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            row["error_plot"] = plot_path.name

            print(
                f"[{i+1}/{len(eval_eps)}] saved {out_mp4.name} | "
                f"mean_err={row['mean_err']:.6f} "
                f"max_err={row['max_err']:.6f} "
                f"final_err={row['final_err']:.6f} | "
                f"ttf={row['divergence_step']} (thr={args.divergence_threshold}) | "
                f"plot={plot_path.name}"
                + (f" | overlay={overlay_name}" if overlay_name else "")
            )
        else:
            print(
                f"[{i+1}/{len(eval_eps)}] "
                f"mean_err={row['mean_err']:.6f} "
                f"max_err={row['max_err']:.6f} "
                f"final_err={row['final_err']:.6f} | "
                f"ttf={row['divergence_step']} (thr={args.divergence_threshold})"
            )

        rollout_rows.append(row)

    ttf = np.asarray([r["divergence_step"] for r in rollout_rows], dtype=np.float32)
    final_err = np.asarray([r["final_err"] for r in rollout_rows], dtype=np.float32)
    diverged = np.asarray([1.0 if r["diverged"] else 0.0 for r in rollout_rows], dtype=np.float32)

    summary = {
        "ckpt": str(ckpt_path),
        "split": args.split,
        "num_rollouts": int(len(eval_eps)),
        "start_idx": int(args.start_idx),
        "rollout_steps": int(args.rollout_steps),
        "speed_max": float(args.speed_max),
        "fixed_speed": None if args.fixed_speed is None else float(args.fixed_speed),
        "fixed_x": None if args.fixed_x is None else float(args.fixed_x),
        "fixed_y": None if args.fixed_y is None else float(args.fixed_y),
        "fixed_vx": None if args.fixed_vx is None else float(args.fixed_vx),
        "fixed_vy": None if args.fixed_vy is None else float(args.fixed_vy),
        "divergence_threshold": float(args.divergence_threshold),
        "divergence_rate": float(np.mean(diverged)),
        "ttf_mean": float(np.mean(ttf)),
        "ttf_median": float(np.median(ttf)),
        "ttf_p10": float(np.quantile(ttf, 0.10)),
        "ttf_p90": float(np.quantile(ttf, 0.90)),
        "final_err_mean": float(np.mean(final_err)),
        "final_err_median": float(np.median(final_err)),
        "final_err_p90": float(np.quantile(final_err, 0.90)),
        "seed": int(args.seed),
        "no_render": bool(args.no_render),
        "per_rollout": rollout_rows,
    }
    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Output dir:", out_dir)
    print("Summary:", summary_path)
    print(
        "Aggregate | "
        f"ttf_median={summary['ttf_median']:.2f} "
        f"ttf_p10={summary['ttf_p10']:.2f} "
        f"divergence_rate={summary['divergence_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
