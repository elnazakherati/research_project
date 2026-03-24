import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_nn_sim.models import ResMLP
from particle_nn_sim.one_particle_data import collect_episodes_1p
from particle_nn_sim.one_particle_rollout import (
    animate_overlay_gt_perturbed_1p,
    animate_side_by_side_1p,
    nn_rollout_residual_1p,
    save_animation_mp4,
)
from particle_nn_sim.simulator import ParticleSim2D


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def resolve_device(flag):
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a one-particle checkpoint on regenerated train/test episodes (no retraining)."
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_1p_resmlp.pt")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--num-episodes", type=int, default=20, help="How many episodes from split to evaluate")
    p.add_argument("--start-idx", type=int, default=0, help="Start offset inside the chosen split")
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=-1,
        help="Override rollout steps. Use -1 to use checkpoint rollout_steps if available, else steps.",
    )
    p.add_argument("--divergence-threshold", type=float, default=0.3)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-render", type=str2bool, default=False)
    p.add_argument("--save-overlay", type=str2bool, default=True)
    p.add_argument("--save-side-by-side", type=str2bool, default=False)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="checkpoints/eval_trainset_1p")
    return p.parse_args()


def get_wall_mode(cfg, meta):
    return str(cfg.get("wall_collision_mode", meta.get("wall_mode", "clamp")))


def main():
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be >= 1")
    if args.start_idx < 0:
        raise ValueError("--start-idx must be >= 0")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.divergence_threshold < 0.0:
        raise ValueError("--divergence-threshold must be >= 0")

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    meta = ckpt["meta"]

    model = ResMLP(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)

    W = float(meta["W"])
    H = float(meta["H"])
    dt = float(meta["dt"])
    radius = float(np.asarray(meta["radii"], dtype=np.float32)[0])
    mass = float(np.asarray(meta["masses"], dtype=np.float32)[0])
    restitution = float(meta["restitution"])
    wall_mode = get_wall_mode(cfg, meta)

    split_indices = ckpt.get("split_indices", None)
    episode_init = ckpt.get("episode_init", None)

    if split_indices is not None and episode_init is not None:
        pos0_all = np.asarray(episode_init["pos0"], dtype=np.float32)  # (E,1,2)
        vel0_all = np.asarray(episode_init["vel0"], dtype=np.float32)  # (E,1,2)
        train_eps = np.asarray(split_indices["train_eps"], dtype=np.int64)
        test_eps = np.asarray(split_indices["test_eps"], dtype=np.int64)
    else:
        # Fallback path for older checkpoints: regenerate dataset from config.
        sim_gen = ParticleSim2D(
            W=W,
            H=H,
            radii=np.asarray(meta["radii"], dtype=float),
            masses=np.asarray(meta["masses"], dtype=float),
            restitution=restitution,
            seed=int(cfg["seed"]),
            wall_mode=wall_mode,
        )
        pos_all, vel_all, coll_all, _ = collect_episodes_1p(
            sim_gen,
            E=int(cfg["episodes"]),
            steps=int(cfg["steps"]),
            dt=float(cfg["dt"]),
            speed_max=float(cfg.get("speed_max", 0.7)),
            seed=int(cfg["seed"]),
            stratified_init=bool(cfg.get("stratified_init", False)),
            pos_grid_n=int(cfg.get("pos_grid_n", 4)),
            angle_bins=int(cfg.get("angle_bins", 8)),
            episodes_per_bucket=cfg.get("episodes_per_bucket", None),
            fixed_speed=cfg.get("fixed_speed", None),
        )
        pos0_all = pos_all[:, 0].astype(np.float32)
        vel0_all = vel_all[:, 0].astype(np.float32)
        E = pos_all.shape[0]
        n_train = int(float(cfg.get("train_split", 0.8)) * E)
        train_eps = np.arange(n_train)
        test_eps = np.arange(n_train, E)

    split_eps = train_eps if args.split == "train" else test_eps
    if len(split_eps) == 0:
        raise RuntimeError(f"No episodes available in split '{args.split}'.")

    end_idx = min(args.start_idx + args.num_episodes, len(split_eps))
    eval_eps = split_eps[args.start_idx:end_idx]
    if len(eval_eps) == 0:
        raise RuntimeError(
            f"Requested empty episode range: split={args.split}, start={args.start_idx}, num={args.num_episodes}"
        )

    if args.rollout_steps >= 0:
        rollout_steps = int(args.rollout_steps)
    else:
        rollout_steps = int(cfg.get("rollout_steps", cfg["steps"]))

    rows = []
    for j, e in enumerate(eval_eps):
        pos0 = pos0_all[e].astype(np.float32)
        vel0 = vel0_all[e].astype(np.float32)

        sim_true = ParticleSim2D(
            W=W,
            H=H,
            radii=np.asarray(meta["radii"], dtype=float),
            masses=np.asarray(meta["masses"], dtype=float),
            restitution=restitution,
            seed=10_000 + int(e),
            wall_mode=wall_mode,
        )
        sim_true.reset(pos0, vel0)
        pos_true, _ = sim_true.rollout(dt=dt, steps=rollout_steps)
        pos_true = pos_true.astype(np.float32)

        pos_pred, _ = nn_rollout_residual_1p(
            model=model,
            pos0=pos0,
            vel0=vel0,
            radius=radius,
            mass=mass,
            steps=rollout_steps,
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
        divergence_step = int(div_idx[0]) if diverged else int(rollout_steps)

        row = {
            "episode_idx_global": int(e),
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

            if args.save_side_by_side:
                side = animate_side_by_side_1p(
                    pos_true=pos_true_v,
                    pos_pred=pos_pred_v,
                    radius=radius,
                    W=W,
                    H=H,
                    dt=dt_v,
                    title_left=f"GT ({args.split}) ep={int(e)}",
                    title_right=f"NN ({args.split}) ep={int(e)}",
                )
                side_path = out_dir / f"{args.split}_ep_{int(e):05d}_gt_vs_pred_1p.mp4"
                save_animation_mp4(side, str(side_path), fps=args.fps)
                row["video"] = side_path.name

            if args.save_overlay:
                overlay = animate_overlay_gt_perturbed_1p(
                    pos_ref=pos_true_v,
                    pos_pert=pos_pred_v,
                    radius=radius,
                    W=W,
                    H=H,
                    dt=dt_v,
                    title=f"GT vs NN ({args.split}) ep={int(e)}",
                    label_ref="GT",
                    label_pert="NN rollout",
                )
                overlay_path = out_dir / f"{args.split}_ep_{int(e):05d}_overlay_1p.mp4"
                save_animation_mp4(overlay, str(overlay_path), fps=args.fps)
                row["overlay_video"] = overlay_path.name

            plot_path = out_dir / f"{args.split}_ep_{int(e):05d}_error_vs_step.png"
            plt.figure(figsize=(7, 4))
            plt.plot(np.arange(len(pos_err)), pos_err, lw=2, color="tab:red")
            plt.xlabel("step")
            plt.ylabel("position error ||x_pred - x_true||")
            plt.title(f"{args.split} episode {int(e)}: Error vs Time Step")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            row["error_plot"] = plot_path.name

        print(
            f"[{j+1}/{len(eval_eps)}] ep={int(e)} "
            f"mean_err={row['mean_err']:.6f} max_err={row['max_err']:.6f} "
            f"final_err={row['final_err']:.6f} ttf={row['divergence_step']}"
        )
        rows.append(row)

    ttf = np.asarray([r["divergence_step"] for r in rows], dtype=np.float32)
    final_err = np.asarray([r["final_err"] for r in rows], dtype=np.float32)
    diverged = np.asarray([1.0 if r["diverged"] else 0.0 for r in rows], dtype=np.float32)

    summary = {
        "ckpt": str(ckpt_path),
        "split": args.split,
        "num_episodes_requested": int(args.num_episodes),
        "num_episodes_evaluated": int(len(rows)),
        "start_idx": int(args.start_idx),
        "rollout_steps": int(rollout_steps),
        "divergence_threshold": float(args.divergence_threshold),
        "divergence_rate": float(np.mean(diverged)),
        "ttf_mean": float(np.mean(ttf)),
        "ttf_median": float(np.median(ttf)),
        "ttf_p10": float(np.quantile(ttf, 0.10)),
        "ttf_p90": float(np.quantile(ttf, 0.90)),
        "final_err_mean": float(np.mean(final_err)),
        "final_err_median": float(np.median(final_err)),
        "final_err_p90": float(np.quantile(final_err, 0.90)),
        "no_render": bool(args.no_render),
        "wall_mode": wall_mode,
        "per_episode": rows,
    }
    summary_path = out_dir / f"eval_{args.split}_summary.json"
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
