import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_nn_sim.one_particle_rollout import (
    animate_overlay_gt_perturbed_1p,
    animate_side_by_side_1p,
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


class HistoryMLP(nn.Module):
    def __init__(self, in_dim, width=1024, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def rollout_history_model(model, pos_true, history_len, device):
    pos_true = np.asarray(pos_true, dtype=np.float32)
    T = pos_true.shape[0]
    H = int(history_len)
    if T <= H:
        raise ValueError(f"T must be > history_len, got T={T}, H={H}")

    pos_pred = np.zeros_like(pos_true, dtype=np.float32)
    pos_pred[:H, 0, :] = pos_true[:H, 0, :]
    model.eval()

    for t in range(H, T):
        hist = pos_pred[t - H : t, 0, :].reshape(1, -1).astype(np.float32)
        x = torch.from_numpy(hist).to(device)
        y = model(x).cpu().numpy()[0].astype(np.float32)
        pos_pred[t, 0, :] = y
        if not np.isfinite(y).all():
            return pos_pred[: t + 1]
    return pos_pred


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate HistoryMLP checkpoint on train/val/test split episodes."
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_1p_history_mlp.pt")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--num-episodes", type=int, default=2)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=-1, help="Use -1 to read from ckpt config.")
    p.add_argument("--divergence-threshold", type=float, default=0.3)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-render", type=str2bool, default=False)
    p.add_argument("--save-overlay", type=str2bool, default=True)
    p.add_argument("--save-side-by-side", type=str2bool, default=False)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="checkpoints/eval_history_trainset_1p")
    return p.parse_args()


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
    if ckpt.get("model_name", "") != "HistoryMLP":
        raise RuntimeError(
            f"Checkpoint model_name is {ckpt.get('model_name')!r}, expected 'HistoryMLP'."
        )

    cfg = ckpt["config"]
    meta = ckpt["meta"]
    model = HistoryMLP(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    H = int(cfg.get("history_len", 10))
    W = float(meta["W"])
    Hbox = float(meta["H"])
    dt = float(meta["dt"])
    radius = float(np.asarray(meta["radii"], dtype=np.float32)[0])
    mass = float(np.asarray(meta["masses"], dtype=np.float32)[0])
    restitution = float(meta["restitution"])
    wall_mode = str(cfg.get("wall_collision_mode", meta.get("wall_mode", "clamp")))

    split_indices = ckpt.get("split_indices", None)
    episode_init = ckpt.get("episode_init", None)
    if split_indices is None or episode_init is None:
        raise RuntimeError("Checkpoint missing split_indices/episode_init; cannot run split eval safely.")

    train_eps = np.asarray(split_indices["train_eps"], dtype=np.int64)
    val_eps = np.asarray(split_indices.get("val_eps", []), dtype=np.int64)
    test_eps = np.asarray(split_indices["test_eps"], dtype=np.int64)
    pos0_all = np.asarray(episode_init["pos0"], dtype=np.float32)
    vel0_all = np.asarray(episode_init["vel0"], dtype=np.float32)

    split_map = {"train": train_eps, "val": val_eps, "test": test_eps}
    split_eps = split_map[args.split]
    if len(split_eps) == 0:
        raise RuntimeError(f"No episodes available in split '{args.split}'.")

    end_idx = min(args.start_idx + args.num_episodes, len(split_eps))
    eval_eps = split_eps[args.start_idx:end_idx]
    if len(eval_eps) == 0:
        raise RuntimeError(
            f"Requested empty episode range: split={args.split}, start={args.start_idx}, num={args.num_episodes}"
        )

    rollout_steps = int(cfg.get("rollout_steps", cfg.get("steps", 700))) if args.rollout_steps < 0 else int(args.rollout_steps)

    rows = []
    for e in eval_eps:
        e = int(e)
        pos0 = pos0_all[e].astype(np.float32)
        vel0 = vel0_all[e].astype(np.float32)
        sim_true = ParticleSim2D(
            W=W,
            H=Hbox,
            radii=np.asarray(meta["radii"], dtype=float),
            masses=np.asarray(meta["masses"], dtype=float),
            restitution=restitution,
            seed=10_000 + e,
            wall_mode=wall_mode,
        )
        sim_true.reset(pos0, vel0)
        pos_true, _ = sim_true.rollout(dt=dt, steps=rollout_steps)
        pos_true = pos_true.astype(np.float32)

        pos_pred = rollout_history_model(model, pos_true=pos_true, history_len=H, device=device)
        n_frames = min(len(pos_true), len(pos_pred))
        pos_true = pos_true[:n_frames]
        pos_pred = pos_pred[:n_frames]

        pos_err = np.linalg.norm(pos_true[:, 0, :] - pos_pred[:, 0, :], axis=1)
        div_idx = np.where(pos_err > float(args.divergence_threshold))[0]
        diverged = bool(len(div_idx) > 0)
        divergence_step = int(div_idx[0]) if diverged else int(n_frames - 1)

        row = {
            "episode_idx_global": e,
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
                    H=Hbox,
                    dt=dt_v,
                    title_left=f"GT ({args.split}) ep={e}",
                    title_right=f"NN ({args.split}) ep={e}",
                )
                side_path = out_dir / f"{args.split}_ep_{e:05d}_gt_vs_pred_1p.mp4"
                save_animation_mp4(side, str(side_path), fps=args.fps)
                row["video_side_by_side"] = side_path.name

            if args.save_overlay:
                overlay = animate_overlay_gt_perturbed_1p(
                    pos_ref=pos_true_v,
                    pos_pert=pos_pred_v,
                    radius=radius,
                    W=W,
                    H=Hbox,
                    dt=dt_v,
                    title=f"GT vs NN rollout ({args.split}) ep={e}",
                    label_ref="GT",
                    label_pert="NN rollout",
                )
                overlay_path = out_dir / f"{args.split}_ep_{e:05d}_gt_vs_pred_overlay_1p.mp4"
                save_animation_mp4(overlay, str(overlay_path), fps=args.fps)
                row["video_overlay"] = overlay_path.name

            plot_path = out_dir / f"{args.split}_ep_{e:05d}_error_vs_step.png"
            plt.figure(figsize=(6, 3.5))
            plt.plot(np.arange(len(pos_err)), pos_err, lw=2, color="tab:red")
            plt.xlabel("step")
            plt.ylabel("||x_pred - x_true||")
            plt.title(f"{args.split} ep={e}: Error vs Time Step")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=140)
            plt.close()
            row["error_plot"] = plot_path.name

        rows.append(row)
        print(
            f"[{len(rows)}/{len(eval_eps)}] ep={e} mean_err={row['mean_err']:.6f} "
            f"max_err={row['max_err']:.6f} final_err={row['final_err']:.6f} "
            f"ttf={row['divergence_step']}"
        )

    div_steps = np.array([r["divergence_step"] for r in rows], dtype=np.float64)
    summary = {
        "split": args.split,
        "num_episodes": int(len(rows)),
        "rollout_steps": int(rollout_steps),
        "divergence_threshold": float(args.divergence_threshold),
        "ttf_median": float(np.median(div_steps)),
        "ttf_p10": float(np.percentile(div_steps, 10)),
        "divergence_rate": float(np.mean([1.0 if r["diverged"] else 0.0 for r in rows])),
        "rows": rows,
        "history_len": int(H),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "best_val_mse_pos": float(ckpt.get("best_val_mse_pos", float("nan"))),
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Output dir:", out_dir)
    print("Summary:", out_dir / "eval_summary.json")
    print(
        f"Aggregate | ttf_median={summary['ttf_median']:.2f} "
        f"ttf_p10={summary['ttf_p10']:.2f} divergence_rate={summary['divergence_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
