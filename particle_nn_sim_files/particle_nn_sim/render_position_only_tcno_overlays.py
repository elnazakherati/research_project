from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

try:
    from particle_nn_sim.one_particle_rollout import save_animation_mp4
    from particle_nn_sim.simulator import ParticleSim2D
    from particle_nn_sim.time_conditioned_collision_model import (
        TimeConditionedCollisionModel,
        TimeConditionedCollisionModelConfig,
        TimeEncodingConfig,
    )
except ModuleNotFoundError:
    from one_particle_rollout import save_animation_mp4
    from simulator import ParticleSim2D
    from time_conditioned_collision_model import (
        TimeConditionedCollisionModel,
        TimeConditionedCollisionModelConfig,
        TimeEncodingConfig,
    )


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render overlay videos for position-only TCNO checkpoints.")
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--dataset",
        default="",
        help="Optional cached NPZ dataset. If provided, GT positions and split indices are read from it directly.",
    )
    p.add_argument("--out-dir", default="results/position_only_tcno_overlays")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-episodes", type=int, default=3)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=-1)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--trace-tail-steps", type=int, default=0)
    p.add_argument("--save-overlay", type=str2bool, default=True)
    p.add_argument("--save-error-plots", type=str2bool, default=True)
    return p.parse_args()


def resolve_device(flag: str) -> str:
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(ckpt: dict[str, Any]) -> TimeConditionedCollisionModel:
    raw = ckpt["model_config"]
    data_cfg = ckpt["config"]["data"]
    default_time_max = float(data_cfg.get("dt", 0.01)) * float(data_cfg.get("steps", 1))
    cfg = TimeConditionedCollisionModelConfig(
        state_dim=int(raw["state_dim"]),
        output_mode=str(raw.get("output_mode", "state")),
        trunk_width=int(raw["trunk_width"]),
        trunk_depth=int(raw["trunk_depth"]),
        activation=str(raw["activation"]),
        dropout=float(raw["dropout"]),
        enforce_t0_anchor=bool(raw.get("enforce_t0_anchor", True)),
        use_event_head=bool(raw.get("use_event_head", False)),
        use_endpoint_velocity_head=bool(raw.get("use_endpoint_velocity_head", False)),
        time_encoding=TimeEncodingConfig(
            mode=str(raw.get("time_encoding_mode", "raw")),
            num_frequencies=int(raw.get("num_frequencies", 4)),
            include_raw_time=bool(raw.get("include_raw_time", True)),
            base_frequency=float(raw.get("base_frequency", 1.0)),
            normalize_time=bool(raw.get("normalize_time", True)),
            max_time=float(raw.get("time_max", default_time_max)),
        ),
    )
    model = TimeConditionedCollisionModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    s0_stats = ckpt.get("s0_standardizer", None)
    y_stats = ckpt.get("state_standardizer", None)
    if bool(cfg.enforce_t0_anchor) and s0_stats is not None and y_stats is not None:
        s0_mean = torch.as_tensor(np.asarray(s0_stats["mean"], dtype=np.float32).reshape(4))
        s0_std = torch.as_tensor(np.asarray(s0_stats["std"], dtype=np.float32).reshape(4))
        y_mean = torch.as_tensor(np.asarray(y_stats["mean"], dtype=np.float32).reshape(4))
        y_std = torch.as_tensor(np.asarray(y_stats["std"], dtype=np.float32).reshape(4))
        out_dim = 2 if str(raw.get("output_mode", "state")) == "position" else 4
        anchor_scale = s0_std[:out_dim] / y_std[:out_dim]
        anchor_bias = (s0_mean[:out_dim] - y_mean[:out_dim]) / y_std[:out_dim]
        model.set_s0_anchor_affine(anchor_scale, anchor_bias)
    return model


def animate_overlay_tail_1p(
    *,
    pos_ref: np.ndarray,
    pos_pert: np.ndarray,
    radius: float,
    W: float,
    H: float,
    dt: float,
    tail_steps: int,
    title: str,
    label_ref: str = "GT",
    label_pert: str = "TCNO",
):
    pos_ref = np.asarray(pos_ref, dtype=np.float32)
    pos_pert = np.asarray(pos_pert, dtype=np.float32)
    n_frames = min(len(pos_ref), len(pos_pert))
    display_radius = max(float(radius), 0.015 * min(float(W), float(H)))
    tail_steps = int(tail_steps)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], lw=2)

    (trace_ref,) = ax.plot([], [], color="tab:green", lw=2.0, alpha=0.9, label=str(label_ref))
    (trace_pert,) = ax.plot([], [], color="tab:orange", lw=2.0, alpha=0.45, label=str(label_pert))
    c_ref = plt.Circle(pos_ref[0, 0], display_radius, color="tab:green", fill=True, alpha=0.9)
    c_pert = plt.Circle(pos_pert[0, 0], display_radius, color="tab:orange", fill=True, alpha=0.45)
    ax.add_patch(c_ref)
    ax.add_patch(c_pert)
    ax.legend(loc="upper right")

    step_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def animate_frame(frame: int):
        start = 0 if tail_steps <= 0 else max(0, frame - tail_steps + 1)
        trace_ref.set_data(pos_ref[start : frame + 1, 0, 0], pos_ref[start : frame + 1, 0, 1])
        trace_pert.set_data(pos_pert[start : frame + 1, 0, 0], pos_pert[start : frame + 1, 0, 1])
        c_ref.center = pos_ref[frame, 0]
        c_pert.center = pos_pert[frame, 0]
        step_text.set_text(f"t={frame * dt:.2f}s | step {frame}/{n_frames - 1}")
        return [trace_ref, trace_pert, c_ref, c_pert, step_text]

    ani = animation.FuncAnimation(fig, animate_frame, frames=n_frames, interval=20, blit=True)
    plt.close(fig)
    return ani


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    data_cfg = ckpt["config"]["data"]
    dataset_npz: dict[str, np.ndarray] = {}
    if str(args.dataset).strip():
        with np.load(args.dataset, allow_pickle=False) as data:
            dataset_npz = {
                "pos_all": data["pos_all"].astype(np.float32),
                "vel_all": data["vel_all"].astype(np.float32),
                "train_eps": data["train_eps"].astype(np.int64),
                "val_eps": data["val_eps"].astype(np.int64),
                "test_eps": data["test_eps"].astype(np.int64),
            }

    split_source = dataset_npz if dataset_npz else ckpt["split_indices"]
    split_eps = np.asarray(split_source[f"{args.split}_eps"], dtype=np.int64)
    eval_eps = split_eps[int(args.start_idx) : int(args.start_idx) + int(args.num_episodes)]
    if len(eval_eps) == 0:
        raise RuntimeError("Requested empty episode range.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = build_model(ckpt).to(device)
    model.eval()

    dt = float(data_cfg["dt"])
    rollout_steps = int(data_cfg["steps"]) if int(args.rollout_steps) < 0 else int(args.rollout_steps)
    T = rollout_steps + 1
    radius = float(data_cfg.get("radius", 0.0))
    radius_eff = radius if radius > 0.0 else 1e-6
    mass = float(data_cfg.get("mass", 1.0))
    wall_mode = str(data_cfg.get("wall_collision_mode", "exact"))

    s0_stats = ckpt.get("s0_standardizer", None)
    y_stats = ckpt.get("state_standardizer", None)
    s0_mean = np.asarray(s0_stats["mean"], dtype=np.float32) if s0_stats is not None else None
    s0_std = np.asarray(s0_stats["std"], dtype=np.float32) if s0_stats is not None else None
    y_mean = np.asarray(y_stats["mean"], dtype=np.float32) if y_stats is not None else None
    y_std = np.asarray(y_stats["std"], dtype=np.float32) if y_stats is not None else None

    if dataset_npz:
        pos0_all = dataset_npz["pos_all"][:, 0].astype(np.float32)
        vel0_all = dataset_npz["vel_all"][:, 0].astype(np.float32)
    else:
        pos0_all = np.asarray(ckpt["episode_init"]["pos0"], dtype=np.float32)
        vel0_all = np.asarray(ckpt["episode_init"]["vel0"], dtype=np.float32)
    rows: list[dict[str, Any]] = []

    t_query = (np.arange(T, dtype=np.float32) * np.float32(dt)).astype(np.float32)
    tq_t = torch.from_numpy(t_query).to(device)

    for i, ep in enumerate(eval_eps, start=1):
        ep_i = int(ep)
        pos0 = pos0_all[ep_i].reshape(1, 2).astype(np.float32)
        vel0 = vel0_all[ep_i].reshape(1, 2).astype(np.float32)
        s0 = np.concatenate([pos0.reshape(2), vel0.reshape(2)], axis=0).astype(np.float32)

        if dataset_npz:
            if rollout_steps > dataset_npz["pos_all"].shape[1] - 1:
                raise ValueError("--rollout-steps exceeds cached dataset length.")
            true_pos = dataset_npz["pos_all"][ep_i, : T].astype(np.float32)
        else:
            sim = ParticleSim2D(
                W=1.0,
                H=1.0,
                radii=[radius_eff],
                masses=[mass],
                restitution=1.0,
                seed=10_000 + ep_i,
                wall_mode=wall_mode,
            )
            sim.reset(pos0, vel0)
            true_pos, _true_vel = sim.rollout(dt=dt, steps=rollout_steps)
            true_pos = true_pos.astype(np.float32)

        s0_batch = np.repeat(s0[None, :], T, axis=0).astype(np.float32)
        if s0_mean is not None and s0_std is not None:
            s0_batch = ((s0_batch - s0_mean.reshape(1, 4)) / s0_std.reshape(1, 4)).astype(np.float32)

        with torch.no_grad():
            out = model(torch.from_numpy(s0_batch).to(device), tq_t)
            pred = out["state"].detach().cpu().numpy().astype(np.float32)

        if y_mean is not None and y_std is not None:
            pred = (pred * y_std.reshape(1, 4)[:, : pred.shape[1]] + y_mean.reshape(1, 4)[:, : pred.shape[1]]).astype(
                np.float32
            )
        pred_pos = pred[:, :2].reshape(T, 1, 2).astype(np.float32)

        pos_err = np.linalg.norm(pred_pos[:, 0, :] - true_pos[:, 0, :], axis=1)
        row = {
            "episode_idx_global": ep_i,
            "mean_pos_err": float(np.mean(pos_err)),
            "max_pos_err": float(np.max(pos_err)),
            "final_pos_err": float(pos_err[-1]),
            "initial_state": [float(x) for x in s0.tolist()],
        }

        if args.save_error_plots:
            step_idx = np.arange(T, dtype=np.int64)
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axs[0].plot(step_idx, pos_err, lw=1.3, color="tab:red")
            axs[0].set_ylabel("||pos err||")
            axs[0].grid(True, alpha=0.3)
            axs[1].plot(step_idx, pred_pos[:, 0, 0] - true_pos[:, 0, 0], lw=1.0, color="tab:blue")
            axs[1].set_ylabel("x err")
            axs[1].grid(True, alpha=0.3)
            axs[2].plot(step_idx, pred_pos[:, 0, 1] - true_pos[:, 0, 1], lw=1.0, color="tab:orange")
            axs[2].set_ylabel("y err")
            axs[2].set_xlabel("step")
            axs[2].grid(True, alpha=0.3)
            fig.suptitle(f"Position error ({args.split}) ep={ep_i}", y=0.995, fontsize=11)
            plt.tight_layout()
            err_name = f"{args.split}_ep_{ep_i:05d}_position_error.png"
            plt.savefig(out_dir / err_name, dpi=140)
            plt.close(fig)
            row["position_error_plot"] = err_name

        if args.save_overlay:
            overlay = animate_overlay_tail_1p(
                pos_ref=true_pos[:: int(args.frame_stride)],
                pos_pert=pred_pos[:: int(args.frame_stride)],
                radius=radius_eff,
                W=1.0,
                H=1.0,
                dt=dt * float(args.frame_stride),
                tail_steps=(
                    0
                    if int(args.trace_tail_steps) <= 0
                    else max(1, int(round(int(args.trace_tail_steps) / int(args.frame_stride))))
                ),
                title=f"GT vs TCNO ({args.split}) ep={ep_i} step={rollout_steps}",
                label_ref="GT",
                label_pert="TCNO",
            )
            video_name = f"{args.split}_ep_{ep_i:05d}_gt_vs_pred_overlay_1p.mp4"
            save_animation_mp4(overlay, str(out_dir / video_name), fps=int(args.fps))
            row["video_overlay"] = video_name

        rows.append(row)
        print(
            f"[{i}/{len(eval_eps)}] ep={ep_i} mean_err={row['mean_pos_err']:.6f} "
            f"max_err={row['max_pos_err']:.6f} final_err={row['final_pos_err']:.6f}",
            flush=True,
        )

    summary = {
        "checkpoint": str(args.ckpt),
        "dataset": str(args.dataset) if str(args.dataset).strip() else None,
        "split": str(args.split),
        "rollout_steps": int(rollout_steps),
        "frame_stride": int(args.frame_stride),
        "trace_tail_steps": int(args.trace_tail_steps),
        "fps": int(args.fps),
        "rows": rows,
        "mean_pos_err": float(np.mean([r["mean_pos_err"] for r in rows])),
        "max_pos_err": float(np.max([r["max_pos_err"] for r in rows])),
        "final_pos_err_mean": float(np.mean([r["final_pos_err"] for r in rows])),
    }
    with open(out_dir / "overlay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Done.")
    print("Output dir:", out_dir)
    print("Summary:", out_dir / "overlay_summary.json")


if __name__ == "__main__":
    main()
