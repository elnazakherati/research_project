from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from particle_nn_sim.time_conditioned_collision_model import (
        TimeConditionedCollisionModel,
        TimeConditionedCollisionModelConfig,
        TimeEncodingConfig,
    )
except ModuleNotFoundError:
    from time_conditioned_collision_model import (
        TimeConditionedCollisionModel,
        TimeConditionedCollisionModelConfig,
        TimeEncodingConfig,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot endpoint velocity-head errors for TCNO checkpoints.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", default="results/endpoint_velocity_errors")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-episodes", type=int, default=3)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--anchor-steps", default="")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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

    # Non-persistent anchor buffers must be reconstructed when checkpoints are loaded.
    s0_stats = ckpt.get("s0_standardizer", None)
    y_stats = ckpt.get("state_standardizer", None)
    if bool(cfg.enforce_t0_anchor) and s0_stats is not None and y_stats is not None:
        s0_mean = torch.as_tensor(np.asarray(s0_stats["mean"], dtype=np.float32).reshape(4))
        s0_std = torch.as_tensor(np.asarray(s0_stats["std"], dtype=np.float32).reshape(4))
        y_mean = torch.as_tensor(np.asarray(y_stats["mean"], dtype=np.float32).reshape(4))
        y_std = torch.as_tensor(np.asarray(y_stats["std"], dtype=np.float32).reshape(4))
        out_dim = 2 if str(raw.get("output_mode", "state")) == "position" else 4
        model.set_s0_anchor_affine(s0_std[:out_dim] / y_std[:out_dim], (s0_mean[:out_dim] - y_mean[:out_dim]) / y_std[:out_dim])
    return model


def parse_anchor_steps(value: str, ckpt: dict[str, Any]) -> list[int]:
    if str(value).strip():
        return [int(x.strip()) for x in str(value).split(",") if x.strip()]
    train_cfg = ckpt.get("config", {}).get("train", {})
    anchors = train_cfg.get("endpoint_velocity_anchor_steps", [])
    if anchors:
        return [int(x) for x in anchors]
    return [int(ckpt["config"]["data"]["steps"])]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not bool(ckpt["model_config"].get("use_endpoint_velocity_head", False)):
        raise RuntimeError("Checkpoint does not have an endpoint velocity head.")

    with np.load(args.dataset, allow_pickle=False) as data:
        pos_all = data["pos_all"].astype(np.float32)
        vel_all = data["vel_all"].astype(np.float32)
        split_eps = data[f"{args.split}_eps"].astype(np.int64)

    anchors = parse_anchor_steps(args.anchor_steps, ckpt)
    max_step = int(pos_all.shape[1] - 1)
    anchors = [a for a in anchors if 0 <= int(a) <= max_step]
    if not anchors:
        raise RuntimeError("No valid anchor steps requested.")

    eval_eps = split_eps[int(args.start_idx) : int(args.start_idx) + int(args.num_episodes)]
    if len(eval_eps) == 0:
        raise RuntimeError("Requested empty episode range.")

    device = resolve_device(args.device)
    model = build_model(ckpt).to(device)
    model.eval()

    dt = float(ckpt["config"]["data"]["dt"])
    s0_mean = np.asarray(ckpt["s0_standardizer"]["mean"], dtype=np.float32)
    s0_std = np.asarray(ckpt["s0_standardizer"]["std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["state_standardizer"]["mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["state_standardizer"]["std"], dtype=np.float32)

    rows: list[dict[str, Any]] = []
    all_errs: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    t = (np.asarray(anchors, dtype=np.float32) * np.float32(dt)).astype(np.float32)
    t_t = torch.from_numpy(t).to(device)

    for i, ep in enumerate(eval_eps, start=1):
        ep_i = int(ep)
        s0 = np.concatenate([pos_all[ep_i, 0, 0], vel_all[ep_i, 0, 0]], axis=0).astype(np.float32)
        s0_batch = np.repeat(s0[None, :], len(anchors), axis=0).astype(np.float32)
        s0_batch = ((s0_batch - s0_mean.reshape(1, 4)) / s0_std.reshape(1, 4)).astype(np.float32)

        with torch.no_grad():
            out = model(torch.from_numpy(s0_batch).to(device), t_t)
            pred_vel_norm = out["endpoint_vel"]
            if pred_vel_norm is None:
                raise RuntimeError("Model returned no endpoint_vel output.")
            pred_vel = pred_vel_norm.detach().cpu().numpy().astype(np.float32) * y_std.reshape(1, 4)[:, 2:4] + y_mean.reshape(1, 4)[:, 2:4]

        true_vel = vel_all[ep_i, anchors, 0, :].astype(np.float32)
        err = pred_vel - true_vel
        err_l2 = np.linalg.norm(err, axis=1)
        all_errs.append(err_l2)
        all_pred.append(pred_vel)
        all_true.append(true_vel)

        fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        axs[0].plot(anchors, true_vel[:, 0], marker="o", label="true vx")
        axs[0].plot(anchors, pred_vel[:, 0], marker="x", label="pred vx")
        axs[0].set_ylabel("vx")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc="best")
        axs[1].plot(anchors, true_vel[:, 1], marker="o", label="true vy")
        axs[1].plot(anchors, pred_vel[:, 1], marker="x", label="pred vy")
        axs[1].set_ylabel("vy")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc="best")
        axs[2].plot(anchors, err_l2, marker="o", color="tab:red", label="||velocity error||")
        axs[2].set_ylabel("L2 error")
        axs[2].set_xlabel("step")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc="best")
        fig.suptitle(f"Endpoint velocity error ({args.split}) ep={ep_i}", y=0.995, fontsize=11)
        plt.tight_layout()
        plot_name = f"{args.split}_ep_{ep_i:05d}_endpoint_velocity_error.png"
        plt.savefig(out_dir / plot_name, dpi=140)
        plt.close(fig)

        row = {
            "episode_idx_global": ep_i,
            "anchor_steps": [int(a) for a in anchors],
            "true_velocity": true_vel.tolist(),
            "pred_velocity": pred_vel.tolist(),
            "velocity_error": err.tolist(),
            "velocity_error_l2": err_l2.tolist(),
            "mean_velocity_error_l2": float(np.mean(err_l2)),
            "max_velocity_error_l2": float(np.max(err_l2)),
            "plot": plot_name,
        }
        rows.append(row)
        print(
            f"[{i}/{len(eval_eps)}] ep={ep_i} mean_vel_err={row['mean_velocity_error_l2']:.6f} "
            f"max_vel_err={row['max_velocity_error_l2']:.6f}",
            flush=True,
        )

    err_mat = np.stack(all_errs, axis=0)
    pred_arr = np.stack(all_pred, axis=0)
    true_arr = np.stack(all_true, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    err_mean = err_mat.mean(axis=0)
    err_std = err_mat.std(axis=0)
    ax.plot(anchors, err_mean, marker="o", label="mean ||velocity error||")
    ax.fill_between(anchors, err_mean - err_std, err_mean + err_std, alpha=0.25, label="±1 std")
    ax.set_xlabel("step")
    ax.set_ylabel("L2 velocity error")
    ax.set_title(f"Endpoint velocity error across {args.split} episodes")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    aggregate_plot = "endpoint_velocity_error_mean_std.png"
    plt.savefig(out_dir / aggregate_plot, dpi=140)
    plt.close(fig)

    summary = {
        "checkpoint": str(args.ckpt),
        "dataset": str(args.dataset),
        "split": str(args.split),
        "num_episodes": int(len(rows)),
        "anchor_steps": [int(a) for a in anchors],
        "rows": rows,
        "mean_velocity_error_l2": float(np.mean(err_mat)),
        "max_velocity_error_l2": float(np.max(err_mat)),
        "velocity_mse": float(np.mean((pred_arr - true_arr) ** 2)),
        "aggregate_plot": aggregate_plot,
    }
    with open(out_dir / "endpoint_velocity_error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Done.")
    print("Output dir:", out_dir)
    print("Summary:", out_dir / "endpoint_velocity_error_summary.json")


if __name__ == "__main__":
    main()
