from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from time_conditioned_collision_model import (
    TimeConditionedCollisionModel,
    TimeConditionedCollisionModelConfig,
    TimeEncodingConfig,
)


def load_checkpoint(path: str) -> dict:
    import numpy as np
    import numpy._core.multiarray as ma
    from torch.serialization import safe_globals

    safe = [
        ma._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype("float32")),
        type(np.dtype("int64")),
        type(np.dtype("uint8")),
    ]
    with safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)


def build_model(ckpt: dict) -> TimeConditionedCollisionModel:
    cfg = ckpt["model_config"]
    model = TimeConditionedCollisionModel(
        TimeConditionedCollisionModelConfig(
            state_dim=int(cfg["state_dim"]),
            trunk_width=int(cfg["trunk_width"]),
            trunk_depth=int(cfg["trunk_depth"]),
            activation=str(cfg["activation"]),
            dropout=float(cfg["dropout"]),
            enforce_t0_anchor=bool(cfg["enforce_t0_anchor"]),
            time_encoding=TimeEncodingConfig(
                mode=str(cfg["time_encoding_mode"]),
                num_frequencies=int(cfg["num_frequencies"]),
                include_raw_time=bool(cfg["include_raw_time"]),
                base_frequency=float(cfg["base_frequency"]),
                normalize_time=bool(cfg["normalize_time"]),
                max_time=float(cfg["time_max"]),
            ),
        )
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_states(model: TimeConditionedCollisionModel, ckpt: dict, s0_raw: np.ndarray, times: np.ndarray) -> np.ndarray:
    s0_mean = np.asarray(ckpt["s0_standardizer"]["mean"], dtype=np.float32).reshape(1, 4)
    s0_std = np.asarray(ckpt["s0_standardizer"]["std"], dtype=np.float32).reshape(1, 4)
    y_mean = np.asarray(ckpt["state_standardizer"]["mean"], dtype=np.float32).reshape(1, 4)
    y_std = np.asarray(ckpt["state_standardizer"]["std"], dtype=np.float32).reshape(1, 4)

    s0_n = ((s0_raw.reshape(1, 4) - s0_mean) / s0_std).astype(np.float32)
    s0_batch = np.repeat(s0_n, len(times), axis=0)
    t_batch = times.astype(np.float32)

    out = model(torch.from_numpy(s0_batch), torch.from_numpy(t_batch))["state"].cpu().numpy()
    return (out * y_std + y_mean).astype(np.float32)


def plot_episode_grid(results: dict, true_states: np.ndarray, times: np.ndarray, episode_ids: list[int], out_path: Path) -> None:
    labels = list(results.keys())
    fig, axs = plt.subplots(len(episode_ids), len(labels), figsize=(6.8 * len(labels), 3.0 * len(episode_ids)), sharex=True)
    axs = np.asarray(axs).reshape(len(episode_ids), len(labels))

    for r, ep in enumerate(episode_ids):
        true_speed = np.linalg.norm(true_states[ep, :, 2:4], axis=1)
        for c, label in enumerate(labels):
            pred_speed = np.linalg.norm(results[label][ep, :, 2:4], axis=1)
            ax = axs[r, c]
            ax.plot(times, true_speed, color="black", lw=1.6, ls="--", label="true |v|")
            ax.plot(times, pred_speed, lw=1.5, label="pred |v|")
            ax.set_title(f"{label} - test episode {ep}")
            ax.set_ylabel("speed")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(0.0, max(0.8, float(np.nanmax(pred_speed)) * 1.08))
            if r == 0:
                ax.legend(loc="upper right", fontsize=8)
            if r == len(episode_ids) - 1:
                ax.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_summary(results: dict, true_states: np.ndarray, times: np.ndarray, out_path: Path) -> None:
    true_speed = np.linalg.norm(true_states[:, :, 2:4], axis=2)
    true_mean = true_speed.mean(axis=0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(times, true_mean, color="black", lw=1.8, ls="--", label="true mean |v|")

    for label, pred_states in results.items():
        pred_speed = np.linalg.norm(pred_states[:, :, 2:4], axis=2)
        mean = pred_speed.mean(axis=0)
        std = pred_speed.std(axis=0)
        signed = (pred_speed - true_speed).mean(axis=0)
        mae = np.abs(pred_speed - true_speed).mean(axis=0)

        axs[0].plot(times, mean, lw=1.5, label=f"{label} pred mean |v|")
        axs[0].fill_between(times, mean - std, mean + std, alpha=0.14)
        axs[1].plot(times, signed, lw=1.5, label=label)
        axs[2].plot(times, mae, lw=1.5, label=label)

    axs[0].set_ylabel("speed")
    axs[0].set_title("Predicted Speed Across Test Episodes")
    axs[1].axhline(0.0, color="black", lw=1.0, alpha=0.5)
    axs[1].set_ylabel("mean pred-true")
    axs[2].set_ylabel("mean abs error")
    axs[2].set_xlabel("time (s)")
    for ax in axs:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_error_scatter(
    results: dict,
    true_states: np.ndarray,
    out_path: Path,
    speed_threshold: float,
    pos_threshold: float,
    vel_threshold: float,
) -> None:
    labels = list(results.keys())
    steps = np.arange(true_states.shape[1], dtype=np.int32)
    x_steps = np.repeat(steps[None, :], true_states.shape[0], axis=0).reshape(-1)

    fig, axs = plt.subplots(3, len(labels), figsize=(6.8 * len(labels), 9.2), sharex=True)
    axs = np.asarray(axs).reshape(3, len(labels))

    true_speed = np.linalg.norm(true_states[:, :, 2:4], axis=2)
    for c, label in enumerate(labels):
        pred_states = results[label]
        pred_speed = np.linalg.norm(pred_states[:, :, 2:4], axis=2)
        speed_err = np.abs(pred_speed - true_speed).reshape(-1)
        pos_inf = np.max(np.abs(pred_states[:, :, 0:2] - true_states[:, :, 0:2]), axis=2).reshape(-1)
        vel_inf = np.max(np.abs(pred_states[:, :, 2:4] - true_states[:, :, 2:4]), axis=2).reshape(-1)

        series = [
            ("abs speed error", speed_err, speed_threshold),
            ("position error, L_inf", pos_inf, pos_threshold),
            ("velocity error, L_inf", vel_inf, vel_threshold),
        ]
        for r, (ylabel, values, threshold) in enumerate(series):
            ax = axs[r, c]
            ax.scatter(x_steps, values, s=4, alpha=0.13, linewidths=0, rasterized=True)
            ax.axhline(float(threshold), color="black", lw=1.0, ls="--", alpha=0.65)
            ax.set_title(label if r == 0 else "")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.22)
            if r == 2:
                ax.set_xlabel("step")

    fig.suptitle("Prediction Error by Step Across Test Episodes", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_joint_accuracy_scatter(
    results: dict,
    true_states: np.ndarray,
    out_path: Path,
    pos_threshold: float,
    vel_threshold: float,
) -> None:
    labels = list(results.keys())
    steps = np.arange(true_states.shape[1], dtype=np.int32)
    x_steps = np.repeat(steps[None, :], true_states.shape[0], axis=0).reshape(-1)

    fig, axs = plt.subplots(2, len(labels), figsize=(6.8 * len(labels), 7.0))
    axs = np.asarray(axs).reshape(2, len(labels))

    for c, label in enumerate(labels):
        pred_states = results[label]
        pos_inf = np.max(np.abs(pred_states[:, :, 0:2] - true_states[:, :, 0:2]), axis=2)
        vel_inf = np.max(np.abs(pred_states[:, :, 2:4] - true_states[:, :, 2:4]), axis=2)
        both_ok = (pos_inf <= float(pos_threshold)) & (vel_inf <= float(vel_threshold))
        frac_ok = both_ok.mean(axis=0)

        ax = axs[0, c]
        colors = np.where(both_ok.reshape(-1), "#2b8a3e", "#8c8c8c")
        ax.scatter(x_steps, pos_inf.reshape(-1), c=colors, s=5, alpha=0.16, linewidths=0, rasterized=True)
        ax.axhline(float(pos_threshold), color="black", lw=1.0, ls="--", alpha=0.65)
        ax.set_title(f"{label}: position error colored by joint accuracy")
        ax.set_xlabel("step")
        ax.set_ylabel("position error, L_inf")
        ax.grid(True, alpha=0.22)

        ax = axs[1, c]
        ax.scatter(steps, frac_ok, s=16, alpha=0.85, linewidths=0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("step")
        ax.set_ylabel("fraction with pos and vel accurate")
        ax.grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def print_accuracy_stats(
    results: dict,
    true_states: np.ndarray,
    speed_threshold: float,
    pos_threshold: float,
    vel_threshold: float,
) -> None:
    true_speed = np.linalg.norm(true_states[:, :, 2:4], axis=2)
    for label, pred_states in results.items():
        pred_speed = np.linalg.norm(pred_states[:, :, 2:4], axis=2)
        speed_ok = np.abs(pred_speed - true_speed) <= float(speed_threshold)
        pos_inf = np.max(np.abs(pred_states[:, :, 0:2] - true_states[:, :, 0:2]), axis=2)
        vel_inf = np.max(np.abs(pred_states[:, :, 2:4] - true_states[:, :, 2:4]), axis=2)
        pos_ok = pos_inf <= float(pos_threshold)
        vel_ok = vel_inf <= float(vel_threshold)
        both_ok = pos_ok & vel_ok
        all_three_ok = both_ok & speed_ok

        best_joint_step = int(np.argmax(both_ok.mean(axis=0)))
        best_all_step = int(np.argmax(all_three_ok.mean(axis=0)))
        print(
            f"{label}: speed_ok={speed_ok.mean():.4f}, "
            f"pos_ok={pos_ok.mean():.4f}, vel_ok={vel_ok.mean():.4f}, "
            f"pos_and_vel_ok={both_ok.mean():.4f}, all_three_ok={all_three_ok.mean():.4f}, "
            f"best_joint_step={best_joint_step} ({both_ok[:, best_joint_step].mean():.4f}), "
            f"best_all_step={best_all_step} ({all_three_ok[:, best_all_step].mean():.4f})"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Plot speed diagnostics for simple TCNO checkpoints.")
    p.add_argument("--dataset", default="datasets/simple_tcno_stratified_1p.npz")
    p.add_argument("--simple-ckpt", default="checkpoints/simple_tcno_1p/model_tc_collision_1p.pt")
    p.add_argument("--f-ckpt", default="checkpoints/simple_tcno_1p_exact_F_fourier4_gaussian/model_tc_collision_1p.pt")
    p.add_argument("--out-dir", default="checkpoints/speed_diagnostics")
    p.add_argument("--num-summary-episodes", type=int, default=64)
    p.add_argument("--num-example-episodes", type=int, default=4)
    p.add_argument("--speed-threshold", type=float, default=0.01)
    p.add_argument("--pos-threshold", type=float, default=0.01)
    p.add_argument("--vel-threshold", type=float, default=0.01)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as data:
        pos_all = data["pos_all"].astype(np.float32)
        vel_all = data["vel_all"].astype(np.float32)
        test_eps = data["test_eps"].astype(np.int64)
        meta_json = str(data["meta_json"])

    import json

    meta = json.loads(meta_json)
    dt = float(meta["dt"])
    times = np.arange(pos_all.shape[1], dtype=np.float32) * dt
    true_states_all = np.concatenate([pos_all[:, :, 0, :], vel_all[:, :, 0, :]], axis=2)

    selected = test_eps[: max(1, int(args.num_summary_episodes))]
    examples = list(range(min(int(args.num_example_episodes), len(selected))))
    true_states = true_states_all[selected]

    ckpt_paths = {
        "simple_tcno_1p": args.simple_ckpt,
        "F_fourier4_gaussian": args.f_ckpt,
    }
    results = {}
    for label, ckpt_path in ckpt_paths.items():
        ckpt = load_checkpoint(ckpt_path)
        model = build_model(ckpt)
        pred = np.zeros_like(true_states)
        for i, ep in enumerate(selected):
            s0 = true_states_all[int(ep), 0]
            pred[i] = predict_states(model, ckpt, s0, times)
        results[label] = pred

    plot_episode_grid(
        results,
        true_states=true_states,
        times=times,
        episode_ids=examples,
        out_path=out_dir / "tcno_speed_examples.png",
    )
    plot_summary(
        results,
        true_states=true_states,
        times=times,
        out_path=out_dir / "tcno_speed_summary.png",
    )
    plot_error_scatter(
        results,
        true_states=true_states,
        out_path=out_dir / "tcno_error_scatter_by_step.png",
        speed_threshold=float(args.speed_threshold),
        pos_threshold=float(args.pos_threshold),
        vel_threshold=float(args.vel_threshold),
    )
    plot_joint_accuracy_scatter(
        results,
        true_states=true_states,
        out_path=out_dir / "tcno_joint_accuracy_scatter.png",
        pos_threshold=float(args.pos_threshold),
        vel_threshold=float(args.vel_threshold),
    )

    for label, pred_states in results.items():
        pred_speed = np.linalg.norm(pred_states[:, :, 2:4], axis=2)
        true_speed = np.linalg.norm(true_states[:, :, 2:4], axis=2)
        print(
            f"{label}: mean_pred_speed={pred_speed.mean():.6f}, "
            f"std_pred_speed={pred_speed.std():.6f}, "
            f"mean_abs_speed_error={np.abs(pred_speed - true_speed).mean():.6f}, "
            f"final_mean_pred_speed={pred_speed[:, -1].mean():.6f}"
        )
    print_accuracy_stats(
        results,
        true_states=true_states,
        speed_threshold=float(args.speed_threshold),
        pos_threshold=float(args.pos_threshold),
        vel_threshold=float(args.vel_threshold),
    )
    print(out_dir / "tcno_speed_examples.png")
    print(out_dir / "tcno_speed_summary.png")
    print(out_dir / "tcno_error_scatter_by_step.png")
    print(out_dir / "tcno_joint_accuracy_scatter.png")


if __name__ == "__main__":
    main()
