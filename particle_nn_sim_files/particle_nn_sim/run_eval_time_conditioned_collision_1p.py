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

from particle_nn_sim.one_particle_data import collect_episodes_1p
from particle_nn_sim.one_particle_rollout import (
    animate_overlay_gt_perturbed_1p,
    animate_side_by_side_1p,
    save_animation_mp4,
)
from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.time_conditioned_collision_model import (
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


def resolve_device(flag: str) -> str:
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def infer_collision_steps_from_velocity(vel_ep: np.ndarray) -> np.ndarray:
    vx_prev = vel_ep[:-1, 0, 0]
    vx_now = vel_ep[1:, 0, 0]
    vy_prev = vel_ep[:-1, 0, 1]
    vy_now = vel_ep[1:, 0, 1]
    hit = (vx_prev * vx_now < 0) | (vy_prev * vy_now < 0)
    return np.where(hit)[0].astype(np.int64) + 1


def extract_collision_steps(coll_ep: np.ndarray | None, vel_ep: np.ndarray) -> np.ndarray:
    if coll_ep is not None:
        c = np.asarray(coll_ep).reshape(-1)
        return np.where(c > 0)[0].astype(np.int64) + 1
    return infer_collision_steps_from_velocity(vel_ep)


def build_event_targets(
    T: int,
    collision_steps: np.ndarray,
    dt: float,
    mode: str,
    eps_steps: int,
    event_window: float,
    sigma_event: float,
) -> np.ndarray:
    labels = np.zeros((T,), dtype=np.float32)
    if len(collision_steps) == 0:
        return labels

    mode = str(mode).strip().lower()
    if mode == "window":
        win_steps = max(int(eps_steps), int(np.ceil(float(event_window) / float(dt))))
        for c in collision_steps:
            lo = int(max(0, int(c) - win_steps))
            hi = int(min(T - 1, int(c) + win_steps))
            labels[lo : hi + 1] = 1.0
        return labels

    if mode == "gaussian":
        sigma_steps = max(float(sigma_event) / float(dt), 1e-6)
        idx = np.arange(T, dtype=np.float32)
        for c in collision_steps:
            d = idx - float(c)
            g = np.exp(-0.5 * (d / sigma_steps) ** 2).astype(np.float32)
            labels = np.maximum(labels, g)
        return np.clip(labels, 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unsupported event_target_mode: {mode}")


def compute_event_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    pred = probs >= float(threshold)
    y = labels >= 0.5
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & (~y)))
    tn = int(np.sum((~pred) & (~y)))
    fn = int(np.sum((~pred) & y))
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "event_accuracy": float(acc),
        "event_precision": float(prec),
        "event_recall": float(rec),
    }


def compute_sign_accuracy(pred_vel: np.ndarray, true_vel: np.ndarray, sign_epsilon: float = 1e-6) -> dict[str, float]:
    mask_x = np.abs(true_vel[:, 0]) > float(sign_epsilon)
    mask_y = np.abs(true_vel[:, 1]) > float(sign_epsilon)
    pred_x = pred_vel[:, 0] >= 0.0
    pred_y = pred_vel[:, 1] >= 0.0
    true_x = true_vel[:, 0] >= 0.0
    true_y = true_vel[:, 1] >= 0.0
    acc_x = float(np.mean(pred_x[mask_x] == true_x[mask_x])) if np.any(mask_x) else 1.0
    acc_y = float(np.mean(pred_y[mask_y] == true_y[mask_y])) if np.any(mask_y) else 1.0
    return {"sign_acc_vx": acc_x, "sign_acc_vy": acc_y}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate TimeConditionedCollisionModel checkpoint on train/val/test split."
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_tc_collision_1p.pt")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=-1,
        help="Override rollout steps. -1 uses training steps from checkpoint config.",
    )
    p.add_argument(
        "--chunk-steps",
        type=int,
        default=0,
        help=(
            "If > 0 with --num-chunks > 0, run chunked autoregressive TCNO rollout: "
            "predict chunk-steps, re-seed with last predicted state, repeat."
        ),
    )
    p.add_argument(
        "--num-chunks",
        type=int,
        default=0,
        help="Number of chunks for chunked autoregressive rollout.",
    )
    p.add_argument(
        "--chunk-anchor-mode",
        type=str,
        default="pred",
        choices=["pred", "gt"],
        help=(
            "Chunk re-anchoring mode when chunk rollout is enabled: "
            "'pred' = next chunk starts from previous chunk's last prediction, "
            "'gt' = next chunk starts from GT state at the chunk boundary."
        ),
    )
    p.add_argument("--divergence-threshold", type=float, default=0.3)
    p.add_argument("--event-prob-threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-render", type=str2bool, default=False)
    p.add_argument("--save-overlay", type=str2bool, default=True)
    p.add_argument("--save-side-by-side", type=str2bool, default=False)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--sign-epsilon", type=float, default=1e-6)
    p.add_argument("--plateau-event-threshold", type=float, default=0.1)
    p.add_argument("--out-dir", type=str, default="checkpoints/eval_tc_collision_1p")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be >= 1")
    if args.start_idx < 0:
        raise ValueError("--start-idx must be >= 0")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.divergence_threshold < 0.0:
        raise ValueError("--divergence-threshold must be >= 0")
    if args.chunk_steps < 0:
        raise ValueError("--chunk-steps must be >= 0")
    if args.num_chunks < 0:
        raise ValueError("--num-chunks must be >= 0")

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_name") != "TimeConditionedCollisionModel":
        raise RuntimeError(
            f"Expected model_name='TimeConditionedCollisionModel', got {ckpt.get('model_name')!r}"
        )

    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    split_indices = ckpt["split_indices"]
    model_cfg_raw = ckpt["model_config"]
    default_time_max = float(data_cfg.get("dt", 0.01)) * float(data_cfg.get("steps", 700))
    model_cfg = TimeConditionedCollisionModelConfig(
        state_dim=int(model_cfg_raw["state_dim"]),
        trunk_width=int(model_cfg_raw["trunk_width"]),
        trunk_depth=int(model_cfg_raw["trunk_depth"]),
        activation=str(model_cfg_raw["activation"]),
        dropout=float(model_cfg_raw["dropout"]),
        model_variant=str(model_cfg_raw.get("model_variant", "gated_tcno")),
        alpha_gate=float(model_cfg_raw.get("alpha_gate", 5.0)),
        time_encoding=TimeEncodingConfig(
            mode=str(model_cfg_raw.get("time_encoding_mode", "fourier")),
            num_frequencies=int(model_cfg_raw.get("num_frequencies", 8)),
            include_raw_time=bool(model_cfg_raw.get("include_raw_time", True)),
            base_frequency=float(model_cfg_raw.get("base_frequency", 1.0)),
            normalize_time=bool(model_cfg_raw.get("normalize_time", True)),
            max_time=float(model_cfg_raw.get("time_max", default_time_max)),
        ),
    )
    model = TimeConditionedCollisionModel(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Regenerate episodes to recover full trajectories for split eval.
    radius_eff = float(data_cfg.get("radius", 0.0))
    radius_eff = radius_eff if radius_eff > 0.0 else 1e-6
    sim = ParticleSim2D(
        W=1.0,
        H=1.0,
        radii=[radius_eff],
        masses=[float(data_cfg.get("mass", 1.0))],
        restitution=1.0,
        seed=int(train_cfg.get("seed", 0)),
        wall_mode=str(data_cfg.get("wall_collision_mode", "clamp")),
    )
    pos_all, vel_all, coll_all, meta = collect_episodes_1p(
        sim,
        E=int(data_cfg["episodes"]),
        steps=int(data_cfg["steps"]),
        dt=float(data_cfg["dt"]),
        speed_max=float(data_cfg.get("speed_max", 0.7)),
        seed=int(train_cfg.get("seed", 0)),
        stratified_init=bool(data_cfg.get("stratified_init", False)),
        pos_grid_n=int(data_cfg.get("pos_grid_n", 4)),
        angle_bins=int(data_cfg.get("angle_bins", 8)),
        episodes_per_bucket=data_cfg.get("episodes_per_bucket", None),
        fixed_speed=data_cfg.get("fixed_speed", None),
        fixed_x=data_cfg.get("fixed_x", None),
        fixed_y=data_cfg.get("fixed_y", None),
        fixed_vx=data_cfg.get("fixed_vx", None),
        fixed_vy=data_cfg.get("fixed_vy", None),
        fixed2_x=data_cfg.get("fixed2_x", None),
        fixed2_y=data_cfg.get("fixed2_y", None),
        fixed2_vx=data_cfg.get("fixed2_vx", None),
        fixed2_vy=data_cfg.get("fixed2_vy", None),
        ball_center_x=data_cfg.get("ball_center_x", None),
        ball_center_y=data_cfg.get("ball_center_y", None),
        ball_radius=data_cfg.get("ball_radius", None),
        fixed_vel_vx=data_cfg.get("fixed_vel_vx", None),
        fixed_vel_vy=data_cfg.get("fixed_vel_vy", None),
    )

    split_map = {
        "train": np.asarray(split_indices["train_eps"], dtype=np.int64),
        "val": np.asarray(split_indices["val_eps"], dtype=np.int64),
        "test": np.asarray(split_indices["test_eps"], dtype=np.int64),
    }
    split_eps = split_map[args.split]
    if len(split_eps) == 0:
        raise RuntimeError(f"No episodes in split '{args.split}'.")
    end_idx = min(args.start_idx + args.num_episodes, len(split_eps))
    eval_eps = split_eps[args.start_idx:end_idx]
    if len(eval_eps) == 0:
        raise RuntimeError(
            f"Requested empty range: split={args.split}, start={args.start_idx}, num={args.num_episodes}"
        )

    chunk_mode = args.chunk_steps > 0 and args.num_chunks > 0
    if (args.chunk_steps > 0) ^ (args.num_chunks > 0):
        raise ValueError("Provide both --chunk-steps and --num-chunks (or leave both as 0).")
    rollout_steps = int(data_cfg["steps"]) if args.rollout_steps < 0 else int(args.rollout_steps)
    if chunk_mode:
        rollout_steps = int(args.chunk_steps) * int(args.num_chunks)

    dt = float(meta["dt"])
    radius = float(np.asarray(meta["radii"], dtype=np.float32)[0])
    W = float(meta["W"])
    H = float(meta["H"])
    mass = float(np.asarray(meta["masses"], dtype=np.float32)[0])
    restitution = float(meta["restitution"])
    wall_mode = str(data_cfg.get("wall_collision_mode", meta.get("wall_mode", "clamp")))

    s0_std_state = ckpt.get("s0_standardizer", None)
    y_std_state = ckpt.get("state_standardizer", None)
    s0_mean = s0_std = y_mean = y_std = None
    if s0_std_state is not None:
        s0_mean = np.asarray(s0_std_state["mean"], dtype=np.float32)
        s0_std = np.asarray(s0_std_state["std"], dtype=np.float32)
    if y_std_state is not None:
        y_mean = np.asarray(y_std_state["mean"], dtype=np.float32)
        y_std = np.asarray(y_std_state["std"], dtype=np.float32)

    rows: list[dict[str, Any]] = []
    all_pred_state: list[np.ndarray] = []
    all_true_state: list[np.ndarray] = []
    all_evt_logits: list[np.ndarray] = []
    all_evt_true: list[np.ndarray] = []
    all_gate: list[np.ndarray] = []

    eps_steps = int(data_cfg.get("coll_epsilon_steps", 2))
    event_target_mode = str(data_cfg.get("event_target_mode", "window"))
    event_window = float(data_cfg.get("event_window", 0.05))
    sigma_event = float(data_cfg.get("sigma_event", 0.03))

    for idx, e in enumerate(eval_eps, start=1):
        e = int(e)
        pos0 = pos_all[e, 0].astype(np.float32)
        vel0 = vel_all[e, 0].astype(np.float32)

        sim_true = ParticleSim2D(
            W=W,
            H=H,
            radii=[radius],
            masses=[mass],
            restitution=restitution,
            seed=10_000 + e,
            wall_mode=wall_mode,
        )
        sim_true.reset(pos0, vel0)
        # Chunked mode predicts states at t=dt..t=chunk_steps*dt for each chunk,
        # so we align against GT[1:] from a rollout of (rollout_steps + 1).
        true_steps = int(rollout_steps + 1) if chunk_mode else int(rollout_steps)
        pos_true, vel_true = sim_true.rollout(dt=dt, steps=true_steps)
        pos_true = pos_true.astype(np.float32)
        vel_true = vel_true.astype(np.float32)
        s0 = np.concatenate([pos_true[0, 0], vel_true[0, 0]], axis=0).astype(np.float32)

        if chunk_mode:
            pred_chunks: list[np.ndarray] = []
            logit_chunks: list[np.ndarray] = []
            gate_chunks: list[np.ndarray] = []
            vpre_chunks: list[np.ndarray] = []
            vpost_chunks: list[np.ndarray] = []
            s0_chunk = s0.copy()
            t_query = (np.arange(1, int(args.chunk_steps) + 1, dtype=np.float32) * dt).astype(np.float32)

            true_state_full = np.concatenate([pos_true[:, 0, :], vel_true[:, 0, :]], axis=1).astype(np.float32)

            with torch.no_grad():
                for chunk_idx in range(int(args.num_chunks)):
                    s0_batch = np.repeat(s0_chunk[None, :], int(args.chunk_steps), axis=0).astype(np.float32)
                    if s0_mean is not None and s0_std is not None:
                        s0_batch = ((s0_batch - s0_mean) / s0_std).astype(np.float32)
                    s0_t = torch.from_numpy(s0_batch).to(device)
                    tq_t = torch.from_numpy(t_query).to(device)
                    out = model(s0_t, tq_t)

                    pred_state_chunk = out["state"].cpu().numpy().astype(np.float32)
                    if y_mean is not None and y_std is not None:
                        pred_state_chunk = (pred_state_chunk * y_std + y_mean).astype(np.float32)
                    pred_chunks.append(pred_state_chunk)

                    logit_chunks.append(out["event_logit"].squeeze(1).cpu().numpy().astype(np.float32))
                    gate_chunks.append(out["gate"].squeeze(1).cpu().numpy().astype(np.float32))

                    v_pre_chunk = out["v_pre"].cpu().numpy().astype(np.float32)
                    v_post_chunk = out["v_post"].cpu().numpy().astype(np.float32)
                    if y_mean is not None and y_std is not None:
                        vel_mean = y_mean[:, 2:]
                        vel_std = y_std[:, 2:]
                        v_pre_chunk = (v_pre_chunk * vel_std + vel_mean).astype(np.float32)
                        v_post_chunk = (v_post_chunk * vel_std + vel_mean).astype(np.float32)
                    vpre_chunks.append(v_pre_chunk)
                    vpost_chunks.append(v_post_chunk)

                    # Re-seed next chunk using requested anchor mode.
                    if args.chunk_anchor_mode == "gt":
                        # Boundary after this chunk in the full GT trajectory.
                        gt_idx = min((chunk_idx + 1) * int(args.chunk_steps), true_state_full.shape[0] - 1)
                        s0_chunk = true_state_full[gt_idx].copy()
                    else:
                        s0_chunk = pred_state_chunk[-1].copy()

            pred_state = np.concatenate(pred_chunks, axis=0)
            evt_logit_np = np.concatenate(logit_chunks, axis=0)
            gate_np = np.concatenate(gate_chunks, axis=0)
            v_pre_np = np.concatenate(vpre_chunks, axis=0)
            v_post_np = np.concatenate(vpost_chunks, axis=0)
            true_state = true_state_full[1 : 1 + pred_state.shape[0]]
        else:
            T = pos_true.shape[0]
            s0_batch = np.repeat(s0[None, :], T, axis=0).astype(np.float32)
            if s0_mean is not None and s0_std is not None:
                s0_batch = ((s0_batch - s0_mean) / s0_std).astype(np.float32)
            t_query = (np.arange(T, dtype=np.float32) * dt).astype(np.float32)

            with torch.no_grad():
                s0_t = torch.from_numpy(s0_batch).to(device)
                tq_t = torch.from_numpy(t_query).to(device)
                out = model(s0_t, tq_t)
                pred_state = out["state"].cpu().numpy().astype(np.float32)
                if y_mean is not None and y_std is not None:
                    pred_state = (pred_state * y_std + y_mean).astype(np.float32)
                evt_logit_np = out["event_logit"].squeeze(1).cpu().numpy().astype(np.float32)
                gate_np = out["gate"].squeeze(1).cpu().numpy().astype(np.float32)
                v_pre_np = out["v_pre"].cpu().numpy().astype(np.float32)
                v_post_np = out["v_post"].cpu().numpy().astype(np.float32)
                if y_mean is not None and y_std is not None:
                    # v_pre/v_post are model-space values; map velocity dims back to data-space.
                    vel_mean = y_mean[:, 2:]
                    vel_std = y_std[:, 2:]
                    v_pre_np = (v_pre_np * vel_std + vel_mean).astype(np.float32)
                    v_post_np = (v_post_np * vel_std + vel_mean).astype(np.float32)
            true_state = np.concatenate([pos_true[:, 0, :], vel_true[:, 0, :]], axis=1).astype(np.float32)

        T = pred_state.shape[0]
        # In chunked long-horizon eval, coll_all[e] may only cover the original
        # dataset episode length, so infer collisions from the full GT rollout.
        coll_steps_full = infer_collision_steps_from_velocity(vel_true)
        coll_steps = coll_steps_full - (1 if chunk_mode else 0)
        coll_steps = coll_steps[(coll_steps >= 0) & (coll_steps < T)]
        evt_true = build_event_targets(
            T=T,
            collision_steps=coll_steps,
            dt=dt,
            mode=event_target_mode,
            eps_steps=eps_steps,
            event_window=event_window,
            sigma_event=sigma_event,
        )

        pred_pos = pred_state[:, :2].reshape(T, 1, 2).astype(np.float32)
        true_pos = true_state[:, :2].reshape(T, 1, 2).astype(np.float32)
        pos_err = np.linalg.norm(true_pos[:, 0, :] - pred_pos[:, 0, :], axis=1)
        div_idx = np.where(pos_err > float(args.divergence_threshold))[0]
        diverged = bool(len(div_idx) > 0)
        divergence_step = int(div_idx[0]) if diverged else int(T - 1)

        row: dict[str, Any] = {
            "episode_idx_global": e,
            "mean_err": float(np.mean(pos_err)),
            "max_err": float(np.max(pos_err)),
            "final_err": float(pos_err[-1]),
            "diverged": diverged,
            "divergence_step": int(divergence_step),
        }

        if not args.no_render:
            true_pos_v = true_pos[:: args.frame_stride]
            pred_pos_v = pred_pos[:: args.frame_stride]
            dt_v = dt * float(args.frame_stride)

            if args.save_side_by_side:
                side = animate_side_by_side_1p(
                    pos_true=true_pos_v,
                    pos_pred=pred_pos_v,
                    radius=radius,
                    W=W,
                    H=H,
                    dt=dt_v,
                    title_left=f"GT ({args.split}) ep={e}",
                    title_right=f"TCNO ({args.split}) ep={e}",
                )
                side_path = out_dir / f"{args.split}_ep_{e:05d}_gt_vs_pred_1p.mp4"
                save_animation_mp4(side, str(side_path), fps=args.fps)
                row["video_side_by_side"] = side_path.name

            if args.save_overlay:
                overlay = animate_overlay_gt_perturbed_1p(
                    pos_ref=true_pos_v,
                    pos_pert=pred_pos_v,
                    radius=radius,
                    W=W,
                    H=H,
                    dt=dt_v,
                    title=f"GT vs TCNO ({args.split}) ep={e}",
                    label_ref="GT",
                    label_pert="TCNO",
                )
                overlay_path = out_dir / f"{args.split}_ep_{e:05d}_gt_vs_pred_overlay_1p.mp4"
                save_animation_mp4(overlay, str(overlay_path), fps=args.fps)
                row["video_overlay"] = overlay_path.name

            fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            axs[0].plot(np.arange(T), pos_err, lw=2, color="tab:red")
            axs[0].set_ylabel("||x_pred-x_true||")
            axs[0].grid(True, alpha=0.3)
            evt_prob = 1.0 / (1.0 + np.exp(-np.clip(evt_logit_np, -60.0, 60.0)))
            axs[1].plot(np.arange(T), evt_prob, lw=2, label="pred p(event)")
            axs[1].plot(np.arange(T), evt_true, lw=1.5, alpha=0.75, label="target event")
            if model_cfg.model_variant == "gated_tcno":
                axs[1].plot(np.arange(T), gate_np, lw=1.7, alpha=0.9, label="gate")
            axs[1].set_ylabel("event")
            axs[1].set_xlabel("step")
            axs[1].set_ylim([-0.05, 1.05])
            axs[1].grid(True, alpha=0.3)
            axs[1].legend(loc="best")
            plt.tight_layout()
            plot_path = out_dir / f"{args.split}_ep_{e:05d}_error_and_event.png"
            plt.savefig(plot_path, dpi=140)
            plt.close(fig)
            row["error_event_plot"] = plot_path.name

            # Main state/event diagnostic for TCNO: monitor position, velocity, and event quality.
            fig_state, axs_state = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
            axs_state[0].plot(np.arange(T), true_state[:, 0], lw=2, label="true")
            axs_state[0].plot(np.arange(T), pred_state[:, 0], lw=1.8, alpha=0.9, label="pred")
            axs_state[0].set_ylabel("x(t)")
            axs_state[0].legend(loc="best")
            axs_state[0].grid(True, alpha=0.3)

            axs_state[1].plot(np.arange(T), true_state[:, 1], lw=2, label="true")
            axs_state[1].plot(np.arange(T), pred_state[:, 1], lw=1.8, alpha=0.9, label="pred")
            axs_state[1].set_ylabel("y(t)")
            axs_state[1].grid(True, alpha=0.3)

            axs_state[2].plot(np.arange(T), true_state[:, 2], lw=2, label="true")
            axs_state[2].plot(np.arange(T), pred_state[:, 2], lw=1.8, alpha=0.9, label="pred")
            axs_state[2].set_ylabel("vx(t)")
            axs_state[2].grid(True, alpha=0.3)

            axs_state[3].plot(np.arange(T), true_state[:, 3], lw=2, label="true")
            axs_state[3].plot(np.arange(T), pred_state[:, 3], lw=1.8, alpha=0.9, label="pred")
            axs_state[3].set_ylabel("vy(t)")
            axs_state[3].grid(True, alpha=0.3)

            axs_state[4].plot(np.arange(T), evt_prob, lw=2, label="pred p(event)")
            axs_state[4].plot(np.arange(T), evt_true, lw=1.6, alpha=0.8, label="event target")
            if model_cfg.model_variant == "gated_tcno":
                axs_state[4].plot(np.arange(T), gate_np, lw=1.6, alpha=0.8, label="gate")
            axs_state[4].axhline(float(args.event_prob_threshold), color="k", ls="--", lw=1.0, alpha=0.6)
            axs_state[4].set_ylabel("event")
            axs_state[4].set_xlabel("step")
            axs_state[4].set_ylim([-0.05, 1.05])
            axs_state[4].grid(True, alpha=0.3)
            axs_state[4].legend(loc="best")

            fig_state.suptitle(
                f"TCNO ({args.split}) ep={e}: state and event diagnostics",
                y=0.995,
                fontsize=11,
            )
            plt.tight_layout()
            state_plot_path = out_dir / f"{args.split}_ep_{e:05d}_state_and_event.png"
            plt.savefig(state_plot_path, dpi=140)
            plt.close(fig_state)
            row["state_event_plot"] = state_plot_path.name

            if model_cfg.model_variant == "gated_tcno":
                fig_dbg, dbg_axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
                dbg_axs[0].plot(np.arange(T), v_pre_np[:, 0], label="v_pre_x", alpha=0.9)
                dbg_axs[0].plot(np.arange(T), v_post_np[:, 0], label="v_post_x", alpha=0.9)
                dbg_axs[0].plot(np.arange(T), true_state[:, 2], label="true_vx", lw=1.8)
                dbg_axs[0].legend(loc="best")
                dbg_axs[0].grid(True, alpha=0.3)
                dbg_axs[1].plot(np.arange(T), v_pre_np[:, 1], label="v_pre_y", alpha=0.9)
                dbg_axs[1].plot(np.arange(T), v_post_np[:, 1], label="v_post_y", alpha=0.9)
                dbg_axs[1].plot(np.arange(T), true_state[:, 3], label="true_vy", lw=1.8)
                dbg_axs[1].legend(loc="best")
                dbg_axs[1].grid(True, alpha=0.3)
                dbg_axs[2].plot(np.arange(T), gate_np, label="gate", lw=1.8)
                dbg_axs[2].plot(np.arange(T), evt_true, label="event target", lw=1.4, alpha=0.8)
                dbg_axs[2].set_xlabel("step")
                dbg_axs[2].legend(loc="best")
                dbg_axs[2].grid(True, alpha=0.3)
                plt.tight_layout()
                dbg_path = out_dir / f"{args.split}_ep_{e:05d}_velocity_heads_and_gate.png"
                plt.savefig(dbg_path, dpi=140)
                plt.close(fig_dbg)
                row["velocity_heads_plot"] = dbg_path.name

        rows.append(row)
        all_pred_state.append(pred_state)
        all_true_state.append(true_state)
        all_evt_logits.append(evt_logit_np)
        all_evt_true.append(evt_true)
        all_gate.append(gate_np)
        print(
            f"[{idx}/{len(eval_eps)}] ep={e} mean_err={row['mean_err']:.6f} "
            f"max_err={row['max_err']:.6f} final_err={row['final_err']:.6f} "
            f"ttf={row['divergence_step']}"
        )

    pred_all = np.concatenate(all_pred_state, axis=0)
    true_all = np.concatenate(all_true_state, axis=0)
    logits_all = np.concatenate(all_evt_logits, axis=0)
    evt_all = np.concatenate(all_evt_true, axis=0)

    state_mse = float(np.mean((pred_all - true_all) ** 2))
    pos_mse = float(np.mean((pred_all[:, :2] - true_all[:, :2]) ** 2))
    vel_mse = float(np.mean((pred_all[:, 2:] - true_all[:, 2:]) ** 2))
    plateau_mask = evt_all < float(args.plateau_event_threshold)
    plateau_vel_mse = (
        float(np.mean((pred_all[plateau_mask, 2:] - true_all[plateau_mask, 2:]) ** 2))
        if np.any(plateau_mask)
        else vel_mse
    )
    evt_metrics = compute_event_metrics(
        logits=logits_all,
        labels=evt_all,
        threshold=float(args.event_prob_threshold),
    )
    sign_metrics = compute_sign_accuracy(pred_all[:, 2:], true_all[:, 2:], sign_epsilon=float(args.sign_epsilon))

    div_steps = np.array([r["divergence_step"] for r in rows], dtype=np.float64)
    summary = {
        "split": args.split,
        "num_episodes": int(len(rows)),
        "rollout_steps": int(rollout_steps),
        "chunk_mode": bool(chunk_mode),
        "chunk_steps": int(args.chunk_steps),
        "num_chunks": int(args.num_chunks),
        "chunk_anchor_mode": str(args.chunk_anchor_mode),
        "divergence_threshold": float(args.divergence_threshold),
        "ttf_median": float(np.median(div_steps)),
        "ttf_p10": float(np.percentile(div_steps, 10)),
        "divergence_rate": float(np.mean([1.0 if r["diverged"] else 0.0 for r in rows])),
        "state_mse": state_mse,
        "position_mse": pos_mse,
        "velocity_mse": vel_mse,
        "plateau_velocity_mse": plateau_vel_mse,
        **evt_metrics,
        **sign_metrics,
        "rows": rows,
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "best_val_state_mse": float(ckpt.get("best_val_state_mse", float("nan"))),
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Output dir:", out_dir)
    print("Summary:", out_dir / "eval_summary.json")
    print(
        f"Aggregate | state_mse={summary['state_mse']:.6f} pos_mse={summary['position_mse']:.6f} "
        f"vel_mse={summary['velocity_mse']:.6f} vel_mse_plateau={summary['plateau_velocity_mse']:.6f} "
        f"evt_acc={summary['event_accuracy']:.4f} "
        f"sign_vx={summary['sign_acc_vx']:.4f} sign_vy={summary['sign_acc_vy']:.4f} "
        f"| ttf_median={summary['ttf_median']:.2f} ttf_p10={summary['ttf_p10']:.2f} "
        f"divergence_rate={summary['divergence_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
