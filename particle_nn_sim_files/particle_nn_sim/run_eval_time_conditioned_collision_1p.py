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
    if mode == "spike":
        valid = collision_steps[(collision_steps >= 0) & (collision_steps < T)].astype(np.int64)
        labels[valid] = 1.0
        return labels

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


def renormalize_velocity_in_state(state: np.ndarray, target_speed: float, eps: float = 1e-12) -> np.ndarray:
    """Return a copy of [x,y,vx,vy] with velocity magnitude set to target_speed."""
    out = state.copy()
    if target_speed <= 0.0:
        return out
    vx, vy = float(out[2]), float(out[3])
    vnorm = float(np.hypot(vx, vy))
    if vnorm <= eps:
        return out
    scale = float(target_speed) / vnorm
    out[2] = np.float32(vx * scale)
    out[3] = np.float32(vy * scale)
    return out


def renormalize_velocity_batch(states: np.ndarray, target_speed: float, eps: float = 1e-12) -> np.ndarray:
    """Return a copy of (T,4) states with velocity magnitude fixed to target_speed."""
    out = states.copy()
    if target_speed <= 0.0:
        return out
    v = out[:, 2:4]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    mask = norms > eps
    scale = np.ones_like(norms, dtype=np.float32)
    scale[mask] = np.float32(target_speed) / norms[mask]
    out[:, 2:4] = v * scale
    return out.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate TimeConditionedCollisionModel checkpoint on train/val/test split."
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_tc_collision_1p.pt")
    p.add_argument(
        "--single-ic",
        type=str2bool,
        default=False,
        help="If true, ignore dataset split and evaluate one user-provided initial condition.",
    )
    p.add_argument("--ic-x", type=float, default=None, help="Initial x in [0,1] for --single-ic mode.")
    p.add_argument("--ic-y", type=float, default=None, help="Initial y in [0,1] for --single-ic mode.")
    p.add_argument("--ic-vx", type=float, default=None, help="Initial vx for --single-ic mode.")
    p.add_argument("--ic-vy", type=float, default=None, help="Initial vy for --single-ic mode.")
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
        choices=["pred", "gt", "nnref"],
        help=(
            "Chunk re-anchoring mode when chunk rollout is enabled: "
            "'pred' = next chunk starts from previous chunk's last prediction, "
            "'gt' = next chunk starts from GT state at the chunk boundary, "
            "'nnref' = GT is re-simulated from NN boundary state at each chunk."
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
    p.add_argument(
        "--report-renorm-velocity",
        type=str2bool,
        default=False,
        help="If true, also report/plot renormalized predicted velocities.",
    )
    p.add_argument(
        "--report-renorm-speed",
        type=float,
        default=0.5,
        help="Target speed used when --report-renorm-velocity=true.",
    )
    p.add_argument(
        "--renorm-speed",
        type=float,
        default=0.0,
        help=(
            "If > 0, renormalize chunk boundary velocity to this magnitude "
            "before feeding it into the next chunk."
        ),
    )
    p.add_argument("--plateau-event-threshold", type=float, default=0.1)
    p.add_argument(
        "--anchor-steps",
        type=str,
        default="",
        help="Comma-separated anchor steps for same-window comparison, e.g. '0,100,250'.",
    )
    p.add_argument("--compare-window-start", type=int, default=-1, help="Global start step for anchor-window compare.")
    p.add_argument("--compare-window-end", type=int, default=-1, help="Global end step (inclusive) for anchor-window compare.")
    p.add_argument(
        "--anchor-source",
        type=str,
        default="gt",
        choices=["gt", "pred"],
        help="Source of anchor states for anchor-window compare: ground-truth or model-predicted trajectory.",
    )
    p.add_argument("--out-dir", type=str, default="checkpoints/eval_tc_collision_1p")
    return p.parse_args()


def _predict_window_from_anchor(
    *,
    model: TimeConditionedCollisionModel,
    anchor_state: np.ndarray,
    query_steps_from_anchor: np.ndarray,
    dt: float,
    s0_mean: np.ndarray | None,
    s0_std: np.ndarray | None,
    y_mean: np.ndarray | None,
    y_std: np.ndarray | None,
    device: str,
) -> np.ndarray:
    t_query = (query_steps_from_anchor.astype(np.float32) * np.float32(dt)).astype(np.float32)
    s0_batch = np.repeat(anchor_state[None, :], len(t_query), axis=0).astype(np.float32)
    if s0_mean is not None and s0_std is not None:
        s0_batch = ((s0_batch - s0_mean) / s0_std).astype(np.float32)
    with torch.no_grad():
        s0_t = torch.from_numpy(s0_batch).to(device)
        tq_t = torch.from_numpy(t_query).to(device)
        out = model(s0_t, tq_t)
        pred = out["state"].cpu().numpy().astype(np.float32)
        if y_mean is not None and y_std is not None:
            pred = (pred * y_std + y_mean).astype(np.float32)
    return pred


def run_single_ic_eval(
    *,
    model: TimeConditionedCollisionModel,
    args: argparse.Namespace,
    ckpt: dict[str, Any],
    out_dir: Path,
    dt: float,
    W: float,
    H: float,
    radius: float,
    mass: float,
    restitution: float,
    wall_mode: str,
    s0_mean: np.ndarray | None,
    s0_std: np.ndarray | None,
    y_mean: np.ndarray | None,
    y_std: np.ndarray | None,
    device: str,
) -> None:
    if any(v is None for v in (args.ic_x, args.ic_y, args.ic_vx, args.ic_vy)):
        raise ValueError("--single-ic requires --ic-x --ic-y --ic-vx --ic-vy.")

    rollout_steps = int(ckpt["config"]["data"]["steps"]) if args.rollout_steps < 0 else int(args.rollout_steps)
    chunk_mode = args.chunk_steps > 0 and args.num_chunks > 0
    if chunk_mode:
        rollout_steps = int(args.chunk_steps) * int(args.num_chunks)
    if rollout_steps < 1:
        raise ValueError("--rollout-steps must be >= 1 in --single-ic mode.")

    s0 = np.array([args.ic_x, args.ic_y, args.ic_vx, args.ic_vy], dtype=np.float32)
    if chunk_mode:
        # Autoregressive-by-chunks: query chunk horizon, feed final prediction as next s0.
        t_query_chunk = (np.arange(1, int(args.chunk_steps) + 1, dtype=np.float32) * float(dt)).astype(np.float32)
        pred_chunks: list[np.ndarray] = []
        evt_chunks: list[np.ndarray] = []
        s0_chunk = s0.copy()
        with torch.no_grad():
            for _chunk_idx in range(int(args.num_chunks)):
                s0_batch = np.repeat(s0_chunk[None, :], int(args.chunk_steps), axis=0).astype(np.float32)
                if s0_mean is not None and s0_std is not None:
                    s0_batch = ((s0_batch - s0_mean) / s0_std).astype(np.float32)
                s0_t = torch.from_numpy(s0_batch).to(device)
                tq_t = torch.from_numpy(t_query_chunk).to(device)
                out = model(s0_t, tq_t)
                pred_chunk = out["state"].cpu().numpy().astype(np.float32)
                if y_mean is not None and y_std is not None:
                    pred_chunk = (pred_chunk * y_std + y_mean).astype(np.float32)
                evt_chunk = out["event_logit"].squeeze(1).cpu().numpy().astype(np.float32)
                pred_chunks.append(pred_chunk)
                evt_chunks.append(evt_chunk)

                s0_chunk = pred_chunk[-1].copy()
                if args.renorm_speed > 0.0:
                    s0_chunk = renormalize_velocity_in_state(s0_chunk, float(args.renorm_speed))

        pred_no_t0 = np.concatenate(pred_chunks, axis=0).astype(np.float32)
        evt_no_t0 = np.concatenate(evt_chunks, axis=0).astype(np.float32)
        pred_state = np.concatenate([s0[None, :], pred_no_t0], axis=0).astype(np.float32)
        evt_logit_np = np.concatenate([np.array([evt_no_t0[0]], dtype=np.float32), evt_no_t0], axis=0).astype(np.float32)
    else:
        T = int(rollout_steps + 1)  # include t=0
        t_query = (np.arange(T, dtype=np.float32) * float(dt)).astype(np.float32)
        s0_batch = np.repeat(s0[None, :], T, axis=0).astype(np.float32)
        if s0_mean is not None and s0_std is not None:
            s0_batch = ((s0_batch - s0_mean) / s0_std).astype(np.float32)

        with torch.no_grad():
            s0_t = torch.from_numpy(s0_batch).to(device)
            tq_t = torch.from_numpy(t_query).to(device)
            out = model(s0_t, tq_t)
            pred_state = out["state"].cpu().numpy().astype(np.float32)
            if y_mean is not None and y_std is not None:
                pred_state = (pred_state * y_std + y_mean).astype(np.float32)
            evt_logit_np = out["event_logit"].squeeze(1).cpu().numpy().astype(np.float32)

    T = pred_state.shape[0]
    evt_prob = 1.0 / (1.0 + np.exp(-np.clip(evt_logit_np, -60.0, 60.0)))
    pred_state_renorm = (
        renormalize_velocity_batch(pred_state, float(args.report_renorm_speed))
        if args.report_renorm_velocity
        else None
    )

    # Build GT from the same IC for optional overlay/error diagnostics.
    sim_true = ParticleSim2D(
        W=W,
        H=H,
        radii=[radius],
        masses=[mass],
        restitution=restitution,
        seed=123456,
        wall_mode=wall_mode,
    )
    sim_true.reset(s0[:2].reshape(1, 2), s0[2:].reshape(1, 2))
    pos_true, vel_true = sim_true.rollout(dt=dt, steps=int(rollout_steps))
    true_state = np.concatenate([pos_true[:, 0, :], vel_true[:, 0, :]], axis=1).astype(np.float32)
    anchor_steps: list[int] = []
    if args.anchor_steps.strip():
        anchor_steps = [int(x.strip()) for x in args.anchor_steps.split(",") if x.strip()]
    pred_pos = pred_state[:, :2].reshape(T, 1, 2).astype(np.float32)
    true_pos = true_state[:, :2].reshape(T, 1, 2).astype(np.float32)

    final_pred = pred_state[-1].tolist()
    final_pred_renorm = pred_state_renorm[-1].tolist() if pred_state_renorm is not None else None
    final_true = true_state[-1].tolist()
    print(
        "Single-IC final state @ step "
        f"{rollout_steps}: pred=[{final_pred[0]:.6f}, {final_pred[1]:.6f}, {final_pred[2]:.6f}, {final_pred[3]:.6f}] "
        f"true=[{final_true[0]:.6f}, {final_true[1]:.6f}, {final_true[2]:.6f}, {final_true[3]:.6f}]"
    )
    if final_pred_renorm is not None:
        print(
            "Single-IC final state (renorm vel) @ step "
            f"{rollout_steps}: pred_renorm=[{final_pred_renorm[0]:.6f}, {final_pred_renorm[1]:.6f}, "
            f"{final_pred_renorm[2]:.6f}, {final_pred_renorm[3]:.6f}]"
        )

    anchor_compare_payload: dict[str, Any] | None = None
    if anchor_steps and args.compare_window_start >= 0:
        ws = int(args.compare_window_start)
        we = int(args.compare_window_end)
        max_step = int(true_state.shape[0] - 1)
        ws = max(0, min(ws, max_step))
        we = max(ws, min(we, max_step))
        gt_win = true_state[ws : we + 1].astype(np.float32)
        anchor_results: dict[str, Any] = {}
        fig_cmp, axs_cmp = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        x_axis = np.arange(ws, we + 1, dtype=np.int64)
        axs_cmp[0].plot(x_axis, gt_win[:, 0], lw=2, label="GT x")
        axs_cmp[1].plot(x_axis, gt_win[:, 1], lw=2, label="GT y")
        anchor_state_source = pred_state if args.anchor_source == "pred" else true_state
        for a in anchor_steps:
            if a > ws or a > max_step:
                continue
            q = np.arange(ws, we + 1, dtype=np.int64) - int(a)
            pred_win = _predict_window_from_anchor(
                model=model,
                anchor_state=anchor_state_source[int(a)].astype(np.float32),
                query_steps_from_anchor=q,
                dt=dt,
                s0_mean=s0_mean,
                s0_std=s0_std,
                y_mean=y_mean,
                y_std=y_std,
                device=device,
            )
            anchor_results[str(a)] = {
                "state_mse": float(np.mean((pred_win - gt_win) ** 2)),
                "position_mse": float(np.mean((pred_win[:, :2] - gt_win[:, :2]) ** 2)),
                "velocity_mse": float(np.mean((pred_win[:, 2:] - gt_win[:, 2:]) ** 2)),
            }
            axs_cmp[0].plot(x_axis, pred_win[:, 0], lw=1.3, alpha=0.9, label=f"pred@a={a}({args.anchor_source})")
            axs_cmp[1].plot(x_axis, pred_win[:, 1], lw=1.3, alpha=0.9, label=f"pred@a={a}({args.anchor_source})")
            if (not args.no_render) and args.save_overlay:
                gt_pos_win = gt_win[:, :2].reshape(gt_win.shape[0], 1, 2).astype(np.float32)
                pred_pos_win = pred_win[:, :2].reshape(pred_win.shape[0], 1, 2).astype(np.float32)
                overlay_cmp = animate_overlay_gt_perturbed_1p(
                    pos_ref=gt_pos_win[:: args.frame_stride],
                    pos_pert=pred_pos_win[:: args.frame_stride],
                    radius=radius,
                    W=W,
                    H=H,
                    dt=float(dt) * float(args.frame_stride),
                    title=f"Single-IC anchor compare [{ws},{we}] a={a} ({args.anchor_source})",
                    label_ref="GT",
                    label_pert=f"pred@a={a}",
                )
                overlay_name = f"single_ic_anchor_{int(a):04d}_{args.anchor_source}_window_{ws}_{we}_overlay.mp4"
                save_animation_mp4(overlay_cmp, str(out_dir / overlay_name), fps=args.fps)
                anchor_results[str(a)]["video_overlay"] = overlay_name
        axs_cmp[0].set_ylabel("x")
        axs_cmp[1].set_ylabel("y")
        axs_cmp[1].set_xlabel("global step")
        axs_cmp[0].grid(True, alpha=0.3)
        axs_cmp[1].grid(True, alpha=0.3)
        axs_cmp[0].legend(loc="best", fontsize=8)
        axs_cmp[1].legend(loc="best", fontsize=8)
        plt.tight_layout()
        cmp_plot = out_dir / "single_ic_anchor_window_compare.png"
        plt.savefig(cmp_plot, dpi=140)
        plt.close(fig_cmp)
        anchor_compare_payload = {
            "window_start": ws,
            "window_end": we,
            "anchor_source": str(args.anchor_source),
            "anchors": anchor_results,
            "plot": cmp_plot.name,
        }

    # Save overlay video and compact diagnostics.
    if not args.no_render and args.save_overlay:
        overlay = animate_overlay_gt_perturbed_1p(
            pos_ref=true_pos[:: args.frame_stride],
            pos_pert=pred_pos[:: args.frame_stride],
            radius=radius,
            W=W,
            H=H,
            dt=float(dt) * float(args.frame_stride),
            title=f"GT vs TCNO (single-ic) step={rollout_steps}",
            label_ref="GT",
            label_pert="TCNO",
        )
        save_animation_mp4(overlay, str(out_dir / "single_ic_gt_vs_pred_overlay_1p.mp4"), fps=args.fps)

    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    idx = np.arange(T)
    axs[0].plot(idx, true_state[:, 0], lw=2, label="true")
    axs[0].plot(idx, pred_state[:, 0], lw=1.8, alpha=0.9, label="pred")
    axs[0].set_ylabel("x(t)")
    axs[0].legend(loc="best")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(idx, true_state[:, 1], lw=2, label="true")
    axs[1].plot(idx, pred_state[:, 1], lw=1.8, alpha=0.9, label="pred")
    axs[1].set_ylabel("y(t)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(idx, true_state[:, 2], lw=2, label="true")
    axs[2].plot(idx, pred_state[:, 2], lw=1.8, alpha=0.9, label="pred")
    if pred_state_renorm is not None:
        axs[2].plot(idx, pred_state_renorm[:, 2], lw=1.2, ls="--", alpha=0.9, label="pred_renorm")
    axs[2].set_ylabel("vx(t)")
    axs[2].legend(loc="best")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(idx, true_state[:, 3], lw=2, label="true")
    axs[3].plot(idx, pred_state[:, 3], lw=1.8, alpha=0.9, label="pred")
    if pred_state_renorm is not None:
        axs[3].plot(idx, pred_state_renorm[:, 3], lw=1.2, ls="--", alpha=0.9, label="pred_renorm")
    axs[3].set_ylabel("vy(t)")
    axs[3].legend(loc="best")
    axs[3].grid(True, alpha=0.3)

    axs[4].plot(idx, evt_prob, lw=2, label="pred p(event)")
    axs[4].axhline(float(args.event_prob_threshold), color="k", ls="--", lw=1.0, alpha=0.6)
    axs[4].set_ylabel("event")
    axs[4].set_xlabel("step")
    axs[4].set_ylim([-0.05, 1.05])
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="best")
    speed_true = np.linalg.norm(true_state[:, 2:4], axis=1)
    speed_pred = np.linalg.norm(pred_state[:, 2:4], axis=1)
    axs[5].plot(idx, speed_true, lw=2, label="|v| true")
    axs[5].plot(idx, speed_pred, lw=1.8, alpha=0.9, label="|v| pred")
    if pred_state_renorm is not None:
        speed_pred_renorm = np.linalg.norm(pred_state_renorm[:, 2:4], axis=1)
        axs[5].plot(idx, speed_pred_renorm, lw=1.2, ls="--", alpha=0.9, label="|v| pred_renorm")
    axs[5].set_ylabel("|v|(t)")
    axs[5].set_xlabel("step")
    axs[5].grid(True, alpha=0.3)
    axs[5].legend(loc="best")
    fig.suptitle("TCNO single-IC diagnostics", y=0.995, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "single_ic_state_and_event.png", dpi=140)
    plt.close(fig)

    summary = {
        "mode": "single_ic",
        "rollout_steps": int(rollout_steps),
        "dt": float(dt),
        "initial_state": [float(x) for x in s0.tolist()],
        "final_state_pred": [float(x) for x in pred_state[-1].tolist()],
        "final_state_pred_renorm": (
            [float(x) for x in pred_state_renorm[-1].tolist()] if pred_state_renorm is not None else None
        ),
        "final_state_true": [float(x) for x in true_state[-1].tolist()],
        "state_mse": float(np.mean((pred_state - true_state) ** 2)),
        "position_mse": float(np.mean((pred_state[:, :2] - true_state[:, :2]) ** 2)),
        "velocity_mse": float(np.mean((pred_state[:, 2:] - true_state[:, 2:]) ** 2)),
        "files": {
            "overlay_video": "single_ic_gt_vs_pred_overlay_1p.mp4" if (not args.no_render and args.save_overlay) else None,
            "state_event_plot": "single_ic_state_and_event.png",
        },
        "report_renorm_velocity": bool(args.report_renorm_velocity),
        "report_renorm_speed": float(args.report_renorm_speed),
        "chunk_mode": bool(chunk_mode),
        "chunk_steps": int(args.chunk_steps),
        "num_chunks": int(args.num_chunks),
        "boundary_renorm_speed": float(args.renorm_speed),
        "anchor_window_compare": anchor_compare_payload,
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "best_val_state_mse": float(ckpt.get("best_val_state_mse", float("nan"))),
    }
    with open(out_dir / "single_ic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", out_dir / "single_ic_summary.json")


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
    if args.renorm_speed < 0.0:
        raise ValueError("--renorm-speed must be >= 0")
    if args.report_renorm_speed < 0.0:
        raise ValueError("--report-renorm-speed must be >= 0")
    if (args.compare_window_start >= 0) ^ (args.compare_window_end >= 0):
        raise ValueError("Provide both --compare-window-start and --compare-window-end (or leave both < 0).")
    if args.compare_window_start >= 0 and args.compare_window_end < args.compare_window_start:
        raise ValueError("--compare-window-end must be >= --compare-window-start")

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
        enforce_t0_anchor=bool(model_cfg_raw.get("enforce_t0_anchor", True)),
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

    if args.single_ic:
        run_single_ic_eval(
            model=model,
            args=args,
            ckpt=ckpt,
            out_dir=out_dir,
            dt=dt,
            W=W,
            H=H,
            radius=radius,
            mass=mass,
            restitution=restitution,
            wall_mode=wall_mode,
            s0_mean=s0_mean,
            s0_std=s0_std,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
        )
        return

    rows: list[dict[str, Any]] = []
    all_pred_state: list[np.ndarray] = []
    all_true_state: list[np.ndarray] = []
    all_evt_true: list[np.ndarray] = []

    eps_steps = int(data_cfg.get("coll_epsilon_steps", 2))
    event_target_mode = str(data_cfg.get("event_target_mode", "window"))
    event_window = float(data_cfg.get("event_window", 0.05))
    sigma_event = float(data_cfg.get("sigma_event", 0.03))
    anchor_steps: list[int] = []
    if args.anchor_steps.strip():
        anchor_steps = [int(x.strip()) for x in args.anchor_steps.split(",") if x.strip()]
        if any(a < 0 for a in anchor_steps):
            raise ValueError("--anchor-steps must be non-negative integers.")
    compare_mode = bool(anchor_steps) and args.compare_window_start >= 0

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
        # Chunked mode predicts states at t=dt..t=chunk_steps*dt for each chunk.
        # For pred/gt anchors, we align against GT[1:] from a rollout of (rollout_steps + 1).
        # For nnref, GT is re-simulated chunk-by-chunk from NN boundary states.
        true_steps = int(rollout_steps + 1) if chunk_mode else int(rollout_steps)
        pos_true, vel_true = sim_true.rollout(dt=dt, steps=true_steps)
        pos_true = pos_true.astype(np.float32)
        vel_true = vel_true.astype(np.float32)
        s0 = np.concatenate([pos_true[0, 0], vel_true[0, 0]], axis=0).astype(np.float32)

        if chunk_mode:
            pred_chunks: list[np.ndarray] = []
            logit_chunks: list[np.ndarray] = []
            s0_chunk = s0.copy()
            t_query = (np.arange(1, int(args.chunk_steps) + 1, dtype=np.float32) * dt).astype(np.float32)

            true_state_full = np.concatenate([pos_true[:, 0, :], vel_true[:, 0, :]], axis=1).astype(np.float32)
            true_chunks: list[np.ndarray] = []

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
                    # Re-seed next chunk using requested anchor mode.
                    if args.chunk_anchor_mode == "gt":
                        # Boundary after this chunk in the full GT trajectory.
                        gt_idx = min((chunk_idx + 1) * int(args.chunk_steps), true_state_full.shape[0] - 1)
                        s0_chunk = true_state_full[gt_idx].copy()
                    else:
                        s0_chunk = pred_state_chunk[-1].copy()
                        if args.renorm_speed > 0.0:
                            s0_chunk = renormalize_velocity_in_state(s0_chunk, float(args.renorm_speed))

            pred_state = np.concatenate(pred_chunks, axis=0)
            evt_logit_np = np.concatenate(logit_chunks, axis=0)
            if args.chunk_anchor_mode == "nnref":
                # Re-simulate GT chunk-by-chunk from NN boundary states.
                s0_ref = s0.copy()
                for chunk_idx in range(int(args.num_chunks)):
                    sim_ref = ParticleSim2D(
                        W=W,
                        H=H,
                        radii=[radius],
                        masses=[mass],
                        restitution=restitution,
                        seed=20_000 + e * 100 + chunk_idx,
                        wall_mode=wall_mode,
                    )
                    sim_ref.reset(s0_ref[:2].reshape(1, 2), s0_ref[2:].reshape(1, 2))
                    # ParticleSim2D.rollout returns steps+1 states including t=0.
                    # To get exactly `chunk_steps` targets after dropping t=0,
                    # request `steps=chunk_steps`.
                    pos_ref, vel_ref = sim_ref.rollout(dt=dt, steps=int(args.chunk_steps))
                    true_chunk = np.concatenate([pos_ref[1:, 0, :], vel_ref[1:, 0, :]], axis=1).astype(np.float32)
                    true_chunks.append(true_chunk)
                    s0_ref = pred_chunks[chunk_idx][-1].copy()
                true_state = np.concatenate(true_chunks, axis=0)
            else:
                true_state = true_state_full[1 : 1 + pred_state.shape[0]]
            pred_state_full = np.concatenate([s0[None, :], pred_state], axis=0).astype(np.float32)
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
            true_state = np.concatenate([pos_true[:, 0, :], vel_true[:, 0, :]], axis=1).astype(np.float32)
            true_state_full = true_state
            pred_state_full = pred_state

        T = pred_state.shape[0]
        if chunk_mode and args.chunk_anchor_mode == "nnref":
            coll_steps = infer_collision_steps_from_velocity(
                true_state[:, 2:].reshape(T, 1, 2).astype(np.float32)
            )
            coll_steps = coll_steps[(coll_steps >= 0) & (coll_steps < T)]
        else:
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
        pred_state_renorm = (
            renormalize_velocity_batch(pred_state, float(args.report_renorm_speed))
            if args.report_renorm_velocity
            else None
        )
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

        if compare_mode:
            ws = int(args.compare_window_start)
            we = int(args.compare_window_end)
            max_step = int(true_state_full.shape[0] - 1)
            ws = max(0, min(ws, max_step))
            we = max(ws, min(we, max_step))
            gt_win = true_state_full[ws : we + 1].astype(np.float32)
            anchor_results: dict[str, Any] = {}
            fig_cmp, axs_cmp = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            x_axis = np.arange(ws, we + 1, dtype=np.int64)
            axs_cmp[0].plot(x_axis, gt_win[:, 0], lw=2, label="GT x")
            axs_cmp[1].plot(x_axis, gt_win[:, 1], lw=2, label="GT y")
            anchor_state_source = pred_state_full if args.anchor_source == "pred" else true_state_full
            for a in anchor_steps:
                if a > ws:
                    continue
                if a > max_step:
                    continue
                q = np.arange(ws, we + 1, dtype=np.int64) - int(a)
                pred_win = _predict_window_from_anchor(
                    model=model,
                    anchor_state=anchor_state_source[int(a)].astype(np.float32),
                    query_steps_from_anchor=q,
                    dt=dt,
                    s0_mean=s0_mean,
                    s0_std=s0_std,
                    y_mean=y_mean,
                    y_std=y_std,
                    device=device,
                )
                mse_state = float(np.mean((pred_win - gt_win) ** 2))
                mse_pos = float(np.mean((pred_win[:, :2] - gt_win[:, :2]) ** 2))
                mse_vel = float(np.mean((pred_win[:, 2:] - gt_win[:, 2:]) ** 2))
                anchor_results[str(a)] = {
                    "state_mse": mse_state,
                    "position_mse": mse_pos,
                    "velocity_mse": mse_vel,
                }
                axs_cmp[0].plot(x_axis, pred_win[:, 0], lw=1.3, alpha=0.9, label=f"pred@a={a}({args.anchor_source})")
                axs_cmp[1].plot(x_axis, pred_win[:, 1], lw=1.3, alpha=0.9, label=f"pred@a={a}({args.anchor_source})")
                if (not args.no_render) and args.save_overlay:
                    gt_pos_win = gt_win[:, :2].reshape(gt_win.shape[0], 1, 2).astype(np.float32)
                    pred_pos_win = pred_win[:, :2].reshape(pred_win.shape[0], 1, 2).astype(np.float32)
                    overlay_cmp = animate_overlay_gt_perturbed_1p(
                        pos_ref=gt_pos_win[:: args.frame_stride],
                        pos_pert=pred_pos_win[:: args.frame_stride],
                        radius=radius,
                        W=W,
                        H=H,
                        dt=float(dt) * float(args.frame_stride),
                        title=f"{args.split} ep={e} anchor compare [{ws},{we}] a={a} ({args.anchor_source})",
                        label_ref="GT",
                        label_pert=f"pred@a={a}",
                    )
                    overlay_name = (
                        f"{args.split}_ep_{e:05d}_anchor_{int(a):04d}_{args.anchor_source}_"
                        f"window_{ws}_{we}_overlay.mp4"
                    )
                    save_animation_mp4(overlay_cmp, str(out_dir / overlay_name), fps=args.fps)
                    anchor_results[str(a)]["video_overlay"] = overlay_name
            axs_cmp[0].set_ylabel("x")
            axs_cmp[1].set_ylabel("y")
            axs_cmp[1].set_xlabel("global step")
            axs_cmp[0].grid(True, alpha=0.3)
            axs_cmp[1].grid(True, alpha=0.3)
            axs_cmp[0].legend(loc="best", fontsize=8)
            axs_cmp[1].legend(loc="best", fontsize=8)
            plt.tight_layout()
            cmp_plot = out_dir / f"{args.split}_ep_{e:05d}_anchor_window_compare.png"
            plt.savefig(cmp_plot, dpi=140)
            plt.close(fig_cmp)
            row["anchor_window_compare"] = {
                "window_start": ws,
                "window_end": we,
                "anchor_source": str(args.anchor_source),
                "anchors": anchor_results,
                "plot": cmp_plot.name,
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
            if pred_state_renorm is not None:
                axs_state[2].plot(np.arange(T), pred_state_renorm[:, 2], lw=1.2, ls="--", alpha=0.9, label="pred_renorm")
            axs_state[2].set_ylabel("vx(t)")
            axs_state[2].legend(loc="best")
            axs_state[2].grid(True, alpha=0.3)

            axs_state[3].plot(np.arange(T), true_state[:, 3], lw=2, label="true")
            axs_state[3].plot(np.arange(T), pred_state[:, 3], lw=1.8, alpha=0.9, label="pred")
            if pred_state_renorm is not None:
                axs_state[3].plot(np.arange(T), pred_state_renorm[:, 3], lw=1.2, ls="--", alpha=0.9, label="pred_renorm")
            axs_state[3].set_ylabel("vy(t)")
            axs_state[3].legend(loc="best")
            axs_state[3].grid(True, alpha=0.3)

            axs_state[4].plot(np.arange(T), evt_prob, lw=2, label="pred p(event)")
            axs_state[4].plot(np.arange(T), evt_true, lw=1.6, alpha=0.8, label="event target")
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

        rows.append(row)
        all_pred_state.append(pred_state)
        all_true_state.append(true_state)
        all_evt_true.append(evt_true)
        print(
            f"[{idx}/{len(eval_eps)}] ep={e} mean_err={row['mean_err']:.6f} "
            f"max_err={row['max_err']:.6f} final_err={row['final_err']:.6f} "
            f"ttf={row['divergence_step']}"
        )

    pred_all = np.concatenate(all_pred_state, axis=0)
    true_all = np.concatenate(all_true_state, axis=0)
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
    div_steps = np.array([r["divergence_step"] for r in rows], dtype=np.float64)
    summary = {
        "split": args.split,
        "num_episodes": int(len(rows)),
        "rollout_steps": int(rollout_steps),
        "chunk_mode": bool(chunk_mode),
        "chunk_steps": int(args.chunk_steps),
        "num_chunks": int(args.num_chunks),
        "chunk_anchor_mode": str(args.chunk_anchor_mode),
        "renorm_speed": float(args.renorm_speed),
        "divergence_threshold": float(args.divergence_threshold),
        "ttf_median": float(np.median(div_steps)),
        "ttf_p10": float(np.percentile(div_steps, 10)),
        "divergence_rate": float(np.mean([1.0 if r["diverged"] else 0.0 for r in rows])),
        "state_mse": state_mse,
        "position_mse": pos_mse,
        "velocity_mse": vel_mse,
        "plateau_velocity_mse": plateau_vel_mse,
        "rows": rows,
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "best_val_state_mse": float(ckpt.get("best_val_state_mse", float("nan"))),
        "report_renorm_velocity": bool(args.report_renorm_velocity),
        "report_renorm_speed": float(args.report_renorm_speed),
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Output dir:", out_dir)
    print("Summary:", out_dir / "eval_summary.json")
    print(
        f"Aggregate | state_mse={summary['state_mse']:.6f} pos_mse={summary['position_mse']:.6f} "
        f"vel_mse={summary['velocity_mse']:.6f} vel_mse_plateau={summary['plateau_velocity_mse']:.6f} "
        f"| ttf_median={summary['ttf_median']:.2f} ttf_p10={summary['ttf_p10']:.2f} "
        f"divergence_rate={summary['divergence_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
