from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from particle_nn_sim.time_conditioned_collision_model import (
        TimeConditionedCollisionModel,
        TimeConditionedCollisionModelConfig,
        TimeEncodingConfig,
    )
    from particle_nn_sim.train import fit_standardizer
except ModuleNotFoundError:
    from time_conditioned_collision_model import (
        TimeConditionedCollisionModel,
        TimeConditionedCollisionModelConfig,
        TimeEncodingConfig,
    )
    from train import fit_standardizer


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
    p = argparse.ArgumentParser(description="Train simple one-particle TCNO from a cached NPZ dataset.")
    p.add_argument("--dataset", type=str, default="datasets/simple_tcno_stratified_1p.npz")
    p.add_argument("--out-dir", type=str, default="checkpoints/simple_tcno_1p")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--samples-per-epoch", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--event-loss-weight", type=float, default=0.2)
    p.add_argument(
        "--speed-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional raw-unit velocity-speed conservation loss. "
            "Useful for fixed-speed exact wall-bounce datasets where rollout drift compounds."
        ),
    )
    p.add_argument(
        "--velocity-loss-weight",
        type=float,
        default=0.0,
        help="Optional raw-unit vector velocity loss applied according to --velocity-loss-mode.",
    )
    p.add_argument(
        "--velocity-loss-mode",
        type=str,
        default="all",
        choices=["all", "endpoint", "anchors"],
        help="Where to apply --velocity-loss-weight: all samples, final step only, or listed anchor steps.",
    )
    p.add_argument(
        "--velocity-anchor-steps",
        type=str,
        default="0,250,500,750,1000",
        help="Comma-separated integer steps used when --velocity-loss-mode=anchors.",
    )
    p.add_argument(
        "--use-event-head",
        type=str2bool,
        default=True,
        help="If false, physically remove the event/collision head and train state prediction only.",
    )
    p.add_argument(
        "--use-endpoint-velocity-head",
        type=str2bool,
        default=False,
        help="Add an auxiliary velocity head trained only at selected endpoint/anchor times.",
    )
    p.add_argument(
        "--endpoint-velocity-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the auxiliary endpoint velocity loss.",
    )
    p.add_argument(
        "--endpoint-velocity-loss-mode",
        type=str,
        default="endpoint",
        choices=["endpoint", "anchors"],
        help="Where to train the endpoint velocity head: final step only or listed anchor steps.",
    )
    p.add_argument(
        "--endpoint-velocity-anchor-steps",
        type=str,
        default="250,500,750,1000",
        help="Comma-separated integer steps for --endpoint-velocity-loss-mode=anchors.",
    )
    p.add_argument("--event-pos-weight", type=float, default=2.0)
    p.add_argument("--coll-epsilon-steps", type=int, default=2)
    p.add_argument("--event-target-mode", type=str, default="window", choices=["spike", "window", "gaussian"])
    p.add_argument("--event-window", type=float, default=0.05)
    p.add_argument("--sigma-event", type=float, default=0.03)
    p.add_argument(
        "--state-loss-mode",
        type=str,
        default="full",
        choices=["full", "position"],
        help="'full' uses [x,y,vx,vy] MSE; 'position' uses only [x,y] MSE for the state loss.",
    )
    p.add_argument(
        "--output-mode",
        type=str,
        default="state",
        choices=["state", "position"],
        help="'state' predicts [x,y,vx,vy]; 'position' physically predicts only [x,y].",
    )
    p.add_argument("--trunk-width", type=int, default=256)
    p.add_argument("--trunk-depth", type=int, default=3)
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--time-encoding-mode", type=str, default="fourier", choices=["raw", "low_freq_fourier", "fourier"])
    p.add_argument("--num-frequencies", type=int, default=8)
    p.add_argument("--include-raw-time", type=str2bool, default=True)
    p.add_argument("--base-frequency", type=float, default=1.0)
    p.add_argument("--normalize-time", type=str2bool, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--print-every", type=int, default=1, help="Print progress every N epochs.")
    p.add_argument("--target-val-mse", type=float, default=0.0, help="Stop early once validation state MSE is <= this value.")
    p.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "plateau"])
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--lr-plateau-patience", type=int, default=25)
    p.add_argument("--lr-plateau-min", type=float, default=1e-5)
    p.add_argument("--wandb", type=str2bool, default=False, help="Log training metrics to Weights & Biases.")
    p.add_argument("--wandb-project", type=str, default="particle-nn-sim")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument(
        "--save-best-checkpoint",
        type=str2bool,
        default=True,
        help="Write a renderer-compatible checkpoint every time validation improves.",
    )
    return p.parse_args()


def resolve_device(flag: str) -> str:
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def clean_scalar_time(value: float) -> float:
    return float(round(float(value), 8))


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
            labels = np.maximum(labels, np.exp(-0.5 * (d / sigma_steps) ** 2).astype(np.float32))
        return np.clip(labels, 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unsupported event_target_mode: {mode}")


def make_event_all(coll_all: np.ndarray, dt: float, args: argparse.Namespace) -> np.ndarray:
    E = coll_all.shape[0]
    T = coll_all.shape[1] + 1
    event_all = np.zeros((E, T), dtype=np.float32)
    for e in range(E):
        collision_steps = np.where(coll_all[e] > 0)[0].astype(np.int64) + 1
        event_all[e] = build_event_targets(
            T=T,
            collision_steps=collision_steps,
            dt=dt,
            mode=args.event_target_mode,
            eps_steps=args.coll_epsilon_steps,
            event_window=args.event_window,
            sigma_event=args.sigma_event,
        )
    return event_all


def gather_states(pos_all: np.ndarray, vel_all: np.ndarray, episode_indices: np.ndarray) -> np.ndarray:
    pos = pos_all[np.asarray(episode_indices, dtype=np.int64), :, 0, :]
    vel = vel_all[np.asarray(episode_indices, dtype=np.int64), :, 0, :]
    return np.concatenate([pos, vel], axis=2).reshape(-1, 4).astype(np.float32)


def gather_initial_states(pos_all: np.ndarray, vel_all: np.ndarray, episode_indices: np.ndarray) -> np.ndarray:
    pos0 = pos_all[np.asarray(episode_indices, dtype=np.int64), 0, 0, :]
    vel0 = vel_all[np.asarray(episode_indices, dtype=np.int64), 0, 0, :]
    return np.concatenate([pos0, vel0], axis=1).astype(np.float32)


def materialize_split(
    pos_all: np.ndarray,
    vel_all: np.ndarray,
    event_all: np.ndarray,
    episode_indices: np.ndarray,
    dt: float,
    s0_mean: np.ndarray,
    s0_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = np.asarray(episode_indices, dtype=np.int64)
    T = int(pos_all.shape[1])
    n_eps = int(len(eps))

    s0 = gather_initial_states(pos_all, vel_all, eps)
    s0 = ((s0 - s0_mean.reshape(1, 4)) / s0_std.reshape(1, 4)).astype(np.float32)
    s0 = np.repeat(s0[:, None, :], T, axis=1).reshape(n_eps * T, 4)

    states = np.concatenate([pos_all[eps, :, 0, :], vel_all[eps, :, 0, :]], axis=2)
    y = ((states - y_mean.reshape(1, 1, 4)) / y_std.reshape(1, 1, 4)).reshape(n_eps * T, 4).astype(np.float32)

    t = (np.arange(T, dtype=np.float32) * float(dt))[None, :]
    t = np.repeat(t, n_eps, axis=0).reshape(n_eps * T).astype(np.float32)
    event = event_all[eps].reshape(n_eps * T).astype(np.float32)

    return (
        torch.from_numpy(s0).to(device),
        torch.from_numpy(t).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(event).to(device),
    )


def raw_speed_mse_from_normalized_state(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
) -> torch.Tensor:
    if pred_state.shape[1] < 4:
        return torch.zeros((pred_state.shape[0],), dtype=pred_state.dtype, device=pred_state.device)
    pred_vel_raw = pred_state[:, 2:4] * y_std_t[:, 2:4] + y_mean_t[:, 2:4]
    target_vel_raw = target_state[:, 2:4] * y_std_t[:, 2:4] + y_mean_t[:, 2:4]
    pred_speed = torch.linalg.norm(pred_vel_raw, dim=1)
    target_speed = torch.linalg.norm(target_vel_raw, dim=1)
    return (pred_speed - target_speed).pow(2)


def raw_velocity_mse_from_normalized_state(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
) -> torch.Tensor:
    if pred_state.shape[1] < 4:
        return torch.zeros((pred_state.shape[0],), dtype=pred_state.dtype, device=pred_state.device)
    pred_vel_raw = pred_state[:, 2:4] * y_std_t[:, 2:4] + y_mean_t[:, 2:4]
    target_vel_raw = target_state[:, 2:4] * y_std_t[:, 2:4] + y_mean_t[:, 2:4]
    return F.mse_loss(pred_vel_raw, target_vel_raw, reduction="none").mean(dim=1)


def raw_endpoint_velocity_mse_from_normalized_velocity(
    pred_vel: torch.Tensor,
    target_state: torch.Tensor,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
) -> torch.Tensor:
    pred_vel_raw = pred_vel * y_std_t[:, 2:4] + y_mean_t[:, 2:4]
    target_vel_raw = target_state[:, 2:4] * y_std_t[:, 2:4] + y_mean_t[:, 2:4]
    return F.mse_loss(pred_vel_raw, target_vel_raw, reduction="none").mean(dim=1)


def parse_anchor_steps(value: str) -> set[int]:
    anchors: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        step = int(part)
        if step < 0:
            raise ValueError("--velocity-anchor-steps must contain non-negative integers")
        anchors.add(step)
    return anchors


def velocity_loss_mask(t: torch.Tensor, dt: float, steps: int, mode: str, anchor_steps: set[int]) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "all":
        return torch.ones_like(t, dtype=torch.bool)
    step_idx = torch.round(t / float(dt)).to(torch.long)
    if mode == "endpoint":
        return step_idx == int(steps)
    if mode == "anchors":
        if not anchor_steps:
            return torch.zeros_like(t, dtype=torch.bool)
        mask = torch.zeros_like(t, dtype=torch.bool)
        for step in anchor_steps:
            mask |= step_idx == int(step)
        return mask
    raise ValueError(f"Unsupported velocity loss mode: {mode}")


def endpoint_velocity_loss_mask(t: torch.Tensor, dt: float, steps: int, mode: str, anchor_steps: set[int]) -> torch.Tensor:
    mode = str(mode).strip().lower()
    step_idx = torch.round(t / float(dt)).to(torch.long)
    if mode == "endpoint":
        return step_idx == int(steps)
    if mode == "anchors":
        if not anchor_steps:
            return torch.zeros_like(t, dtype=torch.bool)
        mask = torch.zeros_like(t, dtype=torch.bool)
        for step in anchor_steps:
            mask |= step_idx == int(step)
        return mask
    raise ValueError(f"Unsupported endpoint velocity loss mode: {mode}")


def state_mse_by_mode(pred_state: torch.Tensor, target_state: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "full":
        if pred_state.shape[1] != target_state.shape[1]:
            raise ValueError("state-loss-mode=full requires a full [x,y,vx,vy] model output.")
        return F.mse_loss(pred_state, target_state, reduction="none").mean(dim=1)
    if mode == "position":
        return F.mse_loss(pred_state[:, 0:2], target_state[:, 0:2], reduction="none").mean(dim=1)
    raise ValueError(f"Unsupported state loss mode: {mode}")


@torch.no_grad()
def evaluate_tensors(
    model,
    tensors,
    batch_size,
    event_loss_weight,
    event_pos_weight,
    speed_loss_weight,
    velocity_loss_weight,
    velocity_loss_mode,
    velocity_anchor_steps,
    endpoint_velocity_loss_weight,
    endpoint_velocity_loss_mode,
    endpoint_velocity_anchor_steps,
    dt,
    steps,
    state_loss_mode,
    y_mean_t,
    y_std_t,
):
    model.eval()
    s0_all, t_all, y_all, event_all = tensors
    mse_sum = 0.0
    bce_sum = 0.0
    speed_sum = 0.0
    vel_sum = 0.0
    vel_n = 0
    endpoint_vel_sum = 0.0
    endpoint_vel_n = 0
    loss_sum = 0.0
    event_correct = 0
    n_total = int(s0_all.shape[0])
    has_event_head = bool(getattr(model.cfg, "use_event_head", True))
    pos_weight = torch.as_tensor(float(event_pos_weight), dtype=torch.float32, device=s0_all.device)
    for start in range(0, n_total, int(batch_size)):
        end = min(start + int(batch_size), n_total)
        s0 = s0_all[start:end]
        t = t_all[start:end]
        y = y_all[start:end]
        event = event_all[start:end]
        out = model(s0, t)
        mse = state_mse_by_mode(out["state"], y, state_loss_mode)
        speed_mse = raw_speed_mse_from_normalized_state(out["state"], y, y_mean_t, y_std_t)
        vel_mse = raw_velocity_mse_from_normalized_state(out["state"], y, y_mean_t, y_std_t)
        vel_mask = velocity_loss_mask(t, dt, steps, velocity_loss_mode, velocity_anchor_steps)
        vel_term = vel_mse[vel_mask].mean() if bool(vel_mask.any()) else torch.zeros((), device=s0_all.device)
        endpoint_vel_term = torch.zeros((), device=s0_all.device)
        endpoint_vel_mse = None
        endpoint_vel_mask = torch.zeros_like(t, dtype=torch.bool)
        if out.get("endpoint_vel") is not None:
            endpoint_vel_mse = raw_endpoint_velocity_mse_from_normalized_velocity(
                out["endpoint_vel"],
                y,
                y_mean_t,
                y_std_t,
            )
            endpoint_vel_mask = endpoint_velocity_loss_mask(
                t,
                dt=dt,
                steps=steps,
                mode=endpoint_velocity_loss_mode,
                anchor_steps=endpoint_velocity_anchor_steps,
            )
            if bool(endpoint_vel_mask.any()):
                endpoint_vel_term = endpoint_vel_mse[endpoint_vel_mask].mean()
        if has_event_head and out["event_logit"] is not None:
            bce = F.binary_cross_entropy_with_logits(
                out["event_logit"].squeeze(1),
                event,
                reduction="none",
                pos_weight=pos_weight,
            )
            loss = mse + float(event_loss_weight) * bce + float(speed_loss_weight) * speed_mse
            if float(velocity_loss_weight) > 0.0:
                loss = loss + float(velocity_loss_weight) * vel_term
            if float(endpoint_velocity_loss_weight) > 0.0:
                loss = loss + float(endpoint_velocity_loss_weight) * endpoint_vel_term
            probs = torch.sigmoid(out["event_logit"].squeeze(1))
            event_correct += int(((probs >= 0.5) == (event >= 0.5)).sum().item())
            bce_sum += float(bce.sum().item())
        else:
            loss = mse + float(speed_loss_weight) * speed_mse
            if float(velocity_loss_weight) > 0.0:
                loss = loss + float(velocity_loss_weight) * vel_term
            if float(endpoint_velocity_loss_weight) > 0.0:
                loss = loss + float(endpoint_velocity_loss_weight) * endpoint_vel_term
        B = int(s0.shape[0])
        mse_sum += float(mse.sum().item())
        speed_sum += float(speed_mse.sum().item())
        if bool(vel_mask.any()):
            vel_sum += float(vel_mse[vel_mask].sum().item())
            vel_n += int(vel_mask.sum().item())
        if endpoint_vel_mse is not None and bool(endpoint_vel_mask.any()):
            endpoint_vel_sum += float(endpoint_vel_mse[endpoint_vel_mask].sum().item())
            endpoint_vel_n += int(endpoint_vel_mask.sum().item())
        loss_sum += float(loss.sum().item())
    return {
        "loss": loss_sum / max(n_total, 1),
        "state_mse": mse_sum / max(n_total, 1),
        "speed_mse": speed_sum / max(n_total, 1),
        "velocity_mse_supervised": vel_sum / max(vel_n, 1) if vel_n > 0 else float("nan"),
        "endpoint_velocity_mse_supervised": endpoint_vel_sum / max(endpoint_vel_n, 1)
        if endpoint_vel_n > 0
        else float("nan"),
        "event_bce": bce_sum / max(n_total, 1) if has_event_head else float("nan"),
        "event_acc": event_correct / max(n_total, 1) if has_event_head else float("nan"),
    }


def main() -> None:
    args = parse_args()
    if int(args.print_every) < 1:
        raise ValueError("--print-every must be >= 1")
    if args.output_mode == "position" and args.state_loss_mode != "position":
        raise ValueError("--output-mode position requires --state-loss-mode position.")
    if args.output_mode == "position" and (
        float(args.speed_loss_weight) > 0.0 or float(args.velocity_loss_weight) > 0.0
    ):
        raise ValueError("Velocity/speed losses require --output-mode state.")
    if bool(args.use_endpoint_velocity_head) and float(args.endpoint_velocity_loss_weight) <= 0.0:
        raise ValueError("--use-endpoint-velocity-head true requires --endpoint-velocity-loss-weight > 0.")
    if not bool(args.use_endpoint_velocity_head) and float(args.endpoint_velocity_loss_weight) > 0.0:
        raise ValueError("--endpoint-velocity-loss-weight > 0 requires --use-endpoint-velocity-head true.")
    if int(args.lr_plateau_patience) < 1:
        raise ValueError("--lr-plateau-patience must be >= 1")
    velocity_anchor_steps = parse_anchor_steps(args.velocity_anchor_steps)
    endpoint_velocity_anchor_steps = parse_anchor_steps(args.endpoint_velocity_anchor_steps)
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.dataset)
    with np.load(data_path, allow_pickle=False) as data:
        pos_all = data["pos_all"].astype(np.float32)
        vel_all = data["vel_all"].astype(np.float32)
        coll_all = data["coll_all"].astype(np.uint8)
        train_eps = data["train_eps"].astype(np.int64)
        val_eps = data["val_eps"].astype(np.int64)
        test_eps = data["test_eps"].astype(np.int64)
        meta = json.loads(str(data["meta_json"]))
        generation_config = json.loads(str(data["generation_config"]))

    meta["radii"] = np.asarray(meta["radii"], dtype=np.float32)
    meta["masses"] = np.asarray(meta["masses"], dtype=np.float32)
    dt = clean_scalar_time(meta["dt"])
    steps = int(pos_all.shape[1] - 1)
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {data_path}", flush=True)
    print(f"Shapes: pos={pos_all.shape}, vel={vel_all.shape}, coll={coll_all.shape}", flush=True)
    print(f"Splits: train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}", flush=True)

    event_all = make_event_all(coll_all, dt=dt, args=args)
    s0_mean, s0_std = fit_standardizer(gather_initial_states(pos_all, vel_all, train_eps))
    y_mean, y_std = fit_standardizer(gather_states(pos_all, vel_all, train_eps))

    train_tensors = materialize_split(pos_all, vel_all, event_all, train_eps, dt, s0_mean, s0_std, y_mean, y_std, device)
    val_tensors = materialize_split(pos_all, vel_all, event_all, val_eps, dt, s0_mean, s0_std, y_mean, y_std, device)
    eval_batch_size = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)
    train_size = int(train_tensors[0].shape[0])
    samples_per_epoch = int(args.samples_per_epoch) if int(args.samples_per_epoch) > 0 else train_size
    print(f"Materialized train samples on {device}: {train_size}", flush=True)
    print(f"Samples per epoch: {samples_per_epoch}", flush=True)

    model_cfg = TimeConditionedCollisionModelConfig(
        state_dim=4,
        output_mode=str(args.output_mode),
        trunk_width=int(args.trunk_width),
        trunk_depth=int(args.trunk_depth),
        activation=str(args.activation),
        dropout=float(args.dropout),
        enforce_t0_anchor=True,
        use_event_head=bool(args.use_event_head),
        use_endpoint_velocity_head=bool(args.use_endpoint_velocity_head),
        time_encoding=TimeEncodingConfig(
            mode=str(args.time_encoding_mode),
            num_frequencies=int(args.num_frequencies),
            include_raw_time=bool(args.include_raw_time),
            base_frequency=float(args.base_frequency),
            normalize_time=bool(args.normalize_time),
            max_time=float(dt * steps),
        ),
    )
    model = TimeConditionedCollisionModel(model_cfg).to(device)

    out_dim = 2 if args.output_mode == "position" else 4
    anchor_scale = torch.as_tensor((s0_std.reshape(4)[:out_dim] / y_std.reshape(4)[:out_dim]), dtype=torch.float32, device=device)
    anchor_bias = torch.as_tensor(((s0_mean.reshape(4)[:out_dim] - y_mean.reshape(4)[:out_dim]) / y_std.reshape(4)[:out_dim]), dtype=torch.float32, device=device)
    model.set_s0_anchor_affine(anchor_scale, anchor_bias)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(args.lr_plateau_factor),
            patience=int(args.lr_plateau_patience),
            min_lr=float(args.lr_plateau_min),
        )
    has_event_head = bool(args.use_event_head)
    y_mean_t = torch.as_tensor(y_mean.reshape(1, 4), dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std.reshape(1, 4), dtype=torch.float32, device=device)
    history = {
        "train_loss": [],
        "train_state_mse": [],
        "train_speed_mse": [],
        "train_velocity_mse_supervised": [],
        "train_endpoint_velocity_mse_supervised": [],
        "train_event_bce": [],
        "val_loss": [],
        "val_state_mse": [],
        "val_speed_mse": [],
        "val_velocity_mse_supervised": [],
        "val_endpoint_velocity_mse_supervised": [],
        "val_event_bce": [],
        "val_event_acc": [],
        "lr": [],
    }
    best = {"epoch": 0, "val_loss": float("inf"), "state_dict": None, "stats": None}
    pos_weight = torch.as_tensor(float(args.event_pos_weight), dtype=torch.float32, device=device)
    wandb_run = None
    if bool(args.wandb):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Weights & Biases is not installed. Install wandb or run with --wandb false.") from exc
        wandb_run = wandb.init(
            project=str(args.wandb_project),
            entity=str(args.wandb_entity) or None,
            name=str(args.wandb_run_name) or None,
            mode=str(args.wandb_mode),
            config={
                **vars(args),
                "dataset": str(data_path),
                "train_samples": train_size,
                "val_samples": int(val_tensors[0].shape[0]),
                "steps": steps,
                "dt": dt,
                "device_resolved": str(device),
            },
        )

    epochs_ran = 0
    ckpt_path = out_dir / "model_tc_collision_1p.pt"
    summary_path = out_dir / "training_summary.json"

    def write_checkpoint(state_dict: dict[str, torch.Tensor], stats: dict[str, float] | None) -> None:
        model_config = {
            "state_dim": 4,
            "output_mode": str(args.output_mode),
            "trunk_width": int(args.trunk_width),
            "trunk_depth": int(args.trunk_depth),
            "activation": str(args.activation),
            "dropout": float(args.dropout),
            "enforce_t0_anchor": True,
            "use_event_head": bool(args.use_event_head),
            "use_endpoint_velocity_head": bool(args.use_endpoint_velocity_head),
            "time_encoding_mode": str(args.time_encoding_mode),
            "num_frequencies": int(args.num_frequencies),
            "include_raw_time": bool(args.include_raw_time),
            "base_frequency": float(args.base_frequency),
            "normalize_time": bool(args.normalize_time),
            "time_max": float(dt * steps),
        }
        data_cfg = {
            "dataset": str(data_path),
            "episodes": int(pos_all.shape[0]),
            "steps": steps,
            "dt": dt,
            "speed_max": float(generation_config.get("speed_max", 0.7)),
            "radius": float(generation_config.get("radius", 0.0)),
            "mass": float(generation_config.get("mass", 1.0)),
            "wall_collision_mode": str(generation_config.get("wall_collision_mode", meta.get("wall_mode", "exact"))),
            "stratified_init": bool(generation_config.get("stratified_init", True)),
            "pos_grid_n": int(generation_config.get("pos_grid_n", 4)),
            "angle_bins": int(generation_config.get("angle_bins", 8)),
            "episodes_per_bucket": int(generation_config.get("episodes_per_bucket", 16)),
            "fixed_speed": generation_config.get("fixed_speed", None),
            "coll_epsilon_steps": int(args.coll_epsilon_steps),
            "event_target_mode": str(args.event_target_mode),
            "event_window": float(args.event_window),
            "sigma_event": float(args.sigma_event),
        }
        train_cfg = {
            "epochs": int(args.epochs),
            "epochs_ran": int(epochs_ran),
            "batch_size": int(args.batch_size),
            "samples_per_epoch": int(args.samples_per_epoch),
            "eval_batch_size": int(eval_batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "event_loss_weight": float(args.event_loss_weight),
            "speed_loss_weight": float(args.speed_loss_weight),
            "velocity_loss_weight": float(args.velocity_loss_weight),
            "velocity_loss_mode": str(args.velocity_loss_mode),
            "velocity_anchor_steps": sorted(int(x) for x in velocity_anchor_steps),
            "endpoint_velocity_loss_weight": float(args.endpoint_velocity_loss_weight),
            "endpoint_velocity_loss_mode": str(args.endpoint_velocity_loss_mode),
            "endpoint_velocity_anchor_steps": sorted(int(x) for x in endpoint_velocity_anchor_steps),
            "use_event_head": bool(args.use_event_head),
            "use_endpoint_velocity_head": bool(args.use_endpoint_velocity_head),
            "output_mode": str(args.output_mode),
            "state_loss_mode": str(args.state_loss_mode),
            "event_pos_weight": float(args.event_pos_weight),
            "seed": int(args.seed),
            "device": str(device),
            "print_every": int(args.print_every),
            "target_val_mse": float(args.target_val_mse),
            "lr_scheduler": str(args.lr_scheduler),
            "lr_plateau_factor": float(args.lr_plateau_factor),
            "lr_plateau_patience": int(args.lr_plateau_patience),
            "lr_plateau_min": float(args.lr_plateau_min),
            "save_best_checkpoint": bool(args.save_best_checkpoint),
        }
        ckpt = {
            "model_name": "TimeConditionedCollisionModel",
            "model_state_dict": state_dict,
            "model_config": model_config,
            "config": {"data": data_cfg, "train": train_cfg},
            "meta": meta,
            "split_indices": {"train_eps": train_eps, "val_eps": val_eps, "test_eps": test_eps},
            "episode_init": {
                "pos0": pos_all[:, 0].astype(np.float32),
                "vel0": vel_all[:, 0].astype(np.float32),
            },
            "s0_standardizer": {"mean": s0_mean, "std": s0_std},
            "state_standardizer": {"mean": y_mean, "std": y_std},
            "history": history,
            "best_epoch": int(best["epoch"]),
            "stats": stats,
        }
        torch.save(ckpt, ckpt_path)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": str(data_path),
                    "best_epoch": int(best["epoch"]),
                    "best_stats": stats,
                    "model_config": model_config,
                    "data": data_cfg,
                    "train": train_cfg,
                },
                f,
                indent=2,
            )

    for ep in range(1, int(args.epochs) + 1):
        epochs_ran = ep
        t0 = time.time()
        model.train()
        loss_sum = mse_sum = speed_sum = vel_sum = endpoint_vel_sum = bce_sum = 0.0
        vel_n = 0
        endpoint_vel_n = 0
        n = 0
        if samples_per_epoch >= train_size:
            order = torch.randperm(train_size, device=device)
        else:
            order = torch.randint(0, train_size, (samples_per_epoch,), device=device)

        for start in range(0, int(order.numel()), int(args.batch_size)):
            batch_idx = order[start : start + int(args.batch_size)]
            s0 = train_tensors[0][batch_idx]
            t = train_tensors[1][batch_idx]
            y = train_tensors[2][batch_idx]
            event = train_tensors[3][batch_idx]
            out = model(s0, t)
            mse = state_mse_by_mode(out["state"], y, args.state_loss_mode)
            speed_mse = raw_speed_mse_from_normalized_state(out["state"], y, y_mean_t, y_std_t)
            vel_mse = raw_velocity_mse_from_normalized_state(out["state"], y, y_mean_t, y_std_t)
            vel_mask = velocity_loss_mask(
                t,
                dt=dt,
                steps=steps,
                mode=args.velocity_loss_mode,
                anchor_steps=velocity_anchor_steps,
            )
            vel_term = vel_mse[vel_mask].mean() if bool(vel_mask.any()) else torch.zeros((), device=device)
            endpoint_vel_term = torch.zeros((), device=device)
            endpoint_vel_mse = None
            endpoint_vel_mask = torch.zeros_like(t, dtype=torch.bool)
            if out.get("endpoint_vel") is not None:
                endpoint_vel_mse = raw_endpoint_velocity_mse_from_normalized_velocity(
                    out["endpoint_vel"],
                    y,
                    y_mean_t,
                    y_std_t,
                )
                endpoint_vel_mask = endpoint_velocity_loss_mask(
                    t,
                    dt=dt,
                    steps=steps,
                    mode=args.endpoint_velocity_loss_mode,
                    anchor_steps=endpoint_velocity_anchor_steps,
                )
                if bool(endpoint_vel_mask.any()):
                    endpoint_vel_term = endpoint_vel_mse[endpoint_vel_mask].mean()
            if has_event_head and out["event_logit"] is not None:
                bce = F.binary_cross_entropy_with_logits(
                    out["event_logit"].squeeze(1),
                    event,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                loss = (
                    mse
                    + float(args.event_loss_weight) * bce
                    + float(args.speed_loss_weight) * speed_mse
                    + float(args.velocity_loss_weight) * vel_term
                    + float(args.endpoint_velocity_loss_weight) * endpoint_vel_term
                ).mean()
            else:
                bce = None
                loss = (mse + float(args.speed_loss_weight) * speed_mse).mean()
                if float(args.velocity_loss_weight) > 0.0:
                    loss = loss + float(args.velocity_loss_weight) * vel_term
                if float(args.endpoint_velocity_loss_weight) > 0.0:
                    loss = loss + float(args.endpoint_velocity_loss_weight) * endpoint_vel_term
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            B = int(s0.shape[0])
            loss_sum += float(loss.item()) * B
            mse_sum += float(mse.sum().item())
            speed_sum += float(speed_mse.sum().item())
            if bool(vel_mask.any()):
                vel_sum += float(vel_mse[vel_mask].sum().item())
                vel_n += int(vel_mask.sum().item())
            if endpoint_vel_mse is not None and bool(endpoint_vel_mask.any()):
                endpoint_vel_sum += float(endpoint_vel_mse[endpoint_vel_mask].sum().item())
                endpoint_vel_n += int(endpoint_vel_mask.sum().item())
            if bce is not None:
                bce_sum += float(bce.sum().item())
            n += B

        train_loss = loss_sum / max(n, 1)
        train_mse = mse_sum / max(n, 1)
        train_speed_mse = speed_sum / max(n, 1)
        train_vel_mse = vel_sum / max(vel_n, 1) if vel_n > 0 else float("nan")
        train_endpoint_vel_mse = endpoint_vel_sum / max(endpoint_vel_n, 1) if endpoint_vel_n > 0 else float("nan")
        train_bce = bce_sum / max(n, 1) if has_event_head else float("nan")
        val_stats = evaluate_tensors(
            model,
            val_tensors,
            eval_batch_size,
            args.event_loss_weight,
            args.event_pos_weight,
            args.speed_loss_weight,
            args.velocity_loss_weight,
            args.velocity_loss_mode,
            velocity_anchor_steps,
            args.endpoint_velocity_loss_weight,
            args.endpoint_velocity_loss_mode,
            endpoint_velocity_anchor_steps,
            dt,
            steps,
            args.state_loss_mode,
            y_mean_t,
            y_std_t,
        )
        history["train_loss"].append(train_loss)
        history["train_state_mse"].append(train_mse)
        history["train_speed_mse"].append(train_speed_mse)
        history["train_velocity_mse_supervised"].append(train_vel_mse)
        history["train_endpoint_velocity_mse_supervised"].append(train_endpoint_vel_mse)
        history["train_event_bce"].append(train_bce)
        history["val_loss"].append(val_stats["loss"])
        history["val_state_mse"].append(val_stats["state_mse"])
        history["val_speed_mse"].append(val_stats["speed_mse"])
        history["val_velocity_mse_supervised"].append(val_stats["velocity_mse_supervised"])
        history["val_endpoint_velocity_mse_supervised"].append(val_stats["endpoint_velocity_mse_supervised"])
        history["val_event_bce"].append(val_stats["event_bce"])
        history["val_event_acc"].append(val_stats["event_acc"])
        lr_now = float(opt.param_groups[0]["lr"])
        history["lr"].append(lr_now)
        if val_stats["loss"] < best["val_loss"]:
            best = {
                "epoch": ep,
                "val_loss": val_stats["loss"],
                "state_dict": deepcopy(model.state_dict()),
                "stats": deepcopy(val_stats),
            }
            if bool(args.save_best_checkpoint):
                write_checkpoint(best["state_dict"], best["stats"])
        if scheduler is not None:
            scheduler.step(float(val_stats["state_mse"]))
            lr_after = float(opt.param_groups[0]["lr"])
        else:
            lr_after = lr_now
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": ep,
                    "train/loss": train_loss,
                    "train/state_mse": train_mse,
                    "train/speed_mse": train_speed_mse,
                    "train/velocity_mse_supervised": train_vel_mse,
                    "train/endpoint_velocity_mse_supervised": train_endpoint_vel_mse,
                    "train/event_bce": train_bce,
                    "val/loss": val_stats["loss"],
                    "val/state_mse": val_stats["state_mse"],
                    "val/speed_mse": val_stats["speed_mse"],
                    "val/velocity_mse_supervised": val_stats["velocity_mse_supervised"],
                    "val/endpoint_velocity_mse_supervised": val_stats["endpoint_velocity_mse_supervised"],
                    "val/event_bce": val_stats["event_bce"],
                    "val/event_acc": val_stats["event_acc"],
                    "best/epoch": int(best["epoch"]),
                    "best/val_loss": float(best["val_loss"]),
                    "lr": lr_after,
                    "epoch_seconds": time.time() - t0,
                },
                step=ep,
            )
        if ep == 1 or ep == int(args.epochs) or ep % int(args.print_every) == 0:
            print(
                f"Epoch {ep:03d} | train_loss={train_loss:.6f} | train_mse={train_mse:.6f} "
                f"| train_speed_mse={train_speed_mse:.6f} | train_endpoint_vel_mse={train_endpoint_vel_mse:.6f} "
                f"| train_event_bce={train_bce:.6f} "
                f"| val_loss={val_stats['loss']:.6f} | val_mse={val_stats['state_mse']:.6f} "
                f"| val_speed_mse={val_stats['speed_mse']:.6f} | val_vel_mse_sup={val_stats['velocity_mse_supervised']:.6f} "
                f"| val_endpoint_vel_mse={val_stats['endpoint_velocity_mse_supervised']:.6f} "
                f"| val_event_bce={val_stats['event_bce']:.6f} "
                f"| val_event_acc={val_stats['event_acc']:.4f} | best_epoch={best['epoch']} "
                f"| lr={lr_after:.2e} "
                f"| sec={time.time() - t0:.2f}"
                ,
                flush=True,
            )
        if float(args.target_val_mse) > 0.0 and float(val_stats["state_mse"]) <= float(args.target_val_mse):
            print(
                f"Target reached at epoch {ep}: val_mse={val_stats['state_mse']:.6g} <= {float(args.target_val_mse):.6g}",
                flush=True,
            )
            break

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    model_config = {
        "state_dim": 4,
        "output_mode": str(args.output_mode),
        "trunk_width": int(args.trunk_width),
        "trunk_depth": int(args.trunk_depth),
        "activation": str(args.activation),
        "dropout": float(args.dropout),
        "enforce_t0_anchor": True,
        "use_event_head": bool(args.use_event_head),
        "use_endpoint_velocity_head": bool(args.use_endpoint_velocity_head),
        "time_encoding_mode": str(args.time_encoding_mode),
        "num_frequencies": int(args.num_frequencies),
        "include_raw_time": bool(args.include_raw_time),
        "base_frequency": float(args.base_frequency),
        "normalize_time": bool(args.normalize_time),
        "time_max": float(dt * steps),
    }
    data_cfg = {
        "dataset": str(data_path),
        "episodes": int(pos_all.shape[0]),
        "steps": steps,
        "dt": dt,
        "speed_max": float(generation_config.get("speed_max", 0.7)),
        "radius": float(generation_config.get("radius", 0.0)),
        "mass": float(generation_config.get("mass", 1.0)),
        "wall_collision_mode": str(generation_config.get("wall_collision_mode", meta.get("wall_mode", "exact"))),
        "stratified_init": bool(generation_config.get("stratified_init", True)),
        "pos_grid_n": int(generation_config.get("pos_grid_n", 4)),
        "angle_bins": int(generation_config.get("angle_bins", 8)),
        "episodes_per_bucket": int(generation_config.get("episodes_per_bucket", 16)),
        "fixed_speed": generation_config.get("fixed_speed", None),
        "coll_epsilon_steps": int(args.coll_epsilon_steps),
        "event_target_mode": str(args.event_target_mode),
        "event_window": float(args.event_window),
        "sigma_event": float(args.sigma_event),
    }
    train_cfg = {
        "epochs": int(args.epochs),
        "epochs_ran": int(epochs_ran),
        "batch_size": int(args.batch_size),
        "samples_per_epoch": int(args.samples_per_epoch),
        "eval_batch_size": int(eval_batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "event_loss_weight": float(args.event_loss_weight),
        "speed_loss_weight": float(args.speed_loss_weight),
        "velocity_loss_weight": float(args.velocity_loss_weight),
        "velocity_loss_mode": str(args.velocity_loss_mode),
        "velocity_anchor_steps": sorted(int(x) for x in velocity_anchor_steps),
        "endpoint_velocity_loss_weight": float(args.endpoint_velocity_loss_weight),
        "endpoint_velocity_loss_mode": str(args.endpoint_velocity_loss_mode),
        "endpoint_velocity_anchor_steps": sorted(int(x) for x in endpoint_velocity_anchor_steps),
        "use_event_head": bool(args.use_event_head),
        "use_endpoint_velocity_head": bool(args.use_endpoint_velocity_head),
        "output_mode": str(args.output_mode),
        "state_loss_mode": str(args.state_loss_mode),
        "event_pos_weight": float(args.event_pos_weight),
        "seed": int(args.seed),
        "device": str(device),
        "print_every": int(args.print_every),
        "target_val_mse": float(args.target_val_mse),
        "lr_scheduler": str(args.lr_scheduler),
        "lr_plateau_factor": float(args.lr_plateau_factor),
        "lr_plateau_patience": int(args.lr_plateau_patience),
        "lr_plateau_min": float(args.lr_plateau_min),
    }
    ckpt = {
        "model_name": "TimeConditionedCollisionModel",
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "config": {"data": data_cfg, "train": train_cfg},
        "meta": meta,
        "split_indices": {"train_eps": train_eps, "val_eps": val_eps, "test_eps": test_eps},
        "episode_init": {
            "pos0": pos_all[:, 0].astype(np.float32),
            "vel0": vel_all[:, 0].astype(np.float32),
        },
        "s0_standardizer": {"mean": s0_mean, "std": s0_std},
        "state_standardizer": {"mean": y_mean, "std": y_std},
        "history": history,
        "best_epoch": int(best["epoch"]),
        "stats": best["stats"],
    }
    ckpt_path = out_dir / "model_tc_collision_1p.pt"
    torch.save(ckpt, ckpt_path)
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(data_path),
                "best_epoch": int(best["epoch"]),
                "best_stats": best["stats"],
                "model_config": model_config,
                "data": data_cfg,
                "train": train_cfg,
            },
            f,
            indent=2,
        )
    print("Training complete.", flush=True)
    print("Checkpoint:", ckpt_path, flush=True)
    print("Summary:", out_dir / "training_summary.json", flush=True)
    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = int(best["epoch"])
        wandb_run.summary["best_val_loss"] = float(best["val_loss"])
        if best["stats"] is not None:
            wandb_run.summary["best_val_mse"] = float(best["stats"]["state_mse"])
        wandb_run.finish()


if __name__ == "__main__":
    main()
