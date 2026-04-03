from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from particle_nn_sim.one_particle_data import collect_episodes_1p
from particle_nn_sim.one_particle_rollout import animate_overlay_gt_perturbed_1p, save_animation_mp4
from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.time_conditioned_collision_model import (
    TimeConditionedCollisionModel,
    TimeConditionedCollisionModelConfig,
    TimeEncodingConfig,
)


@dataclass
class LossWeights:
    lambda_pos: float = 1.0
    lambda_vel: float = 3.0
    lambda_event: float = 1.0
    lambda_sign: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 512
    lr: float = 1e-3
    device: str = "auto"
    seed: int = 0
    train_split: float = 0.7
    val_split: float = 0.15
    normalize: bool = True


@dataclass
class DataConfig:
    episodes: int = 1000
    steps: int = 700
    dt: float = 0.01
    speed_max: float = 0.7
    fixed_speed: float | None = None
    radius: float = 0.0
    mass: float = 1.0
    wall_collision_mode: str = "clamp"
    coll_epsilon_steps: int = 2
    event_target_mode: str = "gaussian"  # {"window","gaussian"}
    event_window: float = 0.05
    sigma_event: float = 0.03
    use_collision_oversampling: bool = True
    collision_sample_fraction: float = 0.6
    collision_window: float = 0.05
    # Optional IC controls to match existing workflows.
    fixed_x: float | None = None
    fixed_y: float | None = None
    fixed_vx: float | None = None
    fixed_vy: float | None = None
    fixed2_x: float | None = None
    fixed2_y: float | None = None
    fixed2_vx: float | None = None
    fixed2_vy: float | None = None
    ball_center_x: float | None = None
    ball_center_y: float | None = None
    ball_radius: float | None = None
    fixed_vel_vx: float | None = None
    fixed_vel_vy: float | None = None
    stratified_init: bool = False
    pos_grid_n: int = 4
    angle_bins: int = 8
    episodes_per_bucket: int | None = None


@dataclass
class RunConfig:
    out_dir: str = "checkpoints/tc_collision_1p"
    plot_episode_idx: int = 0
    event_prob_threshold: float = 0.5
    sign_epsilon: float = 1e-6


class Standardizer:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std = np.maximum(self.std, 1e-6)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer not fitted")
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer not fitted")
        return (x * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray]:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer not fitted")
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)


class EpisodeQueryDataset(Dataset):
    """Dataset of query-time samples: (s0, t, state_t, event_t)."""

    def __init__(
        self,
        s0: np.ndarray,
        t_query: np.ndarray,
        state_t: np.ndarray,
        event_t: np.ndarray,
    ) -> None:
        self.s0 = np.asarray(s0, dtype=np.float32)
        self.t_query = np.asarray(t_query, dtype=np.float32)
        self.state_t = np.asarray(state_t, dtype=np.float32)
        self.event_t = np.asarray(event_t, dtype=np.float32)
        n = len(self.s0)
        if not (len(self.t_query) == len(self.state_t) == len(self.event_t) == n):
            raise ValueError("All arrays must have same length")

    def __len__(self) -> int:
        return len(self.s0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            torch.from_numpy(self.s0[idx]),
            torch.from_numpy(np.asarray(self.t_query[idx])),
            torch.from_numpy(self.state_t[idx]),
            torch.from_numpy(np.asarray(self.event_t[idx])),
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_collision_steps_from_velocity(vel_ep: np.ndarray) -> np.ndarray:
    """
    Fallback collision inference:
      collision at step k if vx or vy sign flips between k-1 and k.
    vel_ep: (T,1,2)
    Returns collision state indices in [1, T-1].
    """
    vx_prev = vel_ep[:-1, 0, 0]
    vx_now = vel_ep[1:, 0, 0]
    vy_prev = vel_ep[:-1, 0, 1]
    vy_now = vel_ep[1:, 0, 1]
    hit = (vx_prev * vx_now < 0) | (vy_prev * vy_now < 0)
    return np.where(hit)[0].astype(np.int64) + 1


def extract_collision_steps(coll_ep: np.ndarray | None, vel_ep: np.ndarray) -> np.ndarray:
    """
    Isolated data-format assumption point.
    If coll_ep exists (shape T-1, collision between t and t+1), convert to state indices t+1.
    Else infer from velocity sign flips.
    """
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


def build_query_samples(
    pos_all: np.ndarray,
    vel_all: np.ndarray,
    coll_all: np.ndarray | None,
    dt: float,
    episode_indices: np.ndarray,
    eps_steps: int,
    event_target_mode: str = "gaussian",
    event_window: float = 0.05,
    sigma_event: float = 0.03,
    use_collision_oversampling: bool = True,
    collision_sample_fraction: float = 0.6,
    collision_window: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds samples:
      input s0=(x0,y0,vx0,vy0), query time t, target state=(x,y,vx,vy), target event in {0,1}.
    """
    s0_list: list[np.ndarray] = []
    t_list: list[np.ndarray] = []
    state_list: list[np.ndarray] = []
    event_list: list[np.ndarray] = []

    for e in episode_indices:
        e = int(e)
        pos_ep = pos_all[e]  # (T,1,2)
        vel_ep = vel_all[e]  # (T,1,2)
        T = pos_ep.shape[0]
        coll_ep = None if coll_all is None else coll_all[e]
        collision_steps = extract_collision_steps(coll_ep, vel_ep)
        event_target = build_event_targets(
            T=T,
            collision_steps=collision_steps,
            dt=float(dt),
            mode=event_target_mode,
            eps_steps=int(eps_steps),
            event_window=float(event_window),
            sigma_event=float(sigma_event),
        )

        s0 = np.concatenate([pos_ep[0, 0], vel_ep[0, 0]], axis=0).astype(np.float32)  # (4,)
        states = np.concatenate([pos_ep[:, 0, :], vel_ep[:, 0, :]], axis=1).astype(np.float32)  # (T,4)
        t_query = (np.arange(T, dtype=np.float32) * float(dt)).astype(np.float32)  # (T,)

        idx_all = np.arange(T, dtype=np.int64)
        if use_collision_oversampling and len(collision_steps) > 0:
            window_steps = int(np.ceil(float(collision_window) / float(dt)))
            near_mask = np.zeros((T,), dtype=bool)
            for c in collision_steps:
                lo = int(max(0, int(c) - window_steps))
                hi = int(min(T - 1, int(c) + window_steps))
                near_mask[lo : hi + 1] = True
            near_idx = idx_all[near_mask]
            if len(near_idx) > 0:
                n_total = T
                n_coll = int(np.round(float(collision_sample_fraction) * n_total))
                n_coll = int(np.clip(n_coll, 0, n_total))
                n_uni = n_total - n_coll
                coll_draw = np.random.choice(near_idx, size=n_coll, replace=True) if n_coll > 0 else np.empty((0,), dtype=np.int64)
                uni_draw = np.random.choice(idx_all, size=n_uni, replace=True) if n_uni > 0 else np.empty((0,), dtype=np.int64)
                idx_use = np.concatenate([coll_draw, uni_draw], axis=0)
            else:
                idx_use = idx_all
        else:
            idx_use = idx_all

        s0_rep = np.repeat(s0[None, :], len(idx_use), axis=0)
        s0_list.append(s0_rep)
        t_list.append(t_query[idx_use])
        state_list.append(states[idx_use])
        event_list.append(event_target[idx_use])

    return (
        np.concatenate(s0_list, axis=0).astype(np.float32),
        np.concatenate(t_list, axis=0).astype(np.float32),
        np.concatenate(state_list, axis=0).astype(np.float32),
        np.concatenate(event_list, axis=0).astype(np.float32),
    )


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


def velocity_sign_loss(pred_vel: Tensor, true_vel: Tensor, sign_epsilon: float = 1e-6) -> Tensor:
    mask = torch.abs(true_vel) > float(sign_epsilon)
    if not torch.any(mask):
        return pred_vel.new_zeros(())
    logits = pred_vel[mask]
    target = (true_vel[mask] >= 0.0).to(dtype=pred_vel.dtype)
    return F.binary_cross_entropy_with_logits(logits, target)


@torch.no_grad()
def evaluate(
    model: TimeConditionedCollisionModel,
    loader: DataLoader,
    device: str,
    state_std: Standardizer | None,
    event_threshold: float = 0.5,
    sign_epsilon: float = 1e-6,
) -> dict[str, float]:
    model.eval()
    state_preds: list[np.ndarray] = []
    state_targets: list[np.ndarray] = []
    event_logits: list[np.ndarray] = []
    event_targets: list[np.ndarray] = []

    for s0, t_q, y_state, y_event in loader:
        s0 = s0.to(device)
        t_q = t_q.to(device)
        y_state = y_state.to(device)
        y_event = y_event.to(device)
        out = model(s0, t_q)
        pred_state = out["state"]
        logit = out["event_logit"]
        state_preds.append(pred_state.cpu().numpy())
        state_targets.append(y_state.cpu().numpy())
        event_logits.append(logit.squeeze(1).cpu().numpy())
        event_targets.append(y_event.cpu().numpy())

    pred_s = np.concatenate(state_preds, axis=0).astype(np.float32)
    true_s = np.concatenate(state_targets, axis=0).astype(np.float32)
    if state_std is not None:
        pred_s = state_std.inverse(pred_s)
        true_s = state_std.inverse(true_s)

    logits = np.concatenate(event_logits, axis=0).astype(np.float32)
    y_evt = np.concatenate(event_targets, axis=0).astype(np.float32)

    pos_mse = float(np.mean((pred_s[:, :2] - true_s[:, :2]) ** 2))
    vel_mse = float(np.mean((pred_s[:, 2:] - true_s[:, 2:]) ** 2))
    state_mse = float(np.mean((pred_s - true_s) ** 2))
    evt = compute_event_metrics(logits=logits, labels=y_evt, threshold=event_threshold)
    sign_metrics = compute_sign_accuracy(pred_s[:, 2:], true_s[:, 2:], sign_epsilon=sign_epsilon)
    return {
        "state_mse": state_mse,
        "position_mse": pos_mse,
        "velocity_mse": vel_mse,
        **evt,
        **sign_metrics,
    }


def plot_state_and_event_over_time(
    t_query: np.ndarray,
    true_state: np.ndarray,
    pred_state: np.ndarray,
    event_prob: np.ndarray,
    event_target: np.ndarray,
    gate: np.ndarray | None,
    out_path: Path,
) -> None:
    n_rows = 6 if gate is not None else 5
    fig, axs = plt.subplots(n_rows, 1, figsize=(10, 14 if gate is not None else 12), sharex=True)
    labels = ["x(t)", "y(t)", "vx(t)", "vy(t)"]
    for i in range(4):
        axs[i].plot(t_query, true_state[:, i], lw=2, label="true")
        axs[i].plot(t_query, pred_state[:, i], lw=1.8, alpha=0.9, label="pred")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc="best")
    evt_ax = axs[4]
    evt_ax.plot(t_query, event_prob, lw=2, label="pred event prob")
    evt_ax.plot(t_query, event_target, lw=1.5, alpha=0.75, label="event target")
    evt_ax.set_ylabel("event")
    evt_ax.set_ylim([-0.05, 1.05])
    evt_ax.grid(True, alpha=0.3)
    evt_ax.legend(loc="best")
    if gate is not None:
        gate_ax = axs[5]
        gate_ax.plot(t_query, gate, lw=2, label="gate(t)")
        gate_ax.set_ylabel("gate")
        gate_ax.set_xlabel("time (s)")
        gate_ax.set_ylim([-0.05, 1.05])
        gate_ax.grid(True, alpha=0.3)
        gate_ax.legend(loc="best")
    else:
        evt_ax.set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train pure-ML time-conditioned collision model (non-autoregressive query).")

    # Data.
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--fixed-speed", type=float, default=None)
    p.add_argument("--radius", type=float, default=0.0)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--wall-collision-mode", type=str, default="clamp", choices=["clamp", "exact"])
    p.add_argument("--coll-epsilon-steps", type=int, default=2)
    p.add_argument("--event-target-mode", type=str, default="gaussian", choices=["window", "gaussian"])
    p.add_argument("--event-window", type=float, default=0.05)
    p.add_argument("--sigma-event", type=float, default=0.03)
    p.add_argument("--use-collision-oversampling", type=str2bool, default=True)
    p.add_argument("--collision-sample-fraction", type=float, default=0.6)
    p.add_argument("--collision-window", type=float, default=0.05)

    p.add_argument("--fixed-x", type=float, default=None)
    p.add_argument("--fixed-y", type=float, default=None)
    p.add_argument("--fixed-vx", type=float, default=None)
    p.add_argument("--fixed-vy", type=float, default=None)
    p.add_argument("--fixed2-x", type=float, default=None)
    p.add_argument("--fixed2-y", type=float, default=None)
    p.add_argument("--fixed2-vx", type=float, default=None)
    p.add_argument("--fixed2-vy", type=float, default=None)
    p.add_argument("--ball-center-x", type=float, default=None)
    p.add_argument("--ball-center-y", type=float, default=None)
    p.add_argument("--ball-radius", type=float, default=None)
    p.add_argument("--fixed-vel-vx", type=float, default=None)
    p.add_argument("--fixed-vel-vy", type=float, default=None)
    p.add_argument("--stratified-init", type=str2bool, default=False)
    p.add_argument("--pos-grid-n", type=int, default=4)
    p.add_argument("--angle-bins", type=int, default=8)
    p.add_argument("--episodes-per-bucket", type=int, default=None)

    # Model.
    p.add_argument("--num-frequencies", type=int, default=8)
    p.add_argument("--time-encoding-mode", type=str, default="raw", choices=["raw", "low_freq_fourier", "fourier"])
    p.add_argument("--base-frequency", type=float, default=1.0)
    p.add_argument("--include-raw-time", type=str2bool, default=True)
    p.add_argument("--normalize-time", type=str2bool, default=True)
    p.add_argument("--time-max", type=float, default=-1.0)
    p.add_argument("--model-variant", type=str, default="simple_tcno", choices=["simple_tcno", "gated_tcno"])
    p.add_argument("--trunk-width", type=int, default=256)
    p.add_argument("--trunk-depth", type=int, default=3)
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--alpha-gate", type=float, default=5.0)

    # Loss.
    p.add_argument("--lambda-pos", type=float, default=1.0)
    p.add_argument("--lambda-vel", type=float, default=3.0)
    p.add_argument("--lambda-event", type=float, default=1.0)
    p.add_argument("--lambda-sign", type=float, default=1.0)

    # Train.
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-split", type=float, default=0.7)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--normalize", type=str2bool, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Eval/plots.
    p.add_argument("--event-prob-threshold", type=float, default=0.5)
    p.add_argument("--sign-epsilon", type=float, default=1e-6)
    p.add_argument("--plot-episode-idx", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="checkpoints/tc_collision_1p")
    p.add_argument("--use-wandb", type=str2bool, default=False)
    p.add_argument("--wandb-project", type=str, default="particle-nn-sim")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-tags", type=str, default="")
    return p


def main() -> None:
    args = make_parser().parse_args()
    if args.coll_epsilon_steps < 0:
        raise ValueError("--coll-epsilon-steps must be >= 0")
    if args.event_window < 0.0:
        raise ValueError("--event-window must be >= 0")
    if args.sigma_event <= 0.0:
        raise ValueError("--sigma-event must be > 0")
    if not (0.0 <= args.collision_sample_fraction <= 1.0):
        raise ValueError("--collision-sample-fraction must be in [0,1]")
    if args.collision_window < 0.0:
        raise ValueError("--collision-window must be >= 0")
    if not (0.0 < args.train_split < 1.0):
        raise ValueError("--train-split must be in (0,1)")
    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("--val-split must be in [0,1)")
    if args.train_split + args.val_split >= 1.0:
        raise ValueError("--train-split + --val-split must be < 1")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"Device: {device} ({gpu_name})")

    data_cfg = DataConfig(
        episodes=args.episodes,
        steps=args.steps,
        dt=args.dt,
        speed_max=args.speed_max,
        fixed_speed=args.fixed_speed,
        radius=args.radius,
        mass=args.mass,
        wall_collision_mode=args.wall_collision_mode,
        coll_epsilon_steps=args.coll_epsilon_steps,
        event_target_mode=args.event_target_mode,
        event_window=args.event_window,
        sigma_event=args.sigma_event,
        use_collision_oversampling=args.use_collision_oversampling,
        collision_sample_fraction=args.collision_sample_fraction,
        collision_window=args.collision_window,
        fixed_x=args.fixed_x,
        fixed_y=args.fixed_y,
        fixed_vx=args.fixed_vx,
        fixed_vy=args.fixed_vy,
        fixed2_x=args.fixed2_x,
        fixed2_y=args.fixed2_y,
        fixed2_vx=args.fixed2_vx,
        fixed2_vy=args.fixed2_vy,
        ball_center_x=args.ball_center_x,
        ball_center_y=args.ball_center_y,
        ball_radius=args.ball_radius,
        fixed_vel_vx=args.fixed_vel_vx,
        fixed_vel_vy=args.fixed_vel_vy,
        stratified_init=args.stratified_init,
        pos_grid_n=args.pos_grid_n,
        angle_bins=args.angle_bins,
        episodes_per_bucket=args.episodes_per_bucket,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        normalize=args.normalize,
    )
    lw = LossWeights(
        lambda_pos=args.lambda_pos,
        lambda_vel=args.lambda_vel,
        lambda_event=args.lambda_event,
        lambda_sign=args.lambda_sign,
    )
    run_cfg = RunConfig(
        out_dir=str(out_dir),
        plot_episode_idx=args.plot_episode_idx,
        event_prob_threshold=args.event_prob_threshold,
        sign_epsilon=args.sign_epsilon,
    )
    time_max = float(args.time_max) if args.time_max > 0.0 else float(data_cfg.dt) * float(data_cfg.steps)
    model_cfg = TimeConditionedCollisionModelConfig(
        state_dim=4,
        trunk_width=args.trunk_width,
        trunk_depth=args.trunk_depth,
        activation=args.activation,
        dropout=args.dropout,
        model_variant=args.model_variant,
        alpha_gate=args.alpha_gate,
        time_encoding=TimeEncodingConfig(
            mode=args.time_encoding_mode,
            num_frequencies=args.num_frequencies,
            include_raw_time=args.include_raw_time,
            base_frequency=args.base_frequency,
            normalize_time=args.normalize_time,
            max_time=time_max,
        ),
    )
    print("Config:")
    print(
        {
            "data": asdict(data_cfg),
            "train": asdict(train_cfg),
            "loss_weights": asdict(lw),
            "model": {
                "trunk_width": model_cfg.trunk_width,
                "trunk_depth": model_cfg.trunk_depth,
                "activation": model_cfg.activation,
                "dropout": model_cfg.dropout,
                "model_variant": model_cfg.model_variant,
                "alpha_gate": model_cfg.alpha_gate,
                "time_encoding_mode": model_cfg.time_encoding.mode,
                "base_frequency": model_cfg.time_encoding.base_frequency,
                "include_raw_time": model_cfg.time_encoding.include_raw_time,
                "normalize_time": model_cfg.time_encoding.normalize_time,
                "time_max": model_cfg.time_encoding.max_time,
                "num_frequencies": model_cfg.time_encoding.num_frequencies,
            },
        }
    )

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "--use-wandb was set but wandb is not available. "
                "Install with `pip install wandb` in your environment."
            ) from e
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        init_kwargs: dict[str, Any] = {
            "project": args.wandb_project,
            "config": {
                "data": asdict(data_cfg),
                "train": asdict(train_cfg),
                "loss_weights": asdict(lw),
                "run": asdict(run_cfg),
                "model": {
                    "state_dim": model_cfg.state_dim,
                    "trunk_width": model_cfg.trunk_width,
                    "trunk_depth": model_cfg.trunk_depth,
                    "activation": model_cfg.activation,
                    "dropout": model_cfg.dropout,
                    "model_variant": model_cfg.model_variant,
                    "alpha_gate": model_cfg.alpha_gate,
                    "time_encoding_mode": model_cfg.time_encoding.mode,
                    "base_frequency": model_cfg.time_encoding.base_frequency,
                    "include_raw_time": model_cfg.time_encoding.include_raw_time,
                    "normalize_time": model_cfg.time_encoding.normalize_time,
                    "time_max": model_cfg.time_encoding.max_time,
                    "num_frequencies": model_cfg.time_encoding.num_frequencies,
                },
            },
            "tags": tags,
            "dir": str(out_dir),
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        wandb_run = wandb.init(**init_kwargs)

    # Data generation.
    radius_eff = data_cfg.radius if data_cfg.radius > 0.0 else 1e-6
    sim = ParticleSim2D(
        W=1.0,
        H=1.0,
        radii=[radius_eff],
        masses=[data_cfg.mass],
        restitution=1.0,
        seed=train_cfg.seed,
        wall_mode=data_cfg.wall_collision_mode,
    )
    pos_all, vel_all, coll_all, meta = collect_episodes_1p(
        sim,
        E=data_cfg.episodes,
        steps=data_cfg.steps,
        dt=data_cfg.dt,
        speed_max=data_cfg.speed_max,
        seed=train_cfg.seed,
        stratified_init=data_cfg.stratified_init,
        pos_grid_n=data_cfg.pos_grid_n,
        angle_bins=data_cfg.angle_bins,
        episodes_per_bucket=data_cfg.episodes_per_bucket,
        fixed_speed=data_cfg.fixed_speed,
        fixed_x=data_cfg.fixed_x,
        fixed_y=data_cfg.fixed_y,
        fixed_vx=data_cfg.fixed_vx,
        fixed_vy=data_cfg.fixed_vy,
        fixed2_x=data_cfg.fixed2_x,
        fixed2_y=data_cfg.fixed2_y,
        fixed2_vx=data_cfg.fixed2_vx,
        fixed2_vy=data_cfg.fixed2_vy,
        ball_center_x=data_cfg.ball_center_x,
        ball_center_y=data_cfg.ball_center_y,
        ball_radius=data_cfg.ball_radius,
        fixed_vel_vx=data_cfg.fixed_vel_vx,
        fixed_vel_vy=data_cfg.fixed_vel_vy,
    )
    print(
        f"Generated episodes: pos_all={pos_all.shape}, vel_all={vel_all.shape}, "
        f"collision_frames={int(coll_all.sum())}/{coll_all.size}"
    )

    E = int(pos_all.shape[0])
    idx = np.arange(E)
    n_train = int(train_cfg.train_split * E)
    n_val = int(train_cfg.val_split * E)
    train_eps = idx[:n_train]
    val_eps = idx[n_train : n_train + n_val]
    test_eps = idx[n_train + n_val :]
    if len(train_eps) == 0 or len(val_eps) == 0 or len(test_eps) == 0:
        raise ValueError("Empty split encountered. Increase episodes or adjust train/val split.")
    print(f"Episode splits: train={len(train_eps)} val={len(val_eps)} test={len(test_eps)}")

    # Build query samples.
    X0_tr, Tq_tr, Y_tr, Evt_tr = build_query_samples(
        pos_all=pos_all,
        vel_all=vel_all,
        coll_all=coll_all,
        dt=meta["dt"],
        episode_indices=train_eps,
        eps_steps=data_cfg.coll_epsilon_steps,
        event_target_mode=data_cfg.event_target_mode,
        event_window=data_cfg.event_window,
        sigma_event=data_cfg.sigma_event,
        use_collision_oversampling=data_cfg.use_collision_oversampling,
        collision_sample_fraction=data_cfg.collision_sample_fraction,
        collision_window=data_cfg.collision_window,
    )
    X0_val, Tq_val, Y_val, Evt_val = build_query_samples(
        pos_all=pos_all,
        vel_all=vel_all,
        coll_all=coll_all,
        dt=meta["dt"],
        episode_indices=val_eps,
        eps_steps=data_cfg.coll_epsilon_steps,
        event_target_mode=data_cfg.event_target_mode,
        event_window=data_cfg.event_window,
        sigma_event=data_cfg.sigma_event,
        use_collision_oversampling=False,
        collision_sample_fraction=data_cfg.collision_sample_fraction,
        collision_window=data_cfg.collision_window,
    )
    X0_te, Tq_te, Y_te, Evt_te = build_query_samples(
        pos_all=pos_all,
        vel_all=vel_all,
        coll_all=coll_all,
        dt=meta["dt"],
        episode_indices=test_eps,
        eps_steps=data_cfg.coll_epsilon_steps,
        event_target_mode=data_cfg.event_target_mode,
        event_window=data_cfg.event_window,
        sigma_event=data_cfg.sigma_event,
        use_collision_oversampling=False,
        collision_sample_fraction=data_cfg.collision_sample_fraction,
        collision_window=data_cfg.collision_window,
    )
    print(
        f"Sample counts: train={len(X0_tr)} val={len(X0_val)} test={len(X0_te)} "
        f"| train_event_frac={Evt_tr.mean():.4f} val_event_frac={Evt_val.mean():.4f} test_event_frac={Evt_te.mean():.4f}"
    )

    # Optional normalization for s0 and target state.
    s0_std = Standardizer() if train_cfg.normalize else None
    y_std = Standardizer() if train_cfg.normalize else None
    if s0_std is not None and y_std is not None:
        s0_std.fit(X0_tr)
        y_std.fit(Y_tr)
        X0_tr = s0_std.transform(X0_tr)
        X0_val = s0_std.transform(X0_val)
        X0_te = s0_std.transform(X0_te)
        Y_tr = y_std.transform(Y_tr)
        Y_val = y_std.transform(Y_val)
        Y_te = y_std.transform(Y_te)

    train_ds = EpisodeQueryDataset(X0_tr, Tq_tr, Y_tr, Evt_tr)
    val_ds = EpisodeQueryDataset(X0_val, Tq_val, Y_val, Evt_val)
    test_ds = EpisodeQueryDataset(X0_te, Tq_te, Y_te, Evt_te)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False)

    model = TimeConditionedCollisionModel(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=1e-6)

    history: dict[str, list[float]] = {
        "train_total_loss": [],
        "train_pos_mse_loss": [],
        "train_vel_mse_loss": [],
        "train_event_bce_loss": [],
        "train_sign_bce_loss": [],
        "val_state_mse": [],
        "val_position_mse": [],
        "val_velocity_mse": [],
        "val_event_accuracy": [],
    }
    best = {"epoch": -1, "val_state_mse": float("inf"), "state_dict": None}

    # Training loop.
    for ep in range(1, train_cfg.epochs + 1):
        model.train()
        run_total = 0.0
        run_pos = 0.0
        run_vel = 0.0
        run_evt = 0.0
        run_sign = 0.0
        n_batches = 0
        for s0, t_q, y_state, y_event in train_loader:
            s0 = s0.to(device)
            t_q = t_q.to(device)
            y_state = y_state.to(device)
            y_event = y_event.to(device)

            out = model(s0, t_q)
            pred_state = out["state"]
            evt_logit = out["event_logit"]
            pos_mse = F.mse_loss(pred_state[:, :2], y_state[:, :2])
            vel_mse = F.mse_loss(pred_state[:, 2:], y_state[:, 2:])
            evt_bce = F.binary_cross_entropy_with_logits(evt_logit.squeeze(1), y_event)
            sign_bce = velocity_sign_loss(pred_state[:, 2:], y_state[:, 2:], sign_epsilon=run_cfg.sign_epsilon)
            loss = (
                lw.lambda_pos * pos_mse
                + lw.lambda_vel * vel_mse
                + lw.lambda_event * evt_bce
                + lw.lambda_sign * sign_bce
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_total += float(loss.item())
            run_pos += float(pos_mse.item())
            run_vel += float(vel_mse.item())
            run_evt += float(evt_bce.item())
            run_sign += float(sign_bce.item())
            n_batches += 1

        train_total = run_total / max(1, n_batches)
        train_pos = run_pos / max(1, n_batches)
        train_vel = run_vel / max(1, n_batches)
        train_evt = run_evt / max(1, n_batches)
        train_sign = run_sign / max(1, n_batches)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            state_std=y_std,
            event_threshold=run_cfg.event_prob_threshold,
            sign_epsilon=run_cfg.sign_epsilon,
        )
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            state_std=y_std,
            event_threshold=run_cfg.event_prob_threshold,
            sign_epsilon=run_cfg.sign_epsilon,
        )

        history["train_total_loss"].append(train_total)
        history["train_pos_mse_loss"].append(train_pos)
        history["train_vel_mse_loss"].append(train_vel)
        history["train_event_bce_loss"].append(train_evt)
        history["train_sign_bce_loss"].append(train_sign)
        history["val_state_mse"].append(val_metrics["state_mse"])
        history["val_position_mse"].append(val_metrics["position_mse"])
        history["val_velocity_mse"].append(val_metrics["velocity_mse"])
        history["val_event_accuracy"].append(val_metrics["event_accuracy"])

        if val_metrics["state_mse"] < best["val_state_mse"]:
            best["val_state_mse"] = val_metrics["state_mse"]
            best["epoch"] = ep
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": ep,
                    "train_total_loss": train_total,
                    "train_pos_mse_loss": train_pos,
                    "train_vel_mse_loss": train_vel,
                    "train_event_bce_loss": train_evt,
                    "train_sign_bce_loss": train_sign,
                    "val_state_mse": val_metrics["state_mse"],
                    "val_position_mse": val_metrics["position_mse"],
                    "val_velocity_mse": val_metrics["velocity_mse"],
                    "val_event_accuracy": val_metrics["event_accuracy"],
                    "val_event_precision": val_metrics["event_precision"],
                    "val_event_recall": val_metrics["event_recall"],
                    "val_sign_acc_vx": val_metrics["sign_acc_vx"],
                    "val_sign_acc_vy": val_metrics["sign_acc_vy"],
                    "test_state_mse": test_metrics["state_mse"],
                    "test_position_mse": test_metrics["position_mse"],
                    "test_velocity_mse": test_metrics["velocity_mse"],
                    "test_event_accuracy": test_metrics["event_accuracy"],
                    "test_event_precision": test_metrics["event_precision"],
                    "test_event_recall": test_metrics["event_recall"],
                    "test_sign_acc_vx": test_metrics["sign_acc_vx"],
                    "test_sign_acc_vy": test_metrics["sign_acc_vy"],
                    "best_val_state_mse_so_far": best["val_state_mse"],
                    "best_epoch_so_far": best["epoch"],
                }
            )

        print(
            f"Epoch {ep:04d} | train_total={train_total:.6f} pos={train_pos:.6f} vel={train_vel:.6f} evt={train_evt:.6f} sign={train_sign:.6f} "
            f"| val_state={val_metrics['state_mse']:.6f} val_evt_acc={val_metrics['event_accuracy']:.4f} "
            f"| test_state={test_metrics['state_mse']:.6f} test_evt_acc={test_metrics['event_accuracy']:.4f} "
            f"| best_epoch={best['epoch']}"
        )

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    # Final metrics using best-by-val model.
    final_train = evaluate(model, train_loader, device, y_std, run_cfg.event_prob_threshold, run_cfg.sign_epsilon)
    final_val = evaluate(model, val_loader, device, y_std, run_cfg.event_prob_threshold, run_cfg.sign_epsilon)
    final_test = evaluate(model, test_loader, device, y_std, run_cfg.event_prob_threshold, run_cfg.sign_epsilon)

    # Visualization for one test episode.
    plot_ep_local = int(np.clip(run_cfg.plot_episode_idx, 0, len(test_eps) - 1))
    ep = int(test_eps[plot_ep_local])
    pos_ep = pos_all[ep]
    vel_ep = vel_all[ep]
    T = pos_ep.shape[0]
    s0 = np.concatenate([pos_ep[0, 0], vel_ep[0, 0]], axis=0).astype(np.float32)
    t_q = (np.arange(T, dtype=np.float32) * float(meta["dt"])).astype(np.float32)
    true_state = np.concatenate([pos_ep[:, 0, :], vel_ep[:, 0, :]], axis=1).astype(np.float32)

    coll_steps = extract_collision_steps(coll_all[ep], vel_ep)
    evt_true = build_event_targets(
        T=T,
        collision_steps=coll_steps,
        dt=float(meta["dt"]),
        mode=data_cfg.event_target_mode,
        eps_steps=data_cfg.coll_epsilon_steps,
        event_window=data_cfg.event_window,
        sigma_event=data_cfg.sigma_event,
    )

    s0_batch = np.repeat(s0[None, :], T, axis=0).astype(np.float32)
    if s0_std is not None:
        s0_batch = s0_std.transform(s0_batch)

    model.eval()
    with torch.no_grad():
        s0_t = torch.from_numpy(s0_batch).to(device)
        tq_t = torch.from_numpy(t_q).to(device)
        out = model(s0_t, tq_t)
        pred_state = out["state"].cpu().numpy().astype(np.float32)
        if y_std is not None:
            pred_state = y_std.inverse(pred_state)
        evt_prob = torch.sigmoid(out["event_logit"].squeeze(1)).cpu().numpy().astype(np.float32)
        gate_np = None
        if model_cfg.model_variant == "gated_tcno":
            gate_np = out["gate"].squeeze(1).cpu().numpy().astype(np.float32)

    plot_state_and_event_over_time(
        t_query=t_q,
        true_state=true_state,
        pred_state=pred_state,
        event_prob=evt_prob,
        event_target=evt_true,
        gate=gate_np,
        out_path=out_dir / "state_and_event_vs_time.png",
    )

    # Also save overlay of predicted vs true positions over time.
    pos_true = true_state[:, :2].reshape(T, 1, 2).astype(np.float32)
    pos_pred = pred_state[:, :2].reshape(T, 1, 2).astype(np.float32)
    overlay = animate_overlay_gt_perturbed_1p(
        pos_ref=pos_true,
        pos_pert=pos_pred,
        radius=float(meta["radii"][0]),
        W=float(meta["W"]),
        H=float(meta["H"]),
        dt=float(meta["dt"]),
        title="TC collision model: GT vs predicted position",
        label_ref="GT",
        label_pert="Pred",
    )
    save_animation_mp4(overlay, str(out_dir / "trajectory_overlay_1p.mp4"), fps=50)

    # Save summary/checkpoint.
    summary = {
        "best_epoch": int(best["epoch"]),
        "best_val_state_mse": float(best["val_state_mse"]),
        "final_metrics": {
            "train": final_train,
            "val": final_val,
            "test": final_test,
        },
        "history": history,
        "configs": {
            "data": asdict(data_cfg),
            "train": asdict(train_cfg),
            "loss_weights": asdict(lw),
            "run": asdict(run_cfg),
            "model": {
                "state_dim": model_cfg.state_dim,
                "trunk_width": model_cfg.trunk_width,
                "trunk_depth": model_cfg.trunk_depth,
                "activation": model_cfg.activation,
                "dropout": model_cfg.dropout,
                "model_variant": model_cfg.model_variant,
                "alpha_gate": model_cfg.alpha_gate,
                "time_encoding_mode": model_cfg.time_encoding.mode,
                "base_frequency": model_cfg.time_encoding.base_frequency,
                "include_raw_time": model_cfg.time_encoding.include_raw_time,
                "normalize_time": model_cfg.time_encoding.normalize_time,
                "time_max": model_cfg.time_encoding.max_time,
                "num_frequencies": model_cfg.time_encoding.num_frequencies,
            },
        },
    }
    with open(out_dir / "summary_tc_collision_1p.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ckpt = {
        "model_name": "TimeConditionedCollisionModel",
        "model_config": {
            "state_dim": model_cfg.state_dim,
            "trunk_width": model_cfg.trunk_width,
            "trunk_depth": model_cfg.trunk_depth,
            "activation": model_cfg.activation,
            "dropout": model_cfg.dropout,
            "model_variant": model_cfg.model_variant,
            "alpha_gate": model_cfg.alpha_gate,
            "time_encoding_mode": model_cfg.time_encoding.mode,
            "base_frequency": model_cfg.time_encoding.base_frequency,
            "include_raw_time": model_cfg.time_encoding.include_raw_time,
            "normalize_time": model_cfg.time_encoding.normalize_time,
            "time_max": model_cfg.time_encoding.max_time,
            "num_frequencies": model_cfg.time_encoding.num_frequencies,
        },
        "model_state_dict": model.state_dict(),
        "s0_standardizer": None if s0_std is None else s0_std.state_dict(),
        "state_standardizer": None if y_std is None else y_std.state_dict(),
        "meta": meta,
        "split_indices": {
            "train_eps": train_eps.astype(np.int64),
            "val_eps": val_eps.astype(np.int64),
            "test_eps": test_eps.astype(np.int64),
        },
        "config": {
            "data": asdict(data_cfg),
            "train": asdict(train_cfg),
            "loss_weights": asdict(lw),
            "run": asdict(run_cfg),
        },
    }
    torch.save(ckpt, out_dir / "model_tc_collision_1p.pt")

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = int(best["epoch"])
        wandb_run.summary["best_val_state_mse"] = float(best["val_state_mse"])
        wandb_run.summary["final_train_state_mse"] = final_train["state_mse"]
        wandb_run.summary["final_val_state_mse"] = final_val["state_mse"]
        wandb_run.summary["final_test_state_mse"] = final_test["state_mse"]
        wandb_run.summary["final_test_position_mse"] = final_test["position_mse"]
        wandb_run.summary["final_test_velocity_mse"] = final_test["velocity_mse"]
        wandb_run.summary["final_test_event_accuracy"] = final_test["event_accuracy"]
        wandb_run.summary["final_test_event_precision"] = final_test["event_precision"]
        wandb_run.summary["final_test_event_recall"] = final_test["event_recall"]
        wandb_run.finish()

    print("Run complete.")
    print("Artifacts:")
    print(" -", out_dir / "model_tc_collision_1p.pt")
    print(" -", out_dir / "summary_tc_collision_1p.json")
    print(" -", out_dir / "state_and_event_vs_time.png")
    print(" -", out_dir / "trajectory_overlay_1p.mp4")


if __name__ == "__main__":
    main()
