import argparse
import json
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import wandb
except Exception:
    wandb = None

from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.one_particle_data import collect_episodes_1p
from particle_nn_sim.one_particle_rollout import (
    animate_overlay_gt_perturbed_1p,
    animate_side_by_side_1p,
    animate_single_rollout_1p,
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


def resolve_device(flag):
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HistoryPosDataset(Dataset):
    """
    Input: last H positions -> shape (2H,)
    Target: current/next position -> shape (2,)
    """

    def __init__(self, pos_all, episode_indices, history_len):
        self.pos_all = np.asarray(pos_all, dtype=np.float32)  # (E,T,1,2)
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.history_len = int(history_len)
        self.T = self.pos_all.shape[1]
        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")
        if self.T <= self.history_len:
            raise ValueError(
                f"Need T > history_len, got T={self.T}, history_len={self.history_len}"
            )
        self.starts_per_episode = self.T - self.history_len

    def __len__(self):
        return len(self.episode_indices) * self.starts_per_episode

    def __getitem__(self, idx):
        ep_idx = idx // self.starts_per_episode
        local_idx = idx % self.starts_per_episode
        e = int(self.episode_indices[ep_idx])
        t = local_idx + self.history_len

        hist = self.pos_all[e, t - self.history_len : t, 0, :]  # (H,2)
        target = self.pos_all[e, t, 0, :]  # (2,)
        x = hist.reshape(-1).astype(np.float32)  # (2H,)
        y = target.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class HistoryMLP(nn.Module):
    """
    Linear(2H,1024) -> ReLU -> Linear(1024,1024) -> ReLU -> Linear(1024,2)
    """

    def __init__(self, in_dim, width=1024, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def rollout_history_model(model, pos_true, history_len, device):
    """
    Autoregressive rollout with seed history from GT.
    pos_true: (T,1,2), T = steps+1
    """
    pos_true = np.asarray(pos_true, dtype=np.float32)
    T = pos_true.shape[0]
    H = int(history_len)
    if T <= H:
        raise ValueError(f"T must be > history_len, got T={T}, H={H}")

    pos_pred = np.zeros_like(pos_true, dtype=np.float32)
    pos_pred[:H, 0, :] = pos_true[:H, 0, :]  # seed with GT history
    model.eval()

    for t in range(H, T):
        hist = pos_pred[t - H : t, 0, :].reshape(1, -1).astype(np.float32)
        x = torch.from_numpy(hist).to(device)
        y = model(x).cpu().numpy()[0].astype(np.float32)
        y1 = y[:2]
        pos_pred[t, 0, :] = y1
        if not np.isfinite(y1).all():
            return pos_pred[: t + 1]
    return pos_pred


def train_model(model, train_loader, val_loader, test_loader, device, epochs, lr, lr_step_size, lr_gamma, wb_run=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=int(lr_step_size), gamma=float(lr_gamma))
    history = {"train_mse_pos": [], "val_mse_pos": [], "test_mse_pos": []}
    best = {"epoch": -1, "val_mse_pos": float("inf"), "state_dict": None}

    def eval_loader(loader):
        model.eval()
        sse = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                mse = ((pred - yb) ** 2).mean(dim=1)
                sse += mse.sum().item()
                n += mse.numel()
        return sse / max(n, 1)

    for ep in range(1, int(epochs) + 1):
        t0 = time.time()
        model.train()
        train_sse = 0.0
        train_n = 0
        lr_now = float(opt.param_groups[0]["lr"])
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            mse = ((pred - yb) ** 2).mean(dim=1)
            loss = mse.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_sse += mse.sum().item()
            train_n += mse.numel()

        train_mse = train_sse / max(train_n, 1)
        val_mse = eval_loader(val_loader)
        test_mse = eval_loader(test_loader)
        history["train_mse_pos"].append(train_mse)
        history["val_mse_pos"].append(val_mse)
        history["test_mse_pos"].append(test_mse)

        if val_mse < best["val_mse_pos"]:
            best["val_mse_pos"] = val_mse
            best["epoch"] = ep
            best["state_dict"] = deepcopy(model.state_dict())

        print(
            f"Epoch {ep:04d} | train_mse_pos={train_mse:.6f} | val_mse_pos={val_mse:.6f} "
            f"| test_mse_pos={test_mse:.6f} | lr={lr_now:.6g} | best_epoch={best['epoch']} "
            f"| sec={time.time()-t0:.2f}"
        )
        if wb_run is not None:
            wb_run.log(
                {
                    "epoch": ep,
                    "train_mse_pos": float(train_mse),
                    "val_mse_pos": float(val_mse),
                    "test_mse_pos": float(test_mse),
                    "lr": float(lr_now),
                    "best_epoch_so_far": int(best["epoch"]),
                    "best_val_mse_pos_so_far": float(best["val_mse_pos"]),
                    "epoch_seconds": float(time.time() - t0),
                }
            )
        sched.step()

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    return model, history, best


def parse_args():
    p = argparse.ArgumentParser(description="History-position MLP pipeline (10 past states -> current position).")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--speed-min", type=float, default=0.0)
    p.add_argument("--fixed-speed", type=float, default=None)
    p.add_argument("--radius", type=float, default=0.0)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--wall-collision-mode", type=str, default="clamp", choices=["clamp", "exact"])

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

    p.add_argument("--history-len", type=int, default=10)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--multistep-horizon", type=int, default=1)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lr-step-size", type=int, default=100)
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--train-split", type=float, default=0.7)
    p.add_argument("--val-split", type=float, default=0.15)

    p.add_argument("--rollout-steps", type=int, default=700)
    p.add_argument("--divergence-threshold", type=float, default=0.3)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--save-overlay-rollout", type=str2bool, default=True)
    p.add_argument("--save-side-by-side-rollout", type=str2bool, default=False)
    p.add_argument("--save-train-episode-preview", type=str2bool, default=True)
    p.add_argument("--preview-episode-idx", type=int, default=0)
    p.add_argument("--preview-fps", type=int, default=50)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="checkpoints/history_mlp_1p")
    p.add_argument("--use-wandb", type=str2bool, default=False)
    p.add_argument("--wandb-project", type=str, default="particle-nn-sim")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-tags", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    if args.history_len < 1:
        raise ValueError("--history-len must be >= 1")
    if args.multistep_horizon < 1:
        raise ValueError("--multistep-horizon must be >= 1")
    if args.multistep_horizon != 1:
        print(
            f"[warn] --multistep-horizon={args.multistep_horizon} requested, "
            "but this pipeline now trains one-step loss only. Using horizon=1."
        )
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
    wb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but --use-wandb=true was provided.")
        tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=(args.wandb_run_name or None),
            tags=tags,
            config=vars(args),
        )

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"Device: {device} ({gpu_name})")

    radius_eff = args.radius if args.radius > 0.0 else 1e-6
    sim = ParticleSim2D(
        W=1.0,
        H=1.0,
        radii=[radius_eff],
        masses=[args.mass],
        restitution=1.0,
        seed=args.seed,
        wall_mode=args.wall_collision_mode,
    )

    pos_all, vel_all, coll_all, meta = collect_episodes_1p(
        sim,
        E=args.episodes,
        steps=args.steps,
        dt=args.dt,
        speed_max=args.speed_max,
        speed_min=args.speed_min,
        seed=args.seed,
        stratified_init=args.stratified_init,
        pos_grid_n=args.pos_grid_n,
        angle_bins=args.angle_bins,
        episodes_per_bucket=args.episodes_per_bucket,
        fixed_speed=args.fixed_speed,
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
    )
    print(
        f"Generated episodes: pos_all={pos_all.shape}, vel_all={vel_all.shape}, "
        f"collision_frames={int(coll_all.sum())}/{coll_all.size}"
    )

    if args.save_train_episode_preview:
        if args.preview_episode_idx < 0 or args.preview_episode_idx >= pos_all.shape[0]:
            raise ValueError(
                f"--preview-episode-idx out of range [0, {pos_all.shape[0]-1}]"
            )
        anim = animate_single_rollout_1p(
            pos_all[int(args.preview_episode_idx)],
            radius=float(meta["radii"][0]),
            W=float(meta["W"]),
            H=float(meta["H"]),
            dt=float(meta["dt"]),
            title="Training Episode Example (GT)",
        )
        save_animation_mp4(anim, str(out_dir / "training_episode_example_1p.mp4"), fps=args.preview_fps)

    E = int(pos_all.shape[0])
    idx = np.arange(E)
    n_train = int(args.train_split * E)
    n_val = int(args.val_split * E)
    train_eps = idx[:n_train]
    val_eps = idx[n_train : n_train + n_val]
    test_eps = idx[n_train + n_val :]
    if len(train_eps) == 0 or len(val_eps) == 0 or len(test_eps) == 0:
        raise ValueError("Empty split encountered. Increase episodes or adjust train/val splits.")

    train_ds = HistoryPosDataset(pos_all, train_eps, args.history_len)
    val_ds = HistoryPosDataset(pos_all, val_eps, args.history_len)
    test_ds = HistoryPosDataset(pos_all, test_eps, args.history_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = HistoryMLP(
        in_dim=2 * int(args.history_len),
        width=int(args.width),
        out_dim=2,
    )
    model, hist, best = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        wb_run=wb_run,
    )

    # Rollout from first test episode (autoregressive with GT seed history).
    e0 = int(test_eps[0])
    pos0 = pos_all[e0, 0].astype(np.float32)
    vel0 = vel_all[e0, 0].astype(np.float32)
    sim_true = ParticleSim2D(
        W=float(meta["W"]),
        H=float(meta["H"]),
        radii=np.asarray(meta["radii"], dtype=float),
        masses=np.asarray(meta["masses"], dtype=float),
        restitution=float(meta["restitution"]),
        seed=args.seed + 123,
        wall_mode=args.wall_collision_mode,
    )
    sim_true.reset(pos0, vel0)
    pos_true, _ = sim_true.rollout(dt=float(meta["dt"]), steps=int(args.rollout_steps))
    pos_true = pos_true.astype(np.float32)
    pos_pred = rollout_history_model(
        model=model,
        pos_true=pos_true,
        history_len=int(args.history_len),
        device=device,
    )

    n_frames = min(len(pos_true), len(pos_pred))
    pos_true = pos_true[:n_frames]
    pos_pred = pos_pred[:n_frames]
    pos_err = np.linalg.norm(pos_true[:, 0, :] - pos_pred[:, 0, :], axis=1)
    div_idx = np.where(pos_err > float(args.divergence_threshold))[0]
    diverged = bool(len(div_idx) > 0)
    divergence_step = int(div_idx[0]) if diverged else int(n_frames - 1)

    if args.save_side_by_side_rollout:
        side = animate_side_by_side_1p(
            pos_true=pos_true,
            pos_pred=pos_pred,
            radius=float(meta["radii"][0]),
            W=float(meta["W"]),
            H=float(meta["H"]),
            dt=float(meta["dt"]),
        )
        save_animation_mp4(side, str(out_dir / "rollout_gt_vs_pred_1p.mp4"), fps=args.fps)
    if args.save_overlay_rollout:
        overlay = animate_overlay_gt_perturbed_1p(
            pos_ref=pos_true,
            pos_pert=pos_pred,
            radius=float(meta["radii"][0]),
            W=float(meta["W"]),
            H=float(meta["H"]),
            dt=float(meta["dt"]),
            title=f"GT vs NN rollout (history={int(args.history_len)})",
            label_ref="GT",
            label_pert="NN rollout",
        )
        save_animation_mp4(overlay, str(out_dir / "rollout_gt_vs_pred_overlay_1p.mp4"), fps=args.fps)

    err_plot_path = out_dir / "error_vs_timestep_1p.png"
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(pos_err)), pos_err, lw=2, color="tab:red")
    plt.xlabel("step")
    plt.ylabel("position error ||x_pred - x_true||")
    plt.title("History-MLP Rollout Error vs Time Step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(err_plot_path, dpi=150)
    plt.close()

    analysis = {
        "mean_position_error": float(np.mean(pos_err)),
        "max_position_error": float(np.max(pos_err)),
        "final_position_error": float(pos_err[-1]),
        "diverged": diverged,
        "divergence_step": int(divergence_step),
        "divergence_threshold": float(args.divergence_threshold),
        "best_epoch": int(best["epoch"]),
        "best_val_mse_pos": float(best["val_mse_pos"]),
        "history": hist,
        "error_plot_path": str(err_plot_path),
        "split_sizes": {
            "train_episodes": int(len(train_eps)),
            "val_episodes": int(len(val_eps)),
            "test_episodes": int(len(test_eps)),
        },
        "shape_checks": {
            "pos_all": list(pos_all.shape),
            "vel_all": list(vel_all.shape),
        },
        "config": vars(args),
    }
    with open(out_dir / "analysis_1p.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    if wb_run is not None:
        wb_run.summary["best_epoch"] = int(best["epoch"])
        wb_run.summary["best_val_mse_pos"] = float(best["val_mse_pos"])
        wb_run.summary["final_train_mse_pos"] = float(hist["train_mse_pos"][-1]) if hist["train_mse_pos"] else float("nan")
        wb_run.summary["final_val_mse_pos"] = float(hist["val_mse_pos"][-1]) if hist["val_mse_pos"] else float("nan")
        wb_run.summary["final_test_mse_pos"] = float(hist["test_mse_pos"][-1]) if hist["test_mse_pos"] else float("nan")

    ckpt = {
        "model_name": "HistoryMLP",
        "model_kwargs": {
            "in_dim": 2 * int(args.history_len),
            "width": int(args.width),
            "out_dim": 2,
        },
        "model_state_dict": model.state_dict(),
        "best_epoch": int(best["epoch"]),
        "best_val_mse_pos": float(best["val_mse_pos"]),
        "meta": meta,
        "split_indices": {
            "train_eps": np.asarray(train_eps, dtype=np.int64),
            "val_eps": np.asarray(val_eps, dtype=np.int64),
            "test_eps": np.asarray(test_eps, dtype=np.int64),
        },
        "episode_init": {
            "pos0": pos_all[:, 0].astype(np.float32),
            "vel0": vel_all[:, 0].astype(np.float32),
        },
        "config": vars(args),
    }
    torch.save(ckpt, out_dir / "model_1p_history_mlp.pt")

    print("Run complete.")
    print("Artifacts:")
    print(" -", out_dir / "model_1p_history_mlp.pt")
    print(" -", out_dir / "analysis_1p.json")
    print(" -", out_dir / "error_vs_timestep_1p.png")
    if args.save_side_by_side_rollout:
        print(" -", out_dir / "rollout_gt_vs_pred_1p.mp4")
    if args.save_overlay_rollout:
        print(" -", out_dir / "rollout_gt_vs_pred_overlay_1p.mp4")
    if args.save_train_episode_preview:
        print(" -", out_dir / "training_episode_example_1p.mp4")
    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
