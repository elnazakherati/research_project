import argparse
import json
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.models import ResMLP
from particle_nn_sim.train import fit_standardizer, apply_standardizer, StepDataset
from particle_nn_sim.one_particle_data import collect_episodes_1p, episodes_to_XY_residual_1p
from particle_nn_sim.one_particle_rollout import (
    animate_single_rollout_1p,
    animate_side_by_side_1p,
    nn_rollout_residual_1p,
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


def parse_args():
    p = argparse.ArgumentParser(description="One-particle wall-bounce training pipeline")

    # Data
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--radius", type=float, default=0.0)
    p.add_argument("--mass", type=float, default=1.0)

    # Model/train
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--collision-weight", type=float, default=10.0)
    p.add_argument("--multistep-horizon", type=int, default=10)
    p.add_argument("--rebalance-sampling", type=str2bool, default=True)
    p.add_argument("--target-collision-frac", type=float, default=0.3)

    # Eval/output
    p.add_argument("--rollout-steps", type=int, default=1000)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--out-dir", type=str, default="checkpoints/one_particle_run")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    # Preview
    p.add_argument("--save-train-episode-preview", type=str2bool, default=True)
    p.add_argument("--preview-episode-idx", type=int, default=0)
    p.add_argument("--preview-fps", type=int, default=50)
    p.add_argument("--use-wandb", type=str2bool, default=False)
    p.add_argument("--wandb-project", type=str, default="particle-nn-sim")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-tags", type=str, default="")

    return p.parse_args()


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


class RolloutDataset1P(Dataset):
    """Provides fixed-horizon state sequences from one-particle episodes."""

    def __init__(self, pos_all, vel_all, coll_all, episode_indices, horizon):
        self.pos_all = pos_all
        self.vel_all = vel_all
        self.coll_all = coll_all
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.horizon = int(horizon)
        self.T = pos_all.shape[1]
        self.starts_per_episode = self.T - self.horizon
        if self.starts_per_episode <= 0:
            raise ValueError(
                f"horizon={self.horizon} too large for T={self.T}. Need horizon < T."
            )

    def __len__(self):
        return len(self.episode_indices) * self.starts_per_episode

    def _decode(self, idx):
        ep_idx = idx // self.starts_per_episode
        t0 = idx % self.starts_per_episode
        e = int(self.episode_indices[ep_idx])
        return e, t0

    def __getitem__(self, idx):
        e, t0 = self._decode(idx)
        state0 = np.concatenate(
            [self.pos_all[e, t0, 0, :], self.vel_all[e, t0, 0, :]], axis=0
        ).astype(np.float32)

        pos_future = self.pos_all[e, t0 + 1 : t0 + 1 + self.horizon, 0, :]  # (H,2)
        vel_future = self.vel_all[e, t0 + 1 : t0 + 1 + self.horizon, 0, :]  # (H,2)
        state_future = np.concatenate([pos_future, vel_future], axis=1).astype(np.float32)  # (H,4)
        coll_future = self.coll_all[e, t0 : t0 + self.horizon].astype(np.uint8)  # (H,)
        return (
            torch.from_numpy(state0).float(),
            torch.from_numpy(state_future).float(),
            torch.from_numpy(coll_future).long(),
        )

    def collision_window_labels(self):
        labels = np.zeros(len(self), dtype=np.int64)
        out_idx = 0
        for e in self.episode_indices:
            c = self.coll_all[int(e)].astype(np.int64)  # (T-1,)
            csum = np.concatenate([[0], np.cumsum(c)])  # (T,)
            # window [t, t+h-1]
            hits = (csum[self.horizon :] - csum[: self.T - self.horizon]) > 0
            n = len(hits)
            labels[out_idx : out_idx + n] = hits.astype(np.int64)
            out_idx += n
        return labels


def make_weighted_sampler(binary_labels, target_collision_frac):
    labels = np.asarray(binary_labels, dtype=np.int64).reshape(-1)
    n_col = int(labels.sum())
    n_non = int((labels == 0).sum())
    if n_col == 0 or n_non == 0:
        return None

    p_col = float(target_collision_frac)
    alpha = (p_col / (1.0 - p_col)) * (n_non / max(1, n_col))
    weights = np.ones_like(labels, dtype=np.float64)
    weights[labels == 1] = alpha
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def train_multistep_1p(
    model,
    train_loader,
    test_loader,
    device,
    epochs,
    lr,
    collision_weight,
    x_mean,
    x_std,
    y_mean,
    y_std,
    dt,
    radius,
    mass,
    wandb_run=None,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.as_tensor(x_std, dtype=torch.float32, device=device)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    dt_t = float(dt)
    r_t = float(radius)
    m_t = float(mass)
    cw = float(collision_weight)

    history = {"train_loss": [], "test_mse_all": [], "test_mse_collision": [], "test_mse_noncollision": []}
    best = {"epoch": -1, "mse_all": float("inf"), "state_dict": None, "stats": None}

    def eval_loader(loader):
        model.eval()
        mse_all_sum = 0.0
        mse_col_sum = 0.0
        mse_non_sum = 0.0
        n_all = n_col = n_non = 0
        with torch.no_grad():
            for state0, future_gt, coll_future in loader:
                state0 = state0.to(device)  # (B,4)
                future_gt = future_gt.to(device)  # (B,H,4)
                coll_future = coll_future.to(device).bool()  # (B,H)

                B, H, _ = future_gt.shape
                state = state0
                per_step_mse = []
                per_step_coll = []
                for k in range(H):
                    x_raw = torch.cat(
                        [
                            state,
                            torch.full((B, 1), r_t, device=device),
                            torch.full((B, 1), m_t, device=device),
                        ],
                        dim=1,
                    )  # (B,6)
                    x_n = (x_raw - x_mean_t) / x_std_t
                    resid_n = model(x_n)
                    resid = resid_n * y_std_t + y_mean_t  # (B,4)

                    pos = state[:, 0:2]
                    vel = state[:, 2:4]
                    pos_free = pos + vel * dt_t
                    state_free = torch.cat([pos_free, vel], dim=1)
                    state = state_free + resid

                    gt_k = future_gt[:, k, :]
                    mse_k = ((state - gt_k) ** 2).mean(dim=1)  # (B,)
                    per_step_mse.append(mse_k)
                    per_step_coll.append(coll_future[:, k])

                mse = torch.stack(per_step_mse, dim=1)  # (B,H)
                coll = torch.stack(per_step_coll, dim=1)  # (B,H)
                non = ~coll

                mse_all_sum += mse.sum().item()
                n_all += mse.numel()

                if coll.any():
                    mse_col_sum += mse[coll].sum().item()
                    n_col += int(coll.sum().item())
                if non.any():
                    mse_non_sum += mse[non].sum().item()
                    n_non += int(non.sum().item())

        return {
            "mse_all": mse_all_sum / max(n_all, 1),
            "mse_collision": mse_col_sum / max(n_col, 1),
            "mse_noncollision": mse_non_sum / max(n_non, 1),
            "n_all": n_all,
            "n_collision": n_col,
            "n_noncollision": n_non,
        }

    for ep in range(1, int(epochs) + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        n_samples = 0

        for state0, future_gt, coll_future in train_loader:
            state0 = state0.to(device)
            future_gt = future_gt.to(device)
            coll_future = coll_future.to(device).float()
            B, H, _ = future_gt.shape

            state = state0
            step_losses = []
            for k in range(H):
                x_raw = torch.cat(
                    [
                        state,
                        torch.full((B, 1), r_t, device=device),
                        torch.full((B, 1), m_t, device=device),
                    ],
                    dim=1,
                )
                x_n = (x_raw - x_mean_t) / x_std_t
                resid_n = model(x_n)
                resid = resid_n * y_std_t + y_mean_t

                pos = state[:, 0:2]
                vel = state[:, 2:4]
                pos_free = pos + vel * dt_t
                state_free = torch.cat([pos_free, vel], dim=1)
                state = state_free + resid

                gt_k = future_gt[:, k, :]
                mse_k = ((state - gt_k) ** 2).mean(dim=1)  # (B,)
                if cw != 1.0:
                    w = torch.where(coll_future[:, k] > 0, torch.full_like(mse_k, cw), torch.ones_like(mse_k))
                    mse_k = w * mse_k
                step_losses.append(mse_k.mean())

            loss = torch.stack(step_losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item() * B
            n_samples += B

        train_loss = running / max(n_samples, 1)
        stats = eval_loader(test_loader)
        history["train_loss"].append(train_loss)
        history["test_mse_all"].append(stats["mse_all"])
        history["test_mse_collision"].append(stats["mse_collision"])
        history["test_mse_noncollision"].append(stats["mse_noncollision"])

        if stats["mse_all"] < best["mse_all"]:
            best["mse_all"] = stats["mse_all"]
            best["epoch"] = ep
            best["state_dict"] = deepcopy(model.state_dict())
            best["stats"] = stats
        epoch_sec = time.time() - t0

        print(
            f"Epoch {ep:03d} | train_loss={train_loss:.6f} "
            f"| test_mse={stats['mse_all']:.6f} "
            f"| test_collision={stats['mse_collision']:.6f} "
            f"| test_noncollision={stats['mse_noncollision']:.6f} "
            f"(n_col={stats['n_collision']}, n_noncol={stats['n_noncollision']}) "
            f"| best_epoch={best['epoch']} | sec={epoch_sec:.2f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": ep,
                    "train_loss": train_loss,
                    "test_mse_all": stats["mse_all"],
                    "test_mse_collision": stats["mse_collision"],
                    "test_mse_noncollision": stats["mse_noncollision"],
                    "test_n_collision": stats["n_collision"],
                    "test_n_noncollision": stats["n_noncollision"],
                    "best_mse_all_so_far": best["mse_all"],
                    "best_epoch_so_far": best["epoch"],
                    "epoch_seconds": epoch_sec,
                }
            )

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    return model, best["stats"], history, best


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "cpu"

    print(f"Device: {device} ({gpu_name})")
    print(
        "Config:",
        {
            "episodes": args.episodes,
            "steps": args.steps,
            "dt": args.dt,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "multistep_horizon": args.multistep_horizon,
            "collision_weight": args.collision_weight,
            "rebalance_sampling": args.rebalance_sampling,
            "target_collision_frac": args.target_collision_frac,
            "seed": args.seed,
        },
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Radius zero can create edge ambiguity at exact wall contacts; use tiny epsilon.
    radius_eff = args.radius if args.radius > 0.0 else 1e-6

    sim = ParticleSim2D(
        W=1.0,
        H=1.0,
        radii=[radius_eff],
        masses=[args.mass],
        restitution=1.0,
        seed=args.seed,
    )

    pos_all, vel_all, coll_all, meta = collect_episodes_1p(
        sim,
        E=args.episodes,
        steps=args.steps,
        dt=args.dt,
        speed_max=args.speed_max,
        seed=args.seed,
    )
    print(
        f"Generated episodes: pos_all={pos_all.shape}, vel_all={vel_all.shape}, "
        f"collision_frames={int(coll_all.sum())}/{coll_all.size}"
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
        init_kwargs = {
            "project": args.wandb_project,
            "config": vars(args),
            "tags": tags,
            "dir": str(out_dir),
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        wandb_run = wandb.init(**init_kwargs)

    # Preview one raw training episode from GT.
    if args.save_train_episode_preview:
        if args.preview_episode_idx < 0 or args.preview_episode_idx >= pos_all.shape[0]:
            raise ValueError(
                f"preview_episode_idx={args.preview_episode_idx} out of range [0, {pos_all.shape[0]-1}]"
            )
        pos_preview = pos_all[args.preview_episode_idx]
        preview_anim = animate_single_rollout_1p(
            pos_preview,
            radius=float(meta["radii"][0]),
            W=float(meta["W"]),
            H=float(meta["H"]),
            dt=float(meta["dt"]),
            title="Training Episode Example (GT)",
        )
        save_animation_mp4(
            preview_anim,
            str(out_dir / "training_episode_example_1p.mp4"),
            fps=args.preview_fps,
        )

    # Train/test split by episode.
    E = pos_all.shape[0]
    idx = np.arange(E)
    n_train = int(args.train_split * E)
    train_eps = idx[:n_train]
    test_eps = idx[n_train:]
    if len(test_eps) == 0:
        raise ValueError("No test episodes. Lower --train-split or increase --episodes.")

    Xtr, Ytr, Ctr = episodes_to_XY_residual_1p(pos_all, vel_all, coll_all, meta, train_eps)
    Xte, Yte, Cte = episodes_to_XY_residual_1p(pos_all, vel_all, coll_all, meta, test_eps)
    if Xtr.shape[1] != 6 or Ytr.shape[1] != 4:
        raise RuntimeError(f"Unexpected shapes: Xtr={Xtr.shape}, Ytr={Ytr.shape}")

    x_mean, x_std = fit_standardizer(Xtr)
    y_mean, y_std = fit_standardizer(Ytr)
    Xtr_n = apply_standardizer(Xtr, x_mean, x_std)
    Ytr_n = apply_standardizer(Ytr, y_mean, y_std)
    Xte_n = apply_standardizer(Xte, x_mean, x_std)
    Yte_n = apply_standardizer(Yte, y_mean, y_std)

    # Build rollout datasets for multi-step training.
    train_roll_ds = RolloutDataset1P(pos_all, vel_all, coll_all, train_eps, args.multistep_horizon)
    test_roll_ds = RolloutDataset1P(pos_all, vel_all, coll_all, test_eps, args.multistep_horizon)
    print(
        f"Dataset windows: train={len(train_roll_ds)}, test={len(test_roll_ds)} "
        f"| one-step train={Xtr.shape[0]}, test={Xte.shape[0]}"
    )

    if args.rebalance_sampling:
        labels = train_roll_ds.collision_window_labels()
        sampler = make_weighted_sampler(labels, args.target_collision_frac)
        print(
            f"Rebalance labels: collision_windows={int(labels.sum())}, "
            f"noncollision_windows={int((labels==0).sum())}, "
            f"sampler={'on' if sampler is not None else 'off'}"
        )
    else:
        sampler = None
        print("Rebalance sampling disabled.")

    train_roll_loader = DataLoader(
        train_roll_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )
    test_roll_loader = DataLoader(
        test_roll_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ResMLP(
        in_dim=6,
        hidden=args.hidden,
        out_dim=4,
        blocks=args.blocks,
        dropout=args.dropout,
    )
    model, stats, hist, best = train_multistep_1p(
        model,
        train_roll_loader,
        test_roll_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        collision_weight=args.collision_weight,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        dt=float(meta["dt"]),
        radius=float(meta["radii"][0]),
        mass=float(meta["masses"][0]),
        wandb_run=wandb_run,
    )

    # Rollout GT and model from the same test initial condition.
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
    )
    sim_true.reset(pos0, vel0)
    pos_true, vel_true = sim_true.rollout(dt=float(meta["dt"]), steps=args.rollout_steps)
    pos_true = pos_true.astype(np.float32)
    vel_true = vel_true.astype(np.float32)

    pos_pred, vel_pred = nn_rollout_residual_1p(
        model=model,
        pos0=pos0,
        vel0=vel0,
        radius=float(meta["radii"][0]),
        mass=float(meta["masses"][0]),
        steps=args.rollout_steps,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        dt=float(meta["dt"]),
    )

    # Save GT-vs-pred animation.
    anim = animate_side_by_side_1p(
        pos_true=pos_true,
        pos_pred=pos_pred,
        radius=float(meta["radii"][0]),
        W=float(meta["W"]),
        H=float(meta["H"]),
        dt=float(meta["dt"]),
    )
    save_animation_mp4(anim, str(out_dir / "rollout_gt_vs_pred_1p.mp4"), fps=args.fps)

    # Analysis metrics.
    pos_err = np.linalg.norm(pos_true[:, 0, :] - pos_pred[:, 0, :], axis=1)
    err_plot_path = out_dir / "error_vs_timestep_1p.png"
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(pos_err)), pos_err, lw=2, color="tab:red")
    plt.xlabel("step")
    plt.ylabel("position error ||x_pred - x_true||")
    plt.title("One-Particle Rollout Error vs Time Step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(err_plot_path, dpi=150)
    plt.close()
    analysis = {
        "mean_position_error": float(np.mean(pos_err)),
        "max_position_error": float(np.max(pos_err)),
        "final_position_error": float(pos_err[-1]),
        "error_plot_path": str(err_plot_path),
        "test_stats": stats,
        "best_epoch": int(best["epoch"]),
        "shape_checks": {
            "pos_all": list(pos_all.shape),
            "vel_all": list(vel_all.shape),
            "Xtr": list(Xtr.shape),
            "Ytr": list(Ytr.shape),
        },
        "config": vars(args),
    }
    with open(out_dir / "analysis_1p.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    # Checkpoint.
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_name": "ResMLP",
        "model_kwargs": {
            "in_dim": 6,
            "hidden": args.hidden,
            "out_dim": 4,
            "blocks": args.blocks,
            "dropout": args.dropout,
        },
        "hist": hist,
        "stats": stats,
        "best_epoch": int(best["epoch"]),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "meta": meta,
        "config": vars(args),
    }
    torch.save(ckpt, out_dir / "model_1p_resmlp.pt")

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = int(best["epoch"])
        wandb_run.summary["mean_position_error"] = analysis["mean_position_error"]
        wandb_run.summary["max_position_error"] = analysis["max_position_error"]
        wandb_run.summary["final_position_error"] = analysis["final_position_error"]
        wandb_run.summary["final_test_mse_all"] = stats["mse_all"]
        wandb_run.summary["final_test_mse_collision"] = stats["mse_collision"]
        wandb_run.summary["final_test_mse_noncollision"] = stats["mse_noncollision"]
        wandb_run.finish()

    print("Run complete.")
    print("Artifacts:")
    print(" -", out_dir / "model_1p_resmlp.pt")
    print(" -", out_dir / "analysis_1p.json")
    print(" -", out_dir / "error_vs_timestep_1p.png")
    print(" -", out_dir / "rollout_gt_vs_pred_1p.mp4")
    if args.save_train_episode_preview:
        print(" -", out_dir / "training_episode_example_1p.mp4")


if __name__ == "__main__":
    main()
