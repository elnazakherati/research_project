import argparse
import json
from copy import deepcopy
from pathlib import Path
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_nn_sim.one_particle_data import collect_episodes_1p
from particle_nn_sim.one_particle_rollout import (
    animate_side_by_side_1p,
    animate_single_rollout_1p,
    save_animation_mp4,
)
from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.train import fit_standardizer


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
    p = argparse.ArgumentParser(description="One-particle RNN (GRU/LSTM) training pipeline")
    p.add_argument("--rnn-type", type=str, default="gru", choices=["gru", "lstm"])

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
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--collision-weight", type=float, default=2.0)
    p.add_argument("--rebalance-sampling", type=str2bool, default=True)
    p.add_argument("--target-collision-frac", type=float, default=0.15)

    # Eval/output
    p.add_argument("--rollout-steps", type=int, default=2000)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--out-dir", type=str, default="checkpoints/one_particle_rnn_run")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    # Preview
    p.add_argument("--save-train-episode-preview", type=str2bool, default=True)
    p.add_argument("--preview-episode-idx", type=int, default=0)
    p.add_argument("--preview-fps", type=int, default=50)
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


class RNNDynamics(nn.Module):
    def __init__(self, in_dim=6, hidden=128, out_dim=4, layers=2, dropout=0.0, rnn_type="gru"):
        super().__init__()
        rnn_dropout = float(dropout) if int(layers) > 1 else 0.0
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden,
                num_layers=layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden,
                num_layers=layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        self.head = nn.Linear(hidden, out_dim)
        self.rnn_type = rnn_type

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, h=None):
        o, h = self.rnn(x, h)
        y = self.head(o)
        return y, h


class SeqDataset1P(Dataset):
    def __init__(self, pos_all, vel_all, coll_all, episode_indices, seq_len, radius, mass):
        self.pos_all = pos_all
        self.vel_all = vel_all
        self.coll_all = coll_all
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.radius = float(radius)
        self.mass = float(mass)
        self.T = pos_all.shape[1]
        self.starts_per_episode = self.T - self.seq_len
        if self.starts_per_episode <= 0:
            raise ValueError(f"seq_len={self.seq_len} too large for T={self.T}. Need seq_len < T.")

    def __len__(self):
        return len(self.episode_indices) * self.starts_per_episode

    def __getitem__(self, idx):
        ep_i = idx // self.starts_per_episode
        t0 = idx % self.starts_per_episode
        e = int(self.episode_indices[ep_i])

        pos = self.pos_all[e, t0 : t0 + self.seq_len + 1, 0, :]  # (L+1,2)
        vel = self.vel_all[e, t0 : t0 + self.seq_len + 1, 0, :]  # (L+1,2)
        coll = self.coll_all[e, t0 : t0 + self.seq_len]  # (L,)

        x_state = np.concatenate([pos[:-1], vel[:-1]], axis=1).astype(np.float32)  # (L,4)
        y_next = np.concatenate([pos[1:], vel[1:]], axis=1).astype(np.float32)  # (L,4)

        x = np.zeros((self.seq_len, 6), dtype=np.float32)
        x[:, 0:4] = x_state
        x[:, 4] = self.radius
        x[:, 5] = self.mass
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y_next).float(),
            torch.from_numpy(coll.astype(np.float32)).float(),
        )

    def collision_window_labels(self):
        labels = np.zeros(len(self), dtype=np.int64)
        out_idx = 0
        for e in self.episode_indices:
            c = self.coll_all[int(e)].astype(np.int64)
            csum = np.concatenate([[0], np.cumsum(c)])
            hits = (csum[self.seq_len :] - csum[: self.T - self.seq_len]) > 0
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


@torch.no_grad()
def rnn_rollout_1p(model, pos0, vel0, radius, mass, steps, x_mean, x_std, y_mean, y_std, device):
    model.eval()
    steps = int(steps)
    x_mean = np.asarray(x_mean, dtype=np.float32).reshape(-1)
    x_std = np.asarray(x_std, dtype=np.float32).reshape(-1)
    y_mean = np.asarray(y_mean, dtype=np.float32).reshape(-1)
    y_std = np.asarray(y_std, dtype=np.float32).reshape(-1)
    pos_pred = np.zeros((steps + 1, 1, 2), dtype=np.float32)
    vel_pred = np.zeros((steps + 1, 1, 2), dtype=np.float32)
    pos_pred[0] = pos0.astype(np.float32)
    vel_pred[0] = vel0.astype(np.float32)

    state = np.concatenate([pos_pred[0, 0], vel_pred[0, 0]], axis=0).astype(np.float32)  # (4,)
    h = None
    for t in range(steps):
        x = np.array([state[0], state[1], state[2], state[3], float(radius), float(mass)], dtype=np.float32)
        x_n = ((x[None, None, :] - x_mean[None, None, :]) / x_std[None, None, :]).astype(np.float32)
        x_t = torch.from_numpy(x_n).to(device)
        y_n, h = model(x_t, h)
        y = (y_n[:, -1, :].cpu().numpy() * y_std[None, :]) + y_mean[None, :]
        state = y[0].astype(np.float32)
        pos_pred[t + 1, 0] = state[0:2]
        vel_pred[t + 1, 0] = state[2:4]
    return pos_pred, vel_pred


def train_rnn(
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
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-6)
    cw = float(collision_weight)

    x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32, device=device).view(1, 1, -1)
    x_std_t = torch.as_tensor(x_std, dtype=torch.float32, device=device).view(1, 1, -1)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device).view(1, 1, -1)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device).view(1, 1, -1)

    history = {"train_loss": [], "test_mse_all": [], "test_mse_collision": [], "test_mse_noncollision": []}
    best = {"epoch": -1, "mse_all": float("inf"), "state_dict": None, "stats": None}

    @torch.no_grad()
    def eval_loader(loader):
        model.eval()
        mse_all_sum = 0.0
        mse_col_sum = 0.0
        mse_non_sum = 0.0
        n_all = n_col = n_non = 0
        for x, y, c in loader:
            x = x.to(device)
            y = y.to(device)
            c = c.to(device).bool()

            x_n = (x - x_mean_t) / x_std_t
            y_n = (y - y_mean_t) / y_std_t
            pred_n, _ = model(x_n, None)
            mse = ((pred_n - y_n) ** 2).mean(dim=2)  # (B,L)

            mse_all_sum += mse.sum().item()
            n_all += mse.numel()

            if c.any():
                mse_col_sum += mse[c].sum().item()
                n_col += int(c.sum().item())
            non = ~c
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
        for x, y, c in train_loader:
            x = x.to(device)
            y = y.to(device)
            c = c.to(device)

            x_n = (x - x_mean_t) / x_std_t
            y_n = (y - y_mean_t) / y_std_t
            pred_n, _ = model(x_n, None)

            mse = ((pred_n - y_n) ** 2).mean(dim=2)  # (B,L)
            if cw != 1.0:
                w = torch.where(c > 0, torch.full_like(mse, cw), torch.ones_like(mse))
                mse = mse * w
            loss = mse.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            B = x.shape[0]
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

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    return model, best["stats"], history, best


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"Device: {device} ({gpu_name})")
    print("Config:", vars(args))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    if args.save_train_episode_preview:
        if args.preview_episode_idx < 0 or args.preview_episode_idx >= pos_all.shape[0]:
            raise ValueError(
                f"preview_episode_idx={args.preview_episode_idx} out of range [0, {pos_all.shape[0]-1}]"
            )
        preview_anim = animate_single_rollout_1p(
            pos_all[args.preview_episode_idx],
            radius=float(meta["radii"][0]),
            W=float(meta["W"]),
            H=float(meta["H"]),
            dt=float(meta["dt"]),
            title="Training Episode Example (GT)",
        )
        save_animation_mp4(preview_anim, str(out_dir / "training_episode_example_1p.mp4"), fps=args.preview_fps)

    E = pos_all.shape[0]
    idx = np.arange(E)
    n_train = int(args.train_split * E)
    train_eps = idx[:n_train]
    test_eps = idx[n_train:]
    if len(test_eps) == 0:
        raise ValueError("No test episodes. Lower --train-split or increase --episodes.")

    train_ds = SeqDataset1P(
        pos_all,
        vel_all,
        coll_all,
        train_eps,
        seq_len=args.seq_len,
        radius=float(meta["radii"][0]),
        mass=float(meta["masses"][0]),
    )
    test_ds = SeqDataset1P(
        pos_all,
        vel_all,
        coll_all,
        test_eps,
        seq_len=args.seq_len,
        radius=float(meta["radii"][0]),
        mass=float(meta["masses"][0]),
    )
    print(f"Dataset windows: train={len(train_ds)}, test={len(test_ds)}")

    if args.rebalance_sampling:
        labels = train_ds.collision_window_labels()
        sampler = make_weighted_sampler(labels, args.target_collision_frac)
        print(
            f"Rebalance labels: collision_windows={int(labels.sum())}, "
            f"noncollision_windows={int((labels==0).sum())}, "
            f"sampler={'on' if sampler is not None else 'off'}"
        )
    else:
        sampler = None
        print("Rebalance sampling disabled.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Standardizers from train episodes (one-step flattened views)
    train_state = np.concatenate([pos_all[train_eps, :-1, 0, :], vel_all[train_eps, :-1, 0, :]], axis=2).reshape(-1, 4)
    train_next = np.concatenate([pos_all[train_eps, 1:, 0, :], vel_all[train_eps, 1:, 0, :]], axis=2).reshape(-1, 4)
    x_feat = np.zeros((train_state.shape[0], 6), dtype=np.float32)
    x_feat[:, 0:4] = train_state
    x_feat[:, 4] = float(meta["radii"][0])
    x_feat[:, 5] = float(meta["masses"][0])
    y_feat = train_next.astype(np.float32)
    x_mean, x_std = fit_standardizer(x_feat)
    y_mean, y_std = fit_standardizer(y_feat)

    model = RNNDynamics(
        in_dim=6,
        hidden=args.hidden,
        out_dim=4,
        layers=args.layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
    )
    model, stats, hist, best = train_rnn(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        collision_weight=args.collision_weight,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

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

    pos_pred, vel_pred = rnn_rollout_1p(
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
    )

    name = args.rnn_type.lower()
    mp4_name = f"rollout_gt_vs_pred_1p_{name}.mp4"
    err_plot_name = f"error_vs_timestep_1p_{name}.png"
    ckpt_name = f"model_1p_{name}.pt"
    analysis_name = f"analysis_1p_{name}.json"

    anim = animate_side_by_side_1p(
        pos_true=pos_true,
        pos_pred=pos_pred,
        radius=float(meta["radii"][0]),
        W=float(meta["W"]),
        H=float(meta["H"]),
        dt=float(meta["dt"]),
    )
    save_animation_mp4(anim, str(out_dir / mp4_name), fps=args.fps)

    pos_err = np.linalg.norm(pos_true[:, 0, :] - pos_pred[:, 0, :], axis=1)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(pos_err)), pos_err, lw=2, color="tab:red")
    plt.xlabel("step")
    plt.ylabel("position error ||x_pred - x_true||")
    plt.title(f"One-Particle {name.upper()} Rollout Error vs Time Step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / err_plot_name, dpi=150)
    plt.close()

    analysis = {
        "rnn_type": name,
        "mean_position_error": float(np.mean(pos_err)),
        "max_position_error": float(np.max(pos_err)),
        "final_position_error": float(pos_err[-1]),
        "test_stats": stats,
        "best_epoch": int(best["epoch"]),
        "shape_checks": {
            "pos_all": list(pos_all.shape),
            "vel_all": list(vel_all.shape),
            "train_windows": int(len(train_ds)),
            "test_windows": int(len(test_ds)),
        },
        "config": vars(args),
    }
    with open(out_dir / analysis_name, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_name": "RNNDynamics",
        "model_kwargs": {
            "in_dim": 6,
            "hidden": args.hidden,
            "out_dim": 4,
            "layers": args.layers,
            "dropout": args.dropout,
            "rnn_type": args.rnn_type,
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
    torch.save(ckpt, out_dir / ckpt_name)

    print("Run complete.")
    print("Artifacts:")
    print(" -", out_dir / ckpt_name)
    print(" -", out_dir / analysis_name)
    print(" -", out_dir / err_plot_name)
    print(" -", out_dir / mp4_name)
    if args.save_train_episode_preview:
        print(" -", out_dir / "training_episode_example_1p.mp4")


if __name__ == "__main__":
    main()
