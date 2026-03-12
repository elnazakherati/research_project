import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

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
    nn_rollout_residual_1p,
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
    p = argparse.ArgumentParser(description="One-particle residual RNN (GRU/LSTM) training pipeline")
    p.add_argument("--rnn-type", type=str, default="gru", choices=["gru", "lstm"])
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--radius", type=float, default=0.0)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--collision-weight", type=float, default=2.0)
    p.add_argument("--rebalance-sampling", type=str2bool, default=True)
    p.add_argument("--target-collision-frac", type=float, default=0.15)
    p.add_argument("--rollout-steps", type=int, default=2000)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--out-dir", type=str, default="checkpoints/one_particle_rnn_residual")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
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


class ResidualRNN(nn.Module):
    def __init__(self, in_dim=6, hidden=128, out_dim=4, layers=2, dropout=0.05, rnn_type="gru"):
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, h=None):
        y, h = self.rnn(x, h)
        return self.head(y), h


class ResidualSeqDataset1P(Dataset):
    def __init__(self, pos_all, vel_all, coll_all, episode_indices, seq_len, dt, radius, mass):
        self.pos_all = pos_all
        self.vel_all = vel_all
        self.coll_all = coll_all
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.dt = float(dt)
        self.radius = float(radius)
        self.mass = float(mass)
        self.T = pos_all.shape[1]
        self.starts_per_episode = self.T - self.seq_len
        if self.starts_per_episode <= 0:
            raise ValueError(f"seq_len={self.seq_len} too large for T={self.T}")

    def __len__(self):
        return len(self.episode_indices) * self.starts_per_episode

    def __getitem__(self, idx):
        ep_i = idx // self.starts_per_episode
        t0 = idx % self.starts_per_episode
        e = int(self.episode_indices[ep_i])

        pos = self.pos_all[e, t0 : t0 + self.seq_len + 1, 0, :].astype(np.float32)
        vel = self.vel_all[e, t0 : t0 + self.seq_len + 1, 0, :].astype(np.float32)
        coll = self.coll_all[e, t0 : t0 + self.seq_len].astype(np.float32)

        pos_t = pos[:-1]
        vel_t = vel[:-1]
        pos_n = pos[1:]
        vel_n = vel[1:]

        x = np.zeros((self.seq_len, 6), dtype=np.float32)
        x[:, 0:2] = pos_t
        x[:, 2:4] = vel_t
        x[:, 4] = self.radius
        x[:, 5] = self.mass

        pos_free = pos_t + vel_t * self.dt
        vel_free = vel_t
        y_res = np.zeros((self.seq_len, 4), dtype=np.float32)
        y_res[:, 0:2] = pos_n - pos_free
        y_res[:, 2:4] = vel_n - vel_free

        return torch.from_numpy(x), torch.from_numpy(y_res), torch.from_numpy(coll)

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


def train_residual_rnn(model, train_loader, test_loader, device, epochs, lr, collision_weight, x_mean, x_std, y_mean, y_std):
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
        mse_all_sum = mse_col_sum = mse_non_sum = 0.0
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
            mse = ((pred_n - y_n) ** 2).mean(dim=2)
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
        print(
            f"Epoch {ep:03d} | train_loss={train_loss:.6f} | test_mse={stats['mse_all']:.6f} "
            f"| test_collision={stats['mse_collision']:.6f} | test_noncollision={stats['mse_noncollision']:.6f} "
            f"(n_col={stats['n_collision']}, n_noncol={stats['n_noncollision']}) | best_epoch={best['epoch']} "
            f"| sec={time.time()-t0:.2f}"
        )

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    return model, best["stats"], history, best


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print("Config:", vars(args))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.rnn_type.lower()

    radius_eff = args.radius if args.radius > 0.0 else 1e-6
    sim = ParticleSim2D(W=1.0, H=1.0, radii=[radius_eff], masses=[args.mass], restitution=1.0, seed=args.seed)
    pos_all, vel_all, coll_all, meta = collect_episodes_1p(
        sim, E=args.episodes, steps=args.steps, dt=args.dt, speed_max=args.speed_max, seed=args.seed
    )

    if args.save_train_episode_preview:
        preview = animate_single_rollout_1p(
            pos_all[args.preview_episode_idx],
            radius=float(meta["radii"][0]),
            W=float(meta["W"]),
            H=float(meta["H"]),
            dt=float(meta["dt"]),
            title="Training Episode Example (GT)",
        )
        save_animation_mp4(preview, str(out_dir / "training_episode_example_1p.mp4"), fps=args.preview_fps)

    E = pos_all.shape[0]
    n_train = int(args.train_split * E)
    idx = np.arange(E)
    train_eps = idx[:n_train]
    test_eps = idx[n_train:]

    train_ds = ResidualSeqDataset1P(
        pos_all, vel_all, coll_all, train_eps, args.seq_len, float(meta["dt"]), float(meta["radii"][0]), float(meta["masses"][0])
    )
    test_ds = ResidualSeqDataset1P(
        pos_all, vel_all, coll_all, test_eps, args.seq_len, float(meta["dt"]), float(meta["radii"][0]), float(meta["masses"][0])
    )

    if args.rebalance_sampling:
        labels = train_ds.collision_window_labels()
        sampler = make_weighted_sampler(labels, args.target_collision_frac)
    else:
        sampler = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # fit standardizers on train windows
    x_list = []
    y_list = []
    for e in train_eps:
        pos = pos_all[e, :, 0, :].astype(np.float32)
        vel = vel_all[e, :, 0, :].astype(np.float32)
        pos_t = pos[:-1]
        vel_t = vel[:-1]
        pos_n = pos[1:]
        vel_n = vel[1:]
        x = np.zeros((len(pos_t), 6), dtype=np.float32)
        x[:, 0:2] = pos_t
        x[:, 2:4] = vel_t
        x[:, 4] = float(meta["radii"][0])
        x[:, 5] = float(meta["masses"][0])
        y = np.zeros((len(pos_t), 4), dtype=np.float32)
        y[:, 0:2] = pos_n - (pos_t + vel_t * float(meta["dt"]))
        y[:, 2:4] = vel_n - vel_t
        x_list.append(x)
        y_list.append(y)
    x_train = np.concatenate(x_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    x_mean, x_std = fit_standardizer(x_train)
    y_mean, y_std = fit_standardizer(y_train)

    model = ResidualRNN(in_dim=6, hidden=args.hidden, out_dim=4, layers=args.layers, dropout=args.dropout, rnn_type=args.rnn_type)
    model, stats, hist, best = train_residual_rnn(
        model, train_loader, test_loader, device, args.epochs, args.lr, args.collision_weight, x_mean, x_std, y_mean, y_std
    )

    # rollout eval (same as residual baseline + predicted residual)
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

    # autoregressive residual rollout through RNN hidden state
    @torch.no_grad()
    def rnn_residual_rollout(pos0_, vel0_):
        model.eval()
        S = int(args.rollout_steps)
        pos_pred = np.zeros((S + 1, 1, 2), dtype=np.float32)
        vel_pred = np.zeros((S + 1, 1, 2), dtype=np.float32)
        pos_pred[0] = pos0_
        vel_pred[0] = vel0_
        state = np.concatenate([pos0_[0], vel0_[0]], axis=0).astype(np.float32)
        h = None
        xm = np.asarray(x_mean, dtype=np.float32).reshape(-1)
        xs = np.asarray(x_std, dtype=np.float32).reshape(-1)
        ym = np.asarray(y_mean, dtype=np.float32).reshape(-1)
        ys = np.asarray(y_std, dtype=np.float32).reshape(-1)
        dt = float(meta["dt"])
        r = float(meta["radii"][0])
        m = float(meta["masses"][0])
        for t in range(S):
            x = np.array([state[0], state[1], state[2], state[3], r, m], dtype=np.float32)
            x_n = ((x[None, None, :] - xm[None, None, :]) / xs[None, None, :]).astype(np.float32)
            y_n, h = model(torch.from_numpy(x_n).to(device), h)
            res = (y_n[:, -1, :].cpu().numpy() * ys[None, :]) + ym[None, :]
            pos = state[0:2]
            vel = state[2:4]
            pos_free = pos + vel * dt
            state = np.array(
                [pos_free[0] + res[0, 0], pos_free[1] + res[0, 1], vel[0] + res[0, 2], vel[1] + res[0, 3]],
                dtype=np.float32,
            )
            pos_pred[t + 1, 0] = state[0:2]
            vel_pred[t + 1, 0] = state[2:4]
        return pos_pred, vel_pred

    pos_pred, vel_pred = rnn_residual_rollout(pos0, vel0)

    anim = animate_side_by_side_1p(
        pos_true=pos_true,
        pos_pred=pos_pred,
        radius=float(meta["radii"][0]),
        W=float(meta["W"]),
        H=float(meta["H"]),
        dt=float(meta["dt"]),
        title_left="Ground Truth",
        title_right=f"{name.upper()} residual rollout",
    )
    mp4_name = f"rollout_gt_vs_pred_1p_{name}_residual.mp4"
    save_animation_mp4(anim, str(out_dir / mp4_name), fps=args.fps)

    pos_err = np.linalg.norm(pos_true[:, 0, :] - pos_pred[:, 0, :], axis=1)
    err_name = f"error_vs_timestep_1p_{name}_residual.png"
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(pos_err)), pos_err, lw=2, color="tab:red")
    plt.xlabel("step")
    plt.ylabel("position error ||x_pred - x_true||")
    plt.title(f"{name.upper()} residual rollout error vs time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / err_name, dpi=150)
    plt.close()

    analysis_name = f"analysis_1p_{name}_residual.json"
    ckpt_name = f"model_1p_{name}_residual.pt"
    analysis = {
        "rnn_type": name,
        "mean_position_error": float(np.mean(pos_err)),
        "max_position_error": float(np.max(pos_err)),
        "final_position_error": float(pos_err[-1]),
        "test_stats": stats,
        "best_epoch": int(best["epoch"]),
        "config": vars(args),
    }
    with open(out_dir / analysis_name, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_name": "ResidualRNN",
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
    print(" -", out_dir / err_name)
    print(" -", out_dir / mp4_name)
    if args.save_train_episode_preview:
        print(" -", out_dir / "training_episode_example_1p.mp4")


if __name__ == "__main__":
    main()

