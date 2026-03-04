import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def fit_standardizer(A, eps=1e-8):
    mean = A.mean(axis=0, keepdims=True)
    std  = A.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(A, mean, std):
    return ((A - mean) / std).astype(np.float32)

class StepDataset(Dataset):
    def __init__(self, X, Y, C=None):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.C = None if C is None else torch.from_numpy(C).long()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        if self.C is None:
            return self.X[i], self.Y[i]
        return self.X[i], self.Y[i], self.C[i]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_all = 0.0
    mse_col = 0.0
    mse_ncol = 0.0
    n_all = 0
    n_col = 0
    n_ncol = 0

    for batch in loader:
        if len(batch) == 2:
            X, Y = batch
            C = None
        else:
            X, Y, C = batch
            C = C.to(device).bool()

        X = X.to(device)
        Y = Y.to(device)

        pred = model(X)
        err = (pred - Y) ** 2
        per_sample = err.mean(dim=1)

        mse_all += per_sample.sum().item()
        n_all += per_sample.numel()

        if C is not None:
            if C.any():
                mse_col += per_sample[C].sum().item()
                n_col += C.sum().item()
            if (~C).any():
                mse_ncol += per_sample[~C].sum().item()
                n_ncol += (~C).sum().item()

    return {
        "mse_all": mse_all / max(n_all, 1),
        "mse_collision": mse_col / max(n_col, 1),
        "mse_noncollision": mse_ncol / max(n_ncol, 1),
        "n_all": n_all,
        "n_collision": n_col,
        "n_noncollision": n_ncol,
    }

def train(model, train_loader, test_loader, device, epochs=50, lr=1e-3,
          collision_weight=1.0, weight_decay=0.0):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "test_mse_all": [],
        "test_mse_collision": [],
        "test_mse_noncollision": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for batch in train_loader:
            if len(batch) == 2:
                X, Y = batch
                C = None
            else:
                X, Y, C = batch
                C = C.to(device).float()

            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)

            # per-sample MSE
            per_sample = ((pred - Y) ** 2).mean(dim=1)

            if C is not None:
                w = torch.ones_like(per_sample)
                w = torch.where(C > 0, w * collision_weight, w)
                loss = (w * per_sample).mean()
            else:
                loss = per_sample.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item() * X.shape[0]
            n += X.shape[0]

        train_loss = running / max(n, 1)
        stats = evaluate(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["test_mse_all"].append(stats["mse_all"])
        history["test_mse_collision"].append(stats["mse_collision"])
        history["test_mse_noncollision"].append(stats["mse_noncollision"])

        print(
            f"Epoch {ep:03d} | train_loss={train_loss:.6f} "
            f"| test_mse={stats['mse_all']:.6f} "
            f"| test_collision={stats['mse_collision']:.6f} "
            f"| test_noncollision={stats['mse_noncollision']:.6f} "
            f"(n_col={stats['n_collision']}, n_noncol={stats['n_noncollision']})"
        )

    return stats, history, opt

