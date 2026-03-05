import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Plot saved training curves from model_1p_resmlp.pt")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_1p_resmlp.pt")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for PNGs (default: checkpoint directory)",
    )
    return p.parse_args()


def save_curve(values, ylabel, title, out_path):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(values) + 1), values, lw=2)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_train_test_overlay(train_values, test_values, out_path):
    epochs = range(1, min(len(train_values), len(test_values)) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_values[: len(epochs)], lw=2, label="train_loss")
    plt.plot(epochs, test_values[: len(epochs)], lw=2, label="test_mse_all")
    plt.xlabel("epoch")
    plt.ylabel("loss / mse")
    plt.title("Train Loss vs Test MSE (All)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hist = ckpt.get("hist", None)
    if not isinstance(hist, dict):
        raise RuntimeError("Checkpoint has no 'hist' dict. Cannot plot training curves.")

    required = [
        "train_loss",
        "test_mse_all",
        "test_mse_collision",
        "test_mse_noncollision",
    ]
    for k in required:
        if k not in hist:
            raise RuntimeError(f"Missing history key '{k}' in checkpoint.")
        if len(hist[k]) == 0:
            raise RuntimeError(f"History key '{k}' is empty.")

    outputs = {
        "train_loss_1p.png": ("loss", "Train Loss", hist["train_loss"]),
        "test_mse_all_1p.png": ("MSE", "Test MSE (All)", hist["test_mse_all"]),
        "test_mse_collision_1p.png": ("MSE", "Test MSE (Collision)", hist["test_mse_collision"]),
        "test_mse_noncollision_1p.png": (
            "MSE",
            "Test MSE (Non-Collision)",
            hist["test_mse_noncollision"],
        ),
    }

    for name, (ylabel, title, values) in outputs.items():
        save_curve(values, ylabel, title, out_dir / name)
    save_train_test_overlay(
        hist["train_loss"],
        hist["test_mse_all"],
        out_dir / "train_vs_test_mse_all_1p.png",
    )

    print("Saved training curves:")
    for name in outputs:
        print(" -", out_dir / name)
    print(" -", out_dir / "train_vs_test_mse_all_1p.png")


if __name__ == "__main__":
    main()
