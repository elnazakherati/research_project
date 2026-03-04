import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.models import ResMLP
from particle_nn_sim.train import fit_standardizer, apply_standardizer, StepDataset, train
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
    p.add_argument("--episodes", type=int, default=300)
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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

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

    train_loader = DataLoader(
        StepDataset(Xtr_n, Ytr_n, Ctr),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        StepDataset(Xte_n, Yte_n, Cte),
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
    stats, hist, _ = train(
        model,
        train_loader,
        test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        collision_weight=1.0,
        weight_decay=1e-6,
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
    analysis = {
        "mean_position_error": float(np.mean(pos_err)),
        "max_position_error": float(np.max(pos_err)),
        "final_position_error": float(pos_err[-1]),
        "test_stats": stats,
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
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "meta": meta,
        "config": vars(args),
    }
    torch.save(ckpt, out_dir / "model_1p_resmlp.pt")

    print("Run complete.")
    print("Artifacts:")
    print(" -", out_dir / "model_1p_resmlp.pt")
    print(" -", out_dir / "analysis_1p.json")
    print(" -", out_dir / "rollout_gt_vs_pred_1p.mp4")
    if args.save_train_episode_preview:
        print(" -", out_dir / "training_episode_example_1p.mp4")


if __name__ == "__main__":
    main()
