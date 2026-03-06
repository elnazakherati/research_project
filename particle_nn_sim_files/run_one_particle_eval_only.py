import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_nn_sim.models import ResMLP
from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.one_particle_data import sample_init_1p
from particle_nn_sim.one_particle_rollout import (
    animate_side_by_side_1p,
    nn_rollout_residual_1p,
    save_animation_mp4,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate one-particle checkpoint on new rollouts only.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_1p_resmlp.pt")
    p.add_argument("--num-rollouts", type=int, default=5, help="Number of new random rollouts")
    p.add_argument("--rollout-steps", type=int, default=1000)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--out-dir", type=str, default="checkpoints/one_particle_eval_only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def resolve_device(flag):
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    # weights_only=False is required because checkpoint stores numpy arrays/config dicts too.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_kwargs = ckpt["model_kwargs"]
    model = ResMLP(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    meta = ckpt["meta"]
    cfg = ckpt.get("config", {})
    target_mode = str(cfg.get("target_mode", ""))
    if target_mode not in {"state_residual", "dv"}:
        target_mode = "dv" if int(np.asarray(y_mean).shape[-1]) == 2 else "state_residual"

    W = float(meta["W"])
    H = float(meta["H"])
    dt = float(meta["dt"])
    radius = float(np.asarray(meta["radii"], dtype=np.float32)[0])
    mass = float(np.asarray(meta["masses"], dtype=np.float32)[0])
    restitution = float(meta["restitution"])

    for i in range(int(args.num_rollouts)):
        seed_i = int(rng.integers(1_000_000_000))
        pos0, vel0 = sample_init_1p(W, H, radius, speed_max=args.speed_max, seed=seed_i)

        sim_true = ParticleSim2D(
            W=W,
            H=H,
            radii=[radius],
            masses=[mass],
            restitution=restitution,
            seed=seed_i + 1,
        )
        sim_true.reset(pos0, vel0)
        pos_true, vel_true = sim_true.rollout(dt=dt, steps=args.rollout_steps)
        pos_true = pos_true.astype(np.float32)
        vel_true = vel_true.astype(np.float32)

        pos_pred, vel_pred = nn_rollout_residual_1p(
            model=model,
            pos0=pos0.astype(np.float32),
            vel0=vel0.astype(np.float32),
            radius=radius,
            mass=mass,
            steps=args.rollout_steps,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            dt=dt,
            target_mode=target_mode,
        )

        anim = animate_side_by_side_1p(
            pos_true=pos_true,
            pos_pred=pos_pred,
            radius=radius,
            W=W,
            H=H,
            dt=dt,
        )
        out_mp4 = out_dir / f"rollout_{i:03d}_gt_vs_pred_1p.mp4"
        save_animation_mp4(anim, str(out_mp4), fps=args.fps)

        pos_err = np.linalg.norm(pos_true[:, 0, :] - pos_pred[:, 0, :], axis=1)
        plot_path = out_dir / f"rollout_{i:03d}_error_vs_step.png"
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(len(pos_err)), pos_err, lw=2, color="tab:red")
        plt.xlabel("step")
        plt.ylabel("position error ||x_pred - x_true||")
        plt.title(f"Rollout {i:03d} Error vs Time Step")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(
            f"[{i+1}/{args.num_rollouts}] saved {out_mp4.name} | "
            f"mean_err={float(np.mean(pos_err)):.6f} "
            f"max_err={float(np.max(pos_err)):.6f} "
            f"final_err={float(pos_err[-1]):.6f} | "
            f"plot={plot_path.name}"
        )

    print("Done.")
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()
