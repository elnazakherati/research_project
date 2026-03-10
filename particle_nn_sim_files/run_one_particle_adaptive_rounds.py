#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from particle_nn_sim.models import ResMLP
from particle_nn_sim.one_particle_data import episodes_to_XY_residual_1p, sample_init_1p
from particle_nn_sim.one_particle_rollout import nn_rollout_residual_1p
from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.train import fit_standardizer

from run_one_particle_pipeline import (
    RolloutDataset1P,
    make_weighted_sampler,
    resolve_device,
    set_seed,
    str2bool,
    train_multistep_1p,
)


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive hard-case training for one-particle simulator")

    # Core data generation
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--shuffle-episodes", type=str2bool, default=True)
    p.add_argument("--radius", type=float, default=0.0)
    p.add_argument("--mass", type=float, default=1.0)

    # Model/train
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--collision-weight", type=float, default=2.0)
    p.add_argument("--multistep-horizon", type=int, default=20)
    p.add_argument("--rebalance-sampling", type=str2bool, default=True)
    p.add_argument("--target-collision-frac", type=float, default=0.15)

    # Adaptive hard-case mining
    p.add_argument("--hard-fraction", type=float, default=0.6, help="Fraction of next-round episodes drawn from hard pool")
    p.add_argument("--hard-quantile", type=float, default=0.85, help="Keep probe rollouts with error >= this quantile")
    p.add_argument("--hard-bank-max", type=int, default=2000)
    p.add_argument("--hard-pos-jitter", type=float, default=0.02)
    p.add_argument("--hard-speed-jitter", type=float, default=0.05)
    p.add_argument("--hard-angle-jitter", type=float, default=0.15)
    p.add_argument("--probe-rollouts", type=int, default=2000)
    p.add_argument("--probe-steps", type=int, default=500)

    # Heatmap diagnostics per round
    p.add_argument("--heatmap-grid-n", type=int, default=15)
    p.add_argument("--heatmap-n-speeds", type=int, default=15)
    p.add_argument("--heatmap-n-angles", type=int, default=24)
    p.add_argument("--heatmap-fixed-speed", type=float, default=0.5)
    p.add_argument("--heatmap-fixed-angle", type=float, default=0.8)
    p.add_argument("--heatmap-fixed-x", type=float, default=0.5)
    p.add_argument("--heatmap-fixed-y", type=float, default=0.5)
    p.add_argument("--heatmap-rollout-steps", type=int, default=500)

    # Runtime/output
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="checkpoints/one_particle_adaptive")

    # Optional W&B
    p.add_argument("--use-wandb", type=str2bool, default=False)
    p.add_argument("--wandb-project", type=str, default="particle-nn-sim")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")

    return p.parse_args()


def _compute_wall_hits(vel_traj):
    vx_prev = vel_traj[:-1, 0, 0]
    vx_now = vel_traj[1:, 0, 0]
    vy_prev = vel_traj[:-1, 0, 1]
    vy_now = vel_traj[1:, 0, 1]
    return (vx_prev * vx_now < 0) | (vy_prev * vy_now < 0)


def _clip_pos(pos, radius, W, H):
    lo = np.array([radius, radius], dtype=np.float32)
    hi = np.array([W - radius, H - radius], dtype=np.float32)
    return np.clip(pos.astype(np.float32), lo, hi)


def _jitter_hard_ic(base_pos, base_vel, rng, radius, W, H, speed_max, pos_jitter, speed_jitter, angle_jitter):
    pos = base_pos + rng.normal(0.0, pos_jitter, size=2).astype(np.float32)
    pos = _clip_pos(pos, radius, W, H)

    speed = float(np.linalg.norm(base_vel)) + float(rng.normal(0.0, speed_jitter))
    speed = float(np.clip(speed, 0.0, speed_max))

    angle = float(np.arctan2(base_vel[1], base_vel[0]))
    angle += float(rng.normal(0.0, angle_jitter))

    vel = np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float32)
    return pos, vel


def collect_episodes_adaptive(
    sim,
    E,
    steps,
    dt,
    speed_max,
    rng,
    hard_bank,
    hard_fraction,
    pos_jitter,
    speed_jitter,
    angle_jitter,
):
    T = int(steps) + 1
    pos_all = np.zeros((E, T, 1, 2), dtype=np.float32)
    vel_all = np.zeros((E, T, 1, 2), dtype=np.float32)
    coll_all = np.zeros((E, T - 1), dtype=np.uint8)

    init_pos = np.zeros((E, 2), dtype=np.float32)
    init_vel = np.zeros((E, 2), dtype=np.float32)
    init_source = np.zeros(E, dtype=np.uint8)  # 0=uniform, 1=hard

    radius = float(sim.radii[0])
    W = float(sim.W)
    H = float(sim.H)

    use_hard = hard_bank is not None and hard_bank.get("pos", np.zeros((0, 2))).shape[0] > 0

    for e in range(E):
        draw_hard = use_hard and (rng.random() < float(hard_fraction))

        if draw_hard:
            k = int(rng.integers(hard_bank["pos"].shape[0]))
            base_pos = hard_bank["pos"][k]
            base_vel = hard_bank["vel"][k]
            pos0, vel0 = _jitter_hard_ic(
                base_pos,
                base_vel,
                rng,
                radius,
                W,
                H,
                speed_max,
                pos_jitter,
                speed_jitter,
                angle_jitter,
            )
            init_source[e] = 1
        else:
            pos0_arr, vel0_arr = sample_init_1p(
                W,
                H,
                radius,
                speed_max=speed_max,
                seed=int(rng.integers(1_000_000_000)),
            )
            pos0 = pos0_arr[0]
            vel0 = vel0_arr[0]
            init_source[e] = 0

        sim.reset(pos0[None, :], vel0[None, :])
        pos_traj, vel_traj = sim.rollout(dt=dt, steps=steps)
        pos_traj = pos_traj.astype(np.float32)
        vel_traj = vel_traj.astype(np.float32)
        wall_hit = _compute_wall_hits(vel_traj)

        pos_all[e] = pos_traj
        vel_all[e] = vel_traj
        coll_all[e] = wall_hit.astype(np.uint8)
        init_pos[e] = pos0
        init_vel[e] = vel0

    meta = {
        "dt": np.float32(dt),
        "W": np.float32(sim.W),
        "H": np.float32(sim.H),
        "radii": sim.radii.astype(np.float32),
        "masses": sim.masses.astype(np.float32),
        "restitution": np.float32(sim.restitution),
    }
    return pos_all, vel_all, coll_all, meta, init_pos, init_vel, init_source


def rollout_final_err(model, pos0, vel0, sim_true, steps, x_mean, x_std, y_mean, y_std, meta, device):
    sim_true.reset(pos0[None, :].astype(np.float32), vel0[None, :].astype(np.float32))
    pos_true, _ = sim_true.rollout(dt=float(meta["dt"]), steps=int(steps))
    pos_true = pos_true.astype(np.float32)

    pos_pred, _ = nn_rollout_residual_1p(
        model=model,
        pos0=pos0[None, :].astype(np.float32),
        vel0=vel0[None, :].astype(np.float32),
        radius=float(np.asarray(meta["radii"], dtype=np.float32)[0]),
        mass=float(np.asarray(meta["masses"], dtype=np.float32)[0]),
        steps=int(steps),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        dt=float(meta["dt"]),
    )

    n = min(len(pos_true), len(pos_pred))
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(pos_true[n - 1, 0, :] - pos_pred[n - 1, 0, :]))


def save_round_heatmaps(model, round_dir, x_mean, x_std, y_mean, y_std, meta, args, device, seed):
    W = float(meta["W"])
    H = float(meta["H"])
    r = float(np.asarray(meta["radii"], dtype=np.float32)[0])

    sim_true = ParticleSim2D(
        W=W,
        H=H,
        radii=np.asarray(meta["radii"], dtype=float),
        masses=np.asarray(meta["masses"], dtype=float),
        restitution=float(meta["restitution"]),
        seed=seed,
    )

    # Heatmap 1: fixed velocity, sweep position.
    vel = np.array(
        [
            float(args.heatmap_fixed_speed) * np.cos(float(args.heatmap_fixed_angle)),
            float(args.heatmap_fixed_speed) * np.sin(float(args.heatmap_fixed_angle)),
        ],
        dtype=np.float32,
    )
    xs = np.linspace(r, W - r, int(args.heatmap_grid_n), dtype=np.float32)
    ys = np.linspace(r, H - r, int(args.heatmap_grid_n), dtype=np.float32)
    err_pos = np.zeros((len(ys), len(xs)), dtype=np.float32)

    ctr = 0
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            pos0 = np.array([x0, y0], dtype=np.float32)
            err_pos[iy, ix] = rollout_final_err(
                model,
                pos0,
                vel,
                sim_true,
                args.heatmap_rollout_steps,
                x_mean,
                x_std,
                y_mean,
                y_std,
                meta,
                device,
            )
            ctr += 1

    plt.figure(figsize=(6, 5))
    im1 = plt.imshow(
        err_pos,
        origin="lower",
        extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im1, label="final position error")
    plt.xlabel("x0")
    plt.ylabel("y0")
    plt.title("Final error over initial position (fixed velocity)")
    plt.tight_layout()
    out_pos = round_dir / "heatmap_pos_fixed_vel.png"
    plt.savefig(out_pos, dpi=180)
    plt.close()

    # Heatmap 2: fixed position, sweep velocity.
    fixed_x = float(np.clip(args.heatmap_fixed_x, r, W - r))
    fixed_y = float(np.clip(args.heatmap_fixed_y, r, H - r))
    pos_fixed = np.array([fixed_x, fixed_y], dtype=np.float32)

    angles = np.linspace(0.0, 2.0 * np.pi, int(args.heatmap_n_angles), endpoint=False, dtype=np.float32)
    speeds = np.linspace(0.0, float(args.speed_max), int(args.heatmap_n_speeds), dtype=np.float32)
    err_vel = np.zeros((len(speeds), len(angles)), dtype=np.float32)

    for ispd, speed in enumerate(speeds):
        for ia, angle in enumerate(angles):
            vel0 = np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float32)
            err_vel[ispd, ia] = rollout_final_err(
                model,
                pos_fixed,
                vel0,
                sim_true,
                args.heatmap_rollout_steps,
                x_mean,
                x_std,
                y_mean,
                y_std,
                meta,
                device,
            )

    plt.figure(figsize=(7, 5))
    im2 = plt.imshow(
        err_vel,
        origin="lower",
        extent=[0.0, 2.0 * np.pi, float(speeds[0]), float(speeds[-1])],
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im2, label="final position error")
    plt.xlabel("velocity angle (rad)")
    plt.ylabel("speed")
    plt.title("Final error over velocity (fixed position)")
    plt.tight_layout()
    out_vel = round_dir / "heatmap_vel_fixed_pos.png"
    plt.savefig(out_vel, dpi=180)
    plt.close()

    return out_pos, out_vel


def mine_hard_pool(model, round_dir, x_mean, x_std, y_mean, y_std, meta, args, device, seed):
    rng = np.random.default_rng(seed)
    W = float(meta["W"])
    H = float(meta["H"])
    radius = float(np.asarray(meta["radii"], dtype=np.float32)[0])

    sim_true = ParticleSim2D(
        W=W,
        H=H,
        radii=np.asarray(meta["radii"], dtype=float),
        masses=np.asarray(meta["masses"], dtype=float),
        restitution=float(meta["restitution"]),
        seed=seed + 1,
    )

    n_probe = int(args.probe_rollouts)
    pos0_all = np.zeros((n_probe, 2), dtype=np.float32)
    vel0_all = np.zeros((n_probe, 2), dtype=np.float32)
    err_all = np.zeros(n_probe, dtype=np.float32)

    for i in range(n_probe):
        pos0_arr, vel0_arr = sample_init_1p(
            W,
            H,
            radius,
            speed_max=float(args.speed_max),
            seed=int(rng.integers(1_000_000_000)),
        )
        pos0 = pos0_arr[0]
        vel0 = vel0_arr[0]

        err = rollout_final_err(
            model,
            pos0,
            vel0,
            sim_true,
            args.probe_steps,
            x_mean,
            x_std,
            y_mean,
            y_std,
            meta,
            device,
        )

        pos0_all[i] = pos0
        vel0_all[i] = vel0
        err_all[i] = float(err)

    # Save full probe table.
    csv_path = round_dir / "probe_rollouts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x0", "y0", "vx0", "vy0", "speed0", "angle0", "final_err"])
        for i in range(n_probe):
            vx, vy = vel0_all[i]
            w.writerow([
                float(pos0_all[i, 0]),
                float(pos0_all[i, 1]),
                float(vx),
                float(vy),
                float(np.linalg.norm(vel0_all[i])),
                float(np.arctan2(vy, vx)),
                float(err_all[i]),
            ])

    # Hard pool thresholding.
    q = float(np.clip(args.hard_quantile, 0.0, 1.0))
    thr = float(np.quantile(err_all, q))
    hard_idx = np.where(err_all >= thr)[0]

    if int(args.hard_bank_max) > 0 and len(hard_idx) > int(args.hard_bank_max):
        sort_idx = hard_idx[np.argsort(err_all[hard_idx])[::-1]]
        hard_idx = sort_idx[: int(args.hard_bank_max)]

    hard_pos = pos0_all[hard_idx]
    hard_vel = vel0_all[hard_idx]
    hard_err = err_all[hard_idx]

    npz_path = round_dir / "hard_pool.npz"
    np.savez(
        npz_path,
        pos=hard_pos,
        vel=hard_vel,
        err=hard_err,
        threshold=np.asarray([thr], dtype=np.float32),
        quantile=np.asarray([q], dtype=np.float32),
    )

    stats = {
        "probe_rollouts": n_probe,
        "probe_error_mean": float(np.mean(err_all)),
        "probe_error_median": float(np.median(err_all)),
        "probe_error_p90": float(np.quantile(err_all, 0.90)),
        "probe_error_max": float(np.max(err_all)),
        "hard_quantile": q,
        "hard_threshold": thr,
        "hard_pool_size": int(len(hard_idx)),
        "probe_csv": str(csv_path),
        "hard_pool_file": str(npz_path),
    }

    hard_bank = {"pos": hard_pos.astype(np.float32), "vel": hard_vel.astype(np.float32)}
    return hard_bank, stats


def maybe_init_wandb(args, round_idx, round_dir):
    if not args.use_wandb:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "--use-wandb was set but wandb is not available. Install with `pip install wandb`."
        ) from e

    init_kwargs = {
        "project": args.wandb_project,
        "config": vars(args),
        "dir": str(round_dir),
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    base_name = args.wandb_run_name.strip()
    if base_name:
        init_kwargs["name"] = f"{base_name}_r{round_idx:02d}"

    return wandb.init(**init_kwargs)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"Device: {device} ({gpu_name})")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    radius_eff = args.radius if args.radius > 0.0 else 1e-6

    sim = ParticleSim2D(
        W=1.0,
        H=1.0,
        radii=[radius_eff],
        masses=[args.mass],
        restitution=1.0,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    hard_bank = None
    summary = []

    for round_idx in range(1, int(args.rounds) + 1):
        round_seed = int(args.seed + 1000 * round_idx)
        round_rng = np.random.default_rng(round_seed)

        round_dir = out_root / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"Round {round_idx}/{args.rounds} | seed={round_seed}")

        pos_all, vel_all, coll_all, meta, init_pos, init_vel, init_source = collect_episodes_adaptive(
            sim=sim,
            E=int(args.episodes),
            steps=int(args.steps),
            dt=float(args.dt),
            speed_max=float(args.speed_max),
            rng=round_rng,
            hard_bank=hard_bank,
            hard_fraction=float(args.hard_fraction),
            pos_jitter=float(args.hard_pos_jitter),
            speed_jitter=float(args.hard_speed_jitter),
            angle_jitter=float(args.hard_angle_jitter),
        )
        print(
            f"Generated episodes: pos_all={pos_all.shape}, collision_frames={int(coll_all.sum())}/{coll_all.size}, "
            f"hard_inits={int(init_source.sum())}/{len(init_source)}"
        )

        E = pos_all.shape[0]
        idx = np.arange(E)
        if args.shuffle_episodes:
            round_rng.shuffle(idx)
        n_train = int(float(args.train_split) * E)
        train_eps = idx[:n_train]
        test_eps = idx[n_train:]
        if len(test_eps) == 0:
            raise ValueError("No test episodes. Lower --train-split or increase --episodes.")

        Xtr, Ytr, _ = episodes_to_XY_residual_1p(pos_all, vel_all, coll_all, meta, train_eps)
        Xte, Yte, _ = episodes_to_XY_residual_1p(pos_all, vel_all, coll_all, meta, test_eps)
        if Xtr.shape[1] != 6 or Ytr.shape[1] != 4:
            raise RuntimeError(f"Unexpected shapes: Xtr={Xtr.shape}, Ytr={Ytr.shape}")

        x_mean, x_std = fit_standardizer(Xtr)
        y_mean, y_std = fit_standardizer(Ytr)

        train_roll_ds = RolloutDataset1P(pos_all, vel_all, coll_all, train_eps, args.multistep_horizon)
        test_roll_ds = RolloutDataset1P(pos_all, vel_all, coll_all, test_eps, args.multistep_horizon)

        if args.rebalance_sampling:
            labels = train_roll_ds.collision_window_labels()
            sampler = make_weighted_sampler(labels, args.target_collision_frac)
        else:
            sampler = None

        train_loader = DataLoader(
            train_roll_ds,
            batch_size=int(args.batch_size),
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=True,
        )
        test_loader = DataLoader(
            test_roll_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
        )

        model = ResMLP(
            in_dim=6,
            hidden=int(args.hidden),
            out_dim=4,
            blocks=int(args.blocks),
            dropout=float(args.dropout),
        )

        wandb_run = maybe_init_wandb(args, round_idx, round_dir)
        model, stats, hist, best = train_multistep_1p(
            model,
            train_loader,
            test_loader,
            device=device,
            epochs=int(args.epochs),
            lr=float(args.lr),
            collision_weight=float(args.collision_weight),
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            dt=float(meta["dt"]),
            radius=float(meta["radii"][0]),
            mass=float(meta["masses"][0]),
            wandb_run=wandb_run,
        )

        # Mine hard initial conditions for next round.
        hard_bank, probe_stats = mine_hard_pool(
            model=model,
            round_dir=round_dir,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            meta=meta,
            args=args,
            device=device,
            seed=round_seed + 77,
        )

        # Round heatmaps.
        out_pos_hm, out_vel_hm = save_round_heatmaps(
            model=model,
            round_dir=round_dir,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            meta=meta,
            args=args,
            device=device,
            seed=round_seed + 99,
        )

        # Save checkpoint and round analysis.
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_name": "ResMLP",
            "model_kwargs": {
                "in_dim": 6,
                "hidden": int(args.hidden),
                "out_dim": 4,
                "blocks": int(args.blocks),
                "dropout": float(args.dropout),
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
            "round": int(round_idx),
        }
        ckpt_path = round_dir / "model_1p_resmlp.pt"
        torch.save(ckpt, ckpt_path)

        round_summary = {
            "round": int(round_idx),
            "seed": int(round_seed),
            "hard_inits_used": int(init_source.sum()),
            "episodes": int(args.episodes),
            "collision_frames": int(coll_all.sum()),
            "collision_total": int(coll_all.size),
            "test_stats": stats,
            "best_epoch": int(best["epoch"]),
            "probe": probe_stats,
            "heatmap_pos": str(out_pos_hm),
            "heatmap_vel": str(out_vel_hm),
            "ckpt": str(ckpt_path),
            "shape_checks": {
                "pos_all": list(pos_all.shape),
                "vel_all": list(vel_all.shape),
                "Xtr": list(Xtr.shape),
                "Ytr": list(Ytr.shape),
                "Xte": list(Xte.shape),
                "Yte": list(Yte.shape),
            },
        }
        with open(round_dir / "analysis_round.json", "w", encoding="utf-8") as f:
            json.dump(round_summary, f, indent=2)

        if wandb_run is not None:
            wandb_run.summary["round"] = int(round_idx)
            wandb_run.summary["best_epoch"] = int(best["epoch"])
            wandb_run.summary["final_test_mse_all"] = float(stats["mse_all"])
            wandb_run.summary["final_test_mse_collision"] = float(stats["mse_collision"])
            wandb_run.summary["final_test_mse_noncollision"] = float(stats["mse_noncollision"])
            wandb_run.summary["probe_error_mean"] = float(probe_stats["probe_error_mean"])
            wandb_run.summary["probe_error_p90"] = float(probe_stats["probe_error_p90"])
            wandb_run.summary["hard_pool_size"] = int(probe_stats["hard_pool_size"])
            wandb_run.finish()

        summary.append(round_summary)
        print(
            f"Round {round_idx} done | best_epoch={best['epoch']} | "
            f"test_mse={stats['mse_all']:.6f} | hard_pool={probe_stats['hard_pool_size']}"
        )

    with open(out_root / "adaptive_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Adaptive training complete.")
    print("Summary:", out_root / "adaptive_summary.json")


if __name__ == "__main__":
    main()
