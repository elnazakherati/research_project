from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from particle_nn_sim.one_particle_data import collect_episodes_1p
    from particle_nn_sim.simulator import ParticleSim2D
except ModuleNotFoundError:
    from one_particle_data import collect_episodes_1p
    from simulator import ParticleSim2D


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and cache a stratified one-particle dataset for simple TCNO.")
    p.add_argument("--out", type=str, default="datasets/simple_tcno_stratified_1p.npz")
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=0.7)
    p.add_argument("--radius", type=float, default=0.0)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wall-collision-mode", type=str, default="exact", choices=["exact"])
    p.add_argument("--pos-grid-n", type=int, default=4)
    p.add_argument("--angle-bins", type=int, default=8)
    p.add_argument("--speed-bins", type=int, default=6)
    p.add_argument("--episodes-per-bucket", type=int, default=3)
    p.add_argument("--fixed-speed", type=float, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--force", type=str2bool, default=False)
    return p.parse_args()


def jsonable_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def clean_scalar_time(value: float) -> float:
    return float(round(float(value), 8))


def make_split_indices(episodes: int, train_frac: float, val_frac: float) -> dict[str, np.ndarray]:
    if episodes < 3:
        raise ValueError("--episodes implied by stratified buckets must be at least 3")
    if not (0.0 < train_frac < 1.0):
        raise ValueError("--train-frac must be in (0, 1)")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("--val-frac must be in [0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("--train-frac + --val-frac must be < 1")

    idx = np.arange(episodes, dtype=np.int64)
    n_train = int(train_frac * episodes)
    n_val = int(val_frac * episodes)
    return {
        "train_eps": idx[:n_train],
        "val_eps": idx[n_train : n_train + n_val],
        "test_eps": idx[n_train + n_val :],
    }


def collect_speed_stratified_episodes(args: argparse.Namespace, radius_eff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    if int(args.speed_bins) < 1:
        raise ValueError("--speed-bins must be >= 1")
    if float(args.speed_max) <= 0.0:
        raise ValueError("--speed-max must be > 0 for speed-stratified generation")

    pos_chunks = []
    vel_chunks = []
    coll_chunks = []
    speed_bin_chunks = []
    meta_out = None

    base_episodes = int(args.pos_grid_n) * int(args.pos_grid_n) * int(args.angle_bins) * int(args.episodes_per_bucket)
    edges = np.linspace(0.0, float(args.speed_max), int(args.speed_bins) + 1, dtype=np.float64)
    speed_values = 0.5 * (edges[:-1] + edges[1:])

    for speed_i, fixed_speed in enumerate(speed_values):
        sim = ParticleSim2D(
            W=1.0,
            H=1.0,
            radii=[radius_eff],
            masses=[float(args.mass)],
            restitution=1.0,
            seed=int(args.seed) + 10_000 * speed_i,
            wall_mode=str(args.wall_collision_mode),
        )
        pos_all, vel_all, coll_all, meta = collect_episodes_1p(
            sim,
            E=base_episodes,
            steps=int(args.steps),
            dt=float(args.dt),
            speed_max=float(args.speed_max),
            seed=int(args.seed) + 10_000 * speed_i,
            stratified_init=True,
            pos_grid_n=int(args.pos_grid_n),
            angle_bins=int(args.angle_bins),
            episodes_per_bucket=int(args.episodes_per_bucket),
            fixed_speed=float(fixed_speed),
        )
        pos_chunks.append(pos_all)
        vel_chunks.append(vel_all)
        coll_chunks.append(coll_all)
        speed_bin_chunks.append(np.full((base_episodes,), speed_i, dtype=np.int64))
        meta_out = meta

    pos_all = np.concatenate(pos_chunks, axis=0).astype(np.float32)
    vel_all = np.concatenate(vel_chunks, axis=0).astype(np.float32)
    coll_all = np.concatenate(coll_chunks, axis=0).astype(np.uint8)
    speed_bin_ids = np.concatenate(speed_bin_chunks, axis=0)

    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(pos_all.shape[0])
    return pos_all[perm], vel_all[perm], coll_all[perm], meta_out, speed_bin_ids[perm]


def collect_fixed_speed_stratified_episodes(
    args: argparse.Namespace,
    radius_eff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    fixed_speed = float(args.fixed_speed)
    if fixed_speed < 0.0:
        raise ValueError("--fixed-speed must be >= 0")
    if fixed_speed > float(args.speed_max):
        raise ValueError("--fixed-speed must be <= --speed-max")

    episodes = int(args.pos_grid_n) * int(args.pos_grid_n) * int(args.angle_bins) * int(args.episodes_per_bucket)
    sim = ParticleSim2D(
        W=1.0,
        H=1.0,
        radii=[radius_eff],
        masses=[float(args.mass)],
        restitution=1.0,
        seed=int(args.seed),
        wall_mode=str(args.wall_collision_mode),
    )
    pos_all, vel_all, coll_all, meta = collect_episodes_1p(
        sim,
        E=episodes,
        steps=int(args.steps),
        dt=float(args.dt),
        speed_max=float(args.speed_max),
        seed=int(args.seed),
        stratified_init=True,
        pos_grid_n=int(args.pos_grid_n),
        angle_bins=int(args.angle_bins),
        episodes_per_bucket=int(args.episodes_per_bucket),
        fixed_speed=fixed_speed,
    )
    speed_bin_ids = np.zeros((episodes,), dtype=np.int64)
    return pos_all.astype(np.float32), vel_all.astype(np.float32), coll_all.astype(np.uint8), meta, speed_bin_ids


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists. Use --force true to overwrite it.")

    if args.fixed_speed is None:
        episodes = (
            int(args.pos_grid_n)
            * int(args.pos_grid_n)
            * int(args.angle_bins)
            * int(args.speed_bins)
            * int(args.episodes_per_bucket)
        )
    else:
        episodes = int(args.pos_grid_n) * int(args.pos_grid_n) * int(args.angle_bins) * int(args.episodes_per_bucket)
    radius_eff = float(args.radius) if float(args.radius) > 0.0 else 1e-6
    config = {
        "dataset_kind": "simple_tcno_stratified_1p",
        "episodes": episodes,
        "steps": int(args.steps),
        "dt": float(args.dt),
        "speed_max": float(args.speed_max),
        "radius": float(args.radius),
        "radius_eff": radius_eff,
        "mass": float(args.mass),
        "seed": int(args.seed),
        "wall_collision_mode": str(args.wall_collision_mode),
        "stratified_init": True,
        "pos_grid_n": int(args.pos_grid_n),
        "angle_bins": int(args.angle_bins),
        "speed_bins": int(args.speed_bins) if args.fixed_speed is None else 1,
        "episodes_per_bucket": int(args.episodes_per_bucket),
        "fixed_speed": None if args.fixed_speed is None else float(args.fixed_speed),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
    }

    if args.fixed_speed is None:
        pos_all, vel_all, coll_all, meta, speed_bin_ids = collect_speed_stratified_episodes(args, radius_eff)
    else:
        pos_all, vel_all, coll_all, meta, speed_bin_ids = collect_fixed_speed_stratified_episodes(args, radius_eff)
    meta["dt"] = clean_scalar_time(args.dt)
    config["dt"] = clean_scalar_time(args.dt)
    split_indices = make_split_indices(episodes, float(args.train_frac), float(args.val_frac))
    initial_speeds = np.linalg.norm(vel_all[:, 0, 0, :], axis=1).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pos_all=pos_all.astype(np.float32),
        vel_all=vel_all.astype(np.float32),
        coll_all=coll_all.astype(np.uint8),
        initial_speeds=initial_speeds,
        speed_bin_ids=speed_bin_ids,
        train_eps=split_indices["train_eps"],
        val_eps=split_indices["val_eps"],
        test_eps=split_indices["test_eps"],
        meta_json=json.dumps(jsonable_meta(meta), sort_keys=True),
        generation_config=json.dumps(config, sort_keys=True),
    )

    print(f"Saved {out_path}")
    print(f"pos_all shape: {pos_all.shape}")
    print(f"vel_all shape: {vel_all.shape}")
    print(f"coll_all shape: {coll_all.shape}")
    print(
        "splits: "
        f"train={len(split_indices['train_eps'])}, "
        f"val={len(split_indices['val_eps'])}, "
        f"test={len(split_indices['test_eps'])}"
    )
    print(f"collisions: {int(coll_all.sum())}")
    print(f"initial speed range: {float(initial_speeds.min()):.6f} to {float(initial_speeds.max()):.6f}")
    print("speed-bin counts:", np.bincount(speed_bin_ids, minlength=int(args.speed_bins)).tolist())


if __name__ == "__main__":
    main()
