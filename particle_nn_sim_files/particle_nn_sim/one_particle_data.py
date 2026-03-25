import numpy as np


def sample_init_1p(W, H, radius, speed_max=0.7, seed=None):
    rng = np.random.default_rng(seed)
    pos = np.array(
        [rng.uniform(radius, W - radius), rng.uniform(radius, H - radius)],
        dtype=np.float32,
    )
    theta = rng.uniform(0.0, 2.0 * np.pi)
    mag = rng.uniform(0.0, speed_max)
    vel = np.array([mag * np.cos(theta), mag * np.sin(theta)], dtype=np.float32)
    return pos[None, :], vel[None, :]


def sample_init_1p_stratified(
    W,
    H,
    radius,
    pos_grid_n,
    angle_bins,
    cell_idx,
    angle_idx,
    speed_max=0.7,
    fixed_speed=None,
    rng=None,
):
    """Sample one initial condition from a chosen (cell, angle-bin)."""
    if rng is None:
        rng = np.random.default_rng(0)
    g = int(pos_grid_n)
    a = int(angle_bins)
    if g < 1 or a < 1:
        raise ValueError("pos_grid_n and angle_bins must be >= 1")

    # Position cell
    try:
        ix, iy = cell_idx
    except Exception as e:
        raise ValueError(f"cell_idx must be a 2-tuple/list, got {cell_idx!r}") from e
    ix = int(ix)
    iy = int(iy)
    if ix < 0 or ix >= g or iy < 0 or iy >= g:
        raise ValueError(f"cell_idx {(ix, iy)} out of range for grid {g}x{g}")
    x0 = radius + (W - 2.0 * radius) * (ix / g)
    x1 = radius + (W - 2.0 * radius) * ((ix + 1) / g)
    y0 = radius + (H - 2.0 * radius) * (iy / g)
    y1 = radius + (H - 2.0 * radius) * ((iy + 1) / g)
    x = rng.uniform(x0, x1)
    y = rng.uniform(y0, y1)

    # Angle bin
    ia = int(angle_idx)
    if ia < 0 or ia >= a:
        raise ValueError(f"angle_idx {ia} out of range for {a} bins")
    th0 = (2.0 * np.pi) * (ia / a)
    th1 = (2.0 * np.pi) * ((ia + 1) / a)
    theta = rng.uniform(th0, th1)

    if fixed_speed is None:
        mag = rng.uniform(0.0, float(speed_max))
    else:
        mag = float(fixed_speed)
    vel = np.array([mag * np.cos(theta), mag * np.sin(theta)], dtype=np.float32)
    pos = np.array([x, y], dtype=np.float32)
    return pos[None, :], vel[None, :]


def collect_episodes_1p(
    sim,
    E=300,
    steps=1000,
    dt=0.01,
    speed_max=0.7,
    seed=0,
    stratified_init=False,
    pos_grid_n=4,
    angle_bins=8,
    episodes_per_bucket=None,
    fixed_speed=None,
    fixed_x=None,
    fixed_y=None,
    fixed_vx=None,
    fixed_vy=None,
):
    rng = np.random.default_rng(seed)
    T = int(steps) + 1

    pos_all = np.zeros((E, T, 1, 2), dtype=np.float32)
    vel_all = np.zeros((E, T, 1, 2), dtype=np.float32)
    coll_all = np.zeros((E, T - 1), dtype=np.uint8)

    radius = float(sim.radii[0])
    W = float(sim.W)
    H = float(sim.H)

    if fixed_speed is not None and float(fixed_speed) < 0.0:
        raise ValueError("fixed_speed must be >= 0 when provided")
    fixed_ic_vals = (fixed_x, fixed_y, fixed_vx, fixed_vy)
    use_fixed_ic = any(v is not None for v in fixed_ic_vals)
    if use_fixed_ic and not all(v is not None for v in fixed_ic_vals):
        raise ValueError(
            "If any of fixed_x/fixed_y/fixed_vx/fixed_vy is provided, all four must be provided."
        )
    if use_fixed_ic:
        fx = float(fixed_x)
        fy = float(fixed_y)
        fvx = float(fixed_vx)
        fvy = float(fixed_vy)
        if not (radius <= fx <= W - radius):
            raise ValueError(f"fixed_x={fx} outside valid range [{radius}, {W-radius}]")
        if not (radius <= fy <= H - radius):
            raise ValueError(f"fixed_y={fy} outside valid range [{radius}, {H-radius}]")

    # Optional stratified schedule over (position-cell, angle-bin) buckets.
    bucket_schedule = None
    if bool(stratified_init) and not use_fixed_ic:
        g = int(pos_grid_n)
        a = int(angle_bins)
        if g < 1 or a < 1:
            raise ValueError("pos_grid_n and angle_bins must be >= 1 for stratified_init")
        n_buckets = g * g * a
        if episodes_per_bucket is not None:
            epb = int(episodes_per_bucket)
            if epb < 1:
                raise ValueError("episodes_per_bucket must be >= 1")
            expected = epb * n_buckets
            if int(E) != expected:
                raise ValueError(
                    f"With episodes_per_bucket={epb}, expected E={expected} for "
                    f"{g}x{g} position grid and {a} angle bins, got E={E}"
                )
            counts = np.full(n_buckets, epb, dtype=np.int64)
        else:
            # Balanced split with at most 1 difference between buckets.
            q = int(E) // n_buckets
            r = int(E) % n_buckets
            counts = np.full(n_buckets, q, dtype=np.int64)
            if r > 0:
                counts[:r] += 1

        bucket_schedule = []
        for bid in range(n_buckets):
            c = int(counts[bid])
            if c <= 0:
                continue
            ix = bid // (g * a)
            rem = bid % (g * a)
            iy = rem // a
            ia = rem % a
            for _ in range(c):
                bucket_schedule.append((ix, iy, ia))
        # Shuffle schedule to avoid deterministic ordering artifacts.
        rng.shuffle(bucket_schedule)

    for e in range(E):
        if use_fixed_ic:
            pos0 = np.array([[fx, fy]], dtype=np.float32)
            vel0 = np.array([[fvx, fvy]], dtype=np.float32)
        elif bucket_schedule is None:
            pos0, vel0 = sample_init_1p(
                W,
                H,
                radius,
                speed_max=speed_max,
                seed=rng.integers(1_000_000_000),
            )
            if fixed_speed is not None:
                theta = np.arctan2(float(vel0[0, 1]), float(vel0[0, 0]))
                s = float(fixed_speed)
                vel0 = np.array([[s * np.cos(theta), s * np.sin(theta)]], dtype=np.float32)
        else:
            ix, iy, ia = bucket_schedule[e]
            pos0, vel0 = sample_init_1p_stratified(
                W=W,
                H=H,
                radius=radius,
                pos_grid_n=int(pos_grid_n),
                angle_bins=int(angle_bins),
                cell_idx=(int(ix), int(iy)),
                angle_idx=int(ia),
                speed_max=speed_max,
                fixed_speed=fixed_speed,
                rng=rng,
            )
        sim.reset(pos0, vel0)

        pos_traj, vel_traj = sim.rollout(dt=dt, steps=steps)
        pos_traj = pos_traj.astype(np.float32)
        vel_traj = vel_traj.astype(np.float32)

        # Wall collision proxy for 1p: velocity sign flip between consecutive frames.
        vx_prev = vel_traj[:-1, 0, 0]
        vx_now = vel_traj[1:, 0, 0]
        vy_prev = vel_traj[:-1, 0, 1]
        vy_now = vel_traj[1:, 0, 1]
        wall_hit = (vx_prev * vx_now < 0) | (vy_prev * vy_now < 0)

        pos_all[e] = pos_traj
        vel_all[e] = vel_traj
        coll_all[e] = wall_hit.astype(np.uint8)

    meta = {
        "dt": np.float32(dt),
        "W": np.float32(sim.W),
        "H": np.float32(sim.H),
        "radii": sim.radii.astype(np.float32),
        "masses": sim.masses.astype(np.float32),
        "restitution": np.float32(sim.restitution),
        "wall_mode": str(getattr(sim, "wall_mode", "clamp")),
        "stratified_init": bool(stratified_init),
        "pos_grid_n": int(pos_grid_n),
        "angle_bins": int(angle_bins),
        "episodes_per_bucket": None if episodes_per_bucket is None else int(episodes_per_bucket),
        "fixed_speed": None if fixed_speed is None else float(fixed_speed),
        "fixed_x": None if fixed_x is None else float(fixed_x),
        "fixed_y": None if fixed_y is None else float(fixed_y),
        "fixed_vx": None if fixed_vx is None else float(fixed_vx),
        "fixed_vy": None if fixed_vy is None else float(fixed_vy),
    }
    return pos_all, vel_all, coll_all, meta


def episodes_to_XY_residual_1p(pos_all, vel_all, coll_all, meta, episode_indices):
    dt = float(meta["dt"])
    radius = float(np.asarray(meta["radii"], dtype=np.float32)[0])
    mass = float(np.asarray(meta["masses"], dtype=np.float32)[0])

    X_list, Y_list, C_list = [], [], []

    for e in episode_indices:
        pos = pos_all[e].astype(np.float32)  # (T,1,2)
        vel = vel_all[e].astype(np.float32)  # (T,1,2)
        C = coll_all[e].astype(np.uint8)  # (T-1,)

        pos_t = pos[:-1, 0, :]
        vel_t = vel[:-1, 0, :]
        pos_n = pos[1:, 0, :]
        vel_n = vel[1:, 0, :]

        Tm1 = pos_t.shape[0]
        X = np.zeros((Tm1, 6), dtype=np.float32)
        Y_next = np.zeros((Tm1, 4), dtype=np.float32)
        Y_free = np.zeros((Tm1, 4), dtype=np.float32)

        X[:, 0:2] = pos_t
        X[:, 2:4] = vel_t
        X[:, 4] = radius
        X[:, 5] = mass

        Y_next[:, 0:2] = pos_n
        Y_next[:, 2:4] = vel_n

        pos_free = pos_t + vel_t * dt
        vel_free = vel_t
        Y_free[:, 0:2] = pos_free
        Y_free[:, 2:4] = vel_free

        Y_resid = (Y_next - Y_free).astype(np.float32)

        X_list.append(X)
        Y_list.append(Y_resid)
        C_list.append(C)

    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(Y_list, axis=0),
        np.concatenate(C_list, axis=0),
    )
