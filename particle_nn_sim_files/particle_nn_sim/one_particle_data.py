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


def collect_episodes_1p(sim, E=300, steps=1000, dt=0.01, speed_max=0.7, seed=0):
    rng = np.random.default_rng(seed)
    T = int(steps) + 1

    pos_all = np.zeros((E, T, 1, 2), dtype=np.float32)
    vel_all = np.zeros((E, T, 1, 2), dtype=np.float32)
    coll_all = np.zeros((E, T - 1), dtype=np.uint8)

    radius = float(sim.radii[0])
    W = float(sim.W)
    H = float(sim.H)

    for e in range(E):
        pos0, vel0 = sample_init_1p(
            W,
            H,
            radius,
            speed_max=speed_max,
            seed=rng.integers(1_000_000_000),
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
