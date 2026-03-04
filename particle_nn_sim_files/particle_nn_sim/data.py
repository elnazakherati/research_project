import numpy as np

def sample_init(W, H, radii, speed_max=0.7, seed=None):
    """Sample non-overlapping initial positions + random velocities (N=2)."""
    rng = np.random.default_rng(seed)
    r0, r1 = radii

    # rejection sample positions to avoid overlap
    for _ in range(10000):
        p0 = np.array([rng.uniform(r0, W-r0), rng.uniform(r0, H-r0)], dtype=np.float32)
        p1 = np.array([rng.uniform(r1, W-r1), rng.uniform(r1, H-r1)], dtype=np.float32)
        if np.linalg.norm(p1 - p0) > (r0 + r1 + 1e-3):
            break
    else:
        raise RuntimeError("Failed to sample non-overlapping initial positions")

    def rand_vel():
        theta = rng.uniform(0, 2*np.pi)
        mag = rng.uniform(0.0, speed_max)
        return np.array([mag*np.cos(theta), mag*np.sin(theta)], dtype=np.float32)

    v0, v1 = rand_vel(), rand_vel()
    return np.stack([p0, p1]), np.stack([v0, v1])

def compute_collision_flags(pos_traj, vel_traj, radii, W, H, eps=1e-3, dv_thresh=1e-3):
    """
    pos_traj, vel_traj: (T,2,2)
    returns collision flags C: (T-1,) uint8
    """
    T = pos_traj.shape[0]
    r0, r1 = radii
    C = np.zeros((T-1,), dtype=np.uint8)

    def near_wall(p, r):
        x, y = p
        return (x <= r + eps) or (x >= W - r - eps) or (y <= r + eps) or (y >= H - r - eps)

    for t in range(T-1):
        pos_t, vel_t = pos_traj[t], vel_traj[t]
        pos_n, vel_n = pos_traj[t+1], vel_traj[t+1]

        wall_hit = (
            near_wall(pos_t[0], r0) or near_wall(pos_t[1], r1) or
            near_wall(pos_n[0], r0) or near_wall(pos_n[1], r1)
        )

        d_t = np.linalg.norm(pos_t[1] - pos_t[0])
        d_n = np.linalg.norm(pos_n[1] - pos_n[0])
        pair_hit = (d_t <= (r0 + r1 + eps)) or (d_n <= (r0 + r1 + eps))

        dv0 = np.linalg.norm(vel_n[0] - vel_t[0])
        dv1 = np.linalg.norm(vel_n[1] - vel_t[1])
        vel_jump = (dv0 > dv_thresh) or (dv1 > dv_thresh)

        C[t] = 1 if (wall_hit or pair_hit or vel_jump) else 0

    return C

def collect_episodes(sim, E=1000, steps=500, dt=0.01, speed_max=0.7, seed=0):
    """
    Generate E episodes of length steps, returning:
      pos_all:  (E, T, 2, 2)
      vel_all:  (E, T, 2, 2)
      coll_all: (E, T-1)
      meta: dict with dt, W, H, radii, masses, restitution
    """
    rng = np.random.default_rng(seed)
    T = steps + 1

    pos_all  = np.zeros((E, T, 2, 2), dtype=np.float32)
    vel_all  = np.zeros((E, T, 2, 2), dtype=np.float32)
    coll_all = np.zeros((E, T-1), dtype=np.uint8)

    for e in range(E):
        pos0, vel0 = sample_init(sim.W, sim.H, sim.radii, speed_max=speed_max, seed=rng.integers(1e9))
        sim.reset(pos0, vel0)

        pos_traj, vel_traj = sim.rollout(dt=dt, steps=steps)
        pos_traj = pos_traj.astype(np.float32)
        vel_traj = vel_traj.astype(np.float32)

        C = compute_collision_flags(pos_traj, vel_traj, sim.radii, sim.W, sim.H)

        pos_all[e] = pos_traj
        vel_all[e] = vel_traj
        coll_all[e] = C

    meta = {
        "dt": np.float32(dt),
        "W": np.float32(sim.W),
        "H": np.float32(sim.H),
        "radii": sim.radii.astype(np.float32),
        "masses": sim.masses.astype(np.float32),
        "restitution": np.float32(sim.restitution),
    }
    return pos_all, vel_all, coll_all, meta

def episodes_to_XY(pos_all, vel_all, coll_all, meta, episode_indices):
    """Absolute target: X is state(t) (12), Y is state(t+1) (8)."""
    radii  = np.asarray(meta["radii"], dtype=np.float32)
    masses = np.asarray(meta["masses"], dtype=np.float32)
    r0, r1 = radii
    m0, m1 = masses

    X_list, Y_list, C_list = [], [], []

    for e in episode_indices:
        pos = pos_all[e].astype(np.float32)  # (T,2,2)
        vel = vel_all[e].astype(np.float32)  # (T,2,2)
        C   = coll_all[e].astype(np.uint8)   # (T-1,)

        pos_t = pos[:-1]
        vel_t = vel[:-1]
        pos_n = pos[1:]
        vel_n = vel[1:]

        Tm1 = pos_t.shape[0]
        X = np.zeros((Tm1, 12), dtype=np.float32)
        Y = np.zeros((Tm1, 8), dtype=np.float32)

        X[:,0] = pos_t[:,0,0];  X[:,1] = pos_t[:,0,1]
        X[:,2] = vel_t[:,0,0];  X[:,3] = vel_t[:,0,1]
        X[:,4] = r0;            X[:,5] = m0
        X[:,6] = pos_t[:,1,0];  X[:,7] = pos_t[:,1,1]
        X[:,8] = vel_t[:,1,0];  X[:,9] = vel_t[:,1,1]
        X[:,10]= r1;            X[:,11]= m1

        Y[:,0] = pos_n[:,0,0];  Y[:,1] = pos_n[:,0,1]
        Y[:,2] = vel_n[:,0,0];  Y[:,3] = vel_n[:,0,1]
        Y[:,4] = pos_n[:,1,0];  Y[:,5] = pos_n[:,1,1]
        Y[:,6] = vel_n[:,1,0];  Y[:,7] = vel_n[:,1,1]

        X_list.append(X)
        Y_list.append(Y)
        C_list.append(C)

    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0), np.concatenate(C_list, axis=0)

def episodes_to_XY_residual(pos_all, vel_all, coll_all, meta, episode_indices):
    """
    Residual target: Y = next_state - free_flight_next_state.
    X is state(t) (12). Y is residual (8).
    """
    radii  = np.asarray(meta["radii"], dtype=np.float32)
    masses = np.asarray(meta["masses"], dtype=np.float32)
    r0, r1 = radii
    m0, m1 = masses
    dt = float(meta["dt"])

    X_list, Y_list, C_list = [], [], []

    for e in episode_indices:
        pos = pos_all[e].astype(np.float32)  # (T,2,2)
        vel = vel_all[e].astype(np.float32)  # (T,2,2)
        C   = coll_all[e].astype(np.uint8)   # (T-1,)

        pos_t = pos[:-1]
        vel_t = vel[:-1]
        pos_n = pos[1:]
        vel_n = vel[1:]

        Tm1 = pos_t.shape[0]
        X = np.zeros((Tm1, 12), dtype=np.float32)
        Y_next = np.zeros((Tm1, 8), dtype=np.float32)
        Y_free = np.zeros((Tm1, 8), dtype=np.float32)

        # pack X
        X[:,0] = pos_t[:,0,0];  X[:,1] = pos_t[:,0,1]
        X[:,2] = vel_t[:,0,0];  X[:,3] = vel_t[:,0,1]
        X[:,4] = r0;            X[:,5] = m0
        X[:,6] = pos_t[:,1,0];  X[:,7] = pos_t[:,1,1]
        X[:,8] = vel_t[:,1,0];  X[:,9] = vel_t[:,1,1]
        X[:,10]= r1;            X[:,11]= m1

        # pack true next state
        Y_next[:,0] = pos_n[:,0,0];  Y_next[:,1] = pos_n[:,0,1]
        Y_next[:,2] = vel_n[:,0,0];  Y_next[:,3] = vel_n[:,0,1]
        Y_next[:,4] = pos_n[:,1,0];  Y_next[:,5] = pos_n[:,1,1]
        Y_next[:,6] = vel_n[:,1,0];  Y_next[:,7] = vel_n[:,1,1]

        # free flight baseline
        pos_free = pos_t + vel_t * dt
        vel_free = vel_t

        Y_free[:,0] = pos_free[:,0,0];  Y_free[:,1] = pos_free[:,0,1]
        Y_free[:,2] = vel_free[:,0,0];  Y_free[:,3] = vel_free[:,0,1]
        Y_free[:,4] = pos_free[:,1,0];  Y_free[:,5] = pos_free[:,1,1]
        Y_free[:,6] = vel_free[:,1,0];  Y_free[:,7] = vel_free[:,1,1]

        Y_resid = (Y_next - Y_free).astype(np.float32)

        X_list.append(X)
        Y_list.append(Y_resid)
        C_list.append(C)

    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0), np.concatenate(C_list, axis=0)
