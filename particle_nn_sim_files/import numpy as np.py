import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())

class ParticleSim2D:
    """
    N circular particles moving in a 2D axis-aligned box [0, W] x [0, H].
    Elastic collisions with walls and between particles.

    State:
      pos: (N,2)
      vel: (N,2)
    """

    def __init__(self, W=1.0, H=1.0, radii=None, masses=None, restitution=1.0, seed=0):
        self.W = float(W)
        self.H = float(H)
        self.restitution = float(restitution)  # 1.0 = perfectly elastic

        self.rng = np.random.default_rng(seed)

        self.radii = np.array(radii, dtype=float) if radii is not None else None
        self.masses = np.array(masses, dtype=float) if masses is not None else None

        self.pos = None
        self.vel = None

    def reset(self, pos, vel):
        pos = np.array(pos, dtype=float)
        vel = np.array(vel, dtype=float)
        assert pos.ndim == 2 and pos.shape[1] == 2
        assert vel.shape == pos.shape

        N = pos.shape[0]
        if self.radii is None:
            self.radii = np.full(N, 0.05, dtype=float)
        if self.masses is None:
            self.masses = np.full(N, 1.0, dtype=float)

        assert self.radii.shape == (N,)
        assert self.masses.shape == (N,)

        self.pos = pos.copy()
        self.vel = vel.copy()
        return self.state()

    def state(self):
        return {
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "radii": self.radii.copy(),
            "masses": self.masses.copy(),
            "W": self.W,
            "H": self.H,
        }

    def _handle_wall_collisions(self):
        # Left / right walls
        x = self.pos[:, 0]
        r = self.radii

        left_pen = r - x
        hit_left = left_pen > 0
        if np.any(hit_left):
            self.pos[hit_left, 0] = r[hit_left]
            self.vel[hit_left, 0] *= -1

        right_pen = x + r - self.W
        hit_right = right_pen > 0
        if np.any(hit_right):
            self.pos[hit_right, 0] = self.W - r[hit_right]
            self.vel[hit_right, 0] *= -1

        # Bottom / top walls
        y = self.pos[:, 1]
        bottom_pen = r - y
        hit_bottom = bottom_pen > 0
        if np.any(hit_bottom):
            self.pos[hit_bottom, 1] = r[hit_bottom]
            self.vel[hit_bottom, 1] *= -1

        top_pen = y + r - self.H
        hit_top = top_pen > 0
        if np.any(hit_top):
            self.pos[hit_top, 1] = self.H - r[hit_top]
            self.vel[hit_top, 1] *= -1

    def _handle_pair_collisions(self):
        """
        Resolve overlaps + apply elastic collision impulse along the normal.
        This is a "discrete-time" collision resolver (works well with small dt).
        """
        N = self.pos.shape[0]
        if N < 2:
            return

        e = self.restitution

        for i in range(N):
            for j in range(i + 1, N):
                xi = self.pos[i]
                xj = self.pos[j]
                ri = self.radii[i]
                rj = self.radii[j]

                d = xj - xi
                dist = np.linalg.norm(d)
                min_dist = ri + rj

                # If identical position (rare), nudge a bit
                if dist < 1e-12:
                   raise RuntimeError(
                        f"Degenerate collision: particles {i} and {j} "
                        f"have nearly identical positions at dist={dist}"
                    ) 

                if dist < min_dist - 1e-9:
                    n = d / dist  # collision normal from i -> j

                    # --- Positional correction (split overlap) ---
                    overlap = min_dist - dist
                    mi = self.masses[i]
                    mj = self.masses[j]
                    w_i = 1.0 / mi
                    w_j = 1.0 / mj
                    w_sum = w_i + w_j

                    # Move particles apart proportional to inverse mass
                    self.pos[i] -= n * overlap * (w_i / w_sum)
                    self.pos[j] += n * overlap * (w_j / w_sum)

                    # --- Velocity update via impulse (elastic) ---
                    vi = self.vel[i]
                    vj = self.vel[j]
                    rel_v = vi - vj
                    rel_vn = np.dot(rel_v, n)

                    # Only apply impulse if they are moving toward each other
                    if rel_vn > 0:
                        j_imp = -(1 + e) * rel_vn / (w_i + w_j)
                        impulse = j_imp * n
                        self.vel[i] += impulse * w_i
                        self.vel[j] -= impulse * w_j

    def step(self, dt):
        dt = float(dt)
        # Integrate
        self.pos += self.vel * dt

        # Handle wall collisions then pair collisions (order matters a bit)
        self._handle_wall_collisions()
        self._handle_pair_collisions()

        # Clamp again in case pair correction pushed slightly outside
        self._handle_wall_collisions()

        return self.state()

    def kinetic_energy(self):
        return 0.5 * np.sum(self.masses[:, None] * self.vel**2)

    def rollout(self, dt, steps):
        """
        Returns trajectory arrays:
          positions: (steps+1, N, 2)
          velocities: (steps+1, N, 2)
        """
        steps = int(steps)
        pos_traj = np.zeros((steps + 1, self.pos.shape[0], 2), dtype=float)
        vel_traj = np.zeros_like(pos_traj)

        pos_traj[0] = self.pos
        vel_traj[0] = self.vel

        for t in range(1, steps + 1):
            self.step(dt)
            pos_traj[t] = self.pos
            vel_traj[t] = self.vel

        return pos_traj, vel_traj

# Initial conditions (2 particles)
pos0 = np.array([
    [0.1, 0.50],
    [0.75, 0.50],
])

vel0 = np.array([
    [ 0.15,  0.20],
    [-0.35, -0.10],
])

sim = ParticleSim2D(W=1.0, H=1.0, radii=[0.06, 0.06], masses=[1.0, 1.0], restitution=1.0, seed=1)
sim.reset(pos0, vel0)

dt = 0.01
steps = 1200
pos_traj, vel_traj = sim.rollout(dt=dt, steps=steps)

# (Optional) quick energy sanity check
KE0 = 0.5 * np.sum(sim.masses[:, None] * vel0**2)
KE_end = 0.5 * np.sum(sim.masses[:, None] * vel_traj[-1]**2)
print("KE0:", KE0, "KE_end:", KE_end, "diff:", KE_end - KE0)

# making the figures and animations

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(0, sim.W)
ax.set_ylim(0, sim.H)
ax.set_aspect("equal")
ax.set_title("2D Particle Simulator (elastic collisions)")

# Draw walls as a box
ax.plot([0, sim.W, sim.W, 0, 0], [0, 0, sim.H, sim.H, 0], lw=2)

# Circles (matplotlib patches)
circles = []
for i in range(pos_traj.shape[1]):
    c = plt.Circle(pos_traj[0, i], sim.radii[i], fill=True)
    ax.add_patch(c)
    circles.append(c)

time_text = ax.text(0.02, 1.02, "", transform=ax.transAxes)

def init():
    for i, c in enumerate(circles):
        c.center = pos_traj[0, i]
    time_text.set_text("")
    return circles + [time_text]

def animate(frame):
    for i, c in enumerate(circles):
        c.center = pos_traj[frame, i]
    time_text.set_text(f"t = {frame*dt:.2f}s")
    return circles + [time_text]

ani = animation.FuncAnimation(
    fig, animate, frames=len(pos_traj), init_func=init,
    interval=20, blit=True
)

plt.close(fig)
ani

HTML(ani.to_jshtml())

# data generation: responsible for creating many episodes and labeling which time steps contain collisions so we can later build training pairs, and oversample or weight collision steps

# sample a random valid initial condition
def sample_init(W, H, radii, speed_max=0.7, seed=None):
    rng = np.random.default_rng(seed)
    r0, r1 = radii

    # rejection sample positions to avoid overlap
    for _ in range(10000):
        p0 = np.array([rng.uniform(r0, W-r0), rng.uniform(r0, H-r0)])
        p1 = np.array([rng.uniform(r1, W-r1), rng.uniform(r1, H-r1)])
        if np.linalg.norm(p1 - p0) > (r0 + r1 + 1e-3):
            break
    else:
        raise RuntimeError("Failed to sample non-overlapping initial positions")

    def rand_vel():
        theta = rng.uniform(0, 2*np.pi)
        mag = rng.uniform(0.0, speed_max)
        return np.array([mag*np.cos(theta), mag*np.sin(theta)])

    v0, v1 = rand_vel(), rand_vel()
    return np.stack([p0, p1]), np.stack([v0, v1])

# labels which steps are "collision steps"
def compute_collision_flags(pos_traj, vel_traj, radii, W, H, eps=1e-3, dv_thresh=1e-3):
    """
    pos_traj, vel_traj: (T,2,2)
    returns collision: (T-1,) uint8
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

# generate a dataset of many trajectories
def collect_episodes(sim, E=1000, steps=500, dt=0.01, speed_max=0.7, seed=0):
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

pos_all, vel_all, coll_all, meta = collect_episodes(sim, E=1000, steps=500, dt=0.01, speed_max=0.7, seed=0)

pos_all.shape

def episodes_to_XY(pos_all, vel_all, coll_all, meta, episode_indices):
    """
    Returns flattened step pairs for the selected episodes:
      X: (M,12) = [p0(x,y), v0, r0, m0,  p1(x,y), v1, r1, m1]
      Y: (M, 8) = next [p0, v0, p1, v1]
      C: (M,)   = collision flag for t->t+1
    """
    dt = float(meta["dt"])
    radii = np.asarray(meta["radii"], dtype=np.float32)   # (2,)
    masses = np.asarray(meta["masses"], dtype=np.float32) # (2,)

    r0, r1 = radii
    m0, m1 = masses

    X_list, Y_list, C_list = [], [], []
    for e in episode_indices:
        pos = pos_all[e]         # (T,2,2)
        vel = vel_all[e]         # (T,2,2)
        col = coll_all[e]        # (T-1,)

        T = pos.shape[0]
        # Steps t = 0..T-2
        # X: use state at t, Y: state at t+1
        X = np.zeros((T-1, 12), dtype=np.float32)
        Y = np.zeros((T-1,  8), dtype=np.float32)

        pos_t = pos[:-1]     # (T-1,2,2)
        vel_t = vel[:-1]
        pos_n = pos[1:]      # (T-1,2,2)
        vel_n = vel[1:]

        # pack X
        X[:, 0] = pos_t[:,0,0]; X[:, 1] = pos_t[:,0,1]
        X[:, 2] = vel_t[:,0,0]; X[:, 3] = vel_t[:,0,1]
        X[:, 4] = r0;          X[:, 5] = m0

        X[:, 6] = pos_t[:,1,0]; X[:, 7] = pos_t[:,1,1]
        X[:, 8] = vel_t[:,1,0]; X[:, 9] = vel_t[:,1,1]
        X[:,10] = r1;           X[:,11] = m1

        # pack Y
        Y[:, 0] = pos_n[:,0,0]; Y[:, 1] = pos_n[:,0,1]
        Y[:, 2] = vel_n[:,0,0]; Y[:, 3] = vel_n[:,0,1]
        Y[:, 4] = pos_n[:,1,0]; Y[:, 5] = pos_n[:,1,1]
        Y[:, 6] = vel_n[:,1,0]; Y[:, 7] = vel_n[:,1,1]

        X_list.append(X)
        Y_list.append(Y)
        C_list.append(col.astype(np.uint8))

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    C = np.concatenate(C_list, axis=0)
    return X, Y, C


# UPDATE
def episodes_to_XY_residual(pos_all, vel_all, coll_all, meta, episode_indices):
    """
    Returns:
      X: (M,12) current state
      Y: (M,8)  residual = next_state - free_flight_next_state
      C: (M,)   collision flag
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
        C   = coll_all[e].astype(bool)       # (T-1,)

        pos_t = pos[:-1]
        vel_t = vel[:-1]
        pos_n = pos[1:]
        vel_n = vel[1:]

        Tm1 = pos_t.shape[0]
        X = np.zeros((Tm1, 12), dtype=np.float32)
        Y_next = np.zeros((Tm1, 8), dtype=np.float32)
        Y_free = np.zeros((Tm1, 8), dtype=np.float32)

        # ---- pack X (state at t) ----
        X[:,0] = pos_t[:,0,0];  X[:,1] = pos_t[:,0,1]
        X[:,2] = vel_t[:,0,0];  X[:,3] = vel_t[:,0,1]
        X[:,4] = r0;            X[:,5] = m0
        X[:,6] = pos_t[:,1,0];  X[:,7] = pos_t[:,1,1]
        X[:,8] = vel_t[:,1,0];  X[:,9] = vel_t[:,1,1]
        X[:,10]= r1;            X[:,11]= m1

        # ---- pack Y_next (true next state) ----
        Y_next[:,0] = pos_n[:,0,0];  Y_next[:,1] = pos_n[:,0,1]
        Y_next[:,2] = vel_n[:,0,0];  Y_next[:,3] = vel_n[:,0,1]
        Y_next[:,4] = pos_n[:,1,0];  Y_next[:,5] = pos_n[:,1,1]
        Y_next[:,6] = vel_n[:,1,0];  Y_next[:,7] = vel_n[:,1,1]

        # ---- build Y_free (free-flight next state) ----
        pos_free = pos_t + vel_t * dt
        vel_free = vel_t

        Y_free[:,0] = pos_free[:,0,0];  Y_free[:,1] = pos_free[:,0,1]
        Y_free[:,2] = vel_free[:,0,0];  Y_free[:,3] = vel_free[:,0,1]
        Y_free[:,4] = pos_free[:,1,0];  Y_free[:,5] = pos_free[:,1,1]
        Y_free[:,6] = vel_free[:,1,0];  Y_free[:,7] = vel_free[:,1,1]

        # ---- residual target ----
        Y_resid = (Y_next - Y_free).astype(np.float32)

        X_list.append(X)
        Y_list.append(Y_resid)
        C_list.append(C)

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    C = np.concatenate(C_list, axis=0).astype(np.uint8)
    return X, Y, C


# -------------------------
# 2) Normalization (fit on TRAIN only)
# -------------------------

def fit_standardizer(A, eps=1e-8):
    mean = A.mean(axis=0, keepdims=True)
    std  = A.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(A, mean, std):
    return ((A - mean) / std).astype(np.float32)


# -------------------------
# 3) Torch Dataset
# -------------------------

class StepDataset(Dataset):
    def __init__(self, X, Y, C):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.C = torch.from_numpy(C.astype(np.uint8))  # keep for metrics
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.C[i]
    
    # -------------------------
# 4) Simple MLP model
# -------------------------


class MLP(nn.Module):
    def __init__(self, in_dim=12, hidden=128, out_dim=8, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, out_dim),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)



class InteractionNet2Walls(nn.Module):
    def __init__(self, hidden=128, W=1.0, H=1.0):
        super().__init__()
        self.W = float(W)
        self.H = float(H)

        node_in = 10  # x,y,vx,vy,r,m, dL,dR,dB,dT
        self.phi_node = nn.Sequential(
            nn.Linear(node_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        edge_in = 2*hidden + 5  # h_i,h_j, dx,dy,dvx,dvy,dist  -> 2H+5
        self.phi_edge = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        self.phi_update = nn.Sequential(
            nn.Linear(2*hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        self.phi_out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)
        )

    def _node_feats(self, X, which):
        # which=0 uses X[:,0:6], which=1 uses X[:,6:12]
        if which == 0:
            x = X[:, 0]; y = X[:, 1]; vx = X[:, 2]; vy = X[:, 3]; r = X[:, 4]; m = X[:, 5]
        else:
            x = X[:, 6]; y = X[:, 7]; vx = X[:, 8]; vy = X[:, 9]; r = X[:,10]; m = X[:,11]

        dL = x - r
        dR = (self.W - r) - x
        dB = y - r
        dT = (self.H - r) - y

        return torch.stack([x,y,vx,vy,r,m,dL,dR,dB,dT], dim=1)

    def forward(self, X):
        # Node features with wall distances
        n0 = self._node_feats(X, 0)
        n1 = self._node_feats(X, 1)

        h0 = self.phi_node(n0)
        h1 = self.phi_node(n1)

        p0 = X[:, 0:2]; v0 = X[:, 2:4]
        p1 = X[:, 6:8]; v1 = X[:, 8:10]

        dp01 = p1 - p0
        dv01 = v1 - v0
        dist = torch.sqrt((dp01**2).sum(dim=1, keepdim=True) + 1e-12)

        dp10 = -dp01
        dv10 = -dv01

        edge10 = torch.cat([h0, h1, dp10, dv10, dist], dim=1)
        edge01 = torch.cat([h1, h0, dp01, dv01, dist], dim=1)

        m10 = self.phi_edge(edge10)
        m01 = self.phi_edge(edge01)

        h0p = self.phi_update(torch.cat([h0, m10], dim=1))
        h1p = self.phi_update(torch.cat([h1, m01], dim=1))

        y0 = self.phi_out(h0p)
        y1 = self.phi_out(h1p)

        return torch.cat([y0, y1], dim=1)

class ResMLP(nn.Module):
    def __init__(self, dim, hidden_mult=2, dropout=0.0):
        super().__init__()
        h = int(dim * hidden_mult)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, h)
        self.fc2 = nn.Linear(h, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x)
        y = F.silu(self.fc1(y))
        y = self.drop(y)
        y = self.fc2(y)
        return x + y

class InteractionNet2Walls_v2(nn.Module):
    def __init__(self, hidden=512, depth=6, W=1.0, H=1.0, dropout=0.05):
        super().__init__()
        self.W = float(W)
        self.H = float(H)

        node_in = 10  # x,y,vx,vy,r,m,dL,dR,dB,dT

        # Node encoder
        self.node_in = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.node_trunk = nn.Sequential(*[ResMLP(hidden, hidden_mult=2, dropout=dropout) for _ in range(depth)])

        # Edge/message encoder (note: distance feature dim changed)
        # rel: dx,dy,dvx,dvy, dist, inv_dist, inv_dist2, dist2  => 8
        rel_dim = 8
        edge_in = 2 * hidden + rel_dim

        self.edge_in = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.edge_trunk = nn.Sequential(*[ResMLP(hidden, hidden_mult=2, dropout=dropout) for _ in range(depth)])

        # Gate: decides how much of message to use (0..1)
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )

        # Update trunk: combine node + gated message
        self.update_in = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.update_trunk = nn.Sequential(*[ResMLP(hidden, hidden_mult=2, dropout=dropout) for _ in range(depth)])

        # Output head
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    def _node_feats(self, X, which):
        if which == 0:
            x = X[:, 0]; y = X[:, 1]; vx = X[:, 2]; vy = X[:, 3]; r = X[:, 4]; m = X[:, 5]
        else:
            x = X[:, 6]; y = X[:, 7]; vx = X[:, 8]; vy = X[:, 9]; r = X[:,10]; m = X[:,11]

        dL = x - r
        dR = (self.W - r) - x
        dB = y - r
        dT = (self.H - r) - y
        return torch.stack([x,y,vx,vy,r,m,dL,dR,dB,dT], dim=1)

    def _rel_feats(self, p_i, v_i, p_j, v_j):
        dp = p_j - p_i
        dv = v_j - v_i
        dist2 = (dp * dp).sum(dim=1, keepdim=True) + 1e-12
        dist  = torch.sqrt(dist2)
        inv_d = 1.0 / (dist + 1e-6)
        inv_d2 = 1.0 / (dist2 + 1e-6)
        return torch.cat([dp, dv, dist, inv_d, inv_d2, dist2], dim=1)  # (B,8)

    def forward(self, X):
        # Node features with wall distances
        n0 = self._node_feats(X, 0)
        n1 = self._node_feats(X, 1)

        h0 = self.node_trunk(self.node_in(n0))
        h1 = self.node_trunk(self.node_in(n1))

        p0 = X[:, 0:2]; v0 = X[:, 2:4]
        p1 = X[:, 6:8]; v1 = X[:, 8:10]

        rel01 = self._rel_feats(p0, v0, p1, v1)  # j=1 relative to i=0
        rel10 = self._rel_feats(p1, v1, p0, v0)  # j=0 relative to i=1

        e01 = torch.cat([h0, h1, rel01], dim=1)
        e10 = torch.cat([h1, h0, rel10], dim=1)

        m01 = self.edge_trunk(self.edge_in(e01))
        m10 = self.edge_trunk(self.edge_in(e10))

        # Gate messages (helps with strong nonlinearity near collision)
        g01 = self.edge_gate(m01)
        g10 = self.edge_gate(m10)
        m01 = g01 * m01
        m10 = g10 * m10

        u0 = self.update_trunk(self.update_in(torch.cat([h0, m01], dim=1)))
        u1 = self.update_trunk(self.update_in(torch.cat([h1, m10], dim=1)))

        y0 = self.out(u0)
        y1 = self.out(u1)
        return torch.cat([y0, y1], dim=1)


# -------------------------
# 5) Train / eval loops
# -------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_all = 0.0
    mse_col = 0.0
    mse_ncol = 0.0
    n_all = 0
    n_col = 0
    n_ncol = 0

    # Also track position/velocity MSE separately
    # Y layout: [p0x,p0y,v0x,v0y, p1x,p1y,v1x,v1y]
    for X, Y, C in loader:
        X = X.to(device)
        Y = Y.to(device)
        C = C.to(device).bool()

        pred = model(X)
        err = (pred - Y) ** 2  # (B,8)
        per_sample = err.mean(dim=1)  # (B,)

        mse_all += per_sample.sum().item()
        n_all += per_sample.numel()

        if C.any():
            mse_col += per_sample[C].sum().item()
            n_col += C.sum().item()
        if (~C).any():
            mse_ncol += per_sample[~C].sum().item()
            n_ncol += (~C).sum().item()

    out = {
        "mse_all": mse_all / max(n_all, 1),
        "mse_collision": mse_col / max(n_col, 1),
        "mse_noncollision": mse_ncol / max(n_ncol, 1),
        "n_all": n_all,
        "n_collision": n_col,
        "n_noncollision": n_ncol,
    }
    return out

def train(model, train_loader, test_loader, device, epochs=20, lr=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for X, Y, C in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            loss = loss_fn(pred, Y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item() * X.shape[0]
            n += X.shape[0]

        train_loss = running / max(n, 1)

        # Evaluate on TEST (since no validation now)
        test_stats = evaluate(model, test_loader, device)

        print(
            f"Epoch {ep:03d} | train_loss={train_loss:.6f} "
            f"| test_mse={test_stats['mse_all']:.6f} "
            f"| test_collision={test_stats['mse_collision']:.6f} "
            f"| test_noncollision={test_stats['mse_noncollision']:.6f} "
            f"(n_col={test_stats['n_collision']}, n_noncol={test_stats['n_noncollision']})"
        )

def train_weighted(model, train_loader, test_loader, device,
                   epochs=50, lr=1e-3, collision_weight=20.0):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for X, Y, C in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            C = C.to(device).bool()

            pred = model(X)

            # per-sample MSE (mean over the 8 output dims)
            per_sample_mse = ((pred - Y) ** 2).mean(dim=1)   # (B,)

            # weights: collision samples get upweighted
            w = torch.ones_like(per_sample_mse)
            w[C] = collision_weight

            loss = (w * per_sample_mse).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item() * X.shape[0]
            n += X.shape[0]

        train_loss = running / max(n, 1)

        test_stats = evaluate(model, test_loader, device)
        print(
            f"Epoch {ep:03d} | train_loss={train_loss:.6f} "
            f"| test_mse={test_stats['mse_all']:.6f} "
            f"| test_collision={test_stats['mse_collision']:.6f} "
            f"| test_noncollision={test_stats['mse_noncollision']:.6f} "
            f"(n_col={test_stats['n_collision']}, n_noncol={test_stats['n_noncollision']})"
        )

# =========================
# 6) Put it all together (TRAIN / TEST only)
# =========================

E = pos_all.shape[0]
idx = np.arange(E)

# Optional: shuffle episode order once (recommended)
rng = np.random.default_rng(0)
rng.shuffle(idx)

n_train = int(0.8 * E)
train_eps = idx[:n_train]
test_eps  = idx[n_train:]

# Build flattened step pairs
Xtr, Ytr, Ctr = episodes_to_XY(pos_all, vel_all, coll_all, meta, train_eps)
Xte, Yte, Cte = episodes_to_XY(pos_all, vel_all, coll_all, meta, test_eps)

print("Train steps:", Xtr.shape[0], "Test steps:", Xte.shape[0])
print("Train collision fraction:", Ctr.mean(), "Test:", Cte.mean())

# UPDATE
Xtr, Ytr, Ctr = episodes_to_XY_residual(pos_all, vel_all, coll_all, meta, train_eps)
Xte, Yte, Cte = episodes_to_XY_residual(pos_all, vel_all, coll_all, meta, test_eps)

# Fit on TRAIN only
x_mean, x_std = fit_standardizer(Xtr)
y_mean, y_std = fit_standardizer(Ytr)

Xtr_n = apply_standardizer(Xtr, x_mean, x_std)
Ytr_n = apply_standardizer(Ytr, y_mean, y_std)

Xte_n = apply_standardizer(Xte, x_mean, x_std)
Yte_n = apply_standardizer(Yte, y_mean, y_std)

train_loader = DataLoader(StepDataset(Xtr_n, Ytr_n, Ctr), batch_size=512, shuffle=True, drop_last=True)
test_loader  = DataLoader(StepDataset(Xte_n, Yte_n, Cte), batch_size=1024, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(in_dim=12, hidden=128, out_dim=8).to(device)
model = model.to(device)

model = InteractionNet2Walls_v2(hidden=512, depth=6, W=meta["W"], H=meta["H"], dropout=0.05).to(device)
model = model.to(device)

train(model, train_loader, test_loader, device, epochs=50, lr=1e-3)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = InteractionNet2Walls(hidden=128, W=meta["W"], H=meta["H"]) #MLP(in_dim=12, hidden=128, out_dim=8)
#train(model, train_loader, test_loader, device, epochs=50, lr=1e-3)

def pack_X(pos, vel, radii, masses):
    # pos, vel: (2,2)
    r0, r1 = radii
    m0, m1 = masses
    return np.array([
        pos[0,0], pos[0,1], vel[0,0], vel[0,1], r0, m0,
        pos[1,0], pos[1,1], vel[1,0], vel[1,1], r1, m1,
    ], dtype=np.float32)

def unpack_Y(y8):
    # y8: (8,)
    pos_next = np.array([[y8[0], y8[1]],
                         [y8[4], y8[5]]], dtype=np.float32)
    vel_next = np.array([[y8[2], y8[3]],
                         [y8[6], y8[7]]], dtype=np.float32)
    return pos_next, vel_next

@torch.no_grad()
def nn_rollout(model, pos0, vel0, radii, masses, steps,
               x_mean, x_std, y_mean, y_std, device):
    """
    Returns:
      pos_pred: (steps+1, 2, 2)
      vel_pred: (steps+1, 2, 2)
    """
    model.eval()

    pos_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)
    vel_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)

    pos_pred[0] = pos0.astype(np.float32)
    vel_pred[0] = vel0.astype(np.float32)

    pos_t = pos_pred[0].copy()
    vel_t = vel_pred[0].copy()

    for t in range(steps):
        x_raw = pack_X(pos_t, vel_t, radii, masses)                 # (12,)
        x_n = (x_raw[None, :] - x_mean) / x_std                     # (1,12)
        x_n_t = torch.from_numpy(x_n.astype(np.float32)).to(device)

        y_n = model(x_n_t).cpu().numpy()                            # (1,8) normalized
        y_raw = (y_n * y_std) + y_mean                              # denormalize -> (1,8)
        y_raw = y_raw[0].astype(np.float32)

        pos_t, vel_t = unpack_Y(y_raw)

        pos_pred[t+1] = pos_t
        vel_pred[t+1] = vel_t

    return pos_pred, vel_pred

# True rollout
sim = ParticleSim2D(W=1.0, H=1.0, radii=[0.06, 0.06], masses=[1.0, 1.0], restitution=1.0, seed=1)
sim.reset(pos0, vel0)
pos_true, vel_true = sim.rollout(dt=dt, steps=steps)

device = "cuda" if torch.cuda.is_available() else "cpu"

pos_nn, vel_nn = nn_rollout(
    model,
    pos0=pos0, vel0=vel0,
    radii=np.array(sim.radii, dtype=np.float32),
    masses=np.array(sim.masses, dtype=np.float32),
    steps=steps,
    x_mean=x_mean, x_std=x_std,
    y_mean=y_mean, y_std=y_std,
    device=device
)

W, H = sim.W, sim.H
radii = sim.radii

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for ax, title in [(ax1, "TRUE simulator"), (ax2, "NN rollout")]:
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], lw=2)

# circles: true
circles_true = []
for i in range(2):
    c = plt.Circle(pos_true[0, i], radii[i], fill=True)
    ax1.add_patch(c)
    circles_true.append(c)

# circles: nn
circles_nn = []
for i in range(2):
    c = plt.Circle(pos_nn[0, i], radii[i], fill=True)
    ax2.add_patch(c)
    circles_nn.append(c)

time_text = fig.text(0.5, 0.98, "", ha="center")

def animate(frame):
    for i, c in enumerate(circles_true):
        c.center = pos_true[frame, i]
    for i, c in enumerate(circles_nn):
        c.center = pos_nn[frame, i]
    time_text.set_text(f"t = {frame*dt:.2f}s")
    return circles_true + circles_nn + [time_text]

ani = animation.FuncAnimation(fig, animate, frames=len(pos_true), interval=20, blit=True)
plt.close(fig)
ani

HTML(ani.to_jshtml())

from IPython.display import HTML

html_str = ani.to_jshtml()

with open("animation.html", "w") as f:
    f.write(html_str)

import os
os.getcwd()

# ---------- helpers to pack/unpack ----------
def pack_X(pos, vel, radii, masses):
    r0, r1 = radii
    m0, m1 = masses
    return np.array([
        pos[0,0], pos[0,1], vel[0,0], vel[0,1], r0, m0,
        pos[1,0], pos[1,1], vel[1,0], vel[1,1], r1, m1
    ], dtype=np.float32)

def unpack_Y(y8):
    pos_next = np.array([[y8[0], y8[1]],
                         [y8[4], y8[5]]], dtype=np.float32)
    vel_next = np.array([[y8[2], y8[3]],
                         [y8[6], y8[7]]], dtype=np.float32)
    return pos_next, vel_next

@torch.no_grad()
def nn_rollout(model, pos0, vel0, radii, masses, steps,
               x_mean, x_std, y_mean, y_std, device):
    model.eval()
    pos_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)
    vel_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)

    pos_pred[0] = pos0.astype(np.float32)
    vel_pred[0] = vel0.astype(np.float32)

    pos_t = pos_pred[0].copy()
    vel_t = vel_pred[0].copy()

    for t in range(steps):
        x_raw = pack_X(pos_t, vel_t, radii, masses)  # (12,)
        x_n = ((x_raw[None, :] - x_mean) / x_std).astype(np.float32)
        x_t = torch.from_numpy(x_n).to(device)

        y_n = model(x_t).cpu().numpy()               # normalized (1,8)
        y_raw = (y_n * y_std) + y_mean               # denormalize (1,8)
        y_raw = y_raw[0].astype(np.float32)

        pos_t, vel_t = unpack_Y(y_raw)
        pos_pred[t+1] = pos_t
        vel_pred[t+1] = vel_t

    return pos_pred, vel_pred

# ---------- choose an initial condition ----------
# Option A: use your manual pos0/vel0
pos0 = np.array([[0.1, 0.50],
                 [0.75, 0.50]], dtype=np.float32)

vel0 = np.array([[ 0.15,  0.20],
                 [-0.35, -0.10]], dtype=np.float32)

# Option B (alternative): start from a test episode state
# e = test_eps[0]
# pos0 = pos_all[e, 0].astype(np.float32)
# vel0 = vel_all[e, 0].astype(np.float32)

# ---------- run true simulator rollout ----------
dt = float(meta["dt"])
steps = 1200

sim_true = ParticleSim2D(W=float(meta["W"]), H=float(meta["H"]),
                         radii=meta["radii"], masses=meta["masses"],
                         restitution=float(meta["restitution"]), seed=123)
sim_true.reset(pos0, vel0)
pos_true, vel_true = sim_true.rollout(dt=dt, steps=steps)

# ---------- run NN rollout ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)  # your trained model already exists

pos_nn, vel_nn = nn_rollout(
    model, pos0, vel0,
    radii=np.asarray(meta["radii"], dtype=np.float32),
    masses=np.asarray(meta["masses"], dtype=np.float32),
    steps=steps,
    x_mean=x_mean, x_std=x_std,
    y_mean=y_mean, y_std=y_std,
    device=device
)

# ---------- plot error growth ----------
pos_err = np.sqrt(((pos_nn - pos_true)**2).sum(axis=2).mean(axis=1))  # (T,)
# plt.figure()
# plt.plot(np.arange(len(pos_err))*dt, pos_err)
# plt.xlabel("time (s)")
# plt.ylabel("RMS position error")
# plt.title("NN rollout error growth")
# plt.show()
plt.figure()
plt.semilogy(np.arange(len(pos_err))*dt, pos_err + 1e-12)
plt.xlabel("time (s)")
plt.ylabel("RMS position error (log scale)")
plt.title("NN rollout error growth (log)")
plt.show()

# ---------- animate side-by-side ----------
W, H = float(meta["W"]), float(meta["H"])
radii = np.asarray(meta["radii"], dtype=np.float32)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for ax, title in [(ax1, "TRUE simulator"), (ax2, "NN rollout")]:
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], lw=2)

circles_true, circles_nn = [], []
for i in range(2):
    c1 = plt.Circle(pos_true[0, i], radii[i], fill=True)
    ax1.add_patch(c1); circles_true.append(c1)

    c2 = plt.Circle(pos_nn[0, i], radii[i], fill=True)
    ax2.add_patch(c2); circles_nn.append(c2)

time_text = fig.text(0.5, 0.98, "", ha="center")

def animate(frame):
    for i, c in enumerate(circles_true):
        c.center = pos_true[frame, i]
    for i, c in enumerate(circles_nn):
        c.center = pos_nn[frame, i]
    time_text.set_text(f"t = {frame*dt:.2f}s")
    return circles_true + circles_nn + [time_text]

ani = animation.FuncAnimation(fig, animate, frames=len(pos_true), interval=20, blit=True)
plt.close(fig)
ani

def wall_hit_flags(pos_true, radii, W, H, eps=1e-4):
    # pos_true: (T,2,2)
    r0, r1 = radii
    hits = np.zeros((pos_true.shape[0],), dtype=bool)
    for i, r in enumerate([r0, r1]):
        x = pos_true[:, i, 0]
        y = pos_true[:, i, 1]
        hits |= (x <= r + eps) | (x >= W - r - eps) | (y <= r + eps) | (y >= H - r - eps)
    return hits

hits = wall_hit_flags(pos_true, radii=np.asarray(meta["radii"]), W=float(meta["W"]), H=float(meta["H"]))
t_hit = np.where(hits)[0]
print("first wall hit frame:", t_hit[0] if len(t_hit) else None, "time:", (t_hit[0]*dt if len(t_hit) else None))

# pick a random test sample index
i = 0
x_true_n = Xte_n[i]          # normalized X
y_true_n = Yte_n[i]          # normalized Y

x_true_n_t = torch.from_numpy(x_true_n[None, :]).float().to(device)
with torch.no_grad():
    y_pred_n = model(x_true_n_t).cpu().numpy()[0]

print("one-step MSE in normalized space:",
      np.mean((y_pred_n - y_true_n)**2))

# now denormalize and compare in physical units
y_pred = (y_pred_n * y_std[0]) + y_mean[0]
y_true = (y_true_n * y_std[0]) + y_mean[0]

print("one-step MSE in physical units:",
      np.mean((y_pred - y_true)**2))

print("y_true:", y_true)
print("y_pred:", y_pred)

speed_true = np.linalg.norm(vel_true, axis=2)  # (T,2)
speed_nn   = np.linalg.norm(vel_nn, axis=2)

plt.figure()
plt.plot(np.arange(speed_true.shape[0])*dt, speed_true.max(axis=1), label="true max speed")
plt.plot(np.arange(speed_nn.shape[0])*dt, speed_nn.max(axis=1), label="nn max speed")
plt.xlabel("time (s)")
plt.ylabel("max particle speed")
plt.legend()
plt.title("Speed blow-up diagnostic")
plt.show()

print("true max speed:", speed_true.max())
print("nn   max speed:", speed_nn.max())

@torch.no_grad()
def nn_rollout_debug(model, pos0, vel0, radii, masses, steps,
                     x_mean, x_std, y_mean, y_std, device,
                     vmax=2.0):
    model.eval()

    pos_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)
    vel_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)
    pos_pred[0] = pos0.astype(np.float32)
    vel_pred[0] = vel0.astype(np.float32)

    pos_t = pos_pred[0].copy()
    vel_t = vel_pred[0].copy()

    for t in range(steps):
        x_raw = pack_X(pos_t, vel_t, radii, masses)  # (12,)
        if not np.isfinite(x_raw).all():
            print("Non-finite X at step", t, x_raw)
            break

        x_n = ((x_raw[None, :] - x_mean) / x_std).astype(np.float32)
        x_torch = torch.from_numpy(x_n).to(device)

        y_n = model(x_torch).cpu().numpy()[0]
        if not np.isfinite(y_n).all():
            print("Non-finite model output (normalized) at step", t, y_n)
            break

        y_raw = (y_n * y_std[0]) + y_mean[0]  # (8,)
        if not np.isfinite(y_raw).all():
            print("Non-finite model output (raw) at step", t, y_raw)
            break

        pos_t, vel_t = unpack_Y(y_raw)

        # optional: clamp velocity to keep rollout from going OOD
        vel_t = np.clip(vel_t, -vmax, vmax)

        pos_pred[t+1] = pos_t
        vel_pred[t+1] = vel_t

    return pos_pred[:t+2], vel_pred[:t+2]

pos_nn_dbg, vel_nn_dbg = nn_rollout_debug(
    model, pos0, vel0,
    radii=np.asarray(meta["radii"], dtype=np.float32),
    masses=np.asarray(meta["masses"], dtype=np.float32),
    steps=steps,
    x_mean=x_mean, x_std=x_std,
    y_mean=y_mean, y_std=y_std,
    device=device,
    vmax=2.0
)
print("Produced steps:", len(pos_nn_dbg)-1)

@torch.no_grad()
def debug_step(model, pos0, vel0, radii, masses, steps,
               x_mean, x_std, y_mean, y_std, device):
    model.eval()
    pos_t = pos0.astype(np.float32).copy()
    vel_t = vel0.astype(np.float32).copy()

    for t in range(steps):
        x_raw = pack_X(pos_t, vel_t, radii, masses)
        x_n = ((x_raw[None,:] - x_mean) / x_std).astype(np.float32)
        x_torch = torch.from_numpy(x_n).to(device)

        y_n = model(x_torch).cpu().numpy()[0]

        if not np.isfinite(y_n).all():
            print(f"\nNaN at step {t}")
            print("x_raw:", x_raw)
            print("x_n min/max:", np.min(x_n), np.max(x_n))
            print("pos_t:", pos_t)
            print("vel_t:", vel_t)
            return t

        y_raw = (y_n * y_std[0]) + y_mean[0]
        pos_t, vel_t = unpack_Y(y_raw)

    print("No NaNs up to", steps)
    return None

bad_t = debug_step(
    model, pos0, vel0,
    radii=np.asarray(meta["radii"], dtype=np.float32),
    masses=np.asarray(meta["masses"], dtype=np.float32),
    steps=300,
    x_mean=x_mean, x_std=x_std,
    y_mean=y_mean, y_std=y_std,
    device=device
)

def episodes_to_XY_residual(pos_all, vel_all, coll_all, meta, episode_indices):
    """
    Returns:
      X: (M,12) current state
      Y: (M,8)  residual = next_state - free_flight_next_state
      C: (M,)   collision flag
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
        C   = coll_all[e].astype(bool)       # (T-1,)

        pos_t = pos[:-1]
        vel_t = vel[:-1]
        pos_n = pos[1:]
        vel_n = vel[1:]

        Tm1 = pos_t.shape[0]
        X = np.zeros((Tm1, 12), dtype=np.float32)
        Y_next = np.zeros((Tm1, 8), dtype=np.float32)
        Y_free = np.zeros((Tm1, 8), dtype=np.float32)

        # ---- pack X (state at t) ----
        X[:,0] = pos_t[:,0,0];  X[:,1] = pos_t[:,0,1]
        X[:,2] = vel_t[:,0,0];  X[:,3] = vel_t[:,0,1]
        X[:,4] = r0;            X[:,5] = m0
        X[:,6] = pos_t[:,1,0];  X[:,7] = pos_t[:,1,1]
        X[:,8] = vel_t[:,1,0];  X[:,9] = vel_t[:,1,1]
        X[:,10]= r1;            X[:,11]= m1

        # ---- pack Y_next (true next state) ----
        Y_next[:,0] = pos_n[:,0,0];  Y_next[:,1] = pos_n[:,0,1]
        Y_next[:,2] = vel_n[:,0,0];  Y_next[:,3] = vel_n[:,0,1]
        Y_next[:,4] = pos_n[:,1,0];  Y_next[:,5] = pos_n[:,1,1]
        Y_next[:,6] = vel_n[:,1,0];  Y_next[:,7] = vel_n[:,1,1]

        # ---- build Y_free (free-flight next state) ----
        pos_free = pos_t + vel_t * dt
        vel_free = vel_t

        Y_free[:,0] = pos_free[:,0,0];  Y_free[:,1] = pos_free[:,0,1]
        Y_free[:,2] = vel_free[:,0,0];  Y_free[:,3] = vel_free[:,0,1]
        Y_free[:,4] = pos_free[:,1,0];  Y_free[:,5] = pos_free[:,1,1]
        Y_free[:,6] = vel_free[:,1,0];  Y_free[:,7] = vel_free[:,1,1]

        # ---- residual target ----
        Y_resid = (Y_next - Y_free).astype(np.float32)

        X_list.append(X)
        Y_list.append(Y_resid)
        C_list.append(C)

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    C = np.concatenate(C_list, axis=0).astype(np.uint8)
    return X, Y, C
