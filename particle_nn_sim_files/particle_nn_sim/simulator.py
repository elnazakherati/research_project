import numpy as np

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

                    # --- Velocity update via impulse ---
                    vi = self.vel[i]
                    vj = self.vel[j]
                    rel_v = vi - vj
                    rel_vn = np.dot(rel_v, n)

                    # Only apply impulse if moving toward each other
                    if rel_vn > 0:
                        j_imp = -(1 + e) * rel_vn / (w_i + w_j)
                        impulse = j_imp * n
                        self.vel[i] += impulse * w_i
                        self.vel[j] -= impulse * w_j

    def step(self, dt):
        dt = float(dt)
        self.pos += self.vel * dt

        # Handle wall collisions then pair collisions
        self._handle_wall_collisions()
        self._handle_pair_collisions()

        # Clamp again in case pair correction pushed outside
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
