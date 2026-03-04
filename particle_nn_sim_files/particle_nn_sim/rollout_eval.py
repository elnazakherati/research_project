import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

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
def nn_rollout_absolute(model, pos0, vel0, radii, masses, steps,
                        x_mean, x_std, y_mean, y_std, device):
    model.eval()
    pos_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)
    vel_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)
    pos_pred[0] = pos0.astype(np.float32)
    vel_pred[0] = vel0.astype(np.float32)

    pos_t = pos_pred[0].copy()
    vel_t = vel_pred[0].copy()

    for t in range(steps):
        x_raw = pack_X(pos_t, vel_t, radii, masses)
        x_n = ((x_raw[None,:] - x_mean) / x_std).astype(np.float32)
        x_torch = torch.from_numpy(x_n).to(device)

        y_n = model(x_torch).cpu().numpy()[0]
        y_raw = (y_n * y_std[0]) + y_mean[0]

        pos_t, vel_t = unpack_Y(y_raw)
        pos_pred[t+1] = pos_t
        vel_pred[t+1] = vel_t

        if not (np.isfinite(pos_t).all() and np.isfinite(vel_t).all()):
            return pos_pred[:t+2], vel_pred[:t+2]

    return pos_pred, vel_pred

@torch.no_grad()
def nn_rollout_residual(model, pos0, vel0, radii, masses, steps,
                        x_mean, x_std, y_mean, y_std, device, dt):
    model.eval()
    pos_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)
    vel_pred = np.zeros((steps+1, 2, 2), dtype=np.float32)
    pos_pred[0] = pos0.astype(np.float32)
    vel_pred[0] = vel0.astype(np.float32)

    pos_t = pos_pred[0].copy()
    vel_t = vel_pred[0].copy()

    for t in range(steps):
        x_raw = pack_X(pos_t, vel_t, radii, masses)
        x_n = ((x_raw[None,:] - x_mean) / x_std).astype(np.float32)
        x_torch = torch.from_numpy(x_n).to(device)

        resid_n = model(x_torch).cpu().numpy()[0]
        resid = (resid_n * y_std[0]) + y_mean[0]

        pos_free = pos_t + vel_t * dt
        vel_free = vel_t
        y_free = np.array([
            pos_free[0,0], pos_free[0,1], vel_free[0,0], vel_free[0,1],
            pos_free[1,0], pos_free[1,1], vel_free[1,0], vel_free[1,1],
        ], dtype=np.float32)

        y_next = y_free + resid.astype(np.float32)
        pos_t, vel_t = unpack_Y(y_next)

        pos_pred[t+1] = pos_t
        vel_pred[t+1] = vel_t

        if not (np.isfinite(pos_t).all() and np.isfinite(vel_t).all()):
            return pos_pred[:t+2], vel_pred[:t+2]

    return pos_pred, vel_pred

def rms_pos_error(pos_a, pos_b):
    err = ((pos_a - pos_b)**2).sum(axis=2)  # (T,2)
    return np.sqrt(err.mean(axis=1))        # (T,)

def plot_rollout_error(pos_true, pos_pred, dt):
    e = rms_pos_error(pos_true, pos_pred)
    plt.figure()
    plt.semilogy(np.arange(len(e))*dt, e + 1e-12)
    plt.xlabel("time (s)")
    plt.ylabel("RMS position error (log)")
    plt.title("Rollout error growth")
    plt.show()
    return e

def animate_side_by_side(pos_true, pos_pred, radii, W, H, dt, interval=20):
    radii = np.asarray(radii, dtype=np.float32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, title in [(ax1, "TRUE simulator"), (ax2, "NN rollout")]:
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], lw=2)

    circles_true, circles_pred = [], []
    for i in range(2):
        c1 = plt.Circle(pos_true[0, i], radii[i], fill=True)
        ax1.add_patch(c1); circles_true.append(c1)

        c2 = plt.Circle(pos_pred[0, i], radii[i], fill=True)
        ax2.add_patch(c2); circles_pred.append(c2)

    time_text = fig.text(0.5, 0.98, "", ha="center")

    def animate(frame):
        for i, c in enumerate(circles_true):
            c.center = pos_true[frame, i]
        for i, c in enumerate(circles_pred):
            c.center = pos_pred[frame, i]
        time_text.set_text(f"t = {frame*dt:.2f}s")
        return circles_true + circles_pred + [time_text]

    ani = animation.FuncAnimation(fig, animate, frames=len(pos_true), interval=interval, blit=True)
    plt.close(fig)
    return ani

def animate_single_rollout(pos, radii, W, H, dt, interval=20, title="Ground truth rollout"):
    radii = np.asarray(radii, dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float32)
    n_particles = pos.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], lw=2)

    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    circles = []
    for i in range(n_particles):
        c = plt.Circle(pos[0, i], radii[i], color=colors[i % len(colors)], fill=True, alpha=0.9)
        ax.add_patch(c)
        circles.append(c)

    time_text = fig.text(0.5, 0.98, "", ha="center")

    def animate(frame):
        for i, c in enumerate(circles):
            c.center = pos[frame, i]
        time_text.set_text(f"t = {frame*dt:.2f}s")
        return circles + [time_text]

    ani = animation.FuncAnimation(fig, animate, frames=len(pos), interval=interval, blit=True)
    plt.close(fig)
    return ani

def extract_collision_events_from_rollout(pos, vel, radii, W, H, dt, wall_tol=1e-5, pair_tol=1e-5):
    """
    Infer collision events from a rollout trajectory.

    Works for simulator and NN rollouts because it only uses (pos, vel).
    Event timestamps are frame-based (time t_k = k * dt).
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    W = float(W)
    H = float(H)
    dt = float(dt)

    T, N, _ = pos.shape
    if vel.shape != pos.shape:
        raise ValueError(f"vel shape {vel.shape} must match pos shape {pos.shape}")
    if radii.shape != (N,):
        raise ValueError(f"radii shape {radii.shape} must be ({N},)")

    frame_events = []

    # Track pair-contact state to avoid counting the same contact across many frames.
    prev_pair_contact = np.zeros((N, N), dtype=bool)

    for k in range(1, T):
        events_k = []

        # Wall collisions: infer from velocity sign flip + being at a wall after update.
        x = pos[k, :, 0]
        y = pos[k, :, 1]
        vx_prev = vel[k - 1, :, 0]
        vy_prev = vel[k - 1, :, 1]
        vx_now = vel[k, :, 0]
        vy_now = vel[k, :, 1]

        hit_x = (vx_prev * vx_now < 0) & (
            (np.abs(x - radii) <= wall_tol) | (np.abs(x - (W - radii)) <= wall_tol)
        )
        hit_y = (vy_prev * vy_now < 0) & (
            (np.abs(y - radii) <= wall_tol) | (np.abs(y - (H - radii)) <= wall_tol)
        )

        for i in np.where(hit_x)[0]:
            side = "left" if abs(x[i] - radii[i]) <= abs(x[i] - (W - radii[i])) else "right"
            events_k.append({
                "frame": k,
                "time": k * dt,
                "type": "wall",
                "particle": int(i),
                "side": side,
                "axis": "x",
            })
        for i in np.where(hit_y)[0]:
            side = "bottom" if abs(y[i] - radii[i]) <= abs(y[i] - (H - radii[i])) else "top"
            events_k.append({
                "frame": k,
                "time": k * dt,
                "type": "wall",
                "particle": int(i),
                "side": side,
                "axis": "y",
            })

        # Pair collisions: infer by entering/contacting overlap threshold.
        d = pos[k][None, :, :] - pos[k][:, None, :]
        dist = np.linalg.norm(d, axis=-1)
        thresh = radii[:, None] + radii[None, :] + pair_tol
        curr_pair_contact = dist <= thresh
        np.fill_diagonal(curr_pair_contact, False)

        for i in range(N):
            for j in range(i + 1, N):
                if curr_pair_contact[i, j] and not prev_pair_contact[i, j]:
                    events_k.append({
                        "frame": k,
                        "time": k * dt,
                        "type": "pair",
                        "pair": (int(i), int(j)),
                    })

        prev_pair_contact = curr_pair_contact
        frame_events.append(events_k)

    events = [ev for frame_list in frame_events for ev in frame_list]
    return events

def summarize_collision_events(events, dt=None):
    events = list(events)
    n_total = len(events)
    n_wall = sum(ev["type"] == "wall" for ev in events)
    n_pair = sum(ev["type"] == "pair" for ev in events)

    # Frame-level timestamps avoid zero inter-event times from multiple events in one frame.
    frame_times = sorted({float(ev["time"]) for ev in events})
    inter_times = np.diff(frame_times) if len(frame_times) >= 2 else np.array([], dtype=np.float64)

    if n_total > 0:
        type_ratio = {
            "wall": n_wall / n_total,
            "pair": n_pair / n_total,
        }
    else:
        type_ratio = {"wall": np.nan, "pair": np.nan}

    out = {
        "events": events,
        "collision_count": int(n_total),
        "wall_count": int(n_wall),
        "pair_count": int(n_pair),
        "collision_type_ratio": type_ratio,
        "inter_collision_times": inter_times.astype(np.float64),
        "collision_frame_count": int(len(frame_times)),
    }
    if dt is not None:
        out["dt"] = float(dt)
    return out

def collision_stats_from_rollout(pos, vel, radii, W, H, dt, wall_tol=1e-5, pair_tol=1e-5):
    events = extract_collision_events_from_rollout(
        pos, vel, radii, W, H, dt, wall_tol=wall_tol, pair_tol=pair_tol
    )
    return summarize_collision_events(events, dt=dt)

def plot_collision_stats_comparison(
    stats_a,
    stats_b,
    labels=("Ground truth", "Perturbed"),
    bins=30,
    figsize=(13, 3.8),
):
    """
    Plot three core rollout collision diagnostics:
      1) total collision count
      2) inter-collision time distribution
      3) collision type ratio (wall vs pair)
    """
    label_a, label_b = labels
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1) Collision count per rollout (single-rollout comparison -> bar chart)
    ax = axes[0]
    ax.bar([0, 1], [stats_a["collision_count"], stats_b["collision_count"]],
           color=["tab:blue", "tab:orange"], alpha=0.8)
    ax.set_xticks([0, 1], [label_a, label_b], rotation=10)
    ax.set_ylabel("count")
    ax.set_title("Collision Count / Rollout")

    # 2) Inter-collision time distribution
    ax = axes[1]
    a = np.asarray(stats_a["inter_collision_times"], dtype=np.float64)
    b = np.asarray(stats_b["inter_collision_times"], dtype=np.float64)
    if len(a) > 0:
        ax.hist(a, bins=bins, alpha=0.5, density=True, label=label_a)
    if len(b) > 0:
        ax.hist(b, bins=bins, alpha=0.5, density=True, label=label_b)
    ax.set_xlabel("time between collision frames (s)")
    ax.set_ylabel("density")
    ax.set_title("Inter-Collision Times")
    if len(a) > 0 or len(b) > 0:
        ax.legend()

    # 3) Collision type ratio
    ax = axes[2]
    cats = ["wall", "pair"]
    xa = np.arange(len(cats))
    width = 0.36
    ra = [stats_a["collision_type_ratio"].get(c, np.nan) for c in cats]
    rb = [stats_b["collision_type_ratio"].get(c, np.nan) for c in cats]
    ax.bar(xa - width/2, ra, width=width, label=label_a, alpha=0.8)
    ax.bar(xa + width/2, rb, width=width, label=label_b, alpha=0.8)
    ax.set_xticks(xa, cats)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of collisions")
    ax.set_title("Collision Type Ratio")
    ax.legend()

    fig.tight_layout()
    return fig, axes

def kinetic_energy_from_rollout(vel, masses):
    """
    Per-frame total kinetic energy from a rollout velocity tensor.

    Args:
        vel: (T, N, 2) velocities
        masses: (N,) particle masses
    Returns:
        ke: (T,) total kinetic energy per frame
    """
    vel = np.asarray(vel, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if vel.ndim != 3 or vel.shape[2] != 2:
        raise ValueError(f"vel must have shape (T,N,2), got {vel.shape}")
    if masses.shape != (vel.shape[1],):
        raise ValueError(f"masses shape {masses.shape} must be ({vel.shape[1]},)")

    speed_sq = np.sum(vel ** 2, axis=2)  # (T, N)
    ke = 0.5 * np.sum(speed_sq * masses[None, :], axis=1)
    return ke

def plot_kinetic_energy(ke, dt, title="Total Kinetic Energy Over Time"):
    ke = np.asarray(ke, dtype=np.float64)
    t = np.arange(len(ke)) * float(dt)
    plt.figure(figsize=(7, 3.5))
    plt.plot(t, ke, lw=1.8)
    plt.xlabel("time (s)")
    plt.ylabel("total kinetic energy")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return ke
