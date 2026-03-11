import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path


@torch.no_grad()
def nn_rollout_residual_1p(
    model,
    pos0,
    vel0,
    radius,
    mass,
    steps,
    x_mean,
    x_std,
    y_mean,
    y_std,
    device,
    dt,
):
    model.eval()
    steps = int(steps)
    dt = float(dt)

    pos_pred = np.zeros((steps + 1, 1, 2), dtype=np.float32)
    vel_pred = np.zeros((steps + 1, 1, 2), dtype=np.float32)
    pos_pred[0] = pos0.astype(np.float32)
    vel_pred[0] = vel0.astype(np.float32)

    pos_t = pos_pred[0, 0].copy()
    vel_t = vel_pred[0, 0].copy()

    for t in range(steps):
        x_raw = np.array(
            [pos_t[0], pos_t[1], vel_t[0], vel_t[1], float(radius), float(mass)],
            dtype=np.float32,
        )
        x_n = ((x_raw[None, :] - x_mean) / x_std).astype(np.float32)
        x_torch = torch.from_numpy(x_n).to(device)

        resid_n = model(x_torch).cpu().numpy()[0]
        resid = (resid_n * y_std[0]) + y_mean[0]

        pos_free = pos_t + vel_t * dt
        vel_free = vel_t
        y_free = np.array([pos_free[0], pos_free[1], vel_free[0], vel_free[1]], dtype=np.float32)
        y_next = y_free + resid.astype(np.float32)

        pos_t = y_next[0:2]
        vel_t = y_next[2:4]

        pos_pred[t + 1, 0] = pos_t
        vel_pred[t + 1, 0] = vel_t

        if not (np.isfinite(pos_t).all() and np.isfinite(vel_t).all()):
            return pos_pred[: t + 2], vel_pred[: t + 2]

    return pos_pred, vel_pred


def animate_single_rollout_1p(pos, radius, W, H, dt, interval=20, title="Ground truth rollout"):
    pos = np.asarray(pos, dtype=np.float32)
    radius = float(radius)
    dt = float(dt)
    display_radius = max(radius, 0.015 * min(float(W), float(H)))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], lw=2)

    (trace_line,) = ax.plot([], [], color="tab:blue", lw=1.8, alpha=0.8)
    c = plt.Circle(pos[0, 0], display_radius, color="tab:blue", fill=True, alpha=0.9)
    ax.add_patch(c)
    step_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def animate(frame):
        trace_line.set_data(pos[: frame + 1, 0, 0], pos[: frame + 1, 0, 1])
        c.center = pos[frame, 0]
        step_text.set_text(f"t={frame * dt:.2f}s | step {frame}/{len(pos)-1}")
        return [trace_line, c, step_text]

    ani = animation.FuncAnimation(fig, animate, frames=len(pos), interval=interval, blit=True)
    plt.close(fig)
    return ani


def animate_side_by_side_1p(pos_true, pos_pred, radius, W, H, dt, interval=20):
    pos_true = np.asarray(pos_true, dtype=np.float32)
    pos_pred = np.asarray(pos_pred, dtype=np.float32)
    radius = float(radius)
    dt = float(dt)
    display_radius = max(radius, 0.015 * min(float(W), float(H)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, title in [(ax1, "TRUE simulator"), (ax2, "NN rollout")]:
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], lw=2)

    (trace_true,) = ax1.plot([], [], color="tab:green", lw=1.8, alpha=0.8)
    (trace_pred,) = ax2.plot([], [], color="tab:orange", lw=1.8, alpha=0.8)
    c_true = plt.Circle(pos_true[0, 0], display_radius, fill=True, alpha=0.9, color="tab:green")
    c_pred = plt.Circle(pos_pred[0, 0], display_radius, fill=True, alpha=0.9, color="tab:orange")
    ax1.add_patch(c_true)
    ax2.add_patch(c_pred)
    step_text_l = ax1.text(
        0.02,
        0.98,
        "",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    step_text_r = ax2.text(
        0.02,
        0.98,
        "",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    n_frames = min(len(pos_true), len(pos_pred))

    def animate(frame):
        trace_true.set_data(pos_true[: frame + 1, 0, 0], pos_true[: frame + 1, 0, 1])
        trace_pred.set_data(pos_pred[: frame + 1, 0, 0], pos_pred[: frame + 1, 0, 1])
        c_true.center = pos_true[frame, 0]
        c_pred.center = pos_pred[frame, 0]
        txt = f"t={frame * dt:.2f}s | step {frame}/{n_frames-1}"
        step_text_l.set_text(txt)
        step_text_r.set_text(txt)
        return [trace_true, trace_pred, c_true, c_pred, step_text_l, step_text_r]

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interval, blit=True)
    plt.close(fig)
    return ani


def animate_overlay_gt_perturbed_1p(
    pos_ref,
    pos_pert,
    radius,
    W,
    H,
    dt,
    interval=20,
    title="Ground Truth vs Perturbed Ground Truth",
):
    pos_ref = np.asarray(pos_ref, dtype=np.float32)
    pos_pert = np.asarray(pos_pert, dtype=np.float32)
    radius = float(radius)
    dt = float(dt)
    display_radius = max(radius, 0.015 * min(float(W), float(H)))

    n_frames = min(len(pos_ref), len(pos_pert))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], lw=2)

    (trace_ref,) = ax.plot([], [], color="tab:green", lw=2.0, alpha=0.9, label="GT")
    (trace_pert,) = ax.plot([], [], color="tab:orange", lw=2.0, alpha=0.35, label="GT (perturbed)")
    c_ref = plt.Circle(pos_ref[0, 0], display_radius, color="tab:green", fill=True, alpha=0.9)
    c_pert = plt.Circle(pos_pert[0, 0], display_radius, color="tab:orange", fill=True, alpha=0.35)
    ax.add_patch(c_ref)
    ax.add_patch(c_pert)
    ax.legend(loc="upper right")

    step_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def animate(frame):
        trace_ref.set_data(pos_ref[: frame + 1, 0, 0], pos_ref[: frame + 1, 0, 1])
        trace_pert.set_data(pos_pert[: frame + 1, 0, 0], pos_pert[: frame + 1, 0, 1])
        c_ref.center = pos_ref[frame, 0]
        c_pert.center = pos_pert[frame, 0]
        step_text.set_text(f"t={frame * dt:.2f}s | step {frame}/{n_frames-1}")
        return [trace_ref, trace_pert, c_ref, c_pert, step_text]

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interval, blit=True)
    plt.close(fig)
    return ani


def save_animation_mp4(anim, out_path, fps=50):
    out_path = str(out_path)
    suffix = Path(out_path).suffix.lower()
    fps = int(fps)

    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "ffmpeg is required to save MP4 but was not found in PATH. "
                "Install ffmpeg (e.g., `conda install -c conda-forge ffmpeg`) "
                "or save as .gif instead."
            )
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(out_path, writer=writer)
        return

    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        return

    raise ValueError(f"Unsupported animation extension: {suffix}. Use .mp4 or .gif.")
