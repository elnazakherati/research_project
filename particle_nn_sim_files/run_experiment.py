import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from particle_nn_sim.simulator import ParticleSim2D
from particle_nn_sim.data import collect_episodes, episodes_to_XY_residual
from particle_nn_sim.train import fit_standardizer, apply_standardizer, StepDataset, train
from particle_nn_sim.models import MLP
from particle_nn_sim.rollout_eval import (
    nn_rollout_residual,
    plot_rollout_error,
    animate_side_by_side,
    animate_single_rollout,
)
from particle_nn_sim.models import ResMLP

# def main():
#     # --- simulator setup ---
#     sim = ParticleSim2D(W=1.0, H=1.0, radii=[0.06, 0.06], masses=[1.0, 1.0], restitution=1.0, seed=1)

#     # --- data ---
#     pos_all, vel_all, coll_all, meta = collect_episodes(sim, E=100, steps=1000, dt=0.01, speed_max=0.7, seed=0)
#     E = pos_all.shape[0]
#     idx = np.arange(E)
#     n_train = int(0.8 * E)
#     train_eps = idx[:n_train]
#     test_eps  = idx[n_train:]

#     Xtr, Ytr, Ctr = episodes_to_XY_residual(pos_all, vel_all, coll_all, meta, train_eps)
#     Xte, Yte, Cte = episodes_to_XY_residual(pos_all, vel_all, coll_all, meta, test_eps)

#     # --- normalize ---
#     x_mean, x_std = fit_standardizer(Xtr)
#     y_mean, y_std = fit_standardizer(Ytr)

#     Xtr_n = apply_standardizer(Xtr, x_mean, x_std)
#     Ytr_n = apply_standardizer(Ytr, y_mean, y_std)
#     Xte_n = apply_standardizer(Xte, x_mean, x_std)
#     Yte_n = apply_standardizer(Yte, y_mean, y_std)

#     # --- loaders ---
#     train_loader = DataLoader(StepDataset(Xtr_n, Ytr_n, Ctr), batch_size=512, shuffle=True, drop_last=True)
#     test_loader  = DataLoader(StepDataset(Xte_n, Yte_n, Cte), batch_size=512, shuffle=False)

#     # --- model ---
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = MLP(in_dim=12, hidden=128, out_dim=8, dropout=0.0)

#     # --- train ---
#     from copy import deepcopy

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # --- Model 1: no collision weighting ---
#     # model_unw = MLP(in_dim=12, hidden=128, out_dim=8, dropout=0.0)
#     # stats_unw, hist_unw = train(
#     #     model_unw, train_loader, test_loader,
#     #     device=device, epochs=50, lr=1e-3,
#     #     collision_weight=1.0,        # no weighting
#     #     weight_decay=1e-6
#     # )
#     # print("FINAL TEST (unweighted):", stats_unw)

#     # # --- Model 2: collision-weighted ---
#     # model_w = MLP(in_dim=12, hidden=128, out_dim=8, dropout=0.0)
#     # stats_w, hist_w = train(
#     #     model_w, train_loader, test_loader,
#     #     device=device, epochs=50, lr=1e-3,
#     #     collision_weight=5.0,        # try 3,5,8
#     #     weight_decay=1e-6
#     # )
#     # print("FINAL TEST (weighted):", stats_w)


#     # --- Model 1: ResMLP, no collision weighting ---
#     model_unw = ResMLP(
#         in_dim=12,
#         hidden=256,
#         out_dim=8,
#         blocks=3,
#         dropout=0.05
#     )

#     stats_unw, hist_unw = train(
#         model_unw, train_loader, test_loader,
#         device=device,
#         epochs=200,
#         lr=1e-3,
#         collision_weight=1.0,
#         weight_decay=1e-6
#     )

#     print("FINAL TEST (ResMLP unweighted):", stats_unw)


#     # --- Model 2: ResMLP, collision-weighted ---
#     model_w = ResMLP(
#         in_dim=12,
#         hidden=256,
#         out_dim=8,
#         blocks=3,
#         dropout=0.05
#     )

#     stats_w, hist_w = train(
#         model_w, train_loader, test_loader,
#         device=device,
#         epochs=50,
#         lr=1e-3,
#         collision_weight=5.0,   # same idea as before
#         weight_decay=1e-6
#     )

#     print("FINAL TEST (ResMLP weighted):", stats_w)



#     # --- Plot comparison ---
#     plot_histories(
#         {"unweighted": hist_unw, "weighted": hist_w},
#         title="MLP comparison (residual targets)",
#         save_path="curves.png"
#     )
#     # stats = train(model, train_loader, test_loader, device, epochs=50, lr=1e-3, collision_weight=5.0)
#     # print("FINAL TEST:", stats)

#     # --- rollout comparison on a random test episode initial state ---
#     e0 = test_eps[0]
#     pos0 = pos_all[e0, 0].astype(np.float32)
#     vel0 = vel_all[e0, 0].astype(np.float32)

#     # True rollout
#     sim_true = ParticleSim2D(W=float(meta["W"]), H=float(meta["H"]),
#                              radii=meta["radii"], masses=meta["masses"],
#                              restitution=float(meta["restitution"]), seed=123)
#     sim_true.reset(pos0, vel0)
#     pos_true, vel_true = sim_true.rollout(dt=float(meta["dt"]), steps=1200)

#     # NN rollout
#     pos_pred, vel_pred = nn_rollout_residual(
#         model, pos0, vel0,
#         radii=np.asarray(meta["radii"], dtype=np.float32),
#         masses=np.asarray(meta["masses"], dtype=np.float32),
#         steps=1200,
#         x_mean=x_mean, x_std=x_std,
#         y_mean=y_mean, y_std=y_std,
#         device=device,
#         dt=float(meta["dt"])
#     )

#     plot_rollout_error(pos_true, pos_pred, dt=float(meta["dt"]))
#     ani = animate_side_by_side(pos_true, pos_pred, meta["radii"], float(meta["W"]), float(meta["H"]), float(meta["dt"]))
#     # In notebooks, display(ani). In scripts, you can save:
#     ani.save("rollout.mp4", fps=50)


def main():
    # --- simulator setup ---
    sim = ParticleSim2D(W=1.0, H=1.0, radii=[0.06, 0.06], masses=[1.0, 1.0], restitution=1.0, seed=1)

    # --- data ---
    pos_all, vel_all, coll_all, meta = collect_episodes(sim, E=100, steps=1000, dt=0.01, speed_max=0.7, seed=0)
    E = pos_all.shape[0]
    idx = np.arange(E)
    n_train = int(0.8 * E)
    train_eps = idx[:n_train]
    test_eps  = idx[n_train:]

    Xtr, Ytr, Ctr = episodes_to_XY_residual(pos_all, vel_all, coll_all, meta, train_eps)
    Xte, Yte, Cte = episodes_to_XY_residual(pos_all, vel_all, coll_all, meta, test_eps)

    # --- normalize ---
    x_mean, x_std = fit_standardizer(Xtr)
    y_mean, y_std = fit_standardizer(Ytr)

    Xtr_n = apply_standardizer(Xtr, x_mean, x_std)
    Ytr_n = apply_standardizer(Ytr, y_mean, y_std)
    Xte_n = apply_standardizer(Xte, x_mean, x_std)
    Yte_n = apply_standardizer(Yte, y_mean, y_std)

    # --- loaders ---
    train_loader = DataLoader(StepDataset(Xtr_n, Ytr_n, Ctr), batch_size=512, shuffle=True, drop_last=True)
    test_loader  = DataLoader(StepDataset(Xte_n, Yte_n, Cte), batch_size=512, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- ONE model ---
    model = ResMLP(in_dim=12, hidden=256, out_dim=8, blocks=3, dropout=0.05)

    # --- train ONCE (200 epochs) ---
    stats, hist = train(
        model, train_loader, test_loader,
        device=device,
        epochs=200,
        lr=1e-3,
        collision_weight=1.0,   # or set 5.0 if you want weighting
        weight_decay=1e-6
    )
    print("FINAL TEST:", stats)

    # --- plot ---
    plot_histories({"model": hist}, title="ResMLP (residual targets)", save_path="curves.png")

    # --- rollout comparison on a random test episode initial state ---
    e0 = test_eps[0]
    pos0 = pos_all[e0, 0].astype(np.float32)
    vel0 = vel_all[e0, 0].astype(np.float32)

    sim_true = ParticleSim2D(W=float(meta["W"]), H=float(meta["H"]),
                             radii=meta["radii"], masses=meta["masses"],
                             restitution=float(meta["restitution"]), seed=123)
    sim_true.reset(pos0, vel0)
    pos_true, vel_true = sim_true.rollout(dt=float(meta["dt"]), steps=1200)

    pos_pred, vel_pred = nn_rollout_residual(
        model, pos0, vel0,   # <-- uses the trained ResMLP now
        radii=np.asarray(meta["radii"], dtype=np.float32),
        masses=np.asarray(meta["masses"], dtype=np.float32),
        steps=1200,
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
        device=device,
        dt=float(meta["dt"])
    )

    plot_rollout_error(pos_true, pos_pred, dt=float(meta["dt"]))
    ani = animate_side_by_side(pos_true, pos_pred, meta["radii"],
                               float(meta["W"]), float(meta["H"]), float(meta["dt"]))
    ani.save("rollout.mp4", fps=50)
    ani_gt = animate_single_rollout(
        pos_true, meta["radii"], float(meta["W"]), float(meta["H"]), float(meta["dt"]),
        title="Ground truth rollout",
    )
    ani_gt.save("rollout_ground_truth.mp4", fps=50)


def plot_histories(histories, title="Training curves", save_path=None):
    plt.figure()
    for label, h in histories.items():
        plt.plot(h["train_loss"], label=f"{label}: train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title + " (train loss)")
    plt.legend()
    if save_path:
        plt.savefig(save_path.replace(".png","_trainloss.png"), dpi=150)
    plt.show()

    plt.figure()
    for label, h in histories.items():
        plt.plot(h["test_mse_all"], label=f"{label}: test_mse_all")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title(title + " (test MSE all)")
    plt.legend()
    if save_path:
        plt.savefig(save_path.replace(".png","_testmse_all.png"), dpi=150)
    plt.show()

    plt.figure()
    for label, h in histories.items():
        plt.plot(h["test_mse_collision"], label=f"{label}: test_collision")
        plt.plot(h["test_mse_noncollision"], label=f"{label}: test_noncollision", linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title(title + " (collision vs non-collision)")
    plt.legend()
    if save_path:
        plt.savefig(save_path.replace(".png","_testmse_col.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
