import torch

from particle_nn_sim.losses import reflection_aware_loss_2d, reflection_aware_loss_2d_softmin


def main():
    # Fake batch with D=4 state [x,y,vx,vy] to verify position-only slicing.
    pred = torch.tensor(
        [
            [1.01, 0.50, 0.10, 0.00],  # near right wall
            [0.25, 0.75, -0.10, 0.00],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [0.99, 0.50, 0.10, 0.00],
            [0.20, 0.80, -0.10, 0.00],
        ],
        dtype=torch.float32,
    )
    Lx, Ly = 1.0, 1.0

    print("pred shape:", tuple(pred.shape))
    print("target shape:", tuple(target.shape))

    plain_mse = ((pred[:, :2] - target[:, :2]) ** 2).sum(dim=1).mean()
    hard = reflection_aware_loss_2d(pred, target, Lx=Lx, Ly=Ly, K=1, reduction="mean")
    soft = reflection_aware_loss_2d_softmin(
        pred, target, Lx=Lx, Ly=Ly, K=1, tau=0.05, reduction="mean"
    )
    print("plain position squared error mean:", float(plain_mse))
    print("reflection hard-min loss:", float(hard))
    print("reflection soft-min loss:", float(soft))

    # Explicit near-wall case:
    # Ground truth near right wall at x=0.99 in [0,1].
    # Prediction at x=-0.99 is physically close by mirror, but plain MSE is large.
    pred_wall = torch.tensor([[-0.99, 0.40]], dtype=torch.float32)
    target_wall = torch.tensor([[0.99, 0.40]], dtype=torch.float32)
    plain_wall = ((pred_wall - target_wall) ** 2).sum(dim=1).mean()
    hard_wall = reflection_aware_loss_2d(
        pred_wall, target_wall, Lx=Lx, Ly=Ly, K=1, reduction="mean"
    )
    soft_wall = reflection_aware_loss_2d_softmin(
        pred_wall, target_wall, Lx=Lx, Ly=Ly, K=1, tau=0.05, reduction="mean"
    )
    print("\nNear-wall explicit case:")
    print("plain squared error:", float(plain_wall))
    print("reflection hard-min:", float(hard_wall))
    print("reflection soft-min:", float(soft_wall))

    # Minimal training-loop style snippet:
    # base = reflection_aware_loss_2d(state_pred, state_gt, Lx, Ly, K=1, reduction='none')
    # weighted = torch.where(collision_mask > 0, base * collision_weight, base)
    # loss = weighted.mean()


if __name__ == "__main__":
    main()
