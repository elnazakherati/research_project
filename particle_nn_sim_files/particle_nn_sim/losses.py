import torch


def _validate_reflection_inputs(
    pred,
    target,
    Lx,
    Ly,
    K,
    reduction,
    debug_checks=False,
):
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("pred and target must be torch.Tensor")
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"pred/target must be rank-2 tensors, got {pred.shape} and {target.shape}")
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shapes must match, got {pred.shape} and {target.shape}")
    if pred.shape[1] < 2:
        raise ValueError(f"pred/target must have D>=2, got D={pred.shape[1]}")
    if float(Lx) <= 0.0 or float(Ly) <= 0.0:
        raise ValueError(f"Lx and Ly must be positive, got Lx={Lx}, Ly={Ly}")
    if int(K) < 0:
        raise ValueError(f"K must be >= 0, got K={K}")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"reduction must be one of none|mean|sum, got {reduction}")
    if debug_checks:
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("pred/target contain non-finite values")


def _build_reflected_copies(target_xy, Lx, Ly, K):
    """Build mirrored copies for each sample.

    target_xy: (B,2)
    returns: copies (B,N,2), where N = 4 * (2K+1)^2
    """
    device = target_xy.device
    dtype = target_xy.dtype
    sx = torch.tensor([1.0, -1.0], device=device, dtype=dtype)
    sy = torch.tensor([1.0, -1.0], device=device, dtype=dtype)
    kk = torch.arange(-int(K), int(K) + 1, device=device, dtype=dtype)
    mm = torch.arange(-int(K), int(K) + 1, device=device, dtype=dtype)
    SX, SY, KK, MM = torch.meshgrid(sx, sy, kk, mm, indexing="ij")
    sx_f = SX.reshape(-1)
    sy_f = SY.reshape(-1)
    kk_f = KK.reshape(-1)
    mm_f = MM.reshape(-1)

    x = target_xy[:, 0:1]  # (B,1)
    y = target_xy[:, 1:2]  # (B,1)
    x_c = sx_f.unsqueeze(0) * x + (2.0 * float(Lx)) * kk_f.unsqueeze(0)  # (B,N)
    y_c = sy_f.unsqueeze(0) * y + (2.0 * float(Ly)) * mm_f.unsqueeze(0)  # (B,N)
    return torch.stack([x_c, y_c], dim=-1)  # (B,N,2)


def _reduce(loss_per_sample, reduction, return_per_sample):
    if return_per_sample or reduction == "none":
        return loss_per_sample
    if reduction == "mean":
        return loss_per_sample.mean()
    return loss_per_sample.sum()


def reflection_aware_loss_2d(
    pred,
    target,
    Lx,
    Ly,
    K=1,
    reduction="mean",
    return_per_sample=False,
    debug_checks=False,
):
    """Hard-min reflection-aware loss on position slice only.

    pred/target are (B,D), D>=2; only [:, :2] participates in this loss.
    """
    _validate_reflection_inputs(pred, target, Lx, Ly, K, reduction, debug_checks=debug_checks)
    pred_xy = pred[:, :2]
    target_xy = target[:, :2]
    copies = _build_reflected_copies(target_xy, Lx=Lx, Ly=Ly, K=K)  # (B,N,2)
    diff = pred_xy.unsqueeze(1) - copies  # (B,N,2)
    d2 = (diff * diff).sum(dim=-1)  # (B,N), squared Euclidean
    per_sample = d2.min(dim=1).values  # (B,)
    return _reduce(per_sample, reduction=reduction, return_per_sample=return_per_sample)


def reflection_aware_loss_2d_softmin(
    pred,
    target,
    Lx,
    Ly,
    K=1,
    tau=0.05,
    reduction="mean",
    return_per_sample=False,
    debug_checks=False,
):
    """Soft-min reflection-aware loss (logsumexp smoothing) on position slice."""
    _validate_reflection_inputs(pred, target, Lx, Ly, K, reduction, debug_checks=debug_checks)
    if float(tau) <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")

    pred_xy = pred[:, :2]
    target_xy = target[:, :2]
    copies = _build_reflected_copies(target_xy, Lx=Lx, Ly=Ly, K=K)  # (B,N,2)
    diff = pred_xy.unsqueeze(1) - copies  # (B,N,2)
    d2 = (diff * diff).sum(dim=-1)  # (B,N)
    t = float(tau)
    per_sample = -t * torch.logsumexp(-d2 / t, dim=1)  # (B,)
    return _reduce(per_sample, reduction=reduction, return_per_sample=return_per_sample)
