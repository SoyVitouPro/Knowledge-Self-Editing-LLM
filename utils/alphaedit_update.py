"""Projected AlphaEdit-style linear update."""
import gc
import torch


def apply_projected_update(
    linear_module,
    key,
    residual,
    null_basis,
    regularization=1.0,
    previous_keys=None,
):
    """
    Shapes:
      key:         [in_features]
      residual:    [out_features]
      null_basis:  [in_features, nullity]
      update:      [out_features, in_features]

    Solves:
      A = P (k k^T + K_prev K_prev^T) + lambda I
      B = P k residual^T
      DeltaW = (solve(A, B))^T
    """
    if linear_module.weight.device.type == "meta":
        raise RuntimeError("Cannot update a linear layer whose weight is on meta device.")

    in_features = linear_module.in_features
    out_features = linear_module.out_features
    key = key.detach().float().cpu().reshape(-1, 1)
    residual = residual.detach().float().cpu().reshape(-1, 1)
    basis = null_basis.detach().float().cpu()

    if key.shape[0] != in_features or basis.shape[0] != in_features:
        raise ValueError(
            f"Input dimension mismatch: key={key.shape[0]}, basis={basis.shape[0]}, "
            f"layer in_features={in_features}"
        )
    if residual.shape[0] != out_features:
        raise ValueError(
            f"Residual dimension {residual.shape[0]} does not match "
            f"layer out_features={out_features}"
        )

    # Projector onto the preservation null space.
    projector = basis @ basis.T
    covariance = key @ key.T
    if previous_keys:
        prev = torch.stack(previous_keys, dim=1).float()
        covariance.add_(prev @ prev.T)

    A = projector @ covariance
    A.diagonal().add_(regularization)
    B = projector @ key @ residual.T

    update = torch.linalg.solve(A, B)
    weight_update = update.T.to(
        device=linear_module.weight.device,
        dtype=linear_module.weight.dtype,
    )
    with torch.no_grad():
        linear_module.weight.add_(weight_update)

    del projector, covariance, A, B, update, weight_update, basis
    gc.collect()
    return key.reshape(-1)
