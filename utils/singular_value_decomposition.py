"""
SVD-based null-space extraction.

Given a square covariance matrix C = U S V^T, the columns of V associated
with singular values <= threshold form the approximate right null space.
For a symmetric covariance matrix, left and right singular vectors coincide
up to numerical sign/rotation choices.
"""
from typing import Dict, Tuple
import torch


def calculate_null_space_svd(
    covariance: torch.Tensor,
    threshold: float = 1e-2,
    relative_threshold: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Covariance must be square; got {tuple(covariance.shape)}")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    matrix = covariance.detach().to(device="cpu", dtype=torch.float32)
    matrix = 0.5 * (matrix + matrix.T)

    # full_matrices=False is sufficient for a square matrix and avoids
    # unnecessary full-sized U/V allocations.
    _u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    cutoff = threshold * float(singular_values.max()) if relative_threshold else threshold
    null_mask = singular_values <= cutoff
    # V = Vh.T; basis columns span the approximate null space.
    null_basis = vh[null_mask, :].T.contiguous()

    diagnostics = {
        "dimension": int(matrix.shape[0]),
        "threshold": float(threshold),
        "effective_cutoff": float(cutoff),
        "relative_threshold": bool(relative_threshold),
        "nullity": int(null_mask.sum().item()),
        "null_space_percentage": float(100.0 * null_mask.sum().item() / matrix.shape[0]),
        "smallest_singular_value": float(singular_values.min().item()),
        "largest_singular_value": float(singular_values.max().item()),
    }
    return null_basis, diagnostics
