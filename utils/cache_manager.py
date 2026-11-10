"""Load existing covariance files and save SVD null-space cache."""
from pathlib import Path
import json
import torch
from .singular_value_decomposition import calculate_null_space_svd


def load_existing_covariance(layer_dir: Path, num_pages: int = 100_000) -> torch.Tensor:
    mean_path = layer_dir / "covariance_mean.pt"
    sum_path = layer_dir / "covariance.pt"

    if mean_path.exists():
        covariance = torch.load(mean_path, map_location="cpu", weights_only=True)
        print(f"Using normalized covariance: {mean_path}")
    elif sum_path.exists():
        covariance = torch.load(sum_path, map_location="cpu", weights_only=True)
        covariance = covariance.float() / float(num_pages)
        torch.save(covariance, mean_path)
        print(f"Normalized existing covariance and saved: {mean_path}")
    else:
        raise FileNotFoundError(
            f"No covariance file in {layer_dir}. Expected covariance_mean.pt "
            "or covariance.pt. Wikipedia activations are not computed by this tool."
        )

    if not isinstance(covariance, torch.Tensor):
        raise TypeError(f"Covariance file in {layer_dir} did not contain a Tensor.")
    return covariance.float().cpu()


def create_layer_null_cache(
    cache_root: Path,
    layer_index: int,
    threshold: float = 1e-2,
    num_pages: int = 100_000,
    relative_threshold: bool = False,
):
    layer_dir = cache_root / f"layer_{layer_index}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer cache directory does not exist: {layer_dir}")

    covariance = load_existing_covariance(layer_dir, num_pages)
    basis, diagnostics = calculate_null_space_svd(
        covariance, threshold=threshold, relative_threshold=relative_threshold
    )
    basis_path = layer_dir / "null_basis_mean.pt"
    torch.save(basis, basis_path)

    diagnostics.update({
        "layer": int(layer_index),
        "covariance_source": "existing cached covariance",
        "method": "SVD",
        "note": "Mean-pooled Wikipedia activation approximation; not exact AlphaEdit paper statistics."
    })
    with open(layer_dir / "null_space_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(
        f"Layer {layer_index}: basis={tuple(basis.shape)}, "
        f"nullity={diagnostics['nullity']}, "
        f"null-space={diagnostics['null_space_percentage']:.2f}%"
    )
    print(f"Saved: {basis_path}")
    return basis, diagnostics


def load_null_bases(cache_root: Path, layer_indices, expected_input_dim=None):
    bases = {}
    missing = []
    for layer_index in layer_indices:
        path = cache_root / f"layer_{layer_index}" / "null_basis_mean.pt"
        if not path.is_file():
            missing.append(path)
            continue
        basis = torch.load(path, map_location="cpu", weights_only=True).float()
        if basis.ndim != 2:
            raise ValueError(f"Null-space basis must be rank 2: {path}")
        if expected_input_dim is not None and basis.shape[0] != expected_input_dim:
            raise ValueError(
                f"Basis dimension mismatch for layer {layer_index}: "
                f"{basis.shape[0]} != configured {expected_input_dim}"
            )
        bases[layer_index] = basis.contiguous()

    if missing:
        print("\nERROR: Null space has not yet been calculated for these layers:")
        for path in missing:
            print(f"  {path}")
        print("\nRun the cache preparation first:")
        print("  python calculate_null_space_cache.py --model gemma-4-E2B")
        print("Then run the editing script again.")
        return None
    return bases
