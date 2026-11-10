#!/usr/bin/env python3
"""Create SVD null-space files from existing cached covariance matrices only."""
import argparse
from pathlib import Path
from utils.model_loader import load_model_config
from utils.cache_manager import create_layer_null_cache

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "models.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model key from config/models.json")
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--relative-threshold", action="store_true")
    parser.add_argument("--num-pages", type=int, default=100_000)
    args = parser.parse_args()

    config = load_model_config(CONFIG_PATH, args.model)
    cache_root = ROOT / config["cache_dir"]
    layers = config["edit_layers"]

    print("=" * 68)
    print("SVD NULL-SPACE CACHE PREPARATION")
    print(f"Model key: {config['model_key']}")
    print(f"Cache: {cache_root}")
    print("Only existing covariance files are used; Wikipedia is not reloaded.")
    print("=" * 68)

    for layer in layers:
        create_layer_null_cache(
            cache_root,
            layer,
            threshold=args.threshold,
            num_pages=args.num_pages,
            relative_threshold=args.relative_threshold,
        )
    print("\nNull-space cache preparation finished.")


if __name__ == "__main__":
    main()
