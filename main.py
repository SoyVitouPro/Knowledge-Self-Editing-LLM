#!/usr/bin/env python3
"""Configuration-driven experimental AlphaEdit-style GSM8K editing."""
from pathlib import Path
import argparse
import json
import sys
import torch
from datasets import load_dataset

from utils.model_loader import (
    load_model_config, load_model_and_processor, get_input_device,
    get_decoder_layers, get_edit_module,
)
from utils.cache_manager import load_null_bases
from utils.target_optimization import optimize_target_delta
from utils.activation_capture import capture_keys_and_target_activation
from utils.alphaedit_update import apply_projected_update

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "models.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model key from config/models.json")
    parser.add_argument("--num-edits", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_model_config(CONFIG_PATH, args.model)
    layer_indices = config["edit_layers"]
    cache_root = ROOT / config["cache_dir"]

    print("=" * 68)
    print("ALPHAEDIT-STYLE GSM8K EDITING")
    print(f"Model: {config['model_id']} [{config['model_key']}]")
    print(f"Layers: {layer_indices}")
    print("=" * 68)

    # IMPORTANT: main script only loads null-space cache. It never calculates it.
    # Input dimension is checked against the actual down_proj in the update module.
    bases = load_null_bases(cache_root, layer_indices)
    if bases is None:
        sys.exit(1)

    processor, model = load_model_and_processor(config)
    decoder_layers = get_decoder_layers(model, config)
    if config.get("num_layers") and len(decoder_layers) != config["num_layers"]:
        raise RuntimeError(
            f"Configured {config['num_layers']} layers, model exposes {len(decoder_layers)}."
        )
    input_device = get_input_device(model)

    edit_modules = {
        idx: get_edit_module(model, config, idx) for idx in layer_indices
    }
    for idx, module in edit_modules.items():
        if bases[idx].shape[0] != module.in_features:
            raise RuntimeError(
                f"Layer {idx} null-space dimension {bases[idx].shape[0]} "
                f"does not match down_proj.in_features {module.in_features}. "
                "This model needs its own covariance cache."
            )
        print(
            f"Layer {idx}: down_proj {module.in_features}->{module.out_features}; "
            f"null basis {tuple(bases[idx].shape)}"
        )

    print("\nLoading GSM8K train split (test split is never used for editing)...")
    train = load_dataset("openai/gsm8k", "main", split="train")
    requests = train.select(range(min(args.num_edits, len(train))))
    print(f"Editing examples: {len(requests)}")

    previous_keys = {idx: [] for idx in layer_indices}
    manifest = {
        "model_key": config["model_key"],
        "model_id": config["model_id"],
        "dataset": "openai/gsm8k",
        "edit_split": "train",
        "layers": layer_indices,
        "method": "experimental AlphaEdit-style adaptation",
        "notes": [
            "Null spaces are based on mean-pooled Wikipedia activation covariance.",
            "GSM8K question-final-token keys are used instead of subject-relation triplets.",
            "Target optimization omits the paper's full KL-preservation prompt set."
        ],
        "edits": []
    }

    final_layer = layer_indices[-1]
    final_module = edit_modules[final_layer]

    for edit_idx, example in enumerate(requests):
        question = example["question"]
        reference_answer = example["answer"]
        prompt_text = config["prompt_template"].format(question=question)
        print(f"\nEdit {edit_idx + 1}/{len(requests)}")

        prompt_batch, prompt_len, delta = optimize_target_delta(
            model=model,
            processor=processor,
            edit_module=final_module,
            prompt_text=prompt_text,
            reference_answer=reference_answer,
            input_device=input_device,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )

        keys, current_target = capture_keys_and_target_activation(
            model, edit_modules, prompt_batch, prompt_len, final_module
        )
        target_activation = current_target + delta

        for position, layer_idx in enumerate(layer_indices):
            # Re-capture current final-layer output after every preceding update.
            current_keys, current_target = capture_keys_and_target_activation(
                model,
                {layer_idx: edit_modules[layer_idx]},
                prompt_batch,
                prompt_len,
                final_module,
            )
            remaining_layers = len(layer_indices) - position
            residual = (target_activation - current_target) / float(remaining_layers)

            used_key = apply_projected_update(
                linear_module=edit_modules[layer_idx],
                key=current_keys[layer_idx],
                residual=residual,
                null_basis=bases[layer_idx],
                regularization=args.regularization,
                previous_keys=previous_keys[layer_idx],
            )
            previous_keys[layer_idx].append(used_key)
            print(f"  Updated layer {layer_idx}")

        manifest["edits"].append({
            "index": edit_idx,
            "question": question,
            "reference_answer": reference_answer,
        })

    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "results" / f"post-edit-{config['model_key']}-alphaedit-gsm8k"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving edited checkpoint: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    with open(output_dir / "edit_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Editing complete. Original Hugging Face checkpoint was not overwritten.")


if __name__ == "__main__":
    main()
