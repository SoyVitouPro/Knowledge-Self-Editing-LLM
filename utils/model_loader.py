"""Configuration-driven model and processor loading."""
import json
from pathlib import Path
import torch
import transformers


def load_model_config(config_path: Path, model_key=None):
    with open(config_path, "r", encoding="utf-8") as f:
        all_config = json.load(f)
    key = model_key or all_config["default_model"]
    if key not in all_config["models"]:
        raise KeyError(
            f"Unknown model key '{key}'. Available: {', '.join(all_config['models'])}"
        )
    config = dict(all_config["models"][key])
    config["model_key"] = key
    return config


def load_model_and_processor(config, device_map=None, dtype="auto"):
    from transformers import AutoProcessor

    model_class_name = config["model_class"]
    model_class = getattr(transformers, model_class_name, None)
    if model_class is None:
        raise RuntimeError(
            f"Transformers does not provide {model_class_name}. "
            "Upgrade Transformers to a version supporting this model architecture."
        )

    model_id = config["model_id"]
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    if device_map is None:
        device_map = config.get("device_map")
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": config.get("trust_remote_code", False),
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = model_class.from_pretrained(
        model_id,
        **model_kwargs,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


def resolve_module_path(root, dotted_path):
    """Resolve dotted paths with numeric components such as layers.9."""
    obj = root
    for part in dotted_path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def get_decoder_layers(model, config):
    return resolve_module_path(model, config["decoder_layers_path"])


def get_edit_module(model, config, layer_index):
    layer = get_decoder_layers(model, config)[layer_index]
    return resolve_module_path(layer, config["mlp_down_proj_path"])


def get_input_device(model):
    try:
        device = model.get_input_embeddings().weight.device
        if device.type != "meta":
            return device
    except Exception:
        pass
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not identify a non-meta model input device.")
