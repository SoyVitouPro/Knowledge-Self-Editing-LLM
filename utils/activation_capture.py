"""Capture down-projection input keys and one layer's output activation."""
import torch


def capture_keys_and_target_activation(model, edit_modules, input_batch,
                                       prompt_length, target_module):
    keys = {}
    target_activation = {}
    handles = []

    for layer_index, module in edit_modules.items():
        def make_pre_hook(index):
            def hook(_module, inputs):
                keys[index] = inputs[0][0, prompt_length - 1].detach().float().cpu()
            return hook
        handles.append(module.register_forward_pre_hook(make_pre_hook(layer_index)))

    def output_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        target_activation["value"] = tensor[0, prompt_length - 1].detach().float().cpu()

    handles.append(target_module.register_forward_hook(output_hook))
    try:
        with torch.no_grad():
            model(**input_batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if len(keys) != len(edit_modules) or "value" not in target_activation:
        raise RuntimeError("Could not capture all edit keys and target activation.")
    return keys, target_activation["value"]
