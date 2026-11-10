"""Optimize a target down-projection output displacement for one GSM8K prompt."""
import torch
import torch.nn as nn


def encode_prompt(processor, prompt_text, device):
    # Deliberately avoid apply_chat_template: base Gemma checkpoints may not
    # ship a chat template. This is plain text and should match evaluation.
    encoded = processor(
        text=prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def optimize_target_delta(
    model,
    processor,
    edit_module,
    prompt_text,
    reference_answer,
    input_device,
    steps=40,
    learning_rate=0.05,
    max_target_tokens=384,
):
    prompt_batch = encode_prompt(processor, prompt_text, input_device)
    prompt_ids = prompt_batch["input_ids"]
    prompt_length = prompt_ids.shape[1]

    target = processor(
        text=reference_answer,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(input_device)[:, :max_target_tokens]
    if target.shape[1] == 0:
        raise ValueError("The reference answer produced no target tokens.")

    model_inputs = {"input_ids": torch.cat([prompt_ids, target], dim=1)}
    if "attention_mask" in prompt_batch:
        model_inputs["attention_mask"] = torch.cat(
            [prompt_batch["attention_mask"], torch.ones_like(target)], dim=1
        )

    # down_proj output dimension is out_features (weight.shape[0]).
    module_device = edit_module.weight.device
    if module_device.type == "meta":
        raise RuntimeError(
            "The selected edit module is on the meta device due to model offloading. "
            "Direct in-place editing requires this module's weights to be materialized. "
            "Try a device_map that keeps the selected decoder layers in GPU/CPU memory."
        )
    delta = nn.Parameter(torch.zeros(
        edit_module.out_features, device=module_device, dtype=torch.float32
    ))
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    def inject_delta(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        changed = tensor.clone()
        changed[:, prompt_length - 1, :] = (
            changed[:, prompt_length - 1, :].float() + delta
        ).to(changed.dtype)
        if isinstance(output, tuple):
            return (changed,) + output[1:]
        return changed

    handle = edit_module.register_forward_hook(inject_delta)
    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**model_inputs, use_cache=False)
            logits = outputs.logits[
                :, prompt_length - 1:prompt_length - 1 + target.shape[1], :
            ]
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                target.reshape(-1),
            ) + 1e-4 * delta.square().sum()
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % 10 == 0:
                print(f"      target optimization {step + 1}/{steps}: {loss.item():.4f}")
    finally:
        handle.remove()

    return prompt_batch, prompt_length, delta.detach().cpu().float()
