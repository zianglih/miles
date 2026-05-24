import re


def _get_ignore_rules(quantization_config) -> list[str]:
    ignore_rules = quantization_config.get("ignore", []) or []
    if isinstance(ignore_rules, str):
        ignore_rules = [ignore_rules]
    exclude_rules = quantization_config.get("exclude_modules", []) or []
    if isinstance(exclude_rules, str):
        exclude_rules = [exclude_rules]
    return list(ignore_rules) + [rule for rule in exclude_rules if rule not in ignore_rules]


def fp4_param_gather_enabled(args) -> bool:
    return bool(getattr(args, "fp4_param", False) or getattr(args, "fp4_param_gather", False))


def assert_no_fp4_param_gather(args) -> None:
    if fp4_param_gather_enabled(args):
        raise NotImplementedError(
            "Miles NVFP4 weight update requires BF16 primary parameters and TE-generated FP4 workspaces; --fp4-param-gather is unsupported."
        )


def _is_ignored(name: str, ignore_rules: list[str]) -> bool:
    for rule in ignore_rules:
        if rule.startswith("re:"):
            if re.match(rule[3:], name):
                return True
            continue
        if name == rule or name.startswith(f"{rule}."):
            return True
    return False


def is_nvfp4_quantization_config(quantization_config) -> bool:
    if quantization_config is None:
        return False
    return quantization_config.get("quant_algo") == "NVFP4" or quantization_config.get("quant_method") == "nvfp4"


def should_quantize_megatron_param_nvfp4(args, megatron_name: str, quantization_config) -> bool:
    if not is_nvfp4_quantization_config(quantization_config):
        return False
    if getattr(args, "extra_high_precision_layers_megatron", False):
        for layer_name in getattr(args, "extra_high_precision_layers_megatron", ()):
            if layer_name in megatron_name:
                return False

    decoder_layers_pattern = r"decoder\.layers\.(\d+)\.(.+)"
    match = re.search(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"mtp\.layers\.(\d+)\.(.+)"
        match = re.search(mtp_layer_pattern, megatron_name)
        if not match:
            return False
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # Skip quantization for BF16 tail of main decoder layers.
    if getattr(args, "first_last_layers_bf16", False):
        num_layers = int(args.num_layers)
        num_layers_at_start_in_bf16 = int(getattr(args, "num_layers_at_start_in_bf16", 0))
        num_layers_at_end_in_bf16 = int(getattr(args, "num_layers_at_end_in_bf16", 0))
        head_end_idx = num_layers_at_start_in_bf16
        tail_start_idx = num_layers - num_layers_at_end_in_bf16
        if int(layer_idx) < head_end_idx or int(layer_idx) >= tail_start_idx:
            return False

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, _ = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            return True

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            return True

    # for other parameters, we just return the original converted_named_params
    return False


def quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config is not None
    assert is_nvfp4_quantization_config(quantization_config)
    assert_no_fp4_param_gather(args)

    if should_quantize_megatron_param_nvfp4(args, megatron_name, quantization_config):
        quantized_names = [
            name
            for name, _param in converted_named_params
            if should_quantize_hf_weight_nvfp4(name, quantization_config)
        ]
        if quantized_names:
            raise RuntimeError(
                "NVFP4 runtime weight update must use TE-generated FP4 workspaces; "
                f"unexpected BF16 fallback conversion for {megatron_name}: {quantized_names}."
            )

    return converted_named_params


def should_quantize_hf_weight_nvfp4(name: str, quantization_config) -> bool:
    if not is_nvfp4_quantization_config(quantization_config):
        return False
    if not name.endswith(".weight"):
        return False
    return not _is_ignored(name, _get_ignore_rules(quantization_config))
