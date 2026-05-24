import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import (
    _extract_te_nvfp4_weight,
    _is_te_nvfp4_tensor,
    _nvfp4_4over6_enabled,
    assert_no_fp4_param_gather,
    is_nvfp4_quantization_config,
    should_quantize_hf_weight_nvfp4,
    should_quantize_megatron_param_nvfp4,
)
from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


def nvfp4_te_workspace_update_enabled(quantization_config) -> bool:
    return is_nvfp4_quantization_config(quantization_config)


def assert_supported_nvfp4_te_workspace_update(args) -> None:
    assert_no_fp4_param_gather(args)
    if get_parallel_state().etp.size != 1:
        raise NotImplementedError(
            "NVFP4 TE workspace weight update currently supports expert tensor parallel size 1 only."
        )


def _weight_scale_names(weight_name: str) -> tuple[str, str, str]:
    return (
        weight_name.replace(".weight", ".weight_scale"),
        weight_name.replace(".weight", ".weight_scale_2"),
        weight_name.replace(".weight", ".input_scale"),
    )


@dataclass(frozen=True)
class _WorkspaceRef:
    module: torch.nn.Module
    index: int
    cache_name: str

    def refresh(self) -> Any:
        weights = self.module._get_weight_tensors()
        quantizers = self.module._get_weight_quantizers()
        weight = weights[self.index]
        quantizer = quantizers[self.index]
        if quantizer is None:
            raise RuntimeError("TE module does not expose an NVFP4 weight quantizer for weight update.")

        quantizer.set_usage(rowwise=True, columnwise=False)

        from transformer_engine.pytorch.module.base import quantize_weight

        workspaces = getattr(self.module, "_fp8_workspaces", None)
        if workspaces is None:
            self.module._fp8_workspaces = {}
            workspaces = self.module._fp8_workspaces

        workspace = workspaces.get(self.cache_name)
        quantized_weight, new_workspace = quantize_weight(
            tensor=weight,
            quantizer=quantizer,
            workspace=workspace,
            update_workspace=True,
            cache=True,
        )
        workspaces[self.cache_name] = new_workspace if new_workspace is not None else quantized_weight
        return quantized_weight


def _cache_names_for_module(module: torch.nn.Module, num_weights: int) -> list[str]:
    if num_weights == 1:
        return ["weight"]
    return [f"weight{i}" for i in range(num_weights)]


class TENvfp4WorkspaceWeightUpdate:
    def __init__(self, args, model_name: str, model: Sequence[torch.nn.Module], quantization_config) -> None:
        self.args = args
        self.model_name = model_name
        self.model = model
        self.quantization_config = quantization_config
        self._weight_refs: dict[int, _WorkspaceRef] = {}
        self._build_weight_refs(model)
        logger.info("Found %d TE weight workspace references for NVFP4 updates.", len(self._weight_refs))

    def _build_weight_refs(self, model: Sequence[torch.nn.Module]) -> None:
        for model_chunk in model:
            for module in model_chunk.modules():
                if not hasattr(module, "_get_weight_tensors") or not hasattr(module, "_get_weight_quantizers"):
                    continue
                try:
                    weights = module._get_weight_tensors()
                    quantizers = module._get_weight_quantizers()
                except Exception:
                    continue
                if len(weights) != len(quantizers):
                    continue
                for weight, cache_name, index in zip(
                    weights, _cache_names_for_module(module, len(weights)), range(len(weights)), strict=True
                ):
                    if quantizers[index] is not None:
                        self._weight_refs[id(weight)] = _WorkspaceRef(
                            module=module, index=index, cache_name=cache_name
                        )

    def convert(self, megatron_name: str, param: torch.nn.Parameter) -> list[tuple[str, torch.Tensor]] | None:
        if not should_quantize_megatron_param_nvfp4(self.args, megatron_name, self.quantization_config):
            return None
        if not self._all_hf_weights_quantized(megatron_name, param.data):
            return None

        ref = self._weight_refs.get(id(param))
        if ref is None:
            self._build_weight_refs(self.model)
            ref = self._weight_refs.get(id(param))
        if ref is None:
            raise RuntimeError(
                f"NVFP4 weight update could not find a TE weight workspace owner for {megatron_name}. "
                "This path requires BF16 primary parameters inside TE modules."
            )

        workspace = ref.refresh()
        return te_nvfp4_workspace_to_hf(
            self.args,
            self.model_name,
            megatron_name,
            workspace,
            self.quantization_config,
        )

    def _all_hf_weights_quantized(self, megatron_name: str, param: torch.Tensor) -> bool:
        hf_weights = _convert_to_hf_core(self.args, self.model_name, megatron_name, param)
        return all(
            should_quantize_hf_weight_nvfp4(weight_name, self.quantization_config) for weight_name, _ in hf_weights
        )


def te_nvfp4_workspace_to_hf(
    args,
    model_name: str,
    megatron_name: str,
    workspace,
    quantization_config,
) -> list[tuple[str, torch.Tensor]]:
    if not _is_te_nvfp4_tensor(workspace):
        raise RuntimeError(
            f"NVFP4 weight update expected TE to emit an NVFP4 weight workspace for {megatron_name}, "
            f"but got {type(workspace).__name__}."
        )

    qweight, block_scale, global_scale = _extract_te_nvfp4_weight(
        workspace,
        use_4over6=_nvfp4_4over6_enabled(),
    )
    hf_weights = _convert_to_hf_core(args, model_name, megatron_name, qweight)
    hf_block_scales = _convert_to_hf_core(args, model_name, megatron_name, block_scale)

    if len(hf_weights) != len(hf_block_scales):
        raise RuntimeError(
            f"TE NVFP4 workspace conversion produced mismatched weight and scale counts for {megatron_name}."
        )

    converted: list[tuple[str, torch.Tensor]] = []
    for (weight_name, weight), (scale_name, block_scale) in zip(hf_weights, hf_block_scales, strict=True):
        if scale_name != weight_name:
            raise RuntimeError(
                f"TE NVFP4 workspace conversion produced inconsistent HF names: {weight_name} vs {scale_name}."
            )
        if not should_quantize_hf_weight_nvfp4(weight_name, quantization_config):
            raise RuntimeError(
                f"TE NVFP4 workspace update reached an ignored HF weight ({weight_name}). "
                "Ignored weights must use the BF16 fallback conversion path."
            )

        block_scale_name, global_scale_name, input_scale_name = _weight_scale_names(weight_name)
        weight_scale_2 = global_scale.contiguous()
        converted.extend(
            [
                (weight_name, weight.contiguous()),
                (block_scale_name, block_scale.contiguous()),
                (global_scale_name, weight_scale_2),
                (input_scale_name, torch.ones_like(weight_scale_2, dtype=torch.float32)),
            ]
        )

    return converted


def named_tensors_nbytes(named_tensors: list[tuple[str, torch.Tensor]]) -> int:
    return sum(tensor.numel() * tensor.element_size() for _name, tensor in named_tensors)


def all_gather_nvfp4_named_tensors(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    names = [name for name, _ in named_tensors]
    ep_size = get_parallel_state().ep.size
    all_names: list[list[str] | None] = [None] * ep_size
    dist.all_gather_object(all_names, names, group=get_parallel_state().ep.group)

    for ep_names in all_names:
        assert ep_names is not None
        assert len(named_tensors) == len(ep_names), f"mismatch names length: {len(named_tensors)} != {len(ep_names)}"

    all_gathered: list[list[tuple[str, torch.Tensor]]] = [[] for _ in range(ep_size)]
    handles = []
    for index, (_name, tensor) in enumerate(named_tensors):
        tensors = [torch.empty_like(tensor, device=torch.cuda.current_device()) for _ in range(ep_size)]
        handle = dist.all_gather(tensors, tensor, group=get_parallel_state().ep.group, async_op=True)
        handles.append(handle)
        for ep_rank, ep_names in enumerate(all_names):
            all_gathered[ep_rank].append((ep_names[index], tensors[ep_rank]))
    for handle in handles:
        handle.wait()

    return sum(all_gathered, [])
