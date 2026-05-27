import inspect
import logging
import math
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from ray.actor import ActorHandle

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix
from miles.backends.megatron_utils.quantized_weight_transfer import QuantizedWeightTransfer, WeightTransferValue
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import ParamInfo

logger = logging.getLogger(__name__)

try:
    from megatron.core.fp8_utils import dequantize_fp8_tensor, is_float8tensor
except ImportError:
    dequantize_fp8_tensor = None

    def is_float8tensor(_tensor: torch.Tensor) -> bool:
        return False


def _dequantize_for_export(name: str, param: torch.Tensor) -> torch.Tensor:
    """Return a dense tensor suitable for Miles weight export."""
    tensor = param.data if hasattr(param, "data") else param
    if not (is_float8tensor(param) or is_float8tensor(tensor)):
        return tensor
    if dequantize_fp8_tensor is None:
        raise RuntimeError(f"Failed to dequantize fp8 parameter before export: {name}")
    fp8_tensor = param if is_float8tensor(param) else tensor
    try:
        return dequantize_fp8_tensor(fp8_tensor)
    except Exception as e:
        raise RuntimeError(f"Failed to dequantize fp8 parameter before export: {name}") from e


def _use_direct_mxfp8_transfer(args: Namespace, name: str, quantization_config: dict | None) -> bool:
    if not getattr(args, "fp8_param_gather", False):
        return False
    if not quantization_config or quantization_config.get("quant_method") != "mxfp8":
        return False
    if getattr(args, "extra_high_precision_layers_megatron", False):
        for layer_name in getattr(args, "extra_high_precision_layers_megatron", ()):
            if layer_name in name:
                return False

    if getattr(args, "first_last_layers_bf16", False):
        match = re.search(r"(?:decoder|mtp)\.layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            num_layers = int(args.num_layers)
            head_layers = int(getattr(args, "num_layers_at_start_in_bf16", 0))
            tail_start = num_layers - int(getattr(args, "num_layers_at_end_in_bf16", 0))
            if layer_idx < head_layers or layer_idx >= tail_start:
                return False

    return True


def _extract_mxfp8_transfer(
    args: Namespace, name: str, param: torch.nn.Parameter, quantization_config: dict | None
) -> QuantizedWeightTransfer | None:
    if not _use_direct_mxfp8_transfer(args, name, quantization_config):
        return None

    tensor = param if is_float8tensor(param) else getattr(param, "data", param)
    if not is_float8tensor(tensor):
        return None

    rowwise_data = getattr(tensor, "_rowwise_data", None)
    rowwise_scale_inv = getattr(tensor, "_rowwise_scale_inv", None)
    if rowwise_data is None or rowwise_scale_inv is None:
        return None
    if rowwise_data.dtype != torch.uint8:
        return None
    if not hasattr(torch, "float8_e4m3fn"):
        return None

    weight_shape = tuple(tensor.shape)
    if weight_shape[-1] % 32 != 0:
        raise ValueError(f"Last dim {weight_shape[-1]} must be divisible by 32 for MXFP8 transfer: {name}")

    scale_rows = math.prod(weight_shape[:-1])
    scale_cols = weight_shape[-1] // 32
    scale = rowwise_scale_inv[:scale_rows, :scale_cols].contiguous()
    scale = scale.view(*weight_shape[:-1], scale_cols)
    return QuantizedWeightTransfer(
        format="mxfp8",
        weight=rowwise_data.contiguous(),
        weight_dtype=torch.float8_e4m3fn,
        aux_tensors={"weight_scale_inv": scale},
    )


def _gather_with_stride(
    param_partitions: list[torch.Tensor], partition_dim: int, partition_stride: int
) -> torch.Tensor:
    """Gather partitions respecting partition_stride (strided/interleaved TP sharding)."""
    if partition_stride == 1:
        return torch.cat(param_partitions, dim=partition_dim)
    # Interleaved (strided) partitioning, e.g. linear_fc1.weight under GLU/SwiGLU
    chunks_per_rank = [p.chunk(partition_stride, dim=partition_dim) for p in param_partitions]
    interleaved = [chunks_per_rank[r][s] for s in range(partition_stride) for r in range(len(param_partitions))]
    return torch.cat(interleaved, dim=partition_dim)


def _gather_transfer_with_stride(
    partitions: list[WeightTransferValue], partition_dim: int, partition_stride: int
) -> WeightTransferValue:
    first = partitions[0]
    if isinstance(first, QuantizedWeightTransfer):
        weight = _gather_with_stride(
            [partition.weight for partition in partitions],
            partition_dim,
            partition_stride,
        )
        aux_tensors = {}
        for aux_name in first.aux_tensors:
            aux_tensors[aux_name] = _gather_with_stride(
                [partition.aux_tensors[aux_name] for partition in partitions],
                partition_dim,
                partition_stride,
            )
        return first.with_tensors(weight=weight, aux_tensors=aux_tensors)

    return _gather_with_stride(partitions, partition_dim, partition_stride)


def _check_and_fix_partition(args: Namespace, name: str, partition_stride: int, partition_dim: int) -> tuple[int, int]:
    """Validate partition_stride values for known parameter patterns.

    After Megatron-LM PR #2708, linear_fc1 correctly reports partition_stride=2
    (GLU/SwiGLU interleaved [gate, up]), so assert partition_stride==2 is removed.
    But TEGroupedLinear still does not set partition_stride/partition_dim correctly for grouped moe gemm
    """
    if "linear_fc1.weight" in name and args.swiglu:
        partition_stride = 2
    elif "linear_fc2.weight" in name:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
        if partition_dim == 0:
            partition_dim = 1
    else:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
    return partition_stride, partition_dim


def all_gather_param(
    args: Namespace,
    name: str,
    param: torch.nn.Parameter,
    quantization_config: dict | None = None,
) -> WeightTransferValue:
    """
    All-gather TP-sharded param to full tensor. expert_bias→param, non-TP/duplicated→param.data.
    Uses expert-TP for ".experts.", else regular-TP. Handles strided partitioning via partition_stride.
    """
    if "expert_bias" in name:
        return param

    export_param = _extract_mxfp8_transfer(args, name, param, quantization_config)
    if export_param is None:
        export_param = _dequantize_for_export(name, param)
    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return export_param

    if ".experts." in name:
        tp_size = get_parallel_state().etp.size
        tp_group = get_parallel_state().etp.group
    else:
        tp_size = get_parallel_state().tp.size
        tp_group = get_parallel_state().tp.group

    partition_dim = param.partition_dim
    partition_stride = param.partition_stride

    partition_stride, partition_dim = _check_and_fix_partition(args, name, partition_stride, partition_dim)
    if isinstance(export_param, QuantizedWeightTransfer):
        param_partitions = [
            export_param.with_tensors(
                weight=torch.empty_like(export_param.weight),
                aux_tensors={
                    aux_name: torch.empty_like(aux_tensor) for aux_name, aux_tensor in export_param.aux_tensors.items()
                },
            )
            for _ in range(tp_size)
        ]
        dist.all_gather([partition.weight for partition in param_partitions], export_param.weight, group=tp_group)
        for aux_name, aux_tensor in export_param.aux_tensors.items():
            dist.all_gather(
                [partition.aux_tensors[aux_name] for partition in param_partitions],
                aux_tensor,
                group=tp_group,
            )
    else:
        param_partitions = [torch.empty_like(export_param) for _ in range(tp_size)]
        dist.all_gather(param_partitions, export_param, group=tp_group)
    param = _gather_transfer_with_stride(param_partitions, partition_dim, partition_stride)
    return param


def all_gather_params_async(
    args: Namespace,
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    """
    Parallel TP all-gather for multiple params. Loop 1: for each TP param, allocate buffers +
    dist.all_gather(async_op=True) on expert-TP/regular-TP group (skip expert_bias/non-TP/duplicated).
    Loop 2: wait all NCCL handles (enables overlap). Loop 3: concat partitions + apply GLU rechunk/MoE dim fix.
    """
    # Phase 1: Start all async all_gather operations
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        export_param = _dequantize_for_export(info.name, param)
        # Prepare async all_gather
        if "expert_bias" in info.name:
            gather_tasks.append((info, export_param, None, None, None, None))
            handles.append(None)
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            gather_tasks.append((info, export_param, None, None, None, None))
            handles.append(None)
        else:
            # Start async all_gather
            if ".experts." in info.name:
                tp_size = get_parallel_state().etp.size
                tp_group = get_parallel_state().etp.group
            else:
                tp_size = get_parallel_state().tp.size
                tp_group = get_parallel_state().tp.group

            param_partitions = [torch.empty_like(export_param) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, export_param, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim, param.partition_stride))
            handles.append(handle)

    # Phase 2: Wait for ALL async operations to complete at once
    # This ensures maximum parallelism by not blocking on individual operations
    for handle in handles:
        if handle is not None:
            handle.wait()

    # Phase 3: Process all results after all communications are done
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim, partition_stride in gather_tasks:
        if handle is None:
            # No all_gather needed
            param = direct_param
        else:
            partition_stride, partition_dim = _check_and_fix_partition(
                args, info.name, partition_stride, partition_dim
            )
            param = _gather_with_stride(param_partitions, partition_dim, partition_stride)

        gathered_params.append(param)

    return gathered_params


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def _maybe_get_cpu_backup(x: torch.Tensor):
    from torch_memory_saver import torch_memory_saver

    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x)) is not None:
        return cpu_tensor

    return x


def _named_params_and_buffers_vanilla(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Yield (global_name, param/buffer) with consistent names across PP/EP. Adjusts indices for
    virtual PP + EP offsets. Handles decoder.layers, mtp.layers (Multi-Token Prediction), expert_bias.
    """
    ep_size = get_parallel_state().ep.size
    ep_rank = get_parallel_state().ep.rank
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # MTP layer indices start from 0
                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer.mlp.experts\.(.+)\.weight(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.weight{expert_idx}", param
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}", param
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer


def collect_named_tensors_for_weight_transfer(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
    is_expert: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:

    for name, tensor in named_params_and_buffers(
        args,
        model,
        convert_to_global_name,
        translate_gpu_to_cpu,
    ):
        if is_expert == (".experts." in name):
            yield name, tensor


def post_process_weights(
    rollout_engines: Sequence[ActorHandle],
    restore_weights_before_load: bool = False,
    post_process_quantization: bool = False,
    post_load_weights: bool = False,
):
    """
    Trigger post-process on all rollout engines,
    including:
        - int4/fp4 quantization
        - post_load_weights (should be enabled when using p2p weights updating)
    """
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
                post_load_weights=post_load_weights,
            )
            for engine in rollout_engines
        ]
    )
