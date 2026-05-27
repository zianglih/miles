from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class QuantizedWeightTransfer:
    """Quantized weight payload used by direct rollout weight sync.

    ``quant_format`` describes the storage contract for ``weight`` and
    ``aux_tensors``. Keeping the primary quantized storage and format-specific
    auxiliary tensors together lets future formats such as NVFP4 reuse the
    same transfer plumbing without going through dense dequantize/quantize
    round trips.
    """

    quant_format: str
    weight: torch.Tensor
    weight_dtype: torch.dtype
    aux_tensors: dict[str, torch.Tensor]

    def replace_tensors(
        self,
        *,
        weight: torch.Tensor | None = None,
        aux_tensors: dict[str, torch.Tensor] | None = None,
    ) -> QuantizedWeightTransfer:
        return replace(
            self,
            weight=self.weight if weight is None else weight,
            aux_tensors=self.aux_tensors if aux_tensors is None else aux_tensors,
        )


WeightTransferValue = torch.Tensor | QuantizedWeightTransfer


def is_quantized_weight_transfer(value: object) -> bool:
    return isinstance(value, QuantizedWeightTransfer)


def transfer_value_nbytes(value: WeightTransferValue) -> int:
    if isinstance(value, QuantizedWeightTransfer):
        return value.weight.numel() * value.weight.element_size() + sum(
            tensor.numel() * tensor.element_size() for tensor in value.aux_tensors.values()
        )
    return value.numel() * value.element_size()


def empty_like_transfer_value(
    value: WeightTransferValue, *, device: torch.device | int | None = None
) -> WeightTransferValue:
    if isinstance(value, QuantizedWeightTransfer):
        return value.replace_tensors(
            weight=torch.empty_like(value.weight, device=device),
            aux_tensors={name: torch.empty_like(tensor, device=device) for name, tensor in value.aux_tensors.items()},
        )
    return torch.empty_like(value, device=device)


def all_gather_transfer_value(
    outputs: list[WeightTransferValue],
    value: WeightTransferValue,
    *,
    group,
    async_op: bool = False,
) -> list:
    if isinstance(value, QuantizedWeightTransfer):
        handles = [
            torch.distributed.all_gather(
                [output.weight for output in outputs],
                value.weight,
                group=group,
                async_op=async_op,
            )
        ]
        for name, tensor in value.aux_tensors.items():
            handles.append(
                torch.distributed.all_gather(
                    [output.aux_tensors[name] for output in outputs],
                    tensor,
                    group=group,
                    async_op=async_op,
                )
            )
        return handles

    return [
        torch.distributed.all_gather(
            outputs,
            value.data,
            group=group,
            async_op=async_op,
        )
    ]
