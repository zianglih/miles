import json
import os
import time
from collections.abc import Callable, Sequence

import safetensors.torch
import torch

_SAVE_FIRST_SYNC = os.getenv("MILES_DEBUG_SAVE_FIRST_WEIGHT_SYNC") == "1"
_STOP_AFTER_SAVE = os.getenv("MILES_DEBUG_STOP_AFTER_FIRST_WEIGHT_SYNC", "1") == "1"
_COMPARE_WITH_SGLANG = os.getenv("MILES_DEBUG_COMPARE_SGLANG", "1") == "1"
_OUTPUT_DIR = os.getenv("MILES_DEBUG_WEIGHT_SYNC_DIR", "/tmp/miles_weight_sync_debug")

_FIRST_SYNC_SAVED = False


def debug_save_first_weight_sync_enabled() -> bool:
    return _SAVE_FIRST_SYNC


def should_stop_after_debug_save() -> bool:
    return _STOP_AFTER_SAVE


def maybe_dump_first_weight_sync(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    fetch_sglang_weight: Callable[[str, int], list] | None = None,
) -> str | None:
    global _FIRST_SYNC_SAVED
    if not _SAVE_FIRST_SYNC or _FIRST_SYNC_SAVED or not named_tensors:
        return None

    _FIRST_SYNC_SAVED = True
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_OUTPUT_DIR, f"weight_sync_first_chunk_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(out_dir, "hf_first_chunk.safetensors")
    meta_path = os.path.join(out_dir, "hf_first_chunk_meta.json")
    compare_path = os.path.join(out_dir, "sglang_compare.json")

    cpu_tensors = {}
    meta = {}
    for name, tensor in named_tensors:
        cpu_tensor = tensor.detach().cpu()
        cpu_tensors[name] = cpu_tensor
        meta[name] = {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
        }

    safetensors.torch.save_file(cpu_tensors, weights_path, metadata={"format": "pt"})
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if _COMPARE_WITH_SGLANG and fetch_sglang_weight is not None:
        comparisons = {}
        for name, tensor in named_tensors:
            truncate_size = int(tensor.shape[0]) if tensor.ndim > 0 else 1
            try:
                sglang_param = fetch_sglang_weight(name, truncate_size)
            except Exception as exc:
                comparisons[name] = {"status": "error", "error": str(exc)}
                continue

            if sglang_param is None:
                comparisons[name] = {"status": "missing"}
                continue

            sglang_tensor = torch.tensor(sglang_param, dtype=torch.float32)
            src_tensor = tensor.detach().cpu().to(torch.float32)

            if sglang_tensor.shape != src_tensor.shape:
                comparisons[name] = {
                    "status": "shape_mismatch",
                    "src_shape": list(src_tensor.shape),
                    "sglang_shape": list(sglang_tensor.shape),
                }
                continue

            diff = (src_tensor - sglang_tensor).abs()
            comparisons[name] = {
                "status": "ok",
                "max_abs": float(diff.max().item()) if diff.numel() > 0 else 0.0,
                "mean_abs": float(diff.mean().item()) if diff.numel() > 0 else 0.0,
            }

        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2)

    return out_dir
