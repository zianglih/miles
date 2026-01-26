import json
import os
import time
from collections.abc import Callable, Sequence

import safetensors.torch
import torch

_SAVE_WEIGHT_SYNC = (
    os.getenv("MILES_DEBUG_SAVE_WEIGHT_SYNC") == "1" or os.getenv("MILES_DEBUG_SAVE_FIRST_WEIGHT_SYNC") == "1"
)
_STOP_AFTER_SAVE = (
    os.getenv("MILES_DEBUG_STOP_AFTER_WEIGHT_SYNC")
    if os.getenv("MILES_DEBUG_STOP_AFTER_WEIGHT_SYNC") is not None
    else os.getenv("MILES_DEBUG_STOP_AFTER_FIRST_WEIGHT_SYNC", "1")
) == "1"
_COMPARE_WITH_SGLANG = os.getenv("MILES_DEBUG_COMPARE_SGLANG", "1") == "1"
_COMPARE_FIRST_CHUNK_ONLY = os.getenv("MILES_DEBUG_COMPARE_SGLANG_FIRST_CHUNK_ONLY", "1") == "1"
_OUTPUT_DIR = os.getenv("MILES_DEBUG_WEIGHT_SYNC_DIR", "/tmp/miles_weight_sync_debug")

_STATE = {
    "out_dir": None,
    "chunk_idx": 0,
    "weight_map": {},
    "total_size": 0,
    "meta": {},
    "compare": {},
    "compared": False,
    "finalized": False,
}


def debug_save_weight_sync_enabled() -> bool:
    return _SAVE_WEIGHT_SYNC


def should_stop_after_debug_save() -> bool:
    return _STOP_AFTER_SAVE


def record_weight_sync_chunk(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    fetch_sglang_weight: Callable[[str, int], list] | None = None,
) -> str | None:
    if not _SAVE_WEIGHT_SYNC or not named_tensors:
        return None

    out_dir = _STATE["out_dir"]
    if out_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(_OUTPUT_DIR, f"weight_sync_full_{timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        _STATE["out_dir"] = out_dir

    chunk_idx = _STATE["chunk_idx"]
    chunk_name = f"hf_chunk_{chunk_idx:05d}.safetensors"
    weights_path = os.path.join(out_dir, chunk_name)

    cpu_tensors = {}
    for name, tensor in named_tensors:
        cpu_tensor = tensor.detach().cpu()
        cpu_tensors[name] = cpu_tensor
        _STATE["meta"][name] = {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
        }
        _STATE["weight_map"][name] = chunk_name
        _STATE["total_size"] += cpu_tensor.numel() * cpu_tensor.element_size()

    safetensors.torch.save_file(cpu_tensors, weights_path, metadata={"format": "pt"})
    _STATE["chunk_idx"] = chunk_idx + 1

    if _COMPARE_WITH_SGLANG and fetch_sglang_weight is not None:
        if not _STATE["compared"] or not _COMPARE_FIRST_CHUNK_ONLY:
            comparisons = _STATE["compare"]
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
            if _COMPARE_FIRST_CHUNK_ONLY:
                _STATE["compared"] = True

    return out_dir


def finalize_weight_sync_dump() -> str | None:
    if not _SAVE_WEIGHT_SYNC:
        return None
    if _STATE["finalized"]:
        return _STATE["out_dir"]
    out_dir = _STATE["out_dir"]
    if out_dir is None:
        return None

    meta_path = os.path.join(out_dir, "hf_full_meta.json")
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    compare_path = os.path.join(out_dir, "sglang_compare.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_STATE["meta"], f, indent=2)

    index_dict = {
        "weight_map": _STATE["weight_map"],
        "metadata": {"total_size": _STATE["total_size"]},
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=2)

    if _STATE["compare"]:
        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(_STATE["compare"], f, indent=2)

    _STATE["finalized"] = True
    return out_dir


# Backward-compatible aliases
def debug_save_first_weight_sync_enabled() -> bool:
    return debug_save_weight_sync_enabled()


def maybe_dump_first_weight_sync(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    fetch_sglang_weight: Callable[[str, int], list] | None = None,
) -> str | None:
    return record_weight_sync_chunk(named_tensors, fetch_sglang_weight=fetch_sglang_weight)
