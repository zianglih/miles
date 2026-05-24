import os
from typing import Any

import torch

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def nvfp4_4over6_weight_scope_enabled(value) -> bool:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("weights", "all"):
            return True
        if value in ("none", "activations"):
            return False
    return str_to_bool(value)


def nvfp4_4over6_enabled() -> bool:
    env_value = os.getenv("NVTE_NVFP4_4OVER6")
    if env_value is not None:
        return nvfp4_4over6_weight_scope_enabled(env_value)
    return False


def nvfp4_4over6_weight_scope(use_4over6: bool) -> str:
    return "weights" if use_4over6 else "none"


def nvfp4_weight_e4m3_max(use_4over6: bool) -> int:
    if use_4over6 and nvfp4_4over6_weight_scope_enabled(os.getenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")):
        return 256
    return int(FP8_E4M3_MAX)


def nvfp4_4over6_err_mode() -> str:
    err_mode = os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper()
    if err_mode not in ("MAE", "MSE"):
        raise ValueError("NVTE_NVFP4_4OVER6_ERR_MODE must be one of: 'MAE', 'MSE'.")
    return err_mode


def nvfp4_global_decode_scale_te(global_amax: torch.Tensor, nvfp4_e4m3_max: int = int(FP8_E4M3_MAX)) -> torch.Tensor:
    fp4_max = torch.tensor(FP4_E2M1_MAX, device=global_amax.device, dtype=torch.float32)
    fp8_max = torch.tensor(float(nvfp4_e4m3_max), device=global_amax.device, dtype=torch.float32)
    global_encode_scale = torch.div(fp8_max * fp4_max, global_amax.to(torch.float32))
    global_encode_scale = torch.min(
        global_encode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=global_encode_scale.device,
            dtype=torch.float32,
        ),
    )
    if global_encode_scale.numel() == 1:
        if global_encode_scale == torch.tensor(0.0, device=global_amax.device, dtype=torch.float32):
            global_encode_scale = torch.tensor(1.0, device=global_amax.device, dtype=torch.float32)
    else:
        global_encode_scale = torch.where(
            global_encode_scale == 0.0,
            torch.ones_like(global_encode_scale),
            global_encode_scale,
        )
    return torch.div(1.0, global_encode_scale)


def is_te_nvfp4_tensor(weight: Any) -> bool:
    return all(
        hasattr(weight, attr)
        for attr in (
            "_rowwise_data",
            "_rowwise_scale_inv",
            "_amax_rowwise",
            "_with_gemm_swizzled_scales",
            "_row_scaled_nvfp4",
            "_nvfp4_use_4over6",
            "_nvfp4_e4m3_max",
        )
    )
