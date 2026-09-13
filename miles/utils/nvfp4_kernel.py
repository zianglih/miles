"""Fused CUDA scalar conversion for exported NVFP4 global scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def _nvfp4_global_decode_scale_scalar_kernel(amax_ptr, output_ptr, E4M3_MAX: tl.constexpr):
    amax = tl.load(amax_ptr)
    # Preserve both rounded FP32 divisions; amax / numerator is not equivalent.
    encoded = tl.div_rn(float(E4M3_MAX) * 6.0, amax)
    encoded = tl.where(encoded > 3.4028234663852886e38, 3.4028234663852886e38, encoded)
    encoded = tl.where(encoded == 0.0, 1.0, encoded)
    decoded = tl.div_rn(1.0, encoded)
    tl.store(output_ptr, decoded)


def nvfp4_global_decode_scale_scalar(global_amax: torch.Tensor, e4m3_max: int) -> torch.Tensor:
    output = torch.empty_like(global_amax)
    _nvfp4_global_decode_scale_scalar_kernel[(1,)](global_amax, output, e4m3_max, num_warps=1, enable_fp_fusion=False)
    return output
