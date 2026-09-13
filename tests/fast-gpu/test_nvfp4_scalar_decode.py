"""Exact export-scale arithmetic and unchanged non-scalar/autograd contracts."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15,
    suite="stage-c-8-gpu-b200",
    labels=["precision"],
    hardware=["blackwell"],
)


import pytest
import torch

from miles.utils.nvfp4 import nvfp4_global_decode_scale_te


def eager_decode(value, maximum):
    encoded = torch.div(float(maximum) * 6.0, value.to(torch.float32))
    encoded = encoded.clamp_max(torch.finfo(torch.float32).max)
    return torch.div(1.0, torch.where(encoded == 0.0, 1.0, encoded))


def assert_bits_equal(actual, expected):
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape
    assert actual.device == expected.device
    assert torch.equal(actual.reshape(-1).view(torch.int32), expected.reshape(-1).view(torch.int32))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("maximum", [256, 448])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_fallback_dtype_batch_and_noncontiguous(device, maximum, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    values = torch.tensor([[0.0, -0.0, 3.5], [1.0, 12.5, float("inf")]], device=device, dtype=dtype).T
    assert not values.is_contiguous()
    assert_bits_equal(nvfp4_global_decode_scale_te(values, maximum), eager_decode(values, maximum))
    scalar = values[2:3, 1:2]
    assert_bits_equal(nvfp4_global_decode_scale_te(scalar, maximum), eager_decode(scalar, maximum))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("maximum", [256, 448])
def test_autograd_preserves_eager_graph(device, maximum):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    actual_input = torch.tensor(3.5, device=device, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_()
    actual = nvfp4_global_decode_scale_te(actual_input, maximum)
    expected = eager_decode(expected_input, maximum)
    assert_bits_equal(actual, expected)
    actual.backward()
    expected.backward()
    assert_bits_equal(actual_input.grad, expected_input.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("maximum", [256, 448])
def test_cuda_scalar_special_and_random_bits(maximum):
    # Include signaling/quiet NaN payloads, signed zero and subnormals. Compare
    # integer views, never an approximate floating-point tolerance.
    bits = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0xFFC00000,
        0x7FA12345,
        0xFFA12345,
    ]
    signed = [value if value < 2**31 else value - 2**32 for value in bits]
    generator = torch.Generator().manual_seed(42)
    random_bits = torch.randint(-(2**31), 2**31, (1024,), generator=generator, dtype=torch.int32)
    values = torch.cat((torch.tensor(signed, dtype=torch.int32), random_bits)).view(torch.float32).cuda()
    expected = eager_decode(values, maximum)
    actual = torch.stack([nvfp4_global_decode_scale_te(value, maximum) for value in values.unbind()])
    assert_bits_equal(actual, expected)
