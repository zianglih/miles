from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=[])


import json

import pytest
import torch
from tools.convert_hf_to_nvfp4 import _update_quantization_config as tool_update_quantization_config
from tools.convert_hf_to_nvfp4 import _write_hf_quant_config as tool_write_hf_quant_config
from tools.convert_hf_to_nvfp4 import quantize_nvfp4 as tool_quantize_nvfp4
from tools.convert_hf_to_nvfp4 import should_quantize as tool_should_quantize_nvfp4
from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import NVFP4QuantizerRef

from miles.backends.megatron_utils.megatron_to_hf.processors import quantize_params
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import (
    quantize_nvfp4 as processor_quantize_nvfp4,
)
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import quantize_params_nvfp4
from miles.backends.megatron_utils.update_weight.nvfp4_direct import (
    nvfp4_te_direct_update_enabled,
    te_nvfp4_workspace_to_hf,
)
from miles.utils.nvfp4 import (
    NVFP4_GROUP_SIZE,
    nvfp4_4over6_err_mode,
    nvfp4_4over6_weight_scope,
    nvfp4_global_decode_scale_te,
    nvfp4_weight_e4m3_max,
)

NVFP4_SHAPES = [
    (1, 64),
    (1, 1024),
    (3, 128),
    (16, 64),
    (64, 128),
    (128, 64),
    (256, 128),
    (512, 256),
    (128, 1024),
    (1024, 2048),
    (7168, 2048),
    (2048, 7168),
    (128, 16384),
]


def _make_weight(init_data: str, dtype: torch.dtype, shape: tuple[int, int], device: str) -> torch.Tensor:
    m, n = shape
    if init_data == "random":
        return torch.randn((m, n), dtype=dtype, device=device)
    if init_data == "boundary":
        base = torch.linspace(-12.0, 12.0, steps=n // 2, dtype=torch.float32, device=device)
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, 1e-4 * torch.ones_like(base))
        row = torch.empty(n, dtype=torch.float32, device=device)
        row[0::2] = base - eps
        row[1::2] = base + eps
        return row.unsqueeze(0).repeat(m, 1).to(dtype=dtype)
    if init_data == "zeros":
        return torch.zeros((m, n), dtype=dtype, device=device)
    if init_data == "maxes":
        return torch.full((m, n), torch.finfo(dtype).max, dtype=dtype, device=device)
    raise ValueError(f"Unknown init_data: {init_data}")


def _te_nvfp4_reference(
    weight: torch.Tensor, use_4over6: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = weight.contiguous()
    global_amax = torch.max(torch.abs(weight.to(torch.float32)))
    return _te_nvfp4_reference_with_global_amax(weight, global_amax, use_4over6=use_4over6)


def _te_nvfp4_reference_with_global_amax(
    weight: torch.Tensor,
    global_amax: torch.Tensor,
    use_4over6: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = weight.contiguous()
    nvfp4_e4m3_max = nvfp4_weight_e4m3_max(use_4over6)
    qweight, block_scale = NVFP4QuantizerRef._quantize_blockwise_reference(
        weight,
        global_amax,
        NVFP4_GROUP_SIZE,
        1,
        pow_2_scales=False,
        nvfp4_use_4over6=use_4over6,
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_err_mode(),
        eps=0.0,
    )
    return qweight, block_scale, nvfp4_global_decode_scale_te(global_amax, nvfp4_e4m3_max)


class FakeTENvfp4Tensor:
    def __init__(
        self,
        *,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        amax_rowwise: torch.Tensor,
        use_4over6: bool,
        e4m3_max: int,
    ):
        self._rowwise_data = rowwise_data
        self._rowwise_scale_inv = rowwise_scale_inv
        self._amax_rowwise = amax_rowwise
        self._with_gemm_swizzled_scales = False
        self._row_scaled_nvfp4 = False
        self._nvfp4_use_4over6 = use_4over6
        self._nvfp4_e4m3_max = e4m3_max
        self.dtype = torch.bfloat16
        self.shape = torch.Size((rowwise_data.shape[0], rowwise_data.shape[1] * 2))

    def dim(self):
        return len(self.shape)


def _qwen3moe_args():
    return type(
        "Args",
        (),
        {
            "hidden_size": 128,
            "kv_channels": None,
            "num_attention_heads": 4,
            "num_query_groups": 1,
            "q_lora_rank": None,
        },
    )()


def test_nvfp4_dispatch_accepts_quant_algo_without_quant_method():
    converted_named_params = [("model.embed_tokens.weight", torch.zeros((1, 1), dtype=torch.bfloat16))]

    out = quantize_params(
        args=None,
        megatron_name="embedding.word_embeddings.weight",
        converted_named_params=converted_named_params,
        quantization_config={"quant_algo": "NVFP4"},
    )

    assert out is converted_named_params


def test_nvfp4_direct_update_is_env_gated(monkeypatch):
    quantization_config = {"quant_method": "nvfp4"}
    monkeypatch.delenv("MILES_FP4_DIRECT_WEIGHT_UPDATE", raising=False)
    assert not nvfp4_te_direct_update_enabled(args=None, quantization_config=quantization_config)

    monkeypatch.setenv("MILES_FP4_DIRECT_WEIGHT_UPDATE", "1")
    assert nvfp4_te_direct_update_enabled(args=None, quantization_config=quantization_config)


def test_nvfp4_quantize_params_requires_complete_gated_pair():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.float32)
    with pytest.raises(ValueError, match="requires gate/up tensors to be quantized together"):
        quantize_params_nvfp4(
            args=None,
            megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
            converted_named_params=[
                ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
            ],
            quantization_config={"quant_method": "nvfp4"},
        )


def test_nvfp4_quantize_params_respects_extra_high_precision_layers_megatron():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
        ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
    ]
    args = type("Args", (), {"extra_high_precision_layers_megatron": ("linear_fc1",)})()

    out = quantize_params_nvfp4(
        args=args,
        megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "nvfp4"},
    )

    assert out is converted_named_params


def test_nvfp4_quantize_params_rejects_fp4_param_gather():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
    args = type("Args", (), {"fp4_param": True})()

    with pytest.raises(NotImplementedError, match="fp4-param-gather is unsupported"):
        quantize_params_nvfp4(
            args=args,
            megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
            converted_named_params=[
                ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
                ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
            ],
            quantization_config={"quant_method": "nvfp4"},
        )


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_nvfp4_quantize_params_respects_first_last_layers_bf16(layer_idx):
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
        ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
    ]
    args = type(
        "Args",
        (),
        {
            "first_last_layers_bf16": True,
            "num_layers": 4,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        },
    )()

    out = quantize_params_nvfp4(
        args=args,
        megatron_name=f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "nvfp4"},
    )

    assert out is converted_named_params


def test_nvfp4_hf_should_quantize_respects_extra_high_precision_layers_hf():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)

    assert not tool_should_quantize_nvfp4(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.0",),
    )
    assert tool_should_quantize_nvfp4(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.1",),
    )


def test_nvfp4_converter_records_4over6_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "weights")
    cfg = {}
    tool_update_quantization_config(cfg, ignore_list=["model.layers.0"])
    assert cfg["quantization_config"]["nvfp4_4over6"] == "weights"

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    tool_write_hf_quant_config(str(output_dir), ignore_list=["model.layers.0"], input_path=str(input_dir))

    hf_quant_config = json.loads((output_dir / "hf_quant_config.json").read_text())
    assert hf_quant_config["quantization"]["nvfp4_4over6"] == "weights"


@pytest.mark.parametrize("use_4over6", [False, True], ids=lambda value: nvfp4_4over6_weight_scope(value))
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
def test_nvfp4_quantize_matches_te_reference_with_4over6_modes(monkeypatch, use_4over6, err_mode):
    device = "cuda"
    torch.manual_seed(42)
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", err_mode)
    if use_4over6:
        monkeypatch.setenv("NVTE_NVFP4_4OVER6", "weights")
    else:
        monkeypatch.delenv("NVTE_NVFP4_4OVER6", raising=False)

    weight = _make_weight("random", torch.bfloat16, (128, 1024), device)
    qweight, block_scale, global_scale = processor_quantize_nvfp4(weight)
    qweight_ref, block_scale_ref, global_scale_ref = _te_nvfp4_reference(weight, use_4over6=use_4over6)

    torch.testing.assert_close(qweight, qweight_ref, rtol=0, atol=0)
    torch.testing.assert_close(block_scale.view(torch.uint8), block_scale_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(global_scale, global_scale_ref, rtol=0, atol=0)


def test_nvfp4_quantize_params_reads_4over6_from_env(monkeypatch):
    device = "cuda"
    torch.manual_seed(42)
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "weights")

    gate = _make_weight("random", torch.bfloat16, (4, NVFP4_GROUP_SIZE), device)
    up = _make_weight("random", torch.bfloat16, (4, NVFP4_GROUP_SIZE), device)
    shared_amax = torch.max(gate.abs().max().to(torch.float32), up.abs().max().to(torch.float32))
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", gate),
        ("model.layers.0.mlp.experts.0.up_proj.weight", up),
    ]

    out = dict(
        quantize_params_nvfp4(
            args=None,
            megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
            converted_named_params=converted_named_params,
            quantization_config={"quant_method": "nvfp4"},
        )
    )
    qweight_ref, block_scale_ref, global_scale_ref = _te_nvfp4_reference_with_global_amax(
        gate, shared_amax, use_4over6=True
    )

    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.gate_proj.weight"], qweight_ref, rtol=0, atol=0)
    torch.testing.assert_close(
        out["model.layers.0.mlp.experts.0.gate_proj.weight_scale"].view(torch.uint8),
        block_scale_ref.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        out["model.layers.0.mlp.experts.0.gate_proj.weight_scale_2"],
        global_scale_ref,
        rtol=0,
        atol=0,
    )


def test_nvfp4_reference_quantizer_does_not_extract_direct_te_buffers(monkeypatch):
    monkeypatch.setenv("MILES_FP4_DIRECT_WEIGHT_UPDATE", "1")
    monkeypatch.delenv("NVTE_NVFP4_4OVER6", raising=False)
    rowwise_data = torch.arange(32, dtype=torch.uint8, device="cuda").reshape(4, 8)
    rowwise_scale_inv = torch.arange(512, dtype=torch.float32, device="cuda").reshape(128, 4)
    amax_rowwise = torch.tensor(12.0, dtype=torch.float32, device="cuda")
    weight = FakeTENvfp4Tensor(
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        amax_rowwise=amax_rowwise,
        use_4over6=False,
        e4m3_max=448,
    )

    with pytest.raises(ValueError, match="dequantizable"):
        processor_quantize_nvfp4(weight)


def test_nvfp4_te_workspace_to_hf_extracts_direct_buffers(monkeypatch):
    monkeypatch.delenv("NVTE_NVFP4_4OVER6", raising=False)
    rowwise_data = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
    rowwise_scale_inv = torch.arange(256, dtype=torch.uint8).reshape(128, 2)
    amax_rowwise = torch.tensor([12.0], dtype=torch.float32)
    workspace = FakeTENvfp4Tensor(
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        amax_rowwise=amax_rowwise,
        use_4over6=False,
        e4m3_max=448,
    )
    args = _qwen3moe_args()

    out = dict(
        te_nvfp4_workspace_to_hf(
            args,
            "qwen3moe",
            "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight7",
            workspace,
            {"quant_method": "nvfp4"},
        )
    )

    torch.testing.assert_close(
        out["model.layers.2.mlp.experts.7.gate_proj.weight"],
        rowwise_data[:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        out["model.layers.2.mlp.experts.7.up_proj.weight"],
        rowwise_data[2:],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        out["model.layers.2.mlp.experts.7.gate_proj.weight_scale"].view(torch.uint8),
        rowwise_scale_inv[:2],
        rtol=0,
        atol=0,
    )
    assert out["model.layers.2.mlp.experts.7.gate_proj.weight_scale"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        out["model.layers.2.mlp.experts.7.up_proj.weight_scale"].view(torch.uint8),
        rowwise_scale_inv[2:4],
        rtol=0,
        atol=0,
    )
    assert out["model.layers.2.mlp.experts.7.up_proj.weight_scale"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        out["model.layers.2.mlp.experts.7.gate_proj.weight_scale_2"],
        nvfp4_global_decode_scale_te(amax_rowwise).reshape(()),
        rtol=0,
        atol=0,
    )
    assert out["model.layers.2.mlp.experts.7.gate_proj.weight_scale_2"].shape == torch.Size([])
    torch.testing.assert_close(
        out["model.layers.2.mlp.experts.7.up_proj.input_scale"],
        torch.ones_like(nvfp4_global_decode_scale_te(amax_rowwise).reshape(())),
        rtol=0,
        atol=0,
    )


def test_nvfp4_te_workspace_to_hf_rejects_4over6_mismatch(monkeypatch):
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "weights")
    workspace = FakeTENvfp4Tensor(
        rowwise_data=torch.zeros((4, 8), dtype=torch.uint8),
        rowwise_scale_inv=torch.zeros((128, 4), dtype=torch.float32),
        amax_rowwise=torch.tensor(12.0, dtype=torch.float32),
        use_4over6=False,
        e4m3_max=448,
    )
    args = _qwen3moe_args()

    with pytest.raises(ValueError, match="4over6 mode does not match"):
        te_nvfp4_workspace_to_hf(
            args,
            "qwen3moe",
            "module.module.decoder.layers.2.mlp.experts.linear_fc2.weight7",
            workspace,
            {"quant_method": "nvfp4"},
        )


@pytest.mark.parametrize(
    "quantize_fn",
    [processor_quantize_nvfp4, tool_quantize_nvfp4],
    ids=["processor", "convert_tool"],
)
@pytest.mark.parametrize("shape", NVFP4_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
def test_nvfp4_quantize_matches_te_reference_bitwise(quantize_fn, shape, dtype, init_data):
    device = "cuda"
    torch.manual_seed(42)

    weight = _make_weight(init_data, dtype, shape, device)
    qweight, block_scale, global_scale = quantize_fn(weight)
    qweight_ref, block_scale_ref, global_scale_ref = _te_nvfp4_reference(weight)

    torch.testing.assert_close(qweight, qweight_ref, rtol=0, atol=0)
    torch.testing.assert_close(block_scale.view(torch.uint8), block_scale_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(global_scale, global_scale_ref, rtol=0, atol=0)
