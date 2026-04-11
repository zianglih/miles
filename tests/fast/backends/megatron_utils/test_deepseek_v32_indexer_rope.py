"""Unit tests for the DeepSeek-v3.2 indexer RoPE layout in Miles.

This test loads the function from `miles_plugins/models/glm5/glm5.py` directly,
then compares its DeepSeek-v3.2 branch to an SGLang-style reference flow.
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _ensure_module(name: str, *, is_package: bool = False) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        if is_package:
            module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    return module


def _install_glm5_import_stubs() -> None:
    class _Dummy:
        pass

    # Namespace packages for relative imports in glm5.py.
    _ensure_module("miles_plugins", is_package=True)
    _ensure_module("miles_plugins.models", is_package=True)
    _ensure_module("miles_plugins.models.glm5", is_package=True)
    _ensure_module("miles_plugins.models.glm5.ops", is_package=True)

    ops_indexer = _ensure_module("miles_plugins.models.glm5.ops.indexer")
    ops_indexer.generate_varlen_mask_params = lambda *args, **kwargs: None
    ops_indexer.lighting_indexer = lambda *args, **kwargs: None

    ops_sparse_mla = _ensure_module("miles_plugins.models.glm5.ops.sparse_mla")
    ops_sparse_mla.SparseMLA = _Dummy

    # Megatron stubs needed only to import glm5.py.
    _ensure_module("megatron", is_package=True)
    core_mod = _ensure_module("megatron.core", is_package=True)
    parallel_state_mod = _ensure_module("megatron.core.parallel_state")
    parallel_state_mod.get_context_parallel_group = lambda: None
    parallel_state_mod.get_context_parallel_world_size = lambda: 1
    parallel_state_mod.get_context_parallel_rank = lambda: 0
    core_mod.parallel_state = parallel_state_mod

    _ensure_module("megatron.core.extensions", is_package=True)
    te_mod = _ensure_module("megatron.core.extensions.transformer_engine")
    te_mod.TEColumnParallelLinear = _Dummy
    te_mod.TELinear = _Dummy
    te_spec_mod = _ensure_module("megatron.core.extensions.transformer_engine_spec_provider")
    te_spec_mod.TESpecProvider = _Dummy

    _ensure_module("megatron.core.models", is_package=True)
    _ensure_module("megatron.core.models.common", is_package=True)
    embeddings_mod = _ensure_module("megatron.core.models.common.embeddings")
    embeddings_mod.RotaryEmbedding = _Dummy
    embeddings_mod.YarnRotaryEmbedding = _Dummy
    embeddings_mod._yarn_get_mscale = lambda *_args, **_kwargs: 1.0
    _ensure_module("megatron.core.models.gpt", is_package=True)
    gpt_specs_mod = _ensure_module("megatron.core.models.gpt.gpt_layer_specs")
    gpt_specs_mod.get_gpt_decoder_block_spec = lambda *_args, **_kwargs: None

    _ensure_module("megatron.core.post_training", is_package=True)
    _ensure_module("megatron.core.post_training.modelopt", is_package=True)
    modelopt_layers_mod = _ensure_module("megatron.core.post_training.modelopt.layers")
    modelopt_layers_mod.Linear = _Dummy

    _ensure_module("megatron.core.tensor_parallel", is_package=True)
    tp_layers_mod = _ensure_module("megatron.core.tensor_parallel.layers")
    tp_layers_mod.ColumnParallelLinear = _Dummy
    tp_mappings_mod = _ensure_module("megatron.core.tensor_parallel.mappings")
    tp_mappings_mod.gather_from_sequence_parallel_region = lambda x, **_kwargs: x
    tp_mappings_mod.scatter_to_sequence_parallel_region = lambda x, **_kwargs: x

    _ensure_module("megatron.core.transformer", is_package=True)
    transformer_attention_mod = _ensure_module("megatron.core.transformer.attention")
    transformer_attention_mod.Attention = _Dummy
    transformer_enums_mod = _ensure_module("megatron.core.transformer.enums")
    transformer_enums_mod.AttnMaskType = _Dummy
    transformer_identity_mod = _ensure_module("megatron.core.transformer.identity_op")
    transformer_identity_mod.IdentityOp = _Dummy
    _ensure_module("megatron.core.transformer.moe", is_package=True)
    transformer_moe_utils_mod = _ensure_module("megatron.core.transformer.moe.moe_utils")
    transformer_moe_utils_mod.RouterGatingLinearFunction = _Dummy
    transformer_spec_utils_mod = _ensure_module("megatron.core.transformer.spec_utils")
    transformer_spec_utils_mod.ModuleSpec = _Dummy
    transformer_spec_utils_mod.build_module = lambda *_args, **_kwargs: None
    transformer_block_mod = _ensure_module("megatron.core.transformer.transformer_block")
    transformer_block_mod.get_num_layers_to_build = lambda *_args, **_kwargs: 1
    transformer_config_mod = _ensure_module("megatron.core.transformer.transformer_config")
    transformer_config_mod.MLATransformerConfig = _Dummy

    transformers_mod = _ensure_module("transformers")
    if not hasattr(transformers_mod, "AutoConfig"):
        transformers_mod.AutoConfig = _Dummy


def _load_glm5_module():
    _install_glm5_import_stubs()
    module_path = Path(__file__).resolve().parents[4] / "miles_plugins" / "models" / "glm5" / "glm5.py"
    module_name = "miles_plugins.models.glm5.glm5_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


build_indexer_query_key_with_rope = _load_glm5_module().build_indexer_query_key_with_rope


def _fake_fuse_rope(
    x: torch.Tensor,
    _cu_seqlens: torch.Tensor,
    *,
    gathered: bool,
    interleaved: bool,
) -> torch.Tensor:
    # Make the transform branch-dependent while keeping arithmetic simple.
    marker = (1000 if gathered else 100) + (10 if interleaved else 1)
    return x + marker


def _sglang_v32_reference(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    qk_pos_emb_head_dim: int,
    index_head_dim: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirrors SGLang v3.2 indexer path:
    # q_pe, q_nope = split([rope, no_rope]), interleaved=False, cat([q_pe, q_nope])
    q_pe, q_nope = torch.split(
        index_q,
        [qk_pos_emb_head_dim, index_head_dim - qk_pos_emb_head_dim],
        dim=-1,
    )
    q_pe = _fake_fuse_rope(q_pe, cu_seqlens_q, gathered=False, interleaved=False)
    ref_query = torch.cat([q_pe, q_nope], dim=-1)

    k_pe, k_nope = torch.split(
        index_k,
        [qk_pos_emb_head_dim, index_head_dim - qk_pos_emb_head_dim],
        dim=-1,
    )
    k_pe = _fake_fuse_rope(k_pe, cu_seqlens_kv, gathered=True, interleaved=False)
    ref_key = torch.cat([k_pe, k_nope], dim=-1)
    return ref_query, ref_key


def _legacy_glm5_reference(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    qk_pos_emb_head_dim: int,
    index_head_dim: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_nope, q_pe = torch.split(
        index_q,
        [index_head_dim - qk_pos_emb_head_dim, qk_pos_emb_head_dim],
        dim=-1,
    )
    q_pe = _fake_fuse_rope(q_pe, cu_seqlens_q, gathered=False, interleaved=True)
    ref_query = torch.cat([q_nope, q_pe], dim=-1)

    k_nope, k_pe = torch.split(
        index_k,
        [index_head_dim - qk_pos_emb_head_dim, qk_pos_emb_head_dim],
        dim=-1,
    )
    k_pe = _fake_fuse_rope(k_pe, cu_seqlens_kv, gathered=True, interleaved=True)
    ref_key = torch.cat([k_nope, k_pe], dim=-1)
    return ref_query, ref_key


def test_deepseek_v32_indexer_style_matches_sglang_reference():
    qk_pos_emb_head_dim = 64
    index_head_dim = 128
    index_q = torch.arange(2 * 3 * index_head_dim, dtype=torch.float32).view(2, 3, index_head_dim)
    index_k = (10000 + torch.arange(2 * index_head_dim, dtype=torch.float32)).view(2, 1, index_head_dim)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 2], dtype=torch.int32)

    index_query, index_key = build_indexer_query_key_with_rope(
        index_q=index_q,
        index_k=index_k,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        index_head_dim=index_head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        fuse_rope=_fake_fuse_rope,
        indexer_rope_interleave=False,
    )
    ref_query, ref_key = _sglang_v32_reference(
        index_q=index_q,
        index_k=index_k,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        index_head_dim=index_head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
    )

    assert torch.equal(index_query, ref_query)
    assert torch.equal(index_key, ref_key)


def test_legacy_glm5_indexer_style_is_preserved():
    qk_pos_emb_head_dim = 64
    index_head_dim = 128
    index_q = torch.arange(2 * 3 * index_head_dim, dtype=torch.float32).view(2, 3, index_head_dim)
    index_k = (20000 + torch.arange(2 * index_head_dim, dtype=torch.float32)).view(2, 1, index_head_dim)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 2], dtype=torch.int32)

    index_query, index_key = build_indexer_query_key_with_rope(
        index_q=index_q,
        index_k=index_k,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        index_head_dim=index_head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        fuse_rope=_fake_fuse_rope,
        indexer_rope_interleave=True,
    )
    ref_query, ref_key = _legacy_glm5_reference(
        index_q=index_q,
        index_k=index_k,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        index_head_dim=index_head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
    )

    assert torch.equal(index_query, ref_query)
    assert torch.equal(index_key, ref_key)
