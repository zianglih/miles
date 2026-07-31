#!/usr/bin/env python3
"""Run small, repeatable Miles numerical comparisons.

Edit the experiment section below, then run:

    python tools/run_numerical_comparison.py

Each variant runs in a fresh process, writes its full log and ``--dump-details``
artifacts, and is compared using Miles' structured metric events.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Experiment: edit these values directly.
# -----------------------------------------------------------------------------

EXPERIMENT_NAME = "glm5_flashinfer_routed_moe_topk_large"
OUTPUT_BASE = Path("/root/shared_data/numerical_comparisons")

# A replay makes trainer-side comparisons controlled and does not start SGLang.
# Set this to None for live-rollout experiments such as comparing SGLang flags.
FIXED_ROLLOUT_DATA: str | None = (
    "/root/shared_data/numerical_comparisons/"
    "glm5_flashinfer_topk_large_fixture_20260729_191100/"
    "megatron/dump_details/rollout_data/{rollout_id}.pt"
)

NUM_ROLLOUTS = 2
METRICS = (
    "train/train_rollout_logprob_abs_diff",
    "train/train_rollout_kl",
    "train/kl_loss",
)

# The first variant is the baseline. Put arbitrary Miles/SGLang flags in
# ``extra_train_args``; they are appended after the common arguments.
VARIANTS = (
    {
        "name": "megatron",
        "env": {
            "MILES_USE_FAST_ACTIVATIONS": "0",
            "MILES_USE_FLASHINFER_MOE": "0",
        },
        "extra_train_args": "",
    },
    {
        "name": "flashinfer",
        "env": {
            "MILES_USE_FAST_ACTIVATIONS": "0",
            "MILES_USE_FLASHINFER_MOE": "1",
            "MILES_FLASHINFER_MOE_DEBUG": "1",
        },
        "extra_train_args": "",
    },
    # Example live-rollout experiment (also set FIXED_ROLLOUT_DATA = None):
    # {
    #     "name": "flashinfer_topk",
    #     "env": {},
    #     "extra_train_args": "--sglang-dsa-topk-backend flashinfer",
    # },
)

COMMON_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "FLASHINFER_NVFP4_4OVER6": "1",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "SGLANG_DSA_FUSE_TOPK": "0",
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "NVSHMEM_DISABLE_NCCL": "1",
    "NCCL_NVLS_ENABLE": "0",
}

MEGATRON_MODEL_TYPE = "glm5.2-744B-A40B_5layer"
MEGATRON_PATH = "/root/TransformerEngine:/root/Megatron-LM"
NUM_GPUS = 8

TRAIN_ARGS = f"""
--hf-checkpoint /root/models/GLM-5.2_5layer-NVFP4/
--ref-load /root/models/GLM-5.2_5layer_torch_dist
--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
--input-key prompt
--label-key label
--apply-chat-template
--rollout-shuffle
--rm-type deepscaler
--num-rollout {NUM_ROLLOUTS}
--rollout-batch-size 8
--n-samples-per-prompt 8
--rollout-max-response-len 100
--rollout-temperature 1
--global-batch-size 64
--optimizer adam
--lr 1e-6
--lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.98
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--kl-coef 0.00
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
--use-tis
--tis-clip-low 0.5
--tis-clip 2.0
--tensor-model-parallel-size 4
--sequence-parallel
--pipeline-model-parallel-size 1
--context-parallel-size 1
--expert-model-parallel-size 4
--expert-tensor-parallel-size 1
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
--use-dynamic-batch-size
--max-tokens-per-gpu 2048
--data-pad-size-multiplier 1024
--log-probs-chunk-size 16384
--sglang-mem-fraction-static 0.7
--sglang-enable-dp-attention
--sglang-attention-backend nsa
--sglang-nsa-decode-backend flashmla_kv
--sglang-nsa-prefill-backend flashmla_sparse
--sglang-dsa-topk-backend flashinfer
--sglang-kv-cache-dtype fp8_e4m3
--sglang-page-size 64
--rollout-num-gpus-per-engine 2
--sglang-moe-runner-backend flashinfer_trtllm_routed
--sglang-ep-size 2
--sglang-dp-size 2
--sglang-moe-dense-tp-size 1
--sglang-enable-dp-lm-head
--sglang-cuda-graph-max-bs 256
--sglang-max-running-requests 512
--sglang-chunked-prefill-size 4096
--sglang-watchdog-timeout 3600
--ci-test
--ci-disable-logprobs-checker
--ci-disable-weight-update-checker
--transformer-impl transformer_engine
--bf16
--fp4-format e2m1
--fp4-recipe nvfp4
--first-last-layers-bf16
--num-layers-at-start-in-bf16 0
--num-layers-at-end-in-bf16 0
--extra-high-precision-layers-hf .shared_experts.
--extra-high-precision-layers-megatron .shared_experts.linear_fc1 .shared_experts.linear_fc2
--use-rollout-routing-replay
--use-miles-router
--sglang-disable-shared-experts-fusion
--attention-dropout 0.0
--hidden-dropout 0.0
--accumulate-allreduce-grads-in-fp32
--attention-softmax-in-fp32
--attention-backend flash
--allgather-cp
--miles-dsa-topk-backend flashinfer
--update-weight-buffer-size {2 * 1024**3}
--actor-num-nodes 1
--actor-num-gpus-per-node 4
--num-gpus-per-node 8
--rollout-num-gpus 4
--use-fault-tolerance
--moe-enable-deepep
--moe-token-dispatcher-type flex
--debug-disable-optimizer
"""

TE_PRECISION_CONFIG = """
configs:
    nvfp4:
        transformer_engine_config_type: "TEQuantizationParams"
        training_recipe:
            fp4_quantization_recipe: "nvfp4"
    bf16:
        transformer_engine_config_type: "TEQuantizationParams"
        training_recipe: {}
matchers:
    routed_experts_fc1_nvfp4:
        type: "glob"
        enabled: true
        pattern: "*.mlp.experts.linear_fc1"
        config: "nvfp4"
    routed_experts_fc2_nvfp4:
        type: "glob"
        enabled: true
        pattern: "*.mlp.experts.linear_fc2"
        config: "nvfp4"
    shared_experts_fc1_bf16:
        type: "glob"
        enabled: true
        pattern: "*.mlp.shared_experts.linear_fc1"
        config: "bf16"
    shared_experts_fc2_bf16:
        type: "glob"
        enabled: true
        pattern: "*.mlp.shared_experts.linear_fc2"
        config: "bf16"
    default_bf16:
        type: "glob"
        enabled: true
        pattern: "*"
        config: "bf16"
""".strip()


_VARIANT_ENV = "_MILES_NUMERICAL_VARIANT"
_OUTPUT_ENV = "_MILES_NUMERICAL_OUTPUT"


def _normalized_args(text: str) -> str:
    return shlex.join(shlex.split(text))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_variant(name: str) -> dict[str, Any]:
    return next(variant for variant in VARIANTS if variant["name"] == name)


def _run_variant(name: str, output_root: Path) -> None:
    variant = _find_variant(name)
    runtime_env = {**COMMON_ENV, **variant["env"]}
    os.environ.update(runtime_env)
    os.environ.setdefault("RAY_TMPDIR", "/tmp/ray")
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    import miles.utils.external_utils.command_utils as U

    dump_dir = output_root / name / "dump_details"
    te_config_path = U.save_to_temp_file(TE_PRECISION_CONFIG, "yaml")
    extra_args = [
        TRAIN_ARGS,
        f"--te-precision-config-file {shlex.quote(te_config_path)}",
        f"--dump-details {shlex.quote(str(dump_dir))}",
    ]
    if FIXED_ROLLOUT_DATA is not None:
        extra_args.extend(
            (
                f"--load-debug-rollout-data {shlex.quote(FIXED_ROLLOUT_DATA)}",
                "--disable-rollout-global-dataset",
            )
        )
    extra_args.append(variant["extra_train_args"])

    U.execute_train(
        train_args=_normalized_args(" ".join(extra_args)),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        megatron_path=MEGATRON_PATH,
        extra_env_vars=runtime_env,
    )


def _run_child(name: str, output_root: Path) -> None:
    variant_dir = output_root / name
    variant_dir.mkdir(parents=True)
    log_path = variant_dir / "run.log"
    child_env = {
        **os.environ,
        **COMMON_ENV,
        **_find_variant(name)["env"],
        _VARIANT_ENV: name,
        _OUTPUT_ENV: str(output_root),
        "PYTHONUNBUFFERED": "1",
    }
    command = [sys.executable, str(Path(__file__).resolve())]
    print(f"\n===== {name}: {' '.join(shlex.quote(part) for part in command)} =====", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Variant {name!r} failed with exit code {return_code}; see {log_path}")


def _read_metrics(dump_dir: Path) -> dict[str, list[dict[str, int | float]]]:
    raw_events: list[dict[str, Any]] = []
    for event_path in sorted((dump_dir / "events").glob("**/*.jsonl")):
        with event_path.open(encoding="utf-8") as event_file:
            raw_events.extend(json.loads(line) for line in event_file if line.strip())

    metric_events = [event for event in raw_events if event.get("type") == "metric"]
    final_attempt: dict[tuple[str, int | None], int] = {}
    for event in metric_events:
        stream = (json.dumps(event["source"], sort_keys=True), event.get("rollout_id"))
        final_attempt[stream] = max(final_attempt.get(stream, 0), event.get("attempt") or 0)

    series: dict[str, dict[int, float]] = {metric: {} for metric in METRICS}
    for event in metric_events:
        stream = (json.dumps(event["source"], sort_keys=True), event.get("rollout_id"))
        if (event.get("attempt") or 0) != final_attempt[stream]:
            continue
        values = event["metrics"]
        for metric in METRICS:
            if metric not in values:
                continue
            step_key = f"{metric.split('/', 1)[0]}/step"
            step = values.get(step_key, event.get("rollout_id"))
            if step is None:
                raise RuntimeError(f"No step found for metric {metric!r}")
            value = float(values[metric])
            previous = series[metric].setdefault(int(step), value)
            if previous != value:
                raise RuntimeError(f"Conflicting values for {metric!r} at step {step}: {previous} and {value}")

    missing = [metric for metric, values in series.items() if not values]
    if missing:
        raise RuntimeError(f"Missing metrics in {dump_dir / 'events'}: {missing}")
    return {
        metric: [{"step": step, "value": value} for step, value in sorted(values.items())]
        for metric, values in series.items()
    }


def _mean(values: list[dict[str, int | float]]) -> float:
    return sum(float(item["value"]) for item in values) / len(values)


def _write_and_print_results(output_root: Path, source_hashes: dict[str, str]) -> None:
    variants = []
    for variant in VARIANTS:
        name = variant["name"]
        series = _read_metrics(output_root / name / "dump_details")
        variants.append(
            {
                "name": name,
                "env": variant["env"],
                "extra_train_args": variant["extra_train_args"],
                "series": series,
                "mean": {metric: _mean(values) for metric, values in series.items()},
            }
        )

    baseline = variants[0]
    rows = []
    for variant in variants:
        for metric in METRICS:
            value = variant["mean"][metric]
            baseline_value = baseline["mean"][metric]
            delta = value - baseline_value
            relative_delta = None if baseline_value == 0 else delta / abs(baseline_value)
            rows.append((variant["name"], metric, value, delta, relative_delta))

    result = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline["name"],
        "num_rollouts": NUM_ROLLOUTS,
        "megatron_model_type": MEGATRON_MODEL_TYPE,
        "megatron_path": MEGATRON_PATH,
        "common_env": COMMON_ENV,
        "common_train_args": _normalized_args(TRAIN_ARGS),
        "fixed_rollout_data": FIXED_ROLLOUT_DATA,
        "fixed_rollout_sha256": source_hashes,
        "variants": variants,
    }
    result_path = output_root / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nResults (candidate delta = candidate mean - baseline mean)")
    print(f"{'variant':<10} {'metric':<42} {'mean':>14} {'delta':>14} {'relative':>12}")
    for name, metric, value, delta, relative_delta in rows:
        relative = "n/a" if relative_delta is None else f"{relative_delta:+.3%}"
        print(f"{name:<10} {metric:<42} {value:>14.7g} {delta:>+14.7g} {relative:>12}")
    print(f"\nArtifacts: {output_root}")
    print(f"Machine-readable results: {result_path}")
    if FIXED_ROLLOUT_DATA is None:
        print("Note: live-rollout variants are exploratory, not paired-input comparisons.")


def _run_all() -> None:
    names = [variant["name"] for variant in VARIANTS]
    if len(names) != len(set(names)):
        raise ValueError(f"Variant names must be unique: {names}")

    source_hashes = {}
    if FIXED_ROLLOUT_DATA is not None:
        if "{rollout_id}" not in FIXED_ROLLOUT_DATA:
            raise ValueError("FIXED_ROLLOUT_DATA must contain {rollout_id}")
        for rollout_id in range(NUM_ROLLOUTS):
            path = Path(FIXED_ROLLOUT_DATA.format(rollout_id=rollout_id))
            if not path.is_file():
                raise FileNotFoundError(path)
            source_hashes[str(path)] = _sha256(path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = OUTPUT_BASE / f"{EXPERIMENT_NAME}_{timestamp}"
    output_root.mkdir(parents=True)
    for name in names:
        _run_child(name, output_root)
    _write_and_print_results(output_root, source_hashes)


def main() -> None:
    if variant_name := os.environ.get(_VARIANT_ENV):
        _run_variant(variant_name, Path(os.environ[_OUTPUT_ENV]))
    else:
        _run_all()


if __name__ == "__main__":
    main()
