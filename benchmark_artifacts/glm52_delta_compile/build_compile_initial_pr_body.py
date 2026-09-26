#!/usr/bin/env python3
"""Build the initial draft PR body for publication after the GPU pair starts.

CPU and preflight evidence must already be complete. This does not claim a GPU
result, commit/push source, edit a PR, or modify the working PR draft.
"""

import hashlib
import json
from pathlib import Path
import re

from build_final_cpu_stage_report import build as cpu_report

ROOT = Path(__file__).resolve().parent


def raw(name):
    return (ROOT / name).read_bytes().decode("utf-8")


def section(text, start, stop, replacement):
    return text[:text.index(start)] + replacement.rstrip() + "\n\n" + text[text.index(stop):]


def details(title, content):
    return f"<details>\n<summary>{title}</summary>\n\n```text\n{content.rstrip()}\n```\n\n</details>\n"


def main():
    source = raw("miles-torch-compile-pr-body-assembly-template.md")
    report, cpu_markdown = cpu_report()
    plan = json.loads(raw("COMPILE_VALIDATION_PLAN.json"))
    preflight = json.loads(raw("artifacts/compile-final-preflight-01/manifest.json"))
    assert preflight["sources"] == plan["sources"] and preflight["standalone_source"] == plan["standalone_source"]
    assert set(preflight["checks"]) == {"focused-tests", "gloo-failures"}
    assert all(check["exit"] == 0 for check in preflight["checks"].values())
    for filename, key in (("run_compiled_preflight.py", "script_sha256"),
                          ("validate_compiled_sender_gloo.py", "gloo_script_sha256")):
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == preflight[key]
    assert raw("artifacts/torch-compile-final-production-tests.exit").strip() == "0"
    assert raw("artifacts/torch-compile-final-production-tested-head.txt").strip() == plan["standalone_source"]
    standalone = raw("artifacts/torch-compile-final-production-tests.log")
    combined = raw("artifacts/compile-final-preflight-01/focused-tests.log")
    gloo = raw("artifacts/compile-final-preflight-01/gloo-failures.log")
    assert "45 passed, 24 warnings in 30.89s" in standalone
    assert "64 passed, 28 warnings in 30.02s" in combined
    assert "PASS: both failure modes drained all four collectives on both ranks without publication" in gloo
    sources = raw("artifacts/torch-compile-final-source-sha256.txt")
    for line in sources.splitlines():
        digest, filename = line.split(maxsplit=1)
        from subprocess import check_output
        assert hashlib.sha256(check_output(["git", "-C", str(ROOT / "miles-torch-compile"), "show",
                                           f"{plan['standalone_source']}:{filename}"])).hexdigest() == digest
    tests = """### Completed selected-source CPU and distributed preflight

Standalone `6fb0e2a9d81fc343863ba6015512fe93992892e4` passed **45 tests, 24 warnings in 30.89 s** in a clean detached checkout inside the pinned C2 image, using normal repository conftest and a fresh compiler cache. Tested source hashes were recorded for all six production/test files. The complete output, including every retained warning and reported slow test, is below.

From `/hai-workspace/glm52-delta/miles-torch-compile-final`:

```bash
CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 \\
  TORCHINDUCTOR_CACHE_DIR=/hai-workspace/glm52-delta/artifacts/torch-compile-final-production-unit-cache \\
  PYTHONPATH=. timeout --kill-after=5 240 /opt/sglang/bin/python -m pytest \\
  tests/fast/utils/test_delta_preparation.py \\
  tests/fast/backends/training_utils/weight_update/test_delta_compiled_cpu.py \\
  tests/fast/backends/training_utils/weight_update/test_delta.py \\
  tests/fast/backends/training_utils/weight_update/test_disk_delta_weight_version.py \\
  tests/fast/backends/training_utils/weight_update/test_disk_delta_engine_calls.py -q
```

""" + details("Complete selected standalone test output", standalone)
    tests += """
Combined `283a1cb4e6ed4a3712a39bc5dbd26060465bdcca` passed **64 tests, 28 warnings in 30.02 s**. Its real two-rank Gloo harness exercised both layout and CPU-worker failures with actual compiled preparation: each rank drained all four collectives, closed its pool, and agreed on the error without publication. GPUs were hidden and the finalizer was mocked; these checks do not establish live CUDA transfer or receiver RPC behavior.

From `/hai-workspace/glm52-delta`, the retained preflight script verifies clean selected source pins, records exact commands/helper hashes, and uses a fresh shared preflight cache before the focused tests and Gloo harness:

```bash
/opt/sglang/bin/python run_compiled_preflight.py \\
  --plan COMPILE_VALIDATION_PLAN.json --run-name compile-final-preflight-01
```

Its environment is `CUDA_VISIBLE_DEVICES=99`, `OMP_NUM_THREADS=1`, `TORCHINDUCTOR_CACHE_DIR=/hai-workspace/glm52-delta/compile-final-preflight-01-cache`, and `PYTHONPATH=/hai-workspace/glm52-delta/sglang/python:/hai-workspace/glm52-delta/miles-torch-compile-validation:/root/Megatron-LM`. Both subprocesses exited 0; measured preflight elapsed times were 59.609147549 s for focused tests and 56.325512886 s for Gloo, including startup.

""" + details("Complete selected combined-source test output", combined)
    tests += "\n" + details("Complete two-rank Gloo failure output", gloo)
    tests += """
Coverage includes exhaustive byte pairs in all four word lanes, signed-byte masks, odd tails/alignment, counts above 2^24, unchanged identity, independent changed outputs, padding reset, concurrent warmed dispatch, bounded work, snapshot promotion, and error coordination. Warnings remain visible; this is not an error-free environment. Earlier 6eca validation (42 tests, 24 warnings in 23.20 s) is historical and is not substituted for these selected-source checks. Hosted CI is separate and not represented as completed here.

The corrected replay helper also passed macOS ARM eager/compiled fixtures and an injected timed-compilation rejection. Those fixtures establish harness behavior only, not C2 performance.

### Generated code and AVX-512

Read-only collection after the final replay retained seven generated source files, three C++ wrappers, and symbol-scoped disassembly excerpts from the same cache. All three wrappers release the GIL; the materialization loop handles 16 int32 words (64 bytes) per vector load and emits XOR plus owned-snapshot stores together. The count phase retains four byte masks, int32 row reductions, explicit int64 conversion and native ATen cumsum/boundary extraction. Counts and conditional materialization remain separate calls, not one traversal. The selected generated sources have no OpenMP team directives.

Actual kernel disassembly includes AVX-512 `vmovdqu64`/`vpxord` in materialization and `vpandd`/`vpcmpneqd`/`vpaddd` in counting. Recorded source/library hashes and full generated source/disassembly remain in `artifacts/torch-compile-final-codegen-complete/`; binaries are deliberately excluded from the evidence bundle. There was no separate compiler-command JSON. ISA/GIL evidence does not imply speedup; the complete replay below reports both the gain and regression.
"""
    start = "### Historical exact-image CPU checks on 6eca"
    source = section(source, start, "## Whole-stage CPU replay", tests)
    # The standalone report is self-contained. The PR already has its exact
    # source/environment table, so remove duplicated prose, never raw rows.
    cpu_paragraphs = cpu_markdown.split("\n\n")
    cpu_paragraphs = [paragraph for paragraph in cpu_paragraphs if not paragraph.startswith("C2 host `hu-pdx-90`")]
    cpu_paragraphs = [
        f"Replay module bytes match selected commit `{report['source_commit']}`. Raw JSONL SHA256 `{report['raw_sha256']}`; "
        "the exact f17 control includes per-tensor and scalar-batch ASTs. Full source/layout hashes remain in provenance."
        if paragraph.startswith("Selected standalone commit") else paragraph for paragraph in cpu_paragraphs
    ]
    cpu_body = "\n".join("#" + line if line.startswith("#") else line for line in "\n\n".join(cpu_paragraphs).splitlines())
    source = section(source, "## Whole-stage CPU replay", "## Completed 6eca CPU replay", cpu_body)
    # Include every historical materialize-first sample in the body without
    # repeating the entire self-contained historical report.
    material = raw("TORCH_COMPILE_MATERIALIZE_FIRST_HISTORY.md")
    material_samples = material[material.index("## All 28 raw samples"):material.index("## Copy and ownership accounting")]
    material_copy = material[material.index("## Copy and ownership accounting"):material.index("## Retained evidence and validation limits")]
    material_copy = material_copy[:material_copy.index("The experiment materializes all 112 groups")]
    material_block = "\n<details>\n<summary>Complete materialize-first historical samples and copy counters</summary>\n\n"
    material_block += material_samples.replace("## ", "### ") + material_copy.replace("## ", "### ") + "\n</details>\n\n"
    source = source.replace("## Combined GLM-5.2 GPU validation — PENDING", material_block + "## Combined GLM-5.2 GPU validation — PENDING", 1)
    gpu = """## Combined GLM-5.2 GPU comparison — running, no result yet

The paired NumPy/compiled campaign is running; **no GPU correctness result, weight-update timing comparison, or speedup is claimed yet**. Both arms use combined Miles `283a1cb4e6ed4a3712a39bc5dbd26060465bdcca` plus receiver `2f5fb2a09f08eb9a44c0fed5a28be0bda2942b0f`, the same C2 image and existing five-layer NVFP4 W4A16 recipe. One 8-B300 node is split 4 trainer + 4 rollout GPUs: trainer TP4/EP4, PP1/CP1/ETP1; two rollout engines each TP2/DP2/EP2.

The two run names are `compile-v1-synthetic-numpy-01` and `compile-v1-synthetic-torch-01`. Each has seven rollouts: startup u0, first post-training u1, and all five steady u2–u6 observations. Per-run fresh cache paths are `/hai-workspace/glm52-delta/<run-name>-cache`; backend, output/cache paths are the intended pair differences. The NumPy/compiled flag is the existing selector; no default changes. Execute the sequential frozen plan from `/hai-workspace/glm52-delta`:

```bash
/opt/sglang/bin/python run_compile_campaign.py COMPILE_VALIDATION_PLAN.json
```

Synthetic balanced `sample.index % 2` rewards use ordinary GRPO/optimizer updates. Dataset `zhuzilin/dapo-math-17k` revision `2e65612930298bde4c5d58fd97b3f23a483aaff9`; prepared JSONL 10,490,834 bytes, SHA256 `cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6`. Workload: 8 prompts × 8 samples, response limit 100, temperature 1, LR 1e-6, trainer/rollout seeds 1234/42. Existing precision/shared-expert exclusions and disabled logprob/KL/GPU-weight-equality CI checks remain unchanged. This is transport validation, not task-quality or equal-trajectory evidence.

Admission requires all 14 raw update rows, four trainer ranks per update, nonzero synthetic gradients/changes, successful training/replay checks, exact sources/helper hashes/recipes, and no new measured graph/cache/kernel activity in u2–u6. Initial and u1 compiler counters are retained. The observer covers the entire updater, outside updater CPU/wall and inside outer driver wall; it is not exclusive to the preparation graph. Publication checks retain indexes, shard hashes and checksum metadata; successful reload/checksum checks do not establish GPU byte equality.

The v2 CPU observer retains trainer process CPU, inventoried receiver-tree CPU over rank-0's update, and driver CPU separately. These are not simultaneous whole-host CPU; membership is non-atomic and transient children can escape. Observer and compiler-record costs are not subtracted. Multi-node scaling/bandwidth/amortization remain unmeasured. Once both arms pass strict admission, this body will include all 14 update rows and the complete comparison; raw journals/manifests/logs will link to an exact evidence commit.
"""
    source = section(source, "## Combined GLM-5.2 GPU validation — PENDING", "## Earlier kernel probes and prior art", gpu)
    source = re.sub(r"<!-- Local publication draft\..*?-->\n\n", "", source, flags=re.S)
    source = re.sub(r"^\| Public evidence commit and files \|.*\n", "", source, flags=re.M)
    source = source.replace("; exact final run manifests **PENDING**", "; paired campaign running, no result yet")
    source = source.replace("(**public evidence links pending**)", "(retained raw evidence)")
    source = source.replace("(**public link pending**)", "(retained raw evidence)")
    source = source.replace("Final-candidate replay and GPU validation remain separate pending results.",
                            "The completed selected-source replay is reported separately above; GPU results are not available yet.")
    source = source.replace("A default replacement requires representative whole-stage and end-to-end evidence; this draft claims no such result yet.",
                            "The selected CPU replay improves unchanged updates but regresses changed updates, so NumPy remains the default; the GPU pair has no result yet.")
    # Historical timing/copy tables stay complete. Reuse the source/environment
    # above instead of repeating an identical machine/image paragraph.
    historical_start = source.index("## Completed 6eca CPU replay")
    historical_stop = source.index("## Materialize-first 8×4 replay")
    historical = source[historical_start:historical_stop]
    historical = "\n\n".join(paragraph for paragraph in historical.split("\n\n")
                               if not paragraph.startswith(("The replay ran on C2 host", "Runtime used 32 workers",
                                                            "The 4,690 named tensors contain")))
    historical = historical.replace("## Completed 6eca CPU replay — negative result, separate from final candidate\n\n",
        "## Completed 6eca CPU replay — negative result, separate from final candidate\n\n"
        "Same pinned C2 environment, 32×1 worker budget, 4,690 tensors and 112 representative layouts as the selected replay.\n\n")
    source = source[:historical_start] + historical + source[historical_stop:]
    assert source.splitlines()[:3] == ["## Summary", "", "@humansand"]
    assert "PENDING" not in source and "placeholder" not in source.lower()
    assert len(source.encode()) < 65000, len(source.encode())
    # Preserve captured raw output in full, with its original progress-control bytes.
    assert standalone.rstrip() in source and combined.rstrip() in source and gloo.rstrip() in source
    output = ROOT / "miles-torch-compile-pr-body-initial.md"
    output.write_bytes(source.encode())
    print(json.dumps({"body": str(output), "bytes": len(source.encode()), "cpu_samples": len(report["samples"]),
                      "preflight_exits": {key: check["exit"] for key, check in preflight["checks"].items()},
                      "gpu_status": "Prepared for publication after pair launch; no GPU result claimed"}))


if __name__ == "__main__":
    main()
