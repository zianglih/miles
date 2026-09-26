#!/usr/bin/env python3
"""Admit and render the complete selected word-two-phase CPU replay."""

import json
from pathlib import Path

from build_materialize_first_history import table
import publish_compile_evidence as evidence

ROOT = Path(__file__).resolve().parent
RAW = "artifacts/torch-compile-word-two-phase-32x1-replay.jsonl"
COMMIT = "6fb0e2a9d81fc343863ba6015512fe93992892e4"


def build():
    _, admission = evidence.validate_replay(ROOT, RAW, COMMIT, ROOT / "miles-torch-compile")
    raw = [json.loads(line) for line in (ROOT / RAW).read_text().splitlines()]
    provenance = raw[0]
    assert provenance["arguments"]["stage_workers"] == 32 and provenance["arguments"]["kernel_threads"] == 1
    samples = [row for row in raw if row["event"] == "sample"]
    warmups = [row for row in raw if row["event"] == "compile_warmup"]
    cases = ("native-unchanged", "synthetic-v1-v2")
    summaries = {row["case"]: row for row in raw if row["event"] == "summary"}
    report = {"validated_final_cpu_replay": True, **admission, "provenance": provenance,
              "workloads": [row for row in raw if row["event"] == "workload"],
              "warmups": warmups, "samples": samples, "complete": raw[-1]}
    lines = ["# Selected word-two-phase 32×1 CPU replay", "",
             "All 28 warmup/measured samples completed with exact byte/count/checksum verification and no timed compiler activity. "
             "The unchanged case used 50.39% less wall time and 48.54% less process CPU than the exact f17 control. "
             "The changed case used 46.44% more wall time and 27.70% more process CPU. NumPy remains the default; "
             "these results support an opt-in experiment, not replacement of the existing backend or a claim of end-to-end speedup.", "",
             "## Exact source, environment and input", "",
             f"Selected standalone commit `{COMMIT}`; module SHA256 `{admission['module_sha256']}`. "
             "The replay loaded the frozen `delta_preparation_word_two_phase_variant.py`; its archived module is byte-identical "
             "to that committed production module. The exact control is `f17ba4bce13bf7d357e7560182dc859c41a7cb37`, "
             "including both `_diff_and_compress` and `_diff_and_compress_batch` ASTs. "
             f"Raw JSONL SHA256 `{admission['raw_sha256']}`; full source/layout/control hashes remain in its provenance.", "",
             "C2 host `hu-pdx-90`: 2 × Intel Xeon 6776P, 64 cores/socket, SMT2, 256 logical CPUs, four NUMA nodes, "
             "AVX-512 available. Image `radixark/miles:dev-202609251434`, amd64 digest "
             "`sha256:7c4c6c8cc9e76be893941e064f0e43924ef9258b04da365c8009709592ff0984`. "
             "Python 3.12.3; PyTorch `2.13.0+cu130`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`; "
             "NumPy 2.3.5, zstandard 0.25.0, xxhash 3.7.1. Both arms use 32 workers; generated candidate kernels use "
             "one thread, `cpp_wrapper=True`, and `cpp.dynamic_threads=False`. Runtime reports Torch intra/inter-op "
             "threads 1/128 and CPU affinity `0–1,11–129,138–255`.", "",
             "The input has 4,690 emitted canonical tensors and 17,900,804,608 valid bytes; `.input_scale` is excluded "
             "from 6,226 checkpoint tensors. Model source: `Pinaster/GLM-5.2_5layer` revision "
             "`1c749139f70e158e4420ba67f342bef1de2e650d`, prepared as NVFP4 using the #3711 workflow. "
             "The unchanged case uses identical canonical bytes. The changed case reconstructs the retained synthetic "
             "published v1→v2 data: 79,500,803 differing byte positions across 2,646 tensors. Wholly unchanged tensors "
             "hold 465,303,040 bytes (2.5993%); changed-byte density is 0.4441%, a different measure.", "",
             "Sorted-name layouts form 112 representative groups, limited to 128 MiB padded storage except a single "
             "larger tensor. Weight rows are 4,096 bytes; FP32 scalar scales have a separate compact four-byte-row group. "
             "Padding is 44,544 bytes, and the two preallocated old/new states own 35,801,698,304 bytes. The f17 control "
             "submits 3,155 tasks, including one exact compact batch for 1,536 scalars. These are reproducible CPU layouts, "
             "not a capture of live iterator subgroup boundaries.", "",
             "## Method and complete medians", "",
             "Each case starts with serial warmup of both graphs against all actual old/incoming layouts, followed by "
             "two alternating-order warmup pairs and five measured pairs. Medians use every measured sample, with no "
             "sample removal. CPU is process CPU-seconds, not utilization; compiler children are excluded.", "",
             "Timed work includes submission/drain, exact counts, owned XOR/snapshot materialization when changed, "
             "lease return, zstd and checksum. Candidate replay diagnostics also build per-name views and compute "
             "copy/materialized-bucket counters; production caches payload-byte totals, omits unused encoder diagnostic "
             "accumulation, and skips unchanged snapshot-view reinstallation. These replay costs are retained in the "
             "numbers rather than subtracted. The CPU replay does not measure those production encoder savings.", ""]
    lines += table(["Case", "f17 wall s", "Compiled wall s", "Wall change", "f17 CPU s", "Compiled CPU s", "CPU change"], [
        [case, f"{summaries[case]['variants']['exact_f17']['median_wall_s']:.9f}",
         f"{summaries[case]['variants']['packed_stage']['median_wall_s']:.9f}",
         f"{-100*summaries[case]['stage_wall_reduction_fraction']:+.2f}%",
         f"{summaries[case]['variants']['exact_f17']['median_process_cpu_s']:.9f}",
         f"{summaries[case]['variants']['packed_stage']['median_process_cpu_s']:.9f}",
         f"{-100*summaries[case]['stage_cpu_reduction_fraction']:+.2f}%"] for case in cases])
    lines += ["", "## Initialization and compiler activity", "",
              "The first workload started with an empty task-specific disk cache; the second reused the initialized "
              "compiler. Layout warmup includes full execution/materialization and is not pure compilation latency. "
              "An empty disk cache does not establish absence of every external/runtime cache.", ""]
    lines += table(["Case", "Actual layouts warmed", "Wall s", "Process CPU s", "New graphs", "New generated kernels"], [
        [row["case"], len(row["actual_layouts_warmed"]), f"{row['wall_s']:.9f}", f"{row['process_cpu_s']:.9f}",
         row["new_compile_counters"].get("stats.unique_graphs", 0), row["new_compile_counters"].get("inductor.generated_kernel_count", 0)]
        for row in warmups])
    lines += ["", "All 28 timed counter deltas are empty. The first warmup records four captured graphs and 26 generated "
              "kernels across layout specializations; these counts do not prove one fused loop or performance. "
              "The count phase uses word-lane XOR/masks, int32 row reductions, and an int64 prefix/boundary gather. "
              "Only groups with nonzero counts run the second compiled word-view XOR/snapshot phase.", "",
              "## All 28 raw samples", "",
              "Rows preserve observed execution order. `exact_f17` is the control; `packed_stage` is the selected module. "
              "Every row passed exact byte/count/checksum verification and had zero new compiler counters.", ""]
    lines += table(["Sequence", "Case", "Pair", "Phase", "Variant", "Wall s", "Process CPU s"], [
        [index, row["case"], row["iteration"], "warmup" if row["warmup"] else "measured", row["variant"],
         f"{row['wall_s']:.9f}", f"{row['process_cpu_s']:.9f}"] for index, row in enumerate(samples)])
    copy_fields = ("copied_unchanged_bytes", "copied_padding_bytes", "snapshot_copy_bytes", "compressed_payload_bytes")
    copy_rows = []
    for case in cases:
        for variant in ("exact_f17", "packed_stage"):
            selected = [row for row in samples if row["case"] == case and row["variant"] == variant]
            assert len({tuple(row[field] for field in copy_fields) for row in selected}) == 1
            if variant == "packed_stage":
                assert all(row["lease_returns"] == 112 and row["lease_return_once_before_compression"] is True for row in selected)
            copy_rows.append([case, variant, *(selected[0][field] for field in copy_fields)])
    lines += ["", "## Copy and wire-work counters", "", "Values are constant across the seven samples in each case/variant.", ""]
    lines += table(["Case", "Variant", "Copied unchanged bytes", "Copied padding bytes", "Snapshot copy bytes", "Compressed payload bytes"], copy_rows)
    lines += ["", "The unchanged candidate materializes zero groups, preserves old snapshot identity and emits no payload. "
              "The changed candidate materializes 110 of 112 groups, including 465,284,608 unchanged valid bytes inside "
              "those groups. Both arms produce 182,769,068 compressed worker payload bytes; safetensors headers/index "
              "are excluded. f17 still copies the 6,144-byte compact scalar batch when it is unchanged. Snapshot-copy "
              "counters exclude input reads, XOR output writes, count buffers and allocator costs, so they are not total memory traffic. "
              "Each candidate sample returns 112 leases exactly once before compression; the control returns 3,155 "
              "tensor/scalar-group leases.", "",
              "## Exact replay command and retained inputs", "",
              "Run from `/hai-workspace/glm52-delta` inside the recorded image. The frozen variant file is identical to "
              "the selected production module. The full launch script, including preceding exactness checks, is `run_word_two_phase_cpu.sh`.", "",
              "```bash", "CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 timeout --kill-after=10 1200 \\",
              "  /opt/sglang/bin/python benchmark_delta_preparation.py \\",
              "  --miles-path miles --preparation-module delta_preparation_word_two_phase_variant.py \\",
              "  --checkpoint models/GLM-5.2_5layer-NVFP4 \\",
              "  --synthetic-source artifacts/synthetic-balanced-disk-delta-01/delta-publication \\",
              "  --expected-tensors 4690 --block-bytes 4096 --bucket-mib 128 \\",
              "  --baseline-workers 32 --stage-workers 32 --kernel-threads 1 \\",
              "  --warmups 2 --repeats 5 \\",
              "  --output artifacts/torch-compile-word-two-phase-32x1-replay.jsonl \\",
              "  > artifacts/torch-compile-word-two-phase-32x1-replay.log 2>&1", "```", "",
              "No explicit cache argument was supplied; the helper used "
              "`artifacts/torch-compile-word-two-phase-32x1-replay-inductor-cache`. "
              f"Replay helper SHA256 `{provenance['script_sha256']}`; reconstruction helper SHA256 "
              f"`{provenance['replay_helper_sha256']}`; layout SHA256 `{provenance['layout_sha256']}`.", "",
              f"Retained raw files: `{RAW}`, adjacent `.log`/`.exit`, and the adjacent `-sources/` folder. "
              "Their complete provenance includes source hashes, compiler counters, tensor/layout distributions and "
              "old/new checksum manifests. Recreating training does not guarantee the same byte workload; use the "
              "retained published v1/v2 artifacts.", "",
              "## Limits", "",
              "Checkpoint reads/reconstruction, initial packing and allocations, worker startup, warmup and verification "
              "are outside sample timing. No GPU gather/D2H, pinned-pool backpressure, production layout selection, "
              "snapshot installation, filesystem publication or receiver reload is measured. This comparison supports "
              "neither GPU equality nor multi-node bandwidth/scaling or amortization claims. CPU replay includes its "
              "own diagnostics and does not substitute for the separately matched NumPy/compiled GPU campaign."]
    return report, "\n".join(lines) + "\n"


def main():
    report, markdown = build()
    (ROOT / "TORCH_COMPILE_FINAL_CPU_RESULTS.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (ROOT / "TORCH_COMPILE_FINAL_CPU_RESULTS.md").write_text(markdown)
    print(json.dumps({"validated_final_cpu_replay": True, "samples": len(report["samples"]),
                      "raw_sha256": report["raw_sha256"], "markdown_bytes": len(markdown.encode())}))


if __name__ == "__main__":
    main()
