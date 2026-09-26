#!/usr/bin/env python3
"""Render the preserved, uncommitted 8x4 negative CPU experiment only."""

import json
from pathlib import Path

import publish_compile_evidence as evidence

ROOT = Path(__file__).resolve().parent
RAW = "artifacts/torch-compile-materialize-first-8x4-replay.jsonl"
MODULE_SHA = "09fc7f41579c5e517a63263bb6a5763c08ad769a16e34dd658f3255ec900dc5d"
OUTPUT = ROOT / "TORCH_COMPILE_MATERIALIZE_FIRST_HISTORY.md"


def table(headers, rows):
    return ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows
    ]


def main():
    _, admission = evidence.validate_replay(ROOT, RAW, f"sha256:{MODULE_SHA}", ROOT / "miles-torch-compile",
                                            allow_uncommitted=True)
    raw = [json.loads(line) for line in (ROOT / RAW).read_text().splitlines()]
    provenance = raw[0]
    samples = [row for row in raw if row["event"] == "sample"]
    warmups = [row for row in raw if row["event"] == "compile_warmup"]
    cases = ("native-unchanged", "synthetic-v1-v2")
    summaries = {row["case"]: row for row in raw if row["event"] == "summary"}
    assert provenance["arguments"]["stage_workers"] == 8 and provenance["arguments"]["kernel_threads"] == 4
    assert (ROOT / "delta_preparation_materialize_first_variant.py").read_bytes() == (
        ROOT / "artifacts/torch-compile-materialize-first-8x4-replay-sources/delta_preparation.py").read_bytes()
    lines = ["# Materialize-first 8×4 CPU replay: retained negative result", "",
             "This uncommitted experiment completed with exit code 0 and verified all 28 warmup/measured samples. "
             "It was rejected for the host CPU objective: changed weights took 30.70% more process CPU and 11.63% more wall time "
             "than the exact f17 control. Unchanged weights took 184.46% more process CPU and 265.04% more wall time. "
             "This report is historical evidence; it does not describe the final PR implementation or its pending validation.", "",
             "## Exact source and environment", "",
             f"The experimental module is `delta_preparation_materialize_first_variant.py`, SHA256 `{MODULE_SHA}`. "
             "It has no source commit. The archived `-sources/delta_preparation.py` matches the project file byte for byte. "
             f"The raw JSONL SHA256 is `{admission['raw_sha256']}`. "
             f"Replay helper SHA256 `{provenance['script_sha256']}`; reconstruction helper SHA256 `{provenance['replay_helper_sha256']}`.", "",
             "C2 host `hu-pdx-90`: 2 × Intel Xeon 6776P, 64 cores per socket, SMT2, 256 logical CPUs, "
             "four NUMA nodes, AVX-512 available. Image `radixark/miles:dev-202609251434`, amd64 digest "
             "`sha256:7c4c6c8cc9e76be893941e064f0e43924ef9258b04da365c8009709592ff0984`. "
             "Python 3.12.3; PyTorch `2.13.0+cu130`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`; "
             "NumPy 2.3.5; zstandard 0.25.0; xxhash 3.7.1. Runtime reported one Torch intra-op thread, "
             "128 inter-op threads, and CPU affinity `0–1,11–129,138–255`.", "",
             "The control uses the exact committed f17 per-tensor and scalar-batch workers with a 32-worker pool. "
             "The experiment uses eight preparation/compression workers and four generated-kernel threads per worker. "
             "The configured generated-kernel budget is 32 threads, but compression concurrency differs (8 versus 32); "
             "this is a comparison of those complete configurations, not an isolated kernel comparison.", "",
             "Exact C2 launch from `/hai-workspace/glm52-delta`:", "", "```bash",
             "CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 timeout --kill-after=10 1200 \\",
             "  /opt/sglang/bin/python benchmark_delta_preparation.py \\",
             "  --miles-path miles \\",
             "  --preparation-module delta_preparation_materialize_first_variant.py \\",
             "  --checkpoint models/GLM-5.2_5layer-NVFP4 \\",
             "  --synthetic-source artifacts/synthetic-balanced-disk-delta-01/delta-publication \\",
             "  --expected-tensors 4690 --block-bytes 4096 --bucket-mib 128 \\",
             "  --baseline-workers 32 --stage-workers 8 --kernel-threads 4 \\",
             "  --warmups 2 --repeats 5 \\",
             "  --output artifacts/torch-compile-materialize-first-8x4-replay.jsonl \\",
             "  > artifacts/torch-compile-materialize-first-8x4-replay.log 2>&1", "```", "",
             "No explicit cache argument was supplied. The helper selected "
             "`artifacts/torch-compile-materialize-first-8x4-replay-inductor-cache` and recorded "
             "`cache_initially_empty=true`.", "",
             "## Workload and design tradeoff", "",
             "The replay selects 4,690 named tensors (17,900,804,608 valid bytes) from 6,226 canonical tensors, excluding "
             "`.input_scale`. Stable sorted names produce 112 representative groups with at most 128 MiB of padded "
             "storage unless a single tensor is larger. Weight rows use 4,096 bytes; the compact FP32 scalar group uses "
             "4-byte rows. Padding totals 44,544 bytes and the two preallocated input states own 35,801,698,304 bytes. "
             "The control submits 3,155 tasks, including one exact batch for 1,536 FP32 scalars (6,144 bytes). "
             "These layouts are representative, not captured live production subdivisions.", "",
             "The first compiled phase materializes an owned XOR and snapshot using int32 word views. The second phase "
             "counts nonzero byte lanes from that owned XOR, reduces rows, and derives per-name int64 counts using a "
             "prefix sum and boundary gathers. The phase boundary avoids recomputing `new ^ old` in a separate reduction "
             "loop, but it always materializes every nonempty group, including completely unchanged groups. "
             "Only names with nonzero counts proceed to native compression/checksum. Graph/kernel totals alone do not prove fusion or acceleration.", "",
             "The unchanged case compares identical bytes. Synthetic v1→v2 changes 79,500,803 byte positions across "
             "2,646 tensors; entirely unchanged tensors contain 465,303,040 valid bytes. Both arms produce "
             "182,769,068 compressed worker payload bytes for the changed case, excluding safetensors headers/index.", "",
             "## Complete medians", "",
             "Medians use all five measured pairs (iterations 2–6), with no sample removal. CPU means process CPU-seconds, "
             "not CPU utilization. Variant order alternates per pair; both warmup pairs remain in the raw table below.", ""]
    medians = []
    for case in cases:
        summary = summaries[case]
        control, stage = (summary["variants"][name] for name in ("exact_f17", "packed_stage"))
        medians.append([case, f"{control['median_wall_s']:.9f}", f"{stage['median_wall_s']:.9f}",
                        f"{-100 * summary['stage_wall_reduction_fraction']:+.2f}%",
                        f"{control['median_process_cpu_s']:.9f}", f"{stage['median_process_cpu_s']:.9f}",
                        f"{-100 * summary['stage_cpu_reduction_fraction']:+.2f}%"])
    lines += table(["Case", "f17 wall s", "8×4 wall s", "Wall change", "f17 CPU s", "8×4 CPU s", "CPU change"], medians)
    lines += ["", "## Cold and already initialized layout warmup", "",
              "Before fanout, both compiled phases run serially against every actual old/incoming layout. The first case "
              "starts with an empty task-specific disk cache; the second reuses the initialized compiler. These times "
              "include full-layout execution/materialization and are not pure compilation latency. Calling-process CPU "
              "excludes compiler-child CPU. The runtime may still have other caches; the empty disk cache is a recorded condition.", ""]
    lines += table(["Case", "Layouts warmed", "Wall s", "Process CPU s", "New graphs", "New generated kernels"], [
        [row["case"], len(row["actual_layouts_warmed"]), f"{row['wall_s']:.9f}", f"{row['process_cpu_s']:.9f}",
         row["new_compile_counters"].get("stats.unique_graphs", 0), row["new_compile_counters"].get("inductor.generated_kernel_count", 0)]
        for row in warmups])
    lines += ["", "Every timed sample has empty compiler counter deltas. Full warmup counter dictionaries remain in the raw JSONL.", "",
              "## All 28 raw samples", "",
              "Rows preserve their observed execution order. `exact_f17` is the control; `packed_stage` is this uncommitted "
              "materialize-first experiment. All rows passed exact byte/count/checksum verification and had no timed compiler activity.", ""]
    lines += table(["Sequence", "Case", "Pair", "Phase", "Variant", "Wall s", "Process CPU s"], [
        [index, row["case"], row["iteration"], "warmup" if row["warmup"] else "measured", row["variant"],
         f"{row['wall_s']:.9f}", f"{row['process_cpu_s']:.9f}"] for index, row in enumerate(samples)])
    fields = ("copied_unchanged_bytes", "copied_padding_bytes", "snapshot_copy_bytes", "compressed_payload_bytes", "lease_returns")
    copy_rows = []
    for case in cases:
        for variant in ("exact_f17", "packed_stage"):
            selected = [row for row in samples if row["case"] == case and row["variant"] == variant]
            assert len({tuple(row[field] for field in fields) for row in selected}) == 1
            if variant == "packed_stage":
                assert all(row["materialized_buckets"] == 112 and row["lease_return_once_before_compression"] is True for row in selected)
            copy_rows.append([case, variant, *(selected[0][field] for field in fields)])
    lines += ["", "## Copy and ownership accounting", "", "These raw counters are constant across all seven samples of each case/variant.", ""]
    lines += table(["Case", "Variant", "Copied unchanged bytes", "Copied padding bytes", "Snapshot copy bytes", "Compressed payload bytes", "Lease returns"], copy_rows)
    lines += ["", "The experiment materializes all 112 groups in both workloads: each call copies 17,900,849,152 bytes "
              "into owned snapshot storage and also allocates/materializes an XOR buffer of the same padded size. "
              "The table's snapshot-copy counter is not total memory traffic; it excludes reads, XOR output writes, "
              "count buffers and allocator effects. In the unchanged case, f17 only snapshots the 6,144-byte compact scalar batch. "
              "This extra full-model materialization explains a clear design cost, though these measurements do not "
              "attribute all timing differences to one operation.", "",
              "The experimental stage returns each of its 112 input leases once before native compression. "
              "The exact control returns 3,155 tensor/scalar-group leases. Inputs remain unchanged; owned output "
              "snapshots, decoded XOR payloads, per-name counts and checksums are verified outside timing.", "",
              "## Retained evidence and validation limits", "",
              f"- Raw replay: `{RAW}`, adjacent `.log` and `.exit`, and the adjacent `-sources/` folder.",
              "- Layout SHA256: `c1b72468d9a164abaabf6689208d01971f964a9adfe10e85aeeb6bbdc8244fe9`.",
              "- Exact control: `f17ba4bce13bf7d357e7560182dc859c41a7cb37`; archived per-tensor/scalar-batch ASTs and full source hashes are retained.",
              "- C2 exactness probe: `artifacts/torch-compile-materialize-first-exactness.log`. Its complete output is:", "", "```text",
              (ROOT / "artifacts/torch-compile-materialize-first-exactness.log").read_text().rstrip(), "```", "",
              "Timed work includes worker submission/drain, native preparation/count/materialization allocations, "
              "zstd/checksum work and lease return. Reads/reconstruction/packing, initial allocations, worker startup, "
              "layout warmup and verification are outside timing. GPU gather/D2H, actual pinned-pool backpressure, "
              "production layout selection, snapshot installation, file publication and receiver reload are unmeasured. "
              "This CPU-only replay establishes neither GPU equality nor multi-node benefit. It remains separate "
              "from the earlier 6eca replay and the final candidate's CPU/GPU validation.", "",
              "Publisher admission for this experiment:", "", "```text",
              f"--historical-replay {RAW}=sha256:{MODULE_SHA}", "```", "",
              "The source is explicitly labeled uncommitted; no source commit is inferred from the archive hash."]
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(OUTPUT), "raw_samples": len(samples), "source_kind": admission["source_kind"],
                      "raw_sha256": admission["raw_sha256"]}))


if __name__ == "__main__":
    main()
