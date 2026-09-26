#!/usr/bin/env python3
"""Audit a complete third-PR evidence bundle; stage only with explicit --stage.

No Git mutations, commits, or pushes. The original glm52_delta_sync folder is
never a destination. Raw input bytes are preserved, not rewritten or redacted.
The strict GPU reporter is rerun in memory against retained local payloads.
Only explicitly admitted UTF-8 text files enter the public bundle.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

import build_compiled_validation_report as reporter
from compile_campaign_metadata import metadata_record, resolve_metadata
from compile_evidence_attribution import current_hosted_ci, selected_preflight, validate_attribution, validate_compatibility

ROOT = Path(__file__).resolve().parent
DESTINATION = Path("miles-evidence/benchmark_artifacts/glm52_delta_compile")
NEGATIVE_REPLAY = "artifacts/torch-compile-stage-replay-v1.jsonl"
NEGATIVE_SOURCE = "6eca79a001e2e2c9b62385b1397f1ad0fc800e81"
MATERIALIZE_REPLAY = "artifacts/torch-compile-materialize-first-8x4-replay.jsonl"
MATERIALIZE_SOURCE = "sha256:09fc7f41579c5e517a63263bb6a5763c08ad769a16e34dd658f3255ec900dc5d"
BASELINE_SOURCE = "f17ba4bce13bf7d357e7560182dc859c41a7cb37"
TENSOR_COUNT = 4690
PAYLOAD_BYTES = 17900804608
MAX_FILE_BYTES = 16 * 1024**2
MAX_BUNDLE_BYTES = 128 * 1024**2
SUFFIXES = {".py", ".sh", ".json", ".jsonl", ".md", ".txt", ".log", ".exit", ".csv", ".cpp", ".h", ".patch"}
FILES = (
    "publish_compile_evidence.py", "test_publish_compile_evidence.py",
    "audit_compile_evidence_readiness.py",
    "build_compiled_validation_report.py", "test_build_compiled_validation_report.py",
    "build_cpu_benchmark_report.py", "build_benchmark_report.py",
    "test_build_cpu_benchmark_report.py", "test_weight_sync_cpu_v2.py",
    "test_weight_sync_probe_compile.py", "run_compile_campaign.py", "run_compiled_preflight.py",
    "compile_campaign_metadata.py", "test_compile_campaign_metadata.py", "test_run_compile_campaign.py",
    "compile_evidence_attribution.py", "test_compile_evidence_attribution.py",
    "COMPILE_SOURCE_ATTRIBUTION.json",
    "COMPILE_CI_COMPATIBILITY.json", "COMPILE_CI_COMPATIBILITY.md",
    "COMPILE_SECRET_SCAN_FIXTURE_ALLOWLIST.json",
    "capture_compile_pr_ci.py", "test_capture_compile_pr_ci.py",
    "artifacts/capture-compile-pr-ci-helper-tests.log", "artifacts/compile-attribution-helper-tests.log",
    "RECOVERY_20260926.md", "recover_weight_sync_inputs.py", "verify_recovered_nvfp4_headers.py",
    "finalize_recovered_inputs.py", "check_recovered_inputs.py",
    "artifacts/recovery-inputs-ready.json", "artifacts/recovery-original-canonical-headers.json",
    "artifacts/recovery-nvfp4-header-verification.json", "artifacts/input-backup-20260926T021916Z-4640.json",
    "artifacts/recovery-20260926T021916Z-4640-input-identity.json",
    "artifacts/compile-campaign-frozen-inputs.json", "artifacts/compile-v2-render-contract.json",
    "artifacts/compile-v2-render-numpy-01/manifest.json", "artifacts/compile-v2-render-torch-compile-01/manifest.json",
    "preserve_compile_campaign.py", "artifacts/compile-preserver-api-fixture-result.json",
    "summarize_weight_sync.py", "summarize_weight_sync_cpu_v2.py", "summarize_training_evidence.py",
    "benchmark_delta_preparation.py", "benchmark_sender_replay.py",
    "validate_compiled_sender_gloo.py", "prepare_weight_sync.py",
    "probe_delta_cpu_kernels.py", "probe_delta_cpu_word_counts.py", "probe_delta_cpu_fused_prepare.py",
    "delta_preparation_fused_variant.py", "delta_preparation_materialize_first_variant.py",
    "test_fused_delta_candidate.py", "test_delta_preparation_materialize_first_candidate.py",
    "COMPILE_VALIDATION_PLAN.json", "CAMPAIGN.json", "environment-image.json",
    "COMPILE_VALIDATION_PLAN_V1.json", "COMPILE_GPU_V1_INTERRUPTION.md",
    *(f"artifacts/compile-v1-synthetic-numpy-01-{name}.json" for name in (
        "summary", "cpu-summary", "training-summary", "manifest-startup")),
    "COMPILED_VALIDATION_RESULTS.json", "COMPILED_VALIDATION_RESULTS.md",
    "HARNESS.md", "CPU_OBSERVATION_V2.md", "TORCH_COMPILE_STAGE_REPLAY_V1.md", "COMPILE_EVIDENCE_PUBLISHING.md",
    "TORCH_COMPILE_MATERIALIZE_FIRST_HISTORY.md", "build_materialize_first_history.py",
    "TORCH_COMPILE_FINAL_CPU_RESULTS.md", "TORCH_COMPILE_FINAL_CPU_RESULTS.json", "build_final_cpu_stage_report.py",
    "build_compile_initial_pr_body.py", "miles-torch-compile-pr-body-assembly-template.md",
    "finalize_compile_pr_body.py", "test_finalize_compile_pr_body.py",
    "run_word_two_phase_cpu.sh", "run_final_compiled_cpu_tests.sh", "delta_preparation_word_two_phase_variant.py",
    "test_delta_preparation_word_two_phase_candidate.py",
    "artifacts/host-cpu.json", "artifacts/cpu-observer-fast-final-calibration.json",
    "artifacts/cpu-observer-v2-fast-final-tests-c2.log",
    "artifacts/compile-probe-attribution-tests.log",
    "artifacts/compile-validation-helper-tests.log", "artifacts/compile-evidence-publisher-tests.log",
    "artifacts/compile-final-body-helper-tests.log",
    "artifacts/compile-v2-metadata-runner-tests.log", "artifacts/compile-v2-evidence-helper-tests.log",
    "artifacts/compile-replacement-host-cpu.json",
    "artifacts/torch-compile-stage-c2-v3-tests.log",
    "artifacts/torch-compile-kernel-probe.jsonl", "artifacts/torch-compile-word-count-probe.jsonl",
    "artifacts/torch-compile-fused-prepare-probe.jsonl", "artifacts/torch-compile-fused-exactness.log",
    "artifacts/torch-compile-materialize-first-exactness.log",
    "artifacts/torch-compile-final-production-tests.log", "artifacts/torch-compile-final-production-tests.exit",
    "artifacts/torch-compile-final-production-tested-head.txt", "artifacts/torch-compile-final-source-sha256.txt",
    "artifacts/torch-compile-word-two-phase-unit.log",
    "artifacts/compile-final-preflight-01/manifest.json", "artifacts/compile-final-preflight-01/focused-tests.log",
    "artifacts/compile-final-preflight-01/gloo-failures.log",
    "HOSTED_CI_REVIEW.md",
    *(f"artifacts/compile-pr-ci-3720-6fb0e2a/{name}" for name in (
        "README.md", "final-review.json", "summary.json", "latest.json",
        "failed-check-run.json", "failed-job-108312293487.json", "failed-job-annotations.json",
        "failed-shard-1.log", "hosted-shard-1-pytest-argv.json", "run-36209271034-artifacts.json",
        "local-torch211-delta-tests.log", "local-torch211-relative-cache-error.log",
    )),
    "artifacts/torch-compile-stage-benchmark-corrected/compiled-fixture.jsonl",
    "artifacts/torch-compile-stage-benchmark-corrected/compiled-fixture.log",
    "artifacts/torch-compile-stage-benchmark-corrected/eager-fixture.jsonl",
    "artifacts/torch-compile-stage-benchmark-corrected/measured-compilation-rejection-fixture.json",
    *(f"artifacts/torch-compile-final-codegen-complete/{name}" for name in (
        "CODEGEN.md", "proof.json",
        "c2wawido3dtafcfncl27mn6bwebanaump7ohuzk3i7yggo7sw5je.main.cpp",
        "c76clulz2qgtjthxdgzq35qxoj4ofhmjfihkgfpldivebo3sm765.main.cpp",
        "cxw6jn3k65kmhrieuem3rjnbovnqxluqmhsfjm4y57c2osnl34tm.main.cpp",
        "c2wawido3dtafcfncl27mn6bwebanaump7ohuzk3i7yggo7sw5je.main.so.avx512.txt",
        "c76clulz2qgtjthxdgzq35qxoj4ofhmjfihkgfpldivebo3sm765.main.so.avx512.txt",
        "cxw6jn3k65kmhrieuem3rjnbovnqxluqmhsfjm4y57c2osnl34tm.main.so.avx512.txt",
        "c7v7ksfyhytxghbbkzs7tow7v3avlzxx5vivi64567xtiziqgtf4.py",
        "cbcajiukrcdieopikltodknnw4sf2rnqiib5hlrythekremhmwup.py",
        "clhouobsam6grgzhamodqyd2hpue362sjkjaouif6lgb3xoh43a3.py",
        "cmg7b37g2xilek63ny7jq3dkpg3wiy52e5jciv4wfxfbja3mzqkw.py",
    )),
)
RUN_FILES = (
    "manifest.json", "summary.json", "training-evidence.json", "training-steps.csv",
    "driver.jsonl", "updates.csv", "cpu-summary.json", "cpu-updates.csv", "cpu-driver.jsonl",
    *(f"trainer-rank{rank}.jsonl" for rank in range(4)),
    *(f"cpu-trainer-rank{rank}.jsonl" for rank in range(4)),
)
SOURCE_FILES = (
    "miles/utils/delta_preparation.py", "miles/utils/arguments.py",
    "miles/backends/training_utils/weight_update/packed_delta.py",
    "miles/backends/training_utils/weight_update/protocols/delta.py",
    "tests/fast/utils/test_delta_preparation.py",
    "tests/fast/backends/training_utils/weight_update/test_delta_compiled_cpu.py",
)
REPLAY_SOURCES = (
    "benchmark_delta_preparation.py", "benchmark_sender_replay.py", "delta_preparation.py",
    "exact-f17-delta.py", "exact-f17-disk_delta.py", "layouts.json",
)
SECRET_PATTERNS = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("cloud-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("service-token", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{25,})\b")),
    ("bearer-value", re.compile(r"(?i)authorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("credential-url", re.compile(r"https?://[^\s/:@]+:[^\s/@]+@")),
    ("credential-assignment", re.compile(
        r"(?i)\b(?:[a-z0-9]+[_-])*(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret[_-]?(?:access[_-]?)?key)"
        r"[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9+/_=.-]{20,})")),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative_name(name):
    path = Path(name)
    require(not path.is_absolute() and path.parts and ".." not in path.parts,
            f"Expected a task-relative file path: {name}")
    require(path.suffix in SUFFIXES, f"Non-text or unsupported file extension: {name}")
    require(not any(part in {".git", ".venv", "models", "datasets", "dump_details", "rollout-checkpoint"}
                    for part in path.parts), f"Excluded artifact path: {name}")
    return path.as_posix()


def check_text(name, data):
    require(len(data) <= MAX_FILE_BYTES, f"Text file exceeds {MAX_FILE_BYTES} bytes: {name}")
    require(b"\0" not in data, f"Binary content is forbidden: {name}")
    content = data.decode("utf-8")
    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(content):
            if label == "credential-url":
                fixture = json.loads((ROOT / "COMPILE_SECRET_SCAN_FIXTURE_ALLOWLIST.json").read_bytes())
                line_text = content[content.rfind("\n", 0, match.start()) + 1:content.find("\n", match.end())]
                if (fixture["files"].get(name) == digest(data)
                        and digest(match.group().encode()) == fixture["matched_value_sha256"]
                        and fixture["test_node"] in line_text):
                    continue
            line = content.count("\n", 0, match.start()) + 1
            # Never reproduce a potentially sensitive matched value.
            raise ValueError(f"Secret scan rejected {name}:{line} ({label}); raw bytes were not altered")


def read_local(root, name):
    name = relative_name(name)
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), f"Source escapes project: {name}")
    require(not any(parent.is_symlink() for parent in (path, *path.parents)
                    if parent.is_relative_to(root)), f"Symlink source is forbidden: {name}")
    require(path.is_file(), f"Required evidence is missing: {name}")
    require(path.stat().st_size <= MAX_FILE_BYTES, f"Text file exceeds {MAX_FILE_BYTES} bytes: {name}")
    data = path.read_bytes()
    check_text(name, data)
    return data


def git_file(repo, commit, name):
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Source snapshot requires a full commit")
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"])


def validate_replay(root, name, source_reference, repo, *, allow_uncommitted=False):
    """Admit all 28 raw samples and exact archived sources, without rewriting them."""
    path = Path(relative_name(name))
    require(path.suffix == ".jsonl", "Replay must be a JSONL file")
    raw = read_local(root, name)
    events = [json.loads(line) for line in raw.decode().splitlines() if line]
    require(events[0]["event"] == "provenance" and events[-1]["event"] == "complete",
            f"Incomplete replay: {name}")
    require(events[-1]["all_results_verified"] is True
            and events[-1]["all_measured_samples_steady_state"] is True, "Replay verification failed")
    provenance = events[0]
    arguments = provenance["arguments"]
    require(provenance["selected_tensor_count"] == TENSOR_COUNT
            and provenance["canonical_tensor_count"] == 6226
            and arguments["exclude_suffix"] == [".input_scale"] and provenance["payload_bytes"] == PAYLOAD_BYTES,
            "Replay is not the complete 4,690-tensor, 17,900,804,608-byte workload")
    require(arguments["fixture"] is False and arguments["no_compile"] is False,
            "Final replay must use compiled production-size data")
    require(arguments["block_bytes"] == 4096 and arguments["bucket_mib"] == 128
            and arguments["baseline_workers"] == 32
            and arguments["stage_workers"] * arguments["kernel_threads"] == 32 and arguments["warmups"] == 2
            and arguments["repeats"] == 5, "Unexpected replay worker/layout/sample contract")
    require(provenance["baseline"]["commit"] == BASELINE_SOURCE
            and provenance["baseline"]["working_tree"] is False, "Replay baseline source drift")
    expected = Counter((case, iteration, variant) for case in ("native-unchanged", "synthetic-v1-v2")
                       for iteration in range(7) for variant in ("exact_f17", "packed_stage"))
    samples = [row for row in events if row["event"] == "sample"]
    require(Counter((row["case"], row["iteration"], row["variant"]) for row in samples) == expected,
            "Replay must retain every one of the 28 unique raw samples")
    for row in samples:
        require(row["warmup"] is (row["iteration"] < 2)
                and row["exact_bytes_counts_checksums_verified"] is True
                and row["new_graphs_or_kernels"] is False and not row["new_compile_counters"],
                "Replay sample failed exactness, phase, or compiler admission")
        require(type(row["wall_s"]) in (int, float) and math.isfinite(row["wall_s"]) and row["wall_s"] > 0
                and type(row["process_cpu_s"]) in (int, float) and math.isfinite(row["process_cpu_s"])
                and row["process_cpu_s"] >= 0, "Replay contains invalid raw timings")
    workloads = [row for row in events if row["event"] == "workload"]
    require(Counter(row["case"] for row in workloads) == Counter({"native-unchanged": 1, "synthetic-v1-v2": 1}),
            "Missing or duplicate workload records")
    for row in workloads:
        require(row["tensor_count"] == TENSOR_COUNT and row["total_bytes"] == PAYLOAD_BYTES
                and row["encoding"] == "xor" and row["checksum"] == "xxh3-128",
                "Replay workload bytes/tensors/wire mode differ")
        expected_changes = (0, 0) if row["case"] == "native-unchanged" else (79500803, 2646)
        require((row["changed_bytes"], row["changed_tensor_count"]) == expected_changes,
                "Replay changed/unchanged workload differs from the retained v1-v2 pair")
    warmups = [row for row in events if row["event"] == "compile_warmup"]
    require(Counter(row["case"] for row in warmups) == Counter({"native-unchanged": 1, "synthetic-v1-v2": 1})
            and all(row["actual_layouts_warmed"] == list(range(provenance["bucket_count"]))
                    for row in warmups), "Missing actual-layout serial warmup")
    summaries = [row for row in events if row["event"] == "summary"]
    require(Counter(row["case"] for row in summaries) == Counter({"native-unchanged": 1, "synthetic-v1-v2": 1}),
            "Missing or duplicate replay summaries")
    for summary in summaries:
        for variant in ("exact_f17", "packed_stage"):
            selected = [row for row in samples if row["case"] == summary["case"]
                        and row["variant"] == variant and not row["warmup"]]
            for field in ("wall_s", "process_cpu_s"):
                values = [row[field] for row in selected]
                require(summary["variants"][variant][f"raw_{field}"] == values
                        and summary["variants"][variant][f"median_{field}"] == statistics.median(values),
                        "Replay summary differs from complete raw samples")
    source_dir = path.with_name(path.stem + "-sources")
    sources = {filename: read_local(root, str(source_dir / filename)) for filename in REPLAY_SOURCES}
    for filename, key in (("benchmark_delta_preparation.py", "script_sha256"),
                          ("benchmark_sender_replay.py", "replay_helper_sha256"),
                          ("delta_preparation.py", "module_sha256"), ("layouts.json", "layout_sha256")):
        require(digest(sources[filename]) == provenance[key], f"Archived replay source mismatch: {filename}")
    uncommitted = source_reference.startswith("sha256:")
    if uncommitted:
        require(allow_uncommitted and re.fullmatch(r"sha256:[0-9a-f]{64}", source_reference),
                "Only an explicitly historical experiment may use an archived-module SHA256")
        require(digest(sources["delta_preparation.py"]) == source_reference.split(":", 1)[1],
                "Archived experimental module differs from its supplied SHA256")
    else:
        require(sources["delta_preparation.py"] == git_file(repo, source_reference, "miles/utils/delta_preparation.py"),
                "Replay module differs from the declared standalone commit")
    require(digest(sources["exact-f17-delta.py"]) == provenance["baseline"]["source_sha256"]
            and digest(sources["exact-f17-disk_delta.py"]) == provenance["baseline"]["helper_source_sha256"],
            "Archived exact f17 baseline mismatch")
    require(read_local(root, str(path.with_suffix(".exit"))).strip() == b"0", "Replay did not exit successfully")
    files = [name, str(path.with_suffix(".log")), str(path.with_suffix(".exit"))]
    files += [str(source_dir / filename) for filename in REPLAY_SOURCES]
    return files, {"source_commit": None if uncommitted else source_reference,
                   "source_kind": "uncommitted historical experiment" if uncommitted else "committed source",
                   "source_reference": source_reference, "raw_sha256": digest(raw), "samples": len(samples),
                   "module_sha256": provenance["module_sha256"], "summaries": summaries}


def assemble(root, final_replay, codegen_files, validation_logs, extra_files, historical_replays=()):
    require(codegen_files and validation_logs, "Explicit final codegen proof and validation log files are required")
    require(final_replay != NEGATIVE_REPLAY, "Final candidate and historical negative replay must be distinct")
    plan = json.loads(read_local(root, "COMPILE_VALIDATION_PLAN.json"))
    planned = reporter.validate_plan(plan)
    attribution = validate_attribution(root, plan)
    require(json.loads(read_local(root, "COMPILE_SOURCE_ATTRIBUTION.json")) == attribution, "Stored source attribution is stale")
    preflight = selected_preflight(root, plan)
    compatibility = validate_compatibility(root, attribution)
    current_ci = current_hosted_ci(root, plan["standalone_source"])
    frozen = json.loads(read_local(root, "artifacts/compile-campaign-frozen-inputs.json"))
    for name, expected_hash in frozen.items():
        require(digest(read_local(root, name)) == expected_hash, f"Campaign-frozen file drift: {name}")
    require(len(planned) == 2 and {run["delta_cpu_backend"] for run in planned.values()} == {"numpy", "torch-compile"}
            and len({run["reward_mode"] for run in planned.values()}) == 1,
            "Publication requires the complete matched pair, with all 14 GPU updates")
    runs = [root / "artifacts" / name for name in planned]
    selected_metadata = resolve_metadata(root / "COMPILE_VALIDATION_PLAN.json", plan)
    campaign = json.loads(read_local(root, str(selected_metadata["campaign_metadata"].relative_to(root.resolve()))))
    if "prepared_inputs" in campaign:
        prepared = campaign["prepared_inputs"]
        require(digest(read_local(root, prepared["path"])) == prepared["sha256"], "Prepared-input seal differs from campaign pin")
        ready = json.loads(read_local(root, prepared["path"]))
        require(ready["status"] == "ready", "Prepared-input seal is not ready")
        for name in ("artifacts/recovery-original-canonical-headers.json", "artifacts/recovery-nvfp4-header-verification.json",
                     "artifacts/input-backup-20260926T021916Z-4640.json", "artifacts/recovery-20260926T021916Z-4640-input-identity.json"):
            require(digest(read_local(root, name)) == ready["evidence"][name]["sha256"], "Recovery evidence differs from sealed hashes")
    report = reporter.build(runs, root / "COMPILE_VALIDATION_PLAN.json")
    # The report build reads all retained shard/debug evidence. Publish the exact
    # previously generated report only when it still matches current raw input.
    stored = json.loads(read_local(root, "COMPILED_VALIDATION_RESULTS.json"))
    require(json.loads(json.dumps(report)) == stored, "Stored compiled JSON report is stale; regenerate it")
    require(read_local(root, "COMPILED_VALIDATION_RESULTS.md") == reporter.markdown(report).encode(),
            "Stored compiled Markdown report is stale; regenerate it")
    require(sum(len(arm["rows"]) for arm in report["arms"].values()) == 14
            and sum(len(arm["compiler_observations"]) for arm in report["arms"].values()) == 56,
            "Missing GPU updates or per-rank compiler observations")
    names = set(FILES) | set(plan["script_sha256"]) | set(codegen_files) | set(validation_logs) | set(extra_files)
    names.update(str(path.relative_to(root.resolve())) for path in selected_metadata.values())
    if preflight is not None:
        names.update(f"{preflight['directory']}/{filename}" for filename in ("manifest.json", "focused-tests.log", "gloo-failures.log"))
    if compatibility is not None:
        names.update(compatibility["files"])
    if current_ci is not None:
        names.update(current_ci["files"])
    for name, expected in plan["script_sha256"].items():
        require(digest(read_local(root, name)) == expected, f"Final helper differs from frozen campaign: {name}")
    replays = {}
    replay_specs = [(NEGATIVE_REPLAY, NEGATIVE_SOURCE), (MATERIALIZE_REPLAY, MATERIALIZE_SOURCE), *historical_replays,
                    (final_replay, attribution["standalone_cpu_replay"]["measured_source"])]
    replay_specs = list(dict.fromkeys(tuple(pair) for pair in replay_specs))
    require(len({name for name, _ in replay_specs}) == len(replay_specs), "Duplicate CPU replay path")
    for name, commit in replay_specs:
        replay_files, metadata = validate_replay(root, name, commit, root / "miles-torch-compile",
                                                 allow_uncommitted=name not in {NEGATIVE_REPLAY, final_replay})
        names.update(replay_files)
        replays[name] = metadata
    for run in runs:
        if "campaign_metadata" in plan:
            names.add(f"artifacts/{run.name}-campaign-metadata.json")
        names.update(str((run / name).relative_to(root)) for name in RUN_FILES)
        names.update(f"artifacts/{run.name}.{suffix}" for suffix in ("log", "exit"))
        names.update(f"artifacts/{run.name}/delta-publication/weight_v{version:06d}/model.safetensors.index.json"
                     for version in range(1, 7))
    data = {name: read_local(root, name) for name in sorted(names)}
    snapshots = {}
    for role, repo, commit in (("standalone", "miles-torch-compile", plan["standalone_source"]),
                                ("combined-validation", "miles-torch-compile-validation", plan["sources"]["miles"]),
                                ("cpu-replay-measured", "miles-torch-compile", attribution["standalone_cpu_replay"]["measured_source"]),
                                ("historical-preflight-measured", "miles-torch-compile-validation", attribution["combined_historical_preflight"]["measured_source"])):
        snapshots[role] = {"commit": commit, "files": {}}
        for filename in SOURCE_FILES:
            name = f"source_snapshots/{role}/{commit}/{filename}"
            content = git_file(root / repo, commit, filename)
            check_text(name, content)
            data[name] = content
            snapshots[role]["files"][filename] = digest(content)
    metadata = {
        "schema_version": 1, "complete": True, "gpu_updates": 14, "gpu_rank_observations": 56,
        "source_pins": plan["sources"], "source_snapshots": snapshots, "cpu_replays": replays,
        "source_attribution": attribution, "selected_preflight": preflight,
        "ci_compatibility": compatibility, "selected_head_hosted_ci": current_ci,
        "campaign_metadata_selection": metadata_record(selected_metadata, relative_to=root),
        "explicit_codegen_files": codegen_files, "explicit_validation_logs": validation_logs,
        "explicit_file_limit": "Codegen/test files are preserved and hashed; final-source attribution and pass status require reviewer verification.",
        "source_policy": "Raw files copied byte for byte; source snapshots are exact git show bytes.",
        "exclusions": ["Weight tensors and compressed delta payloads", "Binary training debug dumps",
                       "Compiled libraries, object files, compiler binary caches, and source archives",
                       "Large complete hosted workflow logs; compact review, failed shard, metadata and targeted local diagnostics are retained"],
        "strict_admission": "GPU reporter rerun against local shard and binary debug evidence; stored JSON/Markdown required to match.",
        "secret_scan": "All UTF-8 files scanned; no unapproved matches or redaction. Exact public malformed-URL test-fixture matches are recorded in COMPILE_SECRET_SCAN_FIXTURE_ALLOWLIST.json, bound to raw file/value hashes and test-node context.",
    }
    data["BUNDLE_METADATA.json"] = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode()
    data["README.md"] = readme(final_replay, list(planned), metadata_record(selected_metadata, relative_to=root)).encode()
    require(sum(len(value) for value in data.values()) <= MAX_BUNDLE_BYTES, "Evidence exceeds 128 MiB text budget")
    for name, value in data.items():
        check_text(name, value)
    hashes = {name: {"sha256": digest(value), "bytes": len(value)} for name, value in sorted(data.items())}
    data["SHA256.json"] = (json.dumps(hashes, indent=2) + "\n").encode()
    return data


def readme(final_replay, runs, selected_metadata):
    return f"""# GLM-5.2 packed CPU delta evidence

This directory belongs to the separate torch.compile Miles PR. Original delta-sync
evidence remains in `../glm52_delta_sync/` at its original published commit.

- [Matched GPU validation](COMPILED_VALIDATION_RESULTS.md) retains all 14 updates.
  Its JSON retains all 56 per-rank compiler records, source/helper identities,
  runtime settings, publication shard SHA256 and per-tensor checksum metadata.
- [Final CPU replay]({final_replay}) retains all 28 raw warmup/measured samples.
- [Historical negative replay](TORCH_COMPILE_STAGE_REPLAY_V1.md) and its raw files
  remain separate from the final candidate. Fixtures/probes are diagnostic evidence,
  not substitutes for the full replay or combined GPU run.
- [Materialize-first experiment](TORCH_COMPILE_MATERIALIZE_FIRST_HISTORY.md) retains
  its separate 28-sample replay and explicitly uncommitted archived module identity.
- `source_snapshots/` contains exact committed production/test file bytes.
  `BUNDLE_METADATA.json` identifies explicit final codegen proof and test logs.
- [Hosted CI review](HOSTED_CI_REVIEW.md) retains the historical failure separately
  from the [compatibility investigation](COMPILE_CI_COMPATIBILITY.md), selected-head
  hosted snapshots and C2 validation. The compact subset includes the raw failed shard; complete
  workflow logs remain local and available through the linked GitHub jobs.
- `SHA256.json` covers every other file, including generated metadata and this page.
  Original raw evidence was neither normalized nor redacted.

The two GPU arms are `{runs[0]}` and `{runs[1]}`. Sources are pinned by
`COMPILE_VALIDATION_PLAN.json`; selected campaign/environment files are
`{selected_metadata['campaign_metadata']['path']}` and
`{selected_metadata['environment_metadata']['path']}`. Replacement plans pin their
metadata hashes; the original `CAMPAIGN.json` / `environment-image.json` and
`COMPILE_VALIDATION_PLAN_V1.json` remain historical evidence for the original host.
`COMPILE_GPU_V1_INTERRUPTION.md` records the lost ephemeral devbox and v1's
summary-only boundary. Those summaries are never substituted into the matched pair.
CPU replay/preflight evidence uses the original image and is labeled separately.
External calibration scope is explicit in the GPU report; per-run observed costs
and inventories belong to the selected campaign.
Only u2-u6 enter steady GPU medians; initial/u1 observations are still retained.
Compiler subprocess CPU is excluded from trainer-process CPU. Graph/kernel counts
alone do not prove fusion, acceleration, or any multi-node benefit.

The local publisher reruns strict admission before copying. Weight payloads,
compressed delta shards, binary debug tensors, and compiled binaries are deliberately
excluded. Publication index bytes and the report's recorded shard hashes/checksums
are retained. Rerunning full strict admission requires the durable local payloads
or repeating the workload; this text bundle alone cannot reconstruct model state,
prove GPU equality, or independently recompute payload checksums.

No commit or push is performed by the publishing helper. Publication targets only
the `fork` remote (`zianglih/miles`); it is not an upstream source change.
"""


def stage(root, data):
    checkout = root / "miles-evidence"
    common = Path(subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True).strip())
    require(common.resolve().is_relative_to(root.resolve()), "Evidence Git store is outside task workspace")
    branch = subprocess.check_output(["git", "-C", str(checkout), "branch", "--show-current"], text=True).strip()
    require(branch == "glm52-delta-sync-evidence", "Unexpected evidence branch")
    fork = subprocess.check_output(["git", "-C", str(checkout), "remote", "get-url", "fork"], text=True).strip()
    require(fork in {"https://github.com/zianglih/miles.git", "git@github.com:zianglih/miles.git"},
            "Evidence fork remote is not zianglih/miles")
    destination = root / DESTINATION
    require(not any(path.is_symlink() for path in (destination, *destination.parents)
                    if path.is_relative_to(root)), "Symlink destination is forbidden")
    if destination.exists():
        existing = {str(path.relative_to(destination)): path for path in destination.rglob("*") if path.is_file()}
        require(set(existing) == set(data) and all(not path.is_symlink() and path.read_bytes() == data[name]
                                                 for name, path in existing.items()),
                "Existing compile evidence differs; review it explicitly before replacing any file")
        return
    destination.mkdir(parents=True)
    for name, content in sorted(data.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        require(target.read_bytes() == content, f"Staged byte verification failed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-replay", required=True, help="Task-relative final 28-sample JSONL; includes adjacent log/exit/source folder")
    parser.add_argument("--codegen-proof", action="append", required=True, help="Exact task-relative UTF-8 proof file; repeatable")
    parser.add_argument("--validation-log", action="append", required=True, help="Exact final-source test/Gloo log; repeatable")
    parser.add_argument("--include", action="append", default=[], help="Additional explicit text artifact; no directories/globs")
    parser.add_argument("--historical-replay", action="append", default=[], metavar="JSONL=SOURCE",
                        help="Completed historical 28-sample replay: JSONL=COMMIT or JSONL=sha256:MODULE_SHA; repeatable")
    parser.add_argument("--stage", action="store_true", help="Copy audited bytes into the dedicated compile evidence folder; no commit/push")
    args = parser.parse_args()
    try:
        historical = [value.rsplit("=", 1) for value in args.historical_replay]
        require(all(len(pair) == 2 for pair in historical), "Historical replay requires JSONL=COMMIT or JSONL=sha256:MODULE_SHA")
        data = assemble(ROOT, args.final_replay, args.codegen_proof, args.validation_log, args.include, historical)
        if args.stage:
            stage(ROOT, data)
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.CalledProcessError) as error:
        print(json.dumps({"complete": False, "staged": False, "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({"complete": True, "staged": args.stage, "files": len(data),
                      "bytes": sum(map(len, data.values())), "sha256_manifest": digest(data["SHA256.json"]),
                      "destination": str(ROOT / DESTINATION)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
