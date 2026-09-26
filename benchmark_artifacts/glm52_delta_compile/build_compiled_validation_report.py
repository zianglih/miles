#!/usr/bin/env python3
"""Admit frozen compiled-backend runs without changing the original CPU reporter.

Required plan: schema_version=1, frozen=true, sources={miles,sglang},
script_sha256 (the original eight frozen helpers plus the two compile helpers),
and runs=[{name,mode:"disk-delta",reward_mode,delta_cpu_backend,num_rollout:7,
compile_cache_dir,compile_cache_state_before:{exists:bool,empty:bool}}].
Every planned run is required. A single backend is validation only; backend
ratios require both backends under the same reward and matched configurations.

No input is rewritten. The original load_arm code runs with an explicit helper
identity and recipe-contract adapter; all of its raw CPU/training admission
checks remain in force. Run the original three summarizers first. Keep all six
delta publication indexes/shards and fourteen binary debug dumps locally.
"""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import re
import shlex
import struct
import sys
import types

import build_cpu_benchmark_report as base
from compile_campaign_metadata import metadata_record, resolve_metadata, validate_specs
from build_benchmark_report import InvalidEvidence, read_json, read_jsonl, require, sha256, table

ROOT = Path(__file__).resolve().parent
COMPILE_HELPERS = {"launch_weight_sync_compile.py", "weight_sync_probe_compile.py"}
COMPILER_ACTIVITY = (
    "frames.total", "frames.ok", "stats.unique_graphs", "stats.calls_captured",
    "aot_autograd.total", "inductor.fxgraph_cache_hit", "inductor.fxgraph_cache_miss",
    "inductor.generated_kernel_count",
)
STEADY = tuple(range(2, 7))


def validate_plan(plan):
    try:
        validate_specs(plan)
    except ValueError as error:
        require(False, str(error))
    require(plan.get("schema_version") == 1 and plan.get("frozen") is True, "Require a frozen schema1 plan")
    require(plan.get("implementation_label", "candidate") == "candidate", "All arms require implementation_label=candidate")
    require(set(plan["sources"]) == {"miles", "sglang"}
            and all(re.fullmatch(r"[0-9a-f]{40}", value) for value in plan["sources"].values()),
            "Plan must pin two full source commits")
    hashes = plan["script_sha256"]
    require(set(hashes) == set(base.FROZEN_MEASUREMENT_SHA256) | COMPILE_HELPERS
            and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()),
            "Plan must pin all original and compile helper hashes")
    require(all(hashes[name] == digest for name, digest in base.FROZEN_MEASUREMENT_SHA256.items()),
            "Original CPU/wall helpers differ from their calibrated freeze")
    runs = plan["runs"]
    require(runs and len({run["name"] for run in runs}) == len(runs), "Plan has missing/duplicate runs")
    require(len({(run["reward_mode"], run["delta_cpu_backend"]) for run in runs}) == len(runs),
            "Plan repeats a reward/backend arm")
    require(len({Path(run["compile_cache_dir"]).resolve() for run in runs}) == len(runs),
            "Every run requires an independent compile cache path")
    for run in runs:
        require(Path(run["name"]).name == run["name"] and run["name"] not in {"", ".", ".."},
                "Run name must be a single path component")
        require(run["mode"] == "disk-delta" and run["num_rollout"] == 7
                and run["reward_mode"] in {"native", "synthetic-balanced"}
                and run["delta_cpu_backend"] in {"numpy", "torch-compile"}, "Unexpected planned workload")
        require(Path(run["compile_cache_dir"]).is_absolute(), "Plan requires absolute cache paths")
        state = run["compile_cache_state_before"]
        require(set(state) == {"exists", "empty"} and all(type(value) is bool for value in state.values())
                and (state["exists"] or state["empty"]), "Invalid planned cache state")
    require(any(run["delta_cpu_backend"] == "torch-compile" for run in runs), "Plan contains no compiled arm")
    return {run["name"]: run for run in runs}


def validate_manifest(manifest, planned):
    """Explicit compile recipe contract; raw manifests are never normalized here."""
    argv = manifest["train_argv"]
    require(shlex.split(manifest["launch_kwargs"]["train_args"]) == argv,
            "Recorded launch command differs from recorded training argv")
    expected = {
        "--num-rollout": "7", "--actor-num-nodes": "1", "--actor-num-gpus-per-node": "4",
        "--rollout-num-gpus": "4", "--rollout-num-gpus-per-engine": "2",
        "--tensor-model-parallel-size": "4", "--expert-model-parallel-size": "4",
        "--sglang-dp-size": "2", "--sglang-ep-size": "2", "--rm-type": "deepscaler",
        "--custom-megatron-init-path": "weight_sync_probe_compile.install",
        "--update-weight-transfer-mode": "disk-delta",
        "--update-weight-delta-cpu-backend": planned["delta_cpu_backend"],
        "--seed": str(manifest["seed"]), "--rollout-seed": str(manifest["rollout_seed"]),
    }
    if planned["reward_mode"] == "synthetic-balanced":
        expected["--custom-rm-path"] = "weight_sync_reward.balanced_index_reward"
    else:
        require("--custom-rm-path" not in argv, "Native reward unexpectedly overridden")
    for flag, value in expected.items():
        require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value,
                f"Compile recipe mismatch: {flag}")
    for flag, default in (("--update-weight-delta-encoding", "xor"),
                          ("--update-weight-delta-checksum", "xxh3-128")):
        require(flag not in argv or (argv.count(flag) == 1 and argv[argv.index(flag) + 1] == default),
                f"Unexpected encoding/checksum override: {flag}")
    env = manifest["launch_kwargs"]["extra_env_vars"]
    require("--bf16" in argv and env["OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"] == "1"
            and env["SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16"] == "1"
            and env["NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING"] == "1"
            and manifest["launch_kwargs"]["megatron_model_type"] == "glm5.2-744B-A40B_5layer"
            and Path(manifest["launch_kwargs"]["train_script"]).name == "train_weight_sync_profiled_cpu_v2.py",
            "Precision/model/CPU driver contract mismatch")
    require(manifest["source_manifest_version"] == 3 and manifest["compile_observer_version"] == 1,
            "Unsupported source/compile observer contract")
    require(all(manifest[key] == planned[key] for key in
                ("mode", "reward_mode", "delta_cpu_backend", "num_rollout", "compile_cache_dir",
                 "compile_cache_state_before")), "Manifest differs from frozen run plan")
    require(env["TORCHINDUCTOR_CACHE_DIR"] == planned["compile_cache_dir"], "Compiler cache environment mismatch")


def compiler_observations(run, manifest):
    require(not any(event["kind"] == "compile_observation" for event in read_jsonl(run / "driver.jsonl")),
            "Trainer compiler observations were misattributed to the driver")
    records = []
    for rank in range(4):
        events = read_jsonl(run / f"trainer-rank{rank}.jsonl")
        compiles = [event for event in events if event["kind"] == "compile_observation"]
        trainers = {event["update_index"]: event for event in events if event["kind"] == "trainer_update"}
        require(Counter(event["update_index"] for event in compiles) == Counter(range(7)),
                f"Rank {rank}: need seven unique compiler observations")
        previous = None
        for event in sorted(compiles, key=lambda value: value["update_index"]):
            index = event["update_index"]
            require(event["rank"] == rank and event["pid"] == trainers[index]["pid"]
                    and event["success"] is True and event["mode"] == "disk-delta"
                    and event["cpu_backend"] == manifest["delta_cpu_backend"],
                    f"Rank {rank} update {index}: compiler identity/status mismatch")
            before, after = event["before"], event["after"]
            require(all(isinstance(values, dict) and "inductor.generated_kernel_count" in values
                        and all(type(value) is int and value >= 0 for value in values.values())
                        for values in (before, after)), "Invalid absolute compiler counters")
            delta = {key: after.get(key, 0) - before.get(key, 0) for key in sorted(set(before) | set(after))
                     if after.get(key, 0) != before.get(key, 0)}
            require(event["delta"] == delta and all(value >= 0 for value in delta.values()),
                    "Compiler counter arithmetic/reset mismatch")
            if previous is not None:
                require(all(before.get(key, 0) >= value for key, value in previous.items()),
                        "Compiler counters reset between updates")
            previous = after
            activity = {key: delta[key] for key in COMPILER_ACTIVITY if delta.get(key, 0)}
            require(index not in STEADY or not activity,
                    f"Rank {rank} measured update {index}: compiler activity {activity}")
            records.append({**event, "measured_steady": index in STEADY, "graph_or_kernel_activity": activity})
    if manifest["delta_cpu_backend"] == "torch-compile":
        require(any(event["update_index"] == 0 and event["graph_or_kernel_activity"] for event in records),
                "Compiled backend has no observed initial graph/cache/kernel activity")
    return sorted(records, key=lambda event: (event["update_index"], event["rank"]))


def publications(run, rows):
    """Check retained shard/header/checksum contracts; do not claim independent reconstruction."""
    result = []
    for version in range(1, 7):
        directory = run / "delta-publication" / f"weight_v{version:06d}"
        index_path = directory / "model.safetensors.index.json"
        index = read_json(index_path)
        metadata, mapping = index["metadata"], index["weight_map"]
        require(int(metadata["version"]) == version and int(metadata["base_version"]) == version - 1
                and metadata["delta_encoding"] == "xor" and metadata["compression_format"] == "zstd"
                and metadata["checksum_format"] == "xxh3-128", "Publication lineage/encoding/checksum mismatch")
        require(isinstance(mapping, dict), "Publication weight_map is not a dictionary")
        shards = sorted(set(mapping.values()))
        require(set(path.name for path in directory.glob("*.safetensors")) == set(shards),
                "Missing or extra publication shard")
        files, count, total = {}, 0, 0
        for filename in shards:
            require(Path(filename).name == filename, "Nonlocal publication shard path")
            path = directory / filename
            with path.open("rb") as stream:
                raw_size = stream.read(8)
                require(len(raw_size) == 8, "Truncated safetensor length")
                header_size = struct.unpack("<Q", raw_size)[0]
                require(header_size <= path.stat().st_size - 8, "Truncated safetensor header")
                header = json.loads(stream.read(header_size))
            names = {name for name, target in mapping.items() if target == filename}
            require(set(header) - {"__metadata__"} == names and set(header.get("__metadata__", {})) == names,
                    "Shard tensor/checksum names differ from authoritative index")
            data_size = path.stat().st_size - 8 - header_size
            ranges = []
            for name in names:
                tensor = header[name]
                begin, end = tensor["data_offsets"]
                require(tensor["dtype"] == "U8" and tensor["shape"] == [end - begin]
                        and 0 <= begin < end <= data_size, "Invalid compressed uint8 tensor range")
                require(re.fullmatch(r"[0-9a-f]{32}", header["__metadata__"][name]),
                        "Missing or malformed per-tensor xxh3-128 checksum")
                ranges.append((begin, end))
            ranges.sort()
            require(ranges and ranges[0][0] == 0 and ranges[-1][1] == data_size
                    and all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])),
                    "Shard tensor ranges overlap or omit data")
            files[filename] = {"bytes": path.stat().st_size, "sha256": sha256(path),
                               "tensor_checksums": header["__metadata__"]}
            count += len(names)
            total += path.stat().st_size
        require(total == rows[version]["wire_bytes"] and bool(count) == bool(rows[version]["changed_bytes"]),
                "Publication bytes/change presence differ from raw sender counters")
        result.append({"version": version, "index_sha256": sha256(index_path), "metadata": metadata,
                       "tensor_count": count, "serialized_shard_bytes": total, "shards": files})
    return result


def load_arm(run, plan, planned):
    # This creates a new function object; base.load_arm and its globals stay frozen.
    bindings = {**base.load_arm.__globals__,
                "validate_manifest_contract": lambda manifest: validate_manifest(manifest, planned),
                "CPU_HELPERS": base.CPU_HELPERS | COMPILE_HELPERS,
                "FROZEN_MEASUREMENT_SHA256": plan["script_sha256"]}
    adapted = types.FunctionType(base.load_arm.__code__, bindings, "load_compiled_cpu_arm")
    arm = adapted(run, plan)
    arm["compiler_observations"] = compiler_observations(run, arm["manifest"])
    arm["publications"] = publications(run, arm["rows"])
    require(arm["training"]["replay_checks"] > 0 and arm["training"]["replay_nonzero_checks"] == 0,
            "Missing or mismatching routing replay evidence")
    return arm


def comparison_config(manifest):
    """Allow only explicit backend/cache and original output-path differences."""
    result = base.normalized_config(manifest)
    tokens, argv = iter(result["argv"]), []
    for token in tokens:
        if token == "--update-weight-delta-cpu-backend":
            require(next(tokens) == manifest["delta_cpu_backend"], "Unexpected backend during comparison")
        else:
            argv.append(token)
    result["argv"] = argv
    result["launch"]["extra_env_vars"].pop("TORCHINDUCTOR_CACHE_DIR")
    return result


def build(runs, plan_path, campaign_path=None, environment_path=None, calibration_path=None):
    plan = read_json(plan_path)
    planned = validate_plan(plan)
    try:
        metadata = resolve_metadata(plan_path, plan, campaign=campaign_path,
                                    environment=environment_path, calibration=calibration_path)
    except ValueError as error:
        require(False, str(error))
    campaign_path, environment_path, calibration_path = (metadata[key] for key in
        ("campaign_metadata", "environment_metadata", "cpu_calibration_metadata"))
    campaign, environment = read_json(campaign_path), read_json(environment_path)
    require(len(runs) == len({path.resolve() for path in runs}) == len(planned)
            and {path.name for path in runs} == set(planned), "Every frozen planned run is required exactly once")
    require(campaign["topology"] == base.TOPOLOGY, "Campaign topology mismatch")
    gpus = list(csv.DictReader(io.StringIO(environment["gpus"]["stdout"]), skipinitialspace=True))
    require(environment["gpus"]["exit_code"] == 0 and len(gpus) == 8
            and all("B300" in row["name"] for row in gpus), "Require eight B300 GPUs")
    calibration = base.load_calibration(calibration_path)
    if "campaign_metadata" in plan:
        expected_metadata = {"plan_sha256": sha256(plan_path),
                             "metadata_selection": metadata_record(metadata, relative_to=Path(plan_path).parent)}
        for run in runs:
            require(read_json(run.parent / f"{run.name}-campaign-metadata.json") == expected_metadata,
                    f"{run.name}: launcher campaign metadata differs from frozen plan")
    arms = [load_arm(run.resolve(), plan, planned[run.name]) for run in runs]
    require(all(arm["manifest"]["cpu_observer"] == arms[0]["manifest"]["cpu_observer"] for arm in arms),
            "Observer configuration differs across arms")
    keyed = {(arm["manifest"]["reward_mode"], arm["manifest"]["delta_cpu_backend"]): arm for arm in arms}
    comparisons = {}
    for reward in ("native", "synthetic-balanced"):
        if (reward, "numpy") not in keyed or (reward, "torch-compile") not in keyed:
            continue
        control, compiled = keyed[(reward, "numpy")], keyed[(reward, "torch-compile")]
        require(comparison_config(control["manifest"]) == comparison_config(compiled["manifest"]),
                f"{reward}: pair differs beyond explicit CPU backend/cache/output paths")
        require(control["runtime_identity"]["normalized_roles"] == compiled["runtime_identity"]["normalized_roles"],
                f"{reward}: process affinity/cgroup/thread settings differ")
        comparisons[reward] = {"numpy": control["name"], "torch_compile": compiled["name"], "metrics": {
            metric: {"numpy": control["steady"][metric], "torch_compile": compiled["steady"][metric],
                     "compiled_over_numpy": compiled["steady"][metric]["median"] / control["steady"][metric]["median"]
                     if control["steady"][metric]["median"] else None} for metric in base.METRICS}}
    return {
        "schema_version": 1, "validated_compiled_campaign": True, "plan": plan,
        "campaign": {key: campaign[key] for key in (
            "image", "image_index_digest", "image_amd64_digest", "cluster", "host", "devbox",
            "megatron_head", "hf_source", "hf_revision", "topology")},
        "environment": environment, "cpu_calibration": calibration,
        "metadata_selection": metadata_record(metadata),
        "external_calibration_scope": (
            "Historical observer calibration on original hu-pdx-90/dev-202609251434; not remeasured on this campaign host. "
            "Actual per-run snapshot costs and inventories remain in each arm."
            if sha256(calibration_path) == "ce8386c75e01125a7e3754844389a82f2e889200df0ce233cd030aff87216707" else
            "Plan-selected external observer calibration; its exact file records its measurement scope. "
            "Actual per-run snapshot costs and inventories remain in each arm."
        ),
        "metadata_sha256": {str(path.resolve()): sha256(path) for path in
                            (plan_path, campaign_path, environment_path, calibration_path)},
        "generator_sha256": sha256(Path(__file__)), "original_cpu_validator_sha256": sha256(Path(base.__file__)),
        "admission_adapter": "Original load_arm code; explicit compile hook/helper contract only; no raw evidence rewriting",
        "comparison_exclusions": ["CPU backend flag", "validated per-run compiler cache path",
                                  "original validator's explicit output/transport paths"],
        "compiler_activity_keys": COMPILER_ACTIVITY, "steady_update_indices": STEADY,
        "arms": {arm["name"]: arm for arm in arms}, "backend_comparisons": comparisons,
        "limitations": [
            "Single-node C2 4+4 correctness and local overhead; no multi-node speedup or amortization inference.",
            "All seven updates are retained; only u2-u6 enter steady medians. Any invalid steady CPU or compiler activity rejects admission.",
            "Compiler counters cover each entire trainer updater, not only packed CPU graphs. Their reads/write lie outside updater timing and inside driver wall.",
            "Initial setup wall includes initialization/compilation; an empty on-disk cache is recorded but cannot prove absence of all external/compiler caches. Trainer process CPU excludes compiler child CPU.",
            "CPU windows cover trainer processes and inventoried receiver trees, not simultaneous whole-host CPU. Non-atomic membership can omit transient children. No observer cost is subtracted.",
            "Publication indexes, shard SHA256, mapping/ranges, and checksum presence are checked. Successful receiver apply supplies runtime checksum checks; this report does not independently reconstruct checkpoint bytes or attest GPU equality.",
            "Synthetic balanced rewards test changed-weight transport, not task quality or equal training trajectories. Disabled original logprob/KL/weight-equality checkers remain disabled.",
            "A single backend arm supports correctness only. Backend performance ratios appear only for matched same-reward pairs in this frozen plan; earlier campaigns are not substituted.",
        ],
    }


def markdown(report):
    lines = ["# Compiled disk-delta validation", "", "All planned runs passed strict source, recipe, seven-update, 28-rank training/CPU, publication, and per-rank compiler admission.", "",
             f"Sources: `{report['plan']['sources']}`. Image `{report['campaign']['image']}`; amd64 `{report['campaign']['image_amd64_digest']}`.", "",
             report["external_calibration_scope"], "",
             "Steady = all u2–u6; no sample removal. Full raw compiler dictionaries, runtime settings, source/helper hashes, publication checksums, and warnings remain in JSON.", ""]
    for reward, comparison in report["backend_comparisons"].items():
        lines += [f"## {reward}: matched CPU backends", "", "Ratios are compiled / NumPy median. Values are wall seconds or component process CPU-seconds.", ""]
        lines += table(["Metric", "NumPy median", "Compiled median", "Ratio"],
                       [[metric, value["numpy"]["median"], value["torch_compile"]["median"], value["compiled_over_numpy"]]
                        for metric, value in comparison["metrics"].items()])
    for name, arm in report["arms"].items():
        lines += ["", f"## {name}", "", f"Backend `{arm['manifest']['delta_cpu_backend']}`; reward `{arm['manifest']['reward_mode']}`; cache before `{arm['manifest']['compile_cache_state_before']}`.", ""]
        lines += table(["Update", "Driver s", "Actor s", "Trainer CPU sum s", "Trainer CPU max s", "Receiver CPU s", "Driver CPU s", "Changed bytes", "Wire bytes", "CPU valid"],
                       [[row["update_index"], row["driver_wall_s"], row["builtin_actor_s"], row["trainer_cpu_sum_s"], row["trainer_cpu_max_s"],
                         row["receiver_total_cpu_s"], row["driver_cpu_s"], row["changed_bytes"], row["wire_bytes"], row["cpu_valid"]] for row in arm["rows"]])
        lines += ["", "Raw per-rank compiler activity (complete counter deltas retained in JSON):", ""]
        lines += table(["Update", "Rank", "PID", "Success", "Steady", "New graph/cache/kernel counters"],
                       [[event["update_index"], event["rank"], event["pid"], event["success"], event["measured_steady"],
                         json.dumps(event["graph_or_kernel_activity"], sort_keys=True)] for event in arm["compiler_observations"]])
        lines += ["", f"Gradient norms: `{arm['training']['gradient_norms']}`. Routing replay checks: `{arm['training']['replay_checks']}`, mismatching checks: `{arm['training']['replay_nonzero_checks']}`.", "",
                  f"Observer snapshot/driver ratios: `{arm['cpu_observer']['snapshot_to_driver_ratios']}`; outer bookkeeping/driver ratios: `{arm['cpu_observer']['outside_trainer_wall_to_driver_ratios']}`. Invalid early CPU indices: `{arm['cpu_observer']['initial_or_first_post_cpu_invalid']}`.", "",
                  f"Warning inventory: `{arm['warnings']['stage_counts']}`. Readiness retries: `{arm['warnings']['readiness_retry_counts']}`."]
    lines += ["", "## Boundaries", ""] + [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--plan", type=Path, default=ROOT / "COMPILE_VALIDATION_PLAN.json")
    parser.add_argument("--campaign-metadata", type=Path, help="Legacy override; must match a plan pin when present")
    parser.add_argument("--environment", type=Path, help="Legacy override; must match a plan pin when present")
    parser.add_argument("--cpu-calibration", type=Path, help="Legacy override; must match a plan pin when present")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        report = build(args.runs, args.plan, args.campaign_metadata, args.environment, args.cpu_calibration)
        outputs = {"COMPILED_VALIDATION_RESULTS.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
                   "COMPILED_VALIDATION_RESULTS.md": markdown(report)}
    except (InvalidEvidence, OSError, KeyError, IndexError, TypeError, ValueError, StopIteration) as error:
        print(json.dumps({"validated_compiled_campaign": False, "output_written": False, "error": str(error)}), file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = args.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    print(json.dumps({"validated_compiled_campaign": True, "runs": list(report["arms"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
