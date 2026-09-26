#!/usr/bin/env python3
"""Validate the three candidate CPU-v2 arms and publish a separate CPU report.

python build_cpu_benchmark_report.py --campaign-metadata CAMPAIGN.json \
  --candidate-plan CANDIDATE_PLAN.json --environment environment-image.json \
  artifacts/NATIVE_DELTA artifacts/SYNTHETIC_BROADCAST artifacts/SYNTHETIC_DELTA

The candidate plan is the JSON consumed by run_cpu_campaign.py, with sources
{miles,sglang} and runs [{name,mode,reward_mode}]. This tool reads local artifacts
only. Missing/incomplete/invalid data produces no new report. v1 CPU is never
inferred. Run wall, training, then CPU summarizers before this report.
"""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import re
import sys

from build_benchmark_report import (
    InvalidEvidence,
    SCRIPT_NAMES,
    TOPOLOGY,
    distribution,
    finite_number,
    normalized_config,
    read_json,
    read_jsonl,
    require,
    sha256,
    table,
    validate_raw_timings,
    warning_inventory,
)
from summarize_training_evidence import engine_metadata, parse_log
from summarize_weight_sync_cpu_v2 import FIELDS, aggregate

ARMS = {
    ("native", "disk-delta"),
    ("synthetic-balanced", "broadcast"),
    ("synthetic-balanced", "disk-delta"),
}
CPU_HELPERS = {
    "launch_weight_sync_cpu_v2.py",
    "weight_sync_probe_cpu_v2.py",
    "train_weight_sync_profiled_cpu_v2.py",
    "process_cpu_clocks_cpu_v2.py",
}
FROZEN_MEASUREMENT_SHA256 = {
    "launch_weight_sync.py": "1e5c4a50a7f17fdcd61f3dc6875349ca7399667a4eb153c6525e193811ae5815",
    "weight_sync_probe.py": "1e3f80082af5821993c57b6efc9624e9b60418b2264513fb0b54442fb8bc7ed1",
    "train_weight_sync_profiled.py": "243b880d31a606ee2684635076040bd5c773eb2c6feb5c137fb7871514739051",
    "weight_sync_reward.py": "2f785003f111cee264f4aa880cdf14127f6240cc8e3971765d4c7567b782e49c",
    "launch_weight_sync_cpu_v2.py": "46f5b845dea53a59d6582f12f0346352adf8ed022900c6770b69d7be4eb6a2ab",
    "weight_sync_probe_cpu_v2.py": "760cc0a1645d5a0a4dd48eaf8a4f26eede18015b21d82cd87eee72e819593e55",
    "train_weight_sync_profiled_cpu_v2.py": "f8fb68508f7b3dd86979b3e66eb57dc265d11e726fdf34abf41f10270145f8ef",
    "process_cpu_clocks_cpu_v2.py": "4a3b3ec5cf3fd1c27096d9979098a80008f69f88306527b7a75dd7b0b36aab81",
}
METRICS = (
    "driver_wall_s",
    "builtin_actor_s",
    "trainer_max_wall_s",
    "generation_pause_upper_bound_s",
    "trainer_cpu_sum_s",
    "trainer_cpu_max_s",
    "receiver_total_cpu_s",
    "receiver_scheduler_cpu_s",
    "receiver_auxiliary_cpu_s",
    "driver_cpu_s",
)


def validate_manifest_contract(manifest):
    argv = manifest["train_argv"]
    expected = {
        "--num-rollout": "7",
        "--actor-num-nodes": "1",
        "--actor-num-gpus-per-node": "4",
        "--rollout-num-gpus": "4",
        "--rollout-num-gpus-per-engine": "2",
        "--tensor-model-parallel-size": "4",
        "--expert-model-parallel-size": "4",
        "--sglang-dp-size": "2",
        "--sglang-ep-size": "2",
        "--rm-type": "deepscaler",
        "--custom-megatron-init-path": "weight_sync_probe_cpu_v2.install",
        "--update-weight-transfer-mode": manifest["mode"],
        "--seed": str(manifest["seed"]),
        "--rollout-seed": str(manifest["rollout_seed"]),
    }
    if manifest["reward_mode"] == "synthetic-balanced":
        expected["--custom-rm-path"] = "weight_sync_reward.balanced_index_reward"
    else:
        require("--custom-rm-path" not in argv, "Native reward unexpectedly overridden")
    for flag, value in expected.items():
        require(
            argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value,
            f"Configured recipe mismatch: {flag}",
        )
    env = manifest["launch_kwargs"]["extra_env_vars"]
    require(
        "--bf16" in argv
        and env["OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"] == "1"
        and env["SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16"] == "1"
        and env["NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING"] == "1"
        and manifest["launch_kwargs"]["megatron_model_type"]
        == "glm5.2-744B-A40B_5layer"
        and Path(manifest["launch_kwargs"]["train_script"]).name
        == "train_weight_sync_profiled_cpu_v2.py",
        "Precision/model/CPU driver contract mismatch",
    )


def normalized_without_reward(manifest):
    normalized = normalized_config(manifest)
    normalized.pop("reward_mode")
    tokens, argv = iter(normalized["argv"]), []
    for token in tokens:
        if token == "--custom-rm-path":
            require(
                next(tokens, None) == "weight_sync_reward.balanced_index_reward",
                "Unexpected synthetic reward",
            )
        else:
            argv.append(token)
    normalized["argv"] = argv
    return normalized


def load_calibration(path):
    raw = read_json(path)
    require(
        raw["independent_psutil_inventory_matches"] is True
        and raw["process_count"] == len(raw["inventory"]) == 144
        and raw["scheduler_count"] == 4
        and Counter(process["role"] for process in raw["inventory"])
        == Counter(scheduler=4, auxiliary=140),
        "CPU calibration inventory is incomplete or differs from independent psutil",
    )
    costs = {}
    for field, expected_iterations in (
        ("self", 1000),
        ("receiver_snapshot_pair", 1000),
        ("membership_refresh_pair", 20),
    ):
        sample = raw[field]
        values = [sample[f"two_reads_{kind}_ns"] for kind in ("min", "median", "max")]
        require(
            sample["iterations"] == expected_iterations
            and all(finite_number(value) and value >= 0 for value in values)
            and values == sorted(values),
            f"Invalid CPU calibration cost: {field}",
        )
        costs[field] = sample
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "process_count": 144,
        "scheduler_count": 4,
        "auxiliary_count": 140,
        "independent_psutil_inventory_matches": True,
        "costs": costs,
        "scope": raw["scope"],
        "interpretation": (
            f"The {costs['membership_refresh_pair']['two_reads_median_ns'] / 1e6:.2f} ms refresh-pair median is approximately "
            f"{costs['membership_refresh_pair']['two_reads_median_ns'] / 1e9 / 1.27 * 100:.1f}% of a 1.27 s historical broadcast update. "
            "This is a scale comparison, not an observer correction or an observer-on/off result; actual v2 per-update bookkeeping costs remain authoritative and no cost is subtracted."
        ),
    }


def normalized_runtime(runtime):
    require(not runtime.get("metadata_error"), "CPU runtime metadata unavailable")
    require(runtime.get("affinity"), "Missing process CPU affinity")
    return {
        "cpu_count": runtime["cpu_count"],
        "affinity": runtime["affinity"],
        "process_clock": runtime["process_clock"],
        "cpu_configuration": {
            re.sub(r"/proc/(?:self|\d+)/cgroup", "/proc/PID/cgroup", key): value
            for key, value in runtime["cpu_configuration"].items()
        },
        "thread_environment": runtime["thread_environment"],
    }


def validate_runtime_sources(manifest, driver_events, trainer_events):
    """Setup-only runtime checks; receiver module imports are not independently attested."""
    setup = [
        event for event in driver_events if event["kind"] == "cpu_driver_installed"
    ]
    trainer_setup = [
        event for event in trainer_events if event["kind"] == "cpu_probe_installed"
    ]
    require(
        len(setup) == 1 and len(trainer_setup) == 4,
        "Missing/duplicate CPU process setup",
    )
    require(
        {event["rank"] for event in trainer_setup} == set(range(4)),
        "Missing CPU setup rank",
    )
    roots = {
        "miles": Path(manifest["repo"]) / "miles",
        "sglang": Path(manifest["sglang_source"]["repo"]) / "python/sglang",
    }
    runtimes = {"driver": setup[0]["runtime"]}
    runtimes.update(
        {f"trainer-rank{event['rank']}": event["runtime"] for event in trainer_setup}
    )
    loaded = {}
    for role, runtime in {"launcher": manifest["launcher_runtime"], **runtimes}.items():
        paths = runtime.get("imported_sources", {})
        if role != "launcher":
            require("miles" in paths, f"{role}: actual Miles module path missing")
        for name, path in paths.items():
            require(
                Path(path).is_relative_to(roots[name.split(".")[0]]),
                f"{role}: imported {name} outside declared checkout: {path}",
            )
        loaded[role] = paths
    require(
        any("sglang" in paths for paths in loaded.values()),
        "No loaded SGLang package path observed in Miles processes",
    )
    inventories = [
        event for event in trainer_events if event["kind"] == "cpu_receiver_inventory"
    ]
    require(inventories, "Missing receiver process inventory")
    by_process = {
        (process["pid"], process["start_ticks"]): process
        for event in inventories
        for process in event["processes"]
    }
    receiver_signatures = []
    for event in trainer_events:
        if (
            event["kind"] != "cpu_trainer_update"
            or event["rank"] != 0
            or event["update_index"] < 2
        ):
            continue
        signature = Counter()
        for process in event["receiver"]["processes"]:
            detail = by_process.get((process["pid"], process["start_ticks"]))
            require(
                detail is not None, "Receiver counter has no matching startup identity"
            )
            runtime = detail["runtime"]
            expected_path = str(roots["sglang"].parent)
            require(
                expected_path in (runtime.get("pythonpath") or "").split(":"),
                f"Receiver PID {process['pid']}: declared SGLang checkout missing from PYTHONPATH",
            )
            normalized = normalized_runtime(runtime)
            signature[(process["role"], json.dumps(normalized, sort_keys=True))] += 1
        receiver_signatures.append(
            [
                (role, json.loads(value), count)
                for (role, value), count in sorted(signature.items())
            ]
        )
    require(
        len(receiver_signatures) == 5
        and all(
            signature == receiver_signatures[0] for signature in receiver_signatures
        ),
        "Receiver runtime/membership counts changed across steady samples",
    )
    normalized = {
        role: normalized_runtime(runtime) for role, runtime in runtimes.items()
    }
    normalized["receiver_roles"] = receiver_signatures[0]
    quota_paths = {
        key
        for runtime in runtimes.values()
        for key in runtime["cpu_configuration"]
        if "cfs_" in key or key.endswith("cpu.max")
    }
    return {
        "normalized_roles": normalized,
        "loaded_module_paths": loaded,
        "receiver_source_check": "Configured SGLang checkout present in process PYTHONPATH; commands retained in inventory. Actual receiver loaded-module files were not independently sampled.",
        "quota_observation": sorted(quota_paths)
        if quota_paths
        else "CPU quota files not exposed at probed cgroup paths; cgroup membership and affinity retained, quota not independently verified.",
    }


def validate_cpu_clocks(drivers, trainers, wall_trainers):
    """Verify clock arithmetic independently of the derived CPU aggregator."""
    require(
        len(drivers) == 7 and len(trainers) == 28,
        "Expected seven driver and 28 trainer CPU records",
    )
    require(
        Counter(event["update_index"] for event in drivers) == Counter(range(7)),
        "Duplicate/missing driver CPU update",
    )
    require(
        Counter((event["update_index"], event["rank"]) for event in trainers)
        == Counter((index, rank) for index in range(7) for rank in range(4)),
        "Duplicate/missing trainer CPU rank",
    )
    original = {
        (event["update_index"], event["rank"]): event for event in wall_trainers
    }
    for event in drivers + trainers:
        index = event["update_index"]
        require(event["success"] is True, f"CPU update {index}: failed operation")
        if index >= 2:
            require(
                event["cpu_valid"] is True,
                f"Steady CPU update {index} invalid: {event.get('error')}",
            )
        if not event["cpu_valid"]:
            continue
        require(
            type(event["cpu_start_ns"]) is int
            and type(event["cpu_end_ns"]) is int
            and event["cpu_start_ns"] >= 0,
            f"CPU update {index}: missing integer clock counters",
        )
        elapsed = (event["cpu_end_ns"] - event["cpu_start_ns"]) / 1e9
        require(
            elapsed >= 0 and event["cpu_s"] == elapsed,
            f"CPU update {index}: counter difference mismatch",
        )
        require(
            finite_number(event["wall_s"])
            and event["wall_s"] >= 0
            and event["wall_s"] == (event["end_ns"] - event["start_ns"]) / 1e9,
            f"CPU update {index}: wall counter mismatch",
        )
        if event["kind"] != "cpu_trainer_update":
            continue
        old = original[(index, event["rank"])]
        require(
            all(
                finite_number(event[field]) and event[field] >= 0
                for field in ("observer_before_wall_ns", "observer_after_wall_ns")
            ),
            f"CPU update {index}: invalid observer duration",
        )
        require(
            old["start_ns"] <= event["start_ns"] <= event["end_ns"] <= old["end_ns"],
            f"CPU update {index}: trainer CPU interval is outside v1 wall interval",
        )
        require(
            set(event["resource_usage"])
            == {
                "ru_utime",
                "ru_stime",
                "ru_minflt",
                "ru_majflt",
                "ru_nvcsw",
                "ru_nivcsw",
            }
            and all(
                finite_number(value) and value >= 0
                for value in event["resource_usage"].values()
            ),
            f"CPU update {index}: invalid resource usage delta",
        )
        if event["rank"] != 0:
            continue
        receiver = event["receiver"]
        require(receiver is not None, f"CPU update {index}: missing receiver")
        before, after = receiver["before"], receiver["after"]
        for snapshot in (before, after):
            require(
                snapshot["duration_ns"]
                == snapshot["end_ns"] - snapshot["start_ns"]
                >= 0,
                "Receiver snapshot duration mismatch",
            )
        require(
            before["end_ns"] <= event["start_ns"]
            and after["start_ns"] >= event["end_ns"],
            "Receiver snapshots do not bracket trainer-rank0 updater",
        )
        processes = receiver["processes"]
        pids = [str(process["pid"]) for process in processes]
        require(
            len(pids) == len(set(pids))
            and set(pids) == set(before["cpu_ns"]) == set(after["cpu_ns"]),
            "Receiver process membership/counter mismatch",
        )
        require(
            sum(process["role"] == "scheduler" for process in processes) == 4
            and len({process["engine"] for process in processes}) == 2,
            "Receiver CPU requires four schedulers across two engines",
        )
        totals = Counter()
        for process in processes:
            pid = str(process["pid"])
            require(
                type(before["cpu_ns"][pid]) is int
                and type(after["cpu_ns"][pid]) is int
                and before["cpu_ns"][pid] >= 0,
                f"Receiver PID {pid}: invalid absolute CPU clocks",
            )
            delta = after["cpu_ns"][pid] - before["cpu_ns"][pid]
            require(
                delta >= 0 and delta == process["cpu_ns"],
                f"Receiver PID {pid}: CPU counter difference mismatch",
            )
            require(
                process["role"] in ("scheduler", "auxiliary"),
                "Unexpected receiver role",
            )
            totals[process["role"]] += delta
        require(
            receiver["scheduler_cpu_ns"] == totals["scheduler"]
            and receiver["auxiliary_cpu_ns"] == totals["auxiliary"]
            and receiver["total_cpu_ns"] == sum(totals.values()),
            "Receiver CPU sum mismatch",
        )
        require(
            receiver["snapshot_cost_ns"]
            == before["duration_ns"] + after["duration_ns"],
            "Receiver snapshot cost mismatch",
        )


def load_arm(run, plan):
    run = run.resolve()
    manifest = read_json(run / "manifest.json")
    wall = read_json(run / "summary.json")
    cpu = read_json(run / "cpu-summary.json")
    evidence = read_json(run / "training-evidence.json")
    validate_manifest_contract(manifest)
    require(
        manifest.get("cpu_observer_version") == 2
        and manifest.get("implementation_label") == "candidate",
        f"{run.name}: require candidate CPU-v2",
    )
    require(
        (manifest["reward_mode"], manifest["mode"]) in ARMS
        and manifest["num_rollout"] == 7
        and manifest["topology"] == TOPOLOGY,
        f"{run.name}: unexpected workload/topology",
    )
    require(
        manifest["cpu_observer"]["backend"] == "process-clock"
        and manifest["cpu_observer"]["receiver_policy"] == "required",
        f"{run.name}: high-resolution required receiver CPU clocks unavailable",
    )
    require(
        manifest["git_head"] == plan["sources"]["miles"]
        and manifest["sglang_source"]["git_head"] == plan["sources"]["sglang"],
        f"{run.name}: candidate source mismatch",
    )
    require(
        manifest["git_diff"] == manifest["sglang_source"]["git_diff"] == "",
        f"{run.name}: dirty tracked source",
    )
    require(
        set(manifest["script_sha256"]) == SCRIPT_NAMES | CPU_HELPERS,
        f"{run.name}: missing helper identity",
    )
    require(
        manifest["script_sha256"] == FROZEN_MEASUREMENT_SHA256,
        f"{run.name}: helper hashes differ from calibrated final freeze",
    )
    require(
        all(
            re.fullmatch(r"[0-9a-f]{64}", value)
            for value in manifest["script_sha256"].values()
        ),
        f"{run.name}: malformed helper SHA256",
    )
    require(cpu["manifest"] == manifest, f"{run.name}: CPU summary has stale manifest")
    require(
        cpu["complete_cpu_steady"] is True
        and cpu["comparison_eligible"] is True
        and cpu["e2e_complete"] is True,
        f"{run.name}: CPU summary incomplete",
    )
    log_path, exit_path = (
        run.parent / f"{run.name}.log",
        run.parent / f"{run.name}.exit",
    )
    require(
        exit_path.is_file() and exit_path.read_text().strip() == "0",
        f"{run.name}: no final exit0",
    )
    require(
        sha256(log_path) == evidence["log_sha256"],
        f"{run.name}: stale training evidence",
    )
    parsed = parse_log(log_path)
    require(
        parsed["ray_terminal"]
        and parsed["ray_terminal"][-1]["status"] == "succeeded"
        and evidence["ray_terminal"] == parsed["ray_terminal"][-1],
        f"{run.name}: missing final Ray success",
    )
    train = parsed["train_events"]
    require(
        len(train) == 28
        and {(event["rollout"], event["rank"]) for event in train}
        == {(rollout, rank) for rollout in range(7) for rank in range(4)},
        f"{run.name}: incomplete training coverage",
    )
    require(
        all(
            event["attempt"] == 0
            and event["step"] == 0
            and event["valid_step"]
            and event["outcome"] == "NORMAL"
            for event in train
        ),
        f"{run.name}: non-normal/retried training",
    )
    require(
        evidence["run_succeeded"]
        and evidence["all_expected_rank_rollouts_completed"]
        and train == evidence["parsed_log_evidence"]["train_events"],
        f"{run.name}: inconsistent training evidence",
    )
    require(
        not parsed["metric_parse_errors"]
        and not parsed["conflicting_step_metric_lines"],
        f"{run.name}: invalid log metrics",
    )
    require(
        len(parsed["rank0_step_metrics"]) == len(evidence["rank0_steps"]) == 7,
        f"{run.name}: missing rank0 step metrics",
    )
    for raw, retained in zip(
        parsed["rank0_step_metrics"], evidence["rank0_steps"], strict=True
    ):
        require(
            all(retained[key] == value for key, value in raw["metrics"].items()),
            f"{run.name}: stale step metrics",
        )
    version_metrics = parsed["rollout_service_metrics"]
    require(
        len(version_metrics) == 7
        and all(
            record["metrics"]["rollout/weight_version/mixed_version_ratio"] == 0
            and record["metrics"]["rollout/weight_version/min"]
            == record["metrics"]["rollout/weight_version/max"]
            for record in version_metrics
        ),
        f"{run.name}: missing/mixed rollout versions",
    )
    dumps = evidence["debug_dumps"]
    require(
        dumps["status"] == "parsed"
        and not dumps["errors"]
        and len(dumps["train"]) == len(dumps["rollout"]) == 7,
        f"{run.name}: incomplete debug evidence",
    )
    require(
        {record["rollout"] for record in dumps["train"]} == set(range(7)),
        f"{run.name}: train dump coverage mismatch",
    )
    advantages = [
        record["fields"]["advantages"]["unmasked"]
        for record in sorted(dumps["train"], key=lambda record: record["rollout"])
    ]
    require(
        all(record["count"] == record["finite_count"] > 0 for record in advantages),
        f"{run.name}: nonfinite/empty advantage evidence",
    )
    gradients = [record["train/grad_norm"] for record in evidence["rank0_steps"]]
    require(
        all(finite_number(value) and value >= 0 for value in gradients),
        f"{run.name}: invalid gradients",
    )
    if manifest["reward_mode"] == "synthetic-balanced":
        require(
            all(value > 0 for value in gradients)
            and all(
                record["mean_abs"] > 0 and record["nonzero_count"] > 0
                for record in advantages
            ),
            f"{run.name}: synthetic gradients/absolute advantages must be nonzero in every rollout",
        )
    else:
        require(
            all(value == 0 for value in gradients)
            and all(
                record["mean_abs"] == 0 and record["nonzero_count"] == 0
                for record in advantages
            ),
            f"{run.name}: native run is not the expected zero-gradient validation",
        )
    validate_raw_timings(run, manifest, wall)
    driver_all = read_jsonl(run / "cpu-driver.jsonl")
    trainer_all = [
        event
        for rank in range(4)
        for event in read_jsonl(run / f"cpu-trainer-rank{rank}.jsonl")
    ]
    drivers = [event for event in driver_all if event["kind"] == "cpu_driver_update"]
    trainers = [event for event in trainer_all if event["kind"] == "cpu_trainer_update"]
    raw_wall = [
        event
        for rank in range(4)
        for event in read_jsonl(run / f"trainer-rank{rank}.jsonl")
        if event["kind"] == "trainer_update"
    ]
    wall_drivers = [
        event
        for event in read_jsonl(run / "driver.jsonl")
        if event["kind"] == "driver_update"
    ]
    require(
        len(wall_drivers) == 7
        and all(
            event["start_ns"] == wall_drivers[event["update_index"]]["start_ns"]
            and event["end_ns"] == wall_drivers[event["update_index"]]["end_ns"]
            for event in drivers
        ),
        f"{run.name}: CPU and wall driver timestamps differ",
    )
    validate_cpu_clocks(drivers, trainers, raw_wall)
    runtime = validate_runtime_sources(manifest, driver_all, trainer_all)
    fresh = aggregate(manifest, wall, driver_all, trainer_all)
    require(
        fresh["complete_cpu_steady"]
        and fresh["rows"] == cpu["rows"]
        and fresh["steady"] == cpu["steady"],
        f"{run.name}: derived CPU summary differs from raw journals",
    )
    require(
        all(stats["count"] == 5 for stats in cpu["steady"].values()),
        f"{run.name}: expected exactly five valid steady CPU samples",
    )
    builtin = [
        record
        for record in parsed["native_perf_metrics"]
        if "perf/update_weights_time" in record["metrics"]
    ]
    require(
        [record["index"] for record in builtin] == list(range(7))
        and all(
            finite_number(record["metrics"]["perf/update_weights_time"])
            and record["metrics"]["perf/update_weights_time"] > 0
            for record in builtin
        ),
        f"{run.name}: missing builtin actor times",
    )
    rows = [
        dict(
            row,
            builtin_actor_s=record["metrics"]["perf/update_weights_time"],
            builtin_log_line=record["line"],
        )
        for row, record in zip(cpu["rows"], builtin, strict=True)
    ]
    for row in rows:
        if manifest["mode"] == "disk-delta" and row["update_index"]:
            require(
                row["changed_bytes"] == 0
                if manifest["reward_mode"] == "native"
                else row["changed_bytes"] > 0,
                f"{run.name}: exact changed bytes contradict workload classification",
            )
    steady = {
        metric: distribution([row[metric] for row in rows[2:]])
        for metric in dict.fromkeys(METRICS + FIELDS)
    }
    overhead = {
        "snapshot_to_driver_ratios": [
            row["receiver_snapshot_cost_s"] / row["driver_wall_s"] for row in rows[2:]
        ],
        "outside_trainer_wall_to_driver_ratios": [
            row["observer_outside_trainer_wall_max_s"] / row["driver_wall_s"]
            for row in rows[2:]
        ],
        "initial_or_first_post_cpu_invalid": cpu["invalid_initial_or_first_post"],
        "setup_records": cpu["setup_records"],
        "journal_flush_cost": "previous_flush_ns is a lagged measurement; final flush cost is not available until a later write. It is not added/subtracted as current-update overhead.",
    }
    paths = [
        run / name
        for name in (
            "manifest.json",
            "summary.json",
            "cpu-summary.json",
            "training-evidence.json",
            "driver.jsonl",
            "cpu-driver.jsonl",
            *(f"trainer-rank{rank}.jsonl" for rank in range(4)),
            *(f"cpu-trainer-rank{rank}.jsonl" for rank in range(4)),
        )
    ] + [log_path, exit_path]
    raw_dumps = sorted((run / "dump_details/train_data").glob("*.pt")) + sorted(
        (run / "dump_details/rollout_data").glob("*.pt")
    )
    require(len(raw_dumps) == 14, f"{run.name}: missing retained debug dumps")
    return {
        "name": run.name,
        "manifest": manifest,
        "validated": True,
        "rows": rows,
        "steady": steady,
        "input_sha256": {str(path): sha256(path) for path in paths + raw_dumps},
        "cpu_observer": overhead,
        "runtime_identity": runtime,
        "training": {
            "gradient_norms": gradients,
            "unmasked_advantages": advantages,
            "rank0_steps": evidence["rank0_steps"],
            "replay_checks": sum(record["checks"] for record in parsed["replay"]),
            "replay_nonzero_checks": sum(
                record["nonzero_checks"] for record in parsed["replay"]
            ),
            "rollout_debug": dumps["rollout"],
            "rank_returns": 28,
            "rollouts": 7,
            "ray": evidence["ray_terminal"],
        },
        "engines": engine_metadata(run),
        "warnings": warning_inventory(log_path, parsed),
    }


def build(runs, campaign_path, plan_path, environment_path, calibration_path=None):
    require(
        len(runs) == len({run.resolve() for run in runs}) == 3,
        "Exactly three distinct candidate CPU-v2 arms are required",
    )
    campaign, plan, environment = (
        read_json(path) for path in (campaign_path, plan_path, environment_path)
    )
    calibration_path = (
        calibration_path
        or campaign_path.parent / "artifacts/cpu-observer-fast-final-calibration.json"
    )
    calibration = load_calibration(calibration_path)
    calibration["report_link"] = (
        str(calibration_path.resolve().relative_to(campaign_path.parent.resolve()))
        if calibration_path.resolve().is_relative_to(campaign_path.parent.resolve())
        else str(calibration_path.resolve())
    )
    require(campaign["topology"] == TOPOLOGY, "Campaign topology mismatch")
    gpu_rows = list(
        csv.DictReader(
            io.StringIO(environment["gpus"]["stdout"]), skipinitialspace=True
        )
    )
    require(
        environment["gpus"]["exit_code"] == 0
        and len(gpu_rows) == 8
        and all("B300" in row["name"] for row in gpu_rows),
        "Environment must verify eight B300 GPUs",
    )
    arms = [load_arm(run, plan) for run in runs]
    keyed = {
        (arm["manifest"]["reward_mode"], arm["manifest"]["mode"]): arm for arm in arms
    }
    require(set(keyed) == ARMS, "Expected native delta and synthetic broadcast/delta")
    planned = {run["name"]: (run["reward_mode"], run["mode"]) for run in plan["runs"]}
    require(
        all(
            planned.get(arm["name"])
            == (arm["manifest"]["reward_mode"], arm["manifest"]["mode"])
            for arm in arms
        ),
        "Run does not match candidate plan",
    )
    require(
        all(
            arm["manifest"]["script_sha256"] == arms[0]["manifest"]["script_sha256"]
            and arm["manifest"]["cpu_observer"] == arms[0]["manifest"]["cpu_observer"]
            for arm in arms
        ),
        "Candidate helper hashes or CPU observer settings differ",
    )
    broadcast, delta = (
        keyed[("synthetic-balanced", "broadcast")],
        keyed[("synthetic-balanced", "disk-delta")],
    )
    require(
        normalized_config(broadcast["manifest"])
        == normalized_config(delta["manifest"]),
        "Synthetic pair configuration differs beyond transport/output",
    )
    require(
        all(
            normalized_without_reward(arm["manifest"])
            == normalized_without_reward(arms[0]["manifest"])
            for arm in arms
        ),
        "Candidate workload configurations differ beyond explicit reward/transport/output",
    )
    require(
        broadcast["runtime_identity"]["normalized_roles"]
        == delta["runtime_identity"]["normalized_roles"],
        "Synthetic pair process affinity/cgroup/thread environment differs",
    )
    comparison = {
        metric: {
            "delta_over_broadcast": delta["steady"][metric]["median"]
            / broadcast["steady"][metric]["median"]
            if broadcast["steady"][metric]["median"]
            else None,
            "broadcast": broadcast["steady"][metric],
            "disk_delta": delta["steady"][metric],
        }
        for metric in METRICS
    }
    stable_environment = {
        key: campaign[key]
        for key in (
            "image",
            "image_index_digest",
            "image_amd64_digest",
            "cluster",
            "host",
            "devbox",
            "megatron_head",
            "hf_source",
            "hf_revision",
            "topology",
        )
    }
    return {
        "schema_version": 1,
        "validated_three_candidate_arms": True,
        "campaign": stable_environment,
        "candidate_plan": plan,
        "environment": environment,
        "metadata_sha256": {
            str(path.resolve()): sha256(path)
            for path in (campaign_path, plan_path, environment_path)
        },
        "generator_sha256": sha256(Path(__file__)),
        "frozen_measurement_sha256": FROZEN_MEASUREMENT_SHA256,
        "cpu_calibration": calibration,
        "arms": {arm["name"]: arm for arm in arms},
        "synthetic_comparison": {
            "broadcast": broadcast["name"],
            "disk_delta": delta["name"],
            "metrics": comparison,
        },
        "native_scope": "Standalone final zero-change path validation; no candidate native broadcast was supplied.",
        "historical_v1_cpu": "unavailable; no end-to-end CPU reduction versus the prior four-arm v1 campaign is claimed",
        "component_replay": "Separate baseline/candidate component CPU evidence; not an input to or substitute for this E2E transport comparison",
        "limitations": [
            "One C2 node, eight B300 GPUs, 4+4 split with two TP2 rollout engines; local shared storage. No cross-node or cross-cluster claim.",
            "This is a correctness and local overhead benchmark for future large multi-node use. No cross-node speedup or amortization projection is made; beating local broadcast is not an admission requirement.",
            "Synthetic balanced reward is an explicit transport-validation workload, not learned-task quality. Every synthetic rollout must have nonzero gradient norm and nonzero absolute unmasked advantages; every post-training delta must contain nonzero changed bytes.",
            "Trainer CPU uses all-thread process clocks inside each updater. Receiver CPU brackets trainer-rank0 updater; auxiliary includes data-parallel controllers and inventoried compile workers. Driver CPU is separate. These windows differ slightly and should not be treated as an exact simultaneous whole-host total.",
            "Driver wall and builtin actor time include observer discovery/validation and trainer CPU journal flushing. Trainer wall excludes outer discovery/validation/flush but includes CPU snapshots. Actual refresh/snapshot costs are shown; no guessed overhead is subtracted. Other waiting ranks can experience observer-induced synchronization delay.",
            "Cached clock-read calibration does not bound membership-refresh cost. Runtime bookkeeping ratios are reported separately; a large ratio limits latency interpretation. Previous journal flush cost is lagged, not an exact same-update correction.",
            "Final idle-engine calibration costs are retained above. Discovery is material for short local broadcast updates; it is included in driver wall, not called negligible or subtracted.",
            "Receiver trees are checked before and after each update using live, non-atomic /proc child lists. A missing previously known PID still alive with the same start time invalidates the scan. Entirely transient children, or newly born children omitted during concurrent exits, may remain unobserved. Receiver source is verified as the configured launch/PYTHONPATH checkout, not independently attested loaded-module files. CPU quota may be unavailable in the container; observed cgroup metadata and per-role affinity/thread settings are compared.",
            "Five steady samples are update indices2..6. Initial sync and first post-training update remain in raw tables but not steady medians. Invalid early CPU observations are disclosed; any invalid steady CPU observation blocks this report.",
            "Delta wire bytes are serialized safetensor lengths, excluding index JSON and baseline checkpoint copy; they are not measured physical network bytes. Broadcast changed/wire bytes are unavailable, not zero.",
            "Version acknowledgements, byte checksums, replay, and subsequent successful training do not establish active GPU byte equality or exact numerical equivalence. Original recipe weight/logprob/KL CI checkers remain disabled.",
            "The earlier v1 campaign lacks CPU counters. This report compares candidate synthetic delta against candidate synthetic broadcast; it does not establish before/after E2E CPU savings. CPU-only component replay remains separately labeled.",
        ],
    }


def markdown(report):
    campaign = report["campaign"]
    lines = [
        "# Candidate CPU and wall benchmark",
        "",
        "Three candidate runs passed completion, source/helper/configuration, raw CPU counter, and five-steady-sample checks. Synthetic delta is compared with synthetic broadcast. Native delta is standalone zero-change path validation.",
        "",
        "**Earlier v1 CPU measurements are unavailable. This report makes no before/after E2E CPU reduction claim.**",
        "",
        f"Environment: {campaign['cluster']} / {campaign['host']} / {campaign['devbox']}; 8 B300, 4 trainer + 4 rollout GPUs, two TP2 engines. Image `{campaign['image']}`; index `{campaign['image_index_digest']}`; amd64 `{campaign['image_amd64_digest']}`.",
        "",
        f"Candidate source pins: `{report['candidate_plan']['sources']}`; Megatron `{campaign['megatron_head']}`. Full configs, helper hashes, package versions, input SHA256 values and process inventories are retained in the JSON.",
        "",
        "## CPU observer calibration",
        "",
        f"[Final calibration]({report['cpu_calibration']['report_link']}) independently matched 144 processes: four schedulers and 140 auxiliary processes. SHA256 `{report['cpu_calibration']['sha256']}`. All run helper hashes must match the calibrated final freeze.",
        "",
    ]
    lines += table(
        ["Two-read operation", "N", "Median ms", "Min ms", "Max ms"],
        [
            [
                name,
                sample["iterations"],
                sample["two_reads_median_ns"] / 1e6,
                sample["two_reads_min_ns"] / 1e6,
                sample["two_reads_max_ns"] / 1e6,
            ]
            for name, sample in report["cpu_calibration"]["costs"].items()
        ],
    )
    lines += [
        "",
        report["cpu_calibration"]["interpretation"],
        "",
        "## Synthetic transport comparison",
        "",
        "Ratios are delta / broadcast median. Wall values are seconds; process CPU values are CPU-seconds summed over the named processes/threads.",
        "",
    ]
    lines += table(
        [
            "Metric",
            "Broadcast median",
            "Broadcast min–max",
            "Delta median",
            "Delta min–max",
            "Delta / broadcast",
        ],
        [
            [
                metric,
                value["broadcast"]["median"],
                f"{value['broadcast']['min']}–{value['broadcast']['max']}",
                value["disk_delta"]["median"],
                f"{value['disk_delta']['min']}–{value['disk_delta']['max']}",
                value["delta_over_broadcast"],
            ]
            for metric, value in report["synthetic_comparison"]["metrics"].items()
        ],
    )
    for arm in report["arms"].values():
        lines += [
            "",
            f"## {arm['name']}",
            "",
            "Exit0 and final Ray success; seven rollouts, 28 normal attempt0 trainer returns; seven raw updates including initial sync. Steady samples are indices2–6.",
            "",
        ]
        lines += table(
            [
                "Update",
                "Driver s",
                "Builtin actor s",
                "Trainer CPU s",
                "Receiver CPU s",
                "Scheduler CPU s",
                "Auxiliary CPU s",
                "Pause s",
                "Changed bytes",
                "Wire bytes",
                "CPU valid",
            ],
            [
                [
                    row["update_index"],
                    row["driver_wall_s"],
                    row["builtin_actor_s"],
                    row.get("trainer_cpu_sum_s"),
                    row.get("receiver_total_cpu_s"),
                    row.get("receiver_scheduler_cpu_s"),
                    row.get("receiver_auxiliary_cpu_s"),
                    row["generation_pause_upper_bound_s"],
                    row["changed_bytes"],
                    row["wire_bytes"],
                    row["cpu_valid"],
                ]
                for row in arm["rows"]
            ],
        )
        lines += ["", "Steady medians and ranges:", ""]
        lines += table(
            ["Metric", "N", "Median", "Min", "Max"],
            [
                [metric, stats["count"], stats["median"], stats["min"], stats["max"]]
                for metric, stats in arm["steady"].items()
            ],
        )
        training, overhead = arm["training"], arm["cpu_observer"]
        lines += [
            "",
            f"Gradient norms: `{training['gradient_norms']}`. Unmasked advantage mean absolute values: `{[row['mean_abs'] for row in training['unmasked_advantages']]}`. Replay: {training['replay_checks']} checks, {training['replay_nonzero_checks']} with mismatches; counts repeat across ranks/stages/layers.",
            "",
            f"Snapshot cost / driver wall, raw steady ratios: `{overhead['snapshot_to_driver_ratios']}`. Outside-trainer-wall bookkeeping max / driver wall: `{overhead['outside_trainer_wall_to_driver_ratios']}`. These are observation costs, not subtracted corrections. Invalid initial/first-post CPU indices: `{overhead['initial_or_first_post_cpu_invalid']}`.",
            "",
            f"Warning/error counts by log stage: `{arm['warnings']['stage_counts']}`. Readiness retries: `{arm['warnings']['readiness_retry_counts']}`. Full line-numbered warnings/errors, CPU setup calibration, per-process clocks, and raw resource usage counters remain in the retained artifacts and JSON.",
            "",
            f"Receiver source check: {arm['runtime_identity']['receiver_source_check']} CPU quota observation: {arm['runtime_identity']['quota_observation']}",
            "",
        ]
    lines += ["## Boundaries", ""] + [f"- {item}" for item in report["limitations"]]
    lines += [
        "",
        "CPU-only sender/receiver component replay is separate evidence and is not combined into these E2E ratios.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--campaign-metadata", required=True, type=Path)
    parser.add_argument("--candidate-plan", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument(
        "--cpu-calibration",
        type=Path,
        help="Defaults to artifacts/cpu-observer-fast-final-calibration.json beside campaign metadata",
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        report = build(
            args.runs,
            args.campaign_metadata,
            args.candidate_plan,
            args.environment,
            args.cpu_calibration,
        )
        text = markdown(report)
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    except (
        InvalidEvidence,
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "validated_three_candidate_arms": False,
                    "error": str(error),
                    "output_written": False,
                }
            ),
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("CPU_BENCHMARK_RESULTS.md", text),
        ("CPU_BENCHMARK_RESULTS.json", serialized),
    ):
        path = args.output_dir / name
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(content)
        temp.replace(path)
    print(
        json.dumps(
            {
                "validated_three_candidate_arms": True,
                "runs": list(report["arms"]),
                "output_dir": str(args.output_dir.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
