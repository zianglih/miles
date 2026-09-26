#!/usr/bin/env python3
"""Validate four retained arms, then write BENCHMARK_RESULTS.md and .json.

This reads local artifacts only and never refreshes summaries or accesses jobs.
Run summarize_training_evidence.py with debug dumps first for every arm.
"""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys

from summarize_training_evidence import ANSI, engine_metadata, parse_log


ARM_KEYS = {
    (reward, mode)
    for reward in ("native", "synthetic-balanced")
    for mode in ("broadcast", "disk-delta")
}
TOPOLOGY = {
    "training_gpus": 4,
    "rollout_gpus": 4,
    "rollout_engines": 2,
    "gpus_per_engine": 2,
}
SCRIPT_NAMES = {
    "launch_weight_sync.py",
    "weight_sync_probe.py",
    "train_weight_sync_profiled.py",
    "weight_sync_reward.py",
}
TIME_METRICS = (
    "driver_wall_s",
    "builtin_actor_s",
    "trainer_max_wall_s",
    "generation_pause_upper_bound_s",
)
RPC_ENDPOINTS = (
    "pull_weights",
    "update_weights_from_disk",
    "begin_weight_update",
    "end_weight_update",
)
PHASES = (
    "_capture_baseline",
    "_begin_encode",
    "after_base_weights",
    "_write_delta_files",
    "_reload_engines",
)
PHASE_METRICS = tuple(f"{name}_max_rpc_s" for name in RPC_ENDPOINTS) + tuple(
    f"delta{name}_max_s" for name in PHASES
)


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def read_json(path):
    require(path.is_file(), f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    require(path.is_file(), f"Missing raw journal: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def distribution(values):
    require(
        all(finite_number(value) for value in values),
        "Nonfinite or missing measurement",
    )
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "raw": values,
    }


def normalized_config(manifest):
    ignored = {
        "--update-weight-transfer-mode",
        "--update-weight-disk-dir",
        "--update-weight-local-checkpoint-dir",
        "--save-debug-rollout-data",
        "--save-debug-train-data",
        "--save-debug-trajectory-data",
    }
    tokens = iter(manifest["train_argv"])
    argv = []
    for token in tokens:
        if token in ignored:
            require(next(tokens, None) is not None, f"Missing argv value for {token}")
        else:
            argv.append(token)
    launch = {
        key: value
        for key, value in manifest["launch_kwargs"].items()
        if key not in ("train_args", "extra_env_vars")
    }
    launch["extra_env_vars"] = {
        key: value
        for key, value in manifest["launch_kwargs"]["extra_env_vars"].items()
        if key not in ("WEIGHT_SYNC_MODE", "WEIGHT_SYNC_RUN_DIR")
    }
    return {
        "argv": argv,
        "launch": launch,
        "topology": manifest["topology"],
        "seed": manifest["seed"],
        "rollout_seed": manifest["rollout_seed"],
        "reward_mode": manifest["reward_mode"],
        "num_rollout": manifest["num_rollout"],
    }


def warning_inventory(log_path, parsed):
    lines = log_path.read_text(errors="replace").split("\n")
    first_train = next(
        (
            index
            for index, line in enumerate(lines, 1)
            if "ft op=train phase=start" in line
        ),
        len(lines) + 1,
    )
    last_train = max(record["line"] for record in parsed["train_events"])
    pattern = re.compile(
        r"\bWARNING\b|\bWARN\b|\bERROR\b|\bUserWarning:|\bFutureWarning:|"
        r"\bDeprecationWarning:|\b(?:[A-Za-z]+Error|Exception):|"
        r"post-warmup freeze_gc failed|phase=attempt_failed"
    )
    records = []
    readiness = Counter()
    for index, raw in enumerate(lines, 1):
        line = ANSI.sub("", raw)
        if "phase=attempt_failed" in line and (match := re.search(r"op=(\w+)", line)):
            readiness[match.group(1)] += 1
        if pattern.search(line):
            records.append(
                {
                    "line": index,
                    "stage": "startup"
                    if index < first_train
                    else "teardown"
                    if index > last_train
                    else "training",
                    "text": line[:1600],
                }
            )
    return {
        "records": records,
        "stage_counts": dict(Counter(record["stage"] for record in records)),
        "readiness_retry_counts": dict(readiness),
        "freeze_gc_failure_lines": [
            index
            for index, line in enumerate(lines, 1)
            if "post-warmup freeze_gc failed" in line
        ],
        "note": "Stage labels follow log ordering around the first training start and last train-step return; warnings are retained, not all classified as fatal.",
    }


def validate_raw_timings(run_dir, manifest, summary):
    drivers = sorted(
        read_jsonl(run_dir / "driver.jsonl"), key=lambda event: event["start_ns"]
    )
    events = [
        event
        for rank in range(4)
        for event in read_jsonl(run_dir / f"trainer-rank{rank}.jsonl")
    ]
    drivers = [event for event in drivers if event["kind"] == "driver_update"]
    require(len(drivers) == 7, f"{run_dir.name}: expected exactly 7 raw driver updates")
    require(
        len(summary["rows"]) == 7, f"{run_dir.name}: expected exactly 7 summary rows"
    )
    require(summary["complete"] is True, f"{run_dir.name}: timing summary incomplete")
    require(
        summary["expected_post_training_updates"]
        == summary["observed_successful_post_training_updates"]
        == 6,
        f"{run_dir.name}: post-training update count mismatch",
    )
    require(
        not any(event.get("success") is False for event in drivers + events),
        f"{run_dir.name}: failed observed operation",
    )
    trainers = [event for event in events if event["kind"] == "trainer_update"]
    require(len(trainers) == 28, f"{run_dir.name}: expected 28 raw updater returns")
    for index, (outer, row) in enumerate(zip(drivers, summary["rows"], strict=True)):
        context = f"{run_dir.name} update {index}"
        require(
            row["update_index"] == index and row["initial"] == (index == 0),
            f"{context}: index/initial mismatch",
        )
        require(
            row["rollout_id"] == (None if index == 0 else index - 1),
            f"{context}: rollout mapping mismatch",
        )
        require(
            row["success"] is True and outer["success"] is True,
            f"{context}: failed update",
        )
        require(
            outer["initial"] == row["initial"]
            and outer["rollout_id"] == row["rollout_id"],
            f"{context}: raw driver mapping mismatch",
        )
        matched = [event for event in events if event.get("update_index") == index]
        train = [event for event in matched if event["kind"] == "trainer_update"]
        require(
            len(train) == 4 and {event["rank"] for event in train} == set(range(4)),
            f"{context}: missing/duplicate trainer rank",
        )
        require(
            row["trainer_rank_count"] == 4, f"{context}: summary rank count mismatch"
        )
        require(
            row["driver_wall_s"] == outer["elapsed_s"]
            and row["trainer_max_wall_s"] == max(event["elapsed_s"] for event in train),
            f"{context}: raw timing mismatch",
        )
        rpcs = [event for event in matched if event["kind"] == "engine_rpc"]
        require(
            len({(event["engine"], event["endpoint"]) for event in rpcs}) == len(rpcs),
            f"{context}: repeated observed engine RPC endpoint",
        )
        pauses = [event for event in rpcs if event["endpoint"] == "pause_generation"]
        resumes = [
            event for event in rpcs if event["endpoint"] == "continue_generation"
        ]
        if index:
            require(
                len(pauses) == len(resumes) == 2
                and {event["engine"] for event in pauses}
                == {event["engine"] for event in resumes}
                and len({event["engine"] for event in pauses}) == 2,
                f"{context}: missing/duplicate pause/resume",
            )
        pause = (
            (
                max(event["end_ns"] for event in resumes)
                - min(event["start_ns"] for event in pauses)
            )
            / 1e9
            if pauses and resumes
            else None
        )
        require(
            row["generation_pause_upper_bound_s"] == pause,
            f"{context}: raw pause mismatch",
        )
        for endpoint in RPC_ENDPOINTS:
            value = max(
                (event["elapsed_s"] for event in rpcs if event["endpoint"] == endpoint),
                default=None,
            )
            require(
                row[f"{endpoint}_max_rpc_s"] == value,
                f"{context}: {endpoint} timing mismatch",
            )
        for phase in PHASES:
            value = max(
                (
                    event["elapsed_s"]
                    for event in matched
                    if event["kind"] == "delta_phase" and event["phase"] == phase
                ),
                default=None,
            )
            require(
                row[f"delta{phase}_max_s"] == value,
                f"{context}: {phase} timing mismatch",
            )
        byte_events = [event for event in matched if event["kind"] == "delta_bytes"]
        if manifest["mode"] == "disk-delta" and index:
            require(
                len(byte_events) == 4
                and {event["rank"] for event in byte_events} == set(range(4)),
                f"{context}: missing/duplicate exact delta byte record",
            )
        for field in ("changed_bytes", "total_bytes", "wire_bytes"):
            value = sum(event[field] for event in byte_events) if byte_events else None
            require(row[field] == value, f"{context}: exact {field} mismatch")
        if byte_events:
            require(
                0 <= row["changed_bytes"] <= row["total_bytes"]
                and row["total_bytes"] > 0
                and row["wire_bytes"] >= 0,
                f"{context}: invalid byte counters",
            )


def load_arm(run_dir, campaign):
    run_dir = run_dir.resolve()
    context = run_dir.name
    manifest = read_json(run_dir / "manifest.json")
    summary = read_json(run_dir / "summary.json")
    evidence = read_json(run_dir / "training-evidence.json")
    require(
        (manifest.get("reward_mode"), manifest["mode"]) in ARM_KEYS,
        f"{context}: unexpected arm",
    )
    require(
        manifest["num_rollout"] == 7
        and manifest["expected_post_training_updates"] == 6,
        f"{context}: requires seven-rollout campaign",
    )
    require(manifest["topology"] == TOPOLOGY, f"{context}: requires exact 4+4 topology")
    argv = manifest["train_argv"]
    for flag, expected in (
        ("--num-rollout", "7"),
        ("--update-weight-transfer-mode", manifest["mode"]),
        ("--rm-type", "deepscaler"),
        ("--seed", str(manifest["seed"])),
        ("--rollout-seed", str(manifest["rollout_seed"])),
        ("--custom-megatron-init-path", "weight_sync_probe.install"),
    ):
        require(
            argv.count(flag) == 1 and argv[argv.index(flag) + 1] == expected,
            f"{context}: inconsistent {flag}",
        )
    custom_reward = "--custom-rm-path"
    if manifest["reward_mode"] == "native":
        require(custom_reward not in argv, f"{context}: native arm has custom reward")
    else:
        require(
            argv.count(custom_reward) == 1
            and argv[argv.index(custom_reward) + 1]
            == "weight_sync_reward.balanced_index_reward",
            f"{context}: synthetic reward callback mismatch",
        )
    require(
        manifest["git_head"] == campaign["miles_head"]
        and manifest["sglang_source"]["git_head"] == campaign["sglang_head"],
        f"{context}: campaign source mismatch",
    )
    require(
        manifest["git_diff"] == manifest["sglang_source"]["git_diff"] == "",
        f"{context}: campaign expects clean tracked sources",
    )
    require(
        set(manifest["script_sha256"]) == SCRIPT_NAMES
        and all(
            re.fullmatch(r"[0-9a-f]{64}", value)
            for value in manifest["script_sha256"].values()
        ),
        f"{context}: incomplete helper identity",
    )
    require(
        summary["mode"] == evidence["mode"] == manifest["mode"]
        and evidence["reward_mode"] == manifest["reward_mode"],
        f"{context}: artifact arm identity mismatch",
    )
    log_path = run_dir.parent / f"{context}.log"
    exit_path = run_dir.parent / f"{context}.exit"
    require(
        log_path.is_file() and exit_path.is_file(), f"{context}: missing final log/exit"
    )
    require(
        int(exit_path.read_text().strip()) == evidence["launcher_exit_code"] == 0,
        f"{context}: launcher did not exit zero",
    )
    require(
        sha256(log_path) == evidence["log_sha256"],
        f"{context}: training evidence is stale relative to log",
    )
    parsed = parse_log(log_path)
    require(
        parsed["ray_terminal"]
        and parsed["ray_terminal"][-1]["status"] == "succeeded"
        and parsed["ray_terminal"][-1] == evidence["ray_terminal"],
        f"{context}: no verified final Ray success",
    )
    require(
        evidence["run_succeeded"] is True
        and evidence["all_expected_rank_rollouts_completed"] is True,
        f"{context}: E2E evidence incomplete",
    )
    expected_pairs = {(rollout, rank) for rollout in range(7) for rank in range(4)}
    train = parsed["train_events"]
    require(
        len(train) == 28
        and {(event["rollout"], event["rank"]) for event in train} == expected_pairs,
        f"{context}: train coverage mismatch",
    )
    require(
        all(
            event["attempt"] == 0
            and event["step"] == 0
            and event["outcome"] == "NORMAL"
            and event["valid_step"]
            for event in train
        ),
        f"{context}: retry/non-normal training step invalidates this simple paired report",
    )
    require(
        train == evidence["parsed_log_evidence"]["train_events"],
        f"{context}: stale train events",
    )
    require(
        not parsed["metric_parse_errors"]
        and not parsed["conflicting_step_metric_lines"],
        f"{context}: invalid/conflicting log metrics",
    )
    require(
        len(parsed["rank0_step_metrics"]) == len(evidence["rank0_steps"]) == 7,
        f"{context}: missing rank0 step metrics",
    )
    for step, retained in zip(
        parsed["rank0_step_metrics"], evidence["rank0_steps"], strict=True
    ):
        require(
            all(retained[key] == value for key, value in step["metrics"].items()),
            f"{context}: retained step metrics differ from log",
        )
    require(
        len(parsed["rollout_service_metrics"]) == 7,
        f"{context}: missing rollout version metrics",
    )
    for record in parsed["rollout_service_metrics"]:
        metrics = record["metrics"]
        require(
            metrics["rollout/weight_version/mixed_version_ratio"] == 0
            and metrics["rollout/weight_version/min"]
            == metrics["rollout/weight_version/max"],
            f"{context}: mixed rollout versions",
        )
    versions = [
        record["metrics"]["rollout/weight_version/min"]
        for record in parsed["rollout_service_metrics"]
    ]
    require(
        versions == list(range(versions[0], versions[0] + 7)),
        f"{context}: nonconsecutive rollout versions",
    )
    engines = engine_metadata(run_dir)
    require(
        engines.get("available") is True
        and engines["records"] == evidence["engines"]["records"]
        and engines["summary"] == evidence["engines"]["summary"],
        f"{context}: stale engine metadata",
    )
    dumps = evidence["debug_dumps"]
    require(
        dumps["status"] == "parsed"
        and not dumps["errors"]
        and len(dumps["train"]) == len(dumps["rollout"]) == 7,
        f"{context}: missing/partial debug-dump evidence",
    )
    require(
        {record["rollout"] for record in dumps["train"]}
        == {record["rollout"] for record in dumps["rollout"]}
        == set(range(7)),
        f"{context}: debug rollout coverage mismatch",
    )
    for record in dumps["train"]:
        require(record["rank"] == 0, f"{context}: unexpected dump rank")
        for field in ("raw_reward", "rewards", "advantages", "returns"):
            values = record["fields"][field]
            require(
                values["count"] > 0 and values["count"] == values["finite_count"],
                f"{context}: nonfinite/missing {field}",
            )
    validate_raw_timings(run_dir, manifest, summary)
    builtin = [
        record
        for record in parsed["native_perf_metrics"]
        if "perf/update_weights_time" in record["metrics"]
    ]
    require(
        len(builtin) == 7 and [record["index"] for record in builtin] == list(range(7)),
        f"{context}: expected native perf0..6",
    )
    rows = [
        dict(
            row,
            builtin_actor_s=record["metrics"]["perf/update_weights_time"],
            builtin_log_line=record["line"],
        )
        for row, record in zip(summary["rows"], builtin, strict=True)
    ]
    for row in rows:
        for metric in TIME_METRICS:
            if row["initial"] and row[metric] is None:
                continue
            require(
                finite_number(row[metric]) and row[metric] > 0,
                f"{context}: invalid {metric}",
            )
    steady = rows[2:]
    stats = {
        metric: distribution([row[metric] for row in steady if row[metric] is not None])
        for metric in TIME_METRICS + PHASE_METRICS
    }
    for key, metric in (
        ("steady_driver", "driver_wall_s"),
        ("steady_trainer", "trainer_max_wall_s"),
        ("steady_generation_pause", "generation_pause_upper_bound_s"),
    ):
        require(
            summary[key]["count"] == 5
            and summary[key]["median_s"] == stats[metric]["median"],
            f"{context}: summary steady selection mismatch",
        )
    inputs = [
        run_dir / name
        for name in (
            "manifest.json",
            "summary.json",
            "training-evidence.json",
            "driver.jsonl",
            *(f"trainer-rank{rank}.jsonl" for rank in range(4)),
        )
    ] + [log_path, exit_path]
    dump_files = sorted(
        (run_dir / "dump_details" / "train_data").glob("*.pt")
    ) + sorted((run_dir / "dump_details" / "rollout_data").glob("*.pt"))
    require(len(dump_files) == 14, f"{context}: expected 14 retained raw debug dumps")
    gradient_norms = [record["train/grad_norm"] for record in evidence["rank0_steps"]]
    advantage_abs = [
        record["fields"]["advantages"]["unmasked"]["mean_abs"]
        for record in dumps["train"]
    ]
    delta_steady = [
        row["changed_bytes"] for row in steady if row["changed_bytes"] is not None
    ]
    numeric = {
        "gradient_norms": gradient_norms,
        "gradient_norm_distribution": distribution(gradient_norms),
        "advantage_mean_abs_by_rollout": advantage_abs,
        "advantage_nonzero_by_rollout": [
            record["fields"]["advantages"]["unmasked"]["nonzero_count"]
            for record in dumps["train"]
        ],
        "all_advantages_zero": all(value == 0 for value in advantage_abs),
        "all_gradient_norms_zero": all(value == 0 for value in gradient_norms),
        "raw_reward_nonzero_by_rollout": [
            record["fields"]["raw_reward"]["nonzero_count"] for record in dumps["train"]
        ],
        "trainable_response_tokens_by_rollout": [
            record["trainable_response_tokens"] for record in dumps["train"]
        ],
        "rank0_steps": evidence["rank0_steps"],
        "rollout_debug": dumps["rollout"],
        "replay_checks": sum(record["checks"] for record in parsed["replay"]),
        "replay_nonzero_checks": sum(
            record["nonzero_checks"] for record in parsed["replay"]
        ),
        "replay_mismatched_token_comparisons": sum(
            record["mismatched_token_comparisons"] for record in parsed["replay"]
        ),
        "all_steady_delta_updates_have_changed_bytes": all(
            value > 0 for value in delta_steady
        )
        if delta_steady
        else None,
        "all_steady_delta_updates_zero_changed_bytes": all(
            value == 0 for value in delta_steady
        )
        if delta_steady
        else None,
        "delta_density_by_update": [
            row["changed_bytes"] / row["total_bytes"] if row["total_bytes"] else None
            for row in rows
        ],
    }
    return {
        "name": context,
        "run_dir": str(run_dir),
        "reward_mode": manifest["reward_mode"],
        "mode": manifest["mode"],
        "validated": True,
        "manifest": manifest,
        "input_sha256": {str(path): sha256(path) for path in inputs + dump_files},
        "completion": {
            "launcher_exit": 0,
            "ray": parsed["ray_terminal"][-1],
            "rollouts": 7,
            "train_rank_returns": 28,
            "training_attempts": [0],
            "steady_updates": 5,
        },
        "rows": rows,
        "steady": stats,
        "numerics": numeric,
        "engine_metadata": engines,
        "warnings": warning_inventory(log_path, parsed),
        "metric_definitions": summary["metric_definitions"],
    }


def compare_pair(broadcast, delta):
    a, b = broadcast["manifest"], delta["manifest"]
    checks = {
        "configuration_except_transport_and_output": normalized_config(a)
        == normalized_config(b),
        "miles_source": (a["git_head"], a["git_diff"])
        == (b["git_head"], b["git_diff"]),
        "sglang_source": (
            a["sglang_source"]["git_head"],
            a["sglang_source"]["git_diff"],
        )
        == (b["sglang_source"]["git_head"], b["sglang_source"]["git_diff"]),
        "observation_and_reward_helpers": a["script_sha256"] == b["script_sha256"],
    }
    require(all(checks.values()), f"{a['reward_mode']}: pair mismatch: {checks}")
    ratios = {}
    for metric in TIME_METRICS:
        base, candidate = (
            broadcast["steady"][metric]["median"],
            delta["steady"][metric]["median"],
        )
        ratios[metric] = {
            "delta_over_broadcast": candidate / base,
            "broadcast_over_delta": base / candidate,
        }
    return {
        "checks": checks,
        "broadcast": broadcast["name"],
        "disk_delta": delta["name"],
        "median_ratios": ratios,
        "normalized_configuration": normalized_config(a),
        "steady_changed_weight_validation": delta["numerics"][
            "all_steady_delta_updates_have_changed_bytes"
        ],
        "steady_zero_change_validation": delta["numerics"][
            "all_steady_delta_updates_zero_changed_bytes"
        ],
    }


def number(value):
    return "—" if value is None else str(value)


def table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ] + [
        "| "
        + " | ".join(
            number(value).replace("|", "\\|").replace("\n", " ") for value in row
        )
        + " |"
        for row in rows
    ]


def markdown(report):
    c, env = report["campaign"], report["environment"]
    lines = [
        "# GLM-5.2 NVFP4 W4A16 weight synchronization benchmark",
        "",
        "All four retained arms passed the report's completion and source/configuration checks. Native and synthetic-balanced reward arms are separate workloads. Ratios are delta / broadcast median time: below 1 means delta took less time.",
        "",
        f"Environment: {c['cluster']} / {c['host']} / {c['devbox']}; one node, eight NVIDIA B300 GPUs, 4 trainer + 4 rollout GPUs (two 2-GPU engines). Image `{c['image']}`; image-index digest `{c['image_index_digest']}`; amd64 digest `{c['image_amd64_digest']}`.",
        "",
        f"Tested sources: Miles `{c['miles_head']}`; SGLang `{c['sglang_head']}`; Megatron `{c['megatron_head']}`. Both tested Miles/SGLang tracked diffs are empty. HF source `{c['hf_source']}` at `{c['hf_revision']}`.",
        "",
        "Container package metadata: "
        + "; ".join(
            f"{key} `{value}`"
            for key, value in env["packages"].items()
            if value is not None
        )
        + ". Miles/SGLang run from the tested source overrides above, rather than the container's baseline source pins.",
        "",
        "Each arm has seven rollouts, six post-training updates, and five steady samples (update indices 2–6). Initial sync and the first post-training update are excluded from steady statistics. All samples, including slow ones, are retained. Seeds, checkpoint paths, prompts, precision settings, batch sizes, and observation helpers match within each pair; only transport/output options differ.",
        "",
        "## Steady timing comparison",
        "",
    ]
    comparison_rows = []
    for reward, pair in report["comparisons"].items():
        broadcast, delta = (
            report["arms"][pair[key]] for key in ("broadcast", "disk_delta")
        )
        for metric in TIME_METRICS:
            a, b = broadcast["steady"][metric], delta["steady"][metric]
            comparison_rows.append(
                [
                    reward,
                    metric,
                    a["median"],
                    f"{a['min']}–{a['max']}",
                    b["median"],
                    f"{b['min']}–{b['max']}",
                    pair["median_ratios"][metric]["delta_over_broadcast"],
                ]
            )
    lines += table(
        [
            "Reward",
            "Metric (s)",
            "Broadcast median",
            "Broadcast min–max",
            "Delta median",
            "Delta min–max",
            "Delta / broadcast",
        ],
        comparison_rows,
    )
    lines += ["", "## Raw updates and training evidence", ""]
    for arm in report["arms"].values():
        n = arm["numerics"]
        lines += [
            f"### {arm['name']}",
            "",
            "Exit 0; final Ray success; 28 normal train-step returns across seven rollouts × four ranks, all attempt 0; seven successful updater returns on every rank; six post-training and five steady samples.",
            "",
        ]
        lines += table(
            [
                "Update",
                "After train rollout",
                "Driver s",
                "Built-in actor s",
                "Trainer max s",
                "Pause upper bound s",
                "Pull RPC max s",
                "Reload RPC max s",
                "Changed bytes",
                "Total bytes",
                "Delta wire bytes",
            ],
            [
                [
                    row["update_index"],
                    row["rollout_id"],
                    row["driver_wall_s"],
                    row["builtin_actor_s"],
                    row["trainer_max_wall_s"],
                    row["generation_pause_upper_bound_s"],
                    row["pull_weights_max_rpc_s"],
                    row["update_weights_from_disk_max_rpc_s"],
                    row["changed_bytes"],
                    row["total_bytes"],
                    row["wire_bytes"],
                ]
                for row in arm["rows"]
            ],
        )
        lines += [
            "",
            f"Gradient norms: `{n['gradient_norms']}`. Unmasked advantage mean absolute values: `{n['advantage_mean_abs_by_rollout']}`; nonzero counts: `{n['advantage_nonzero_by_rollout']}`. Trainable response-token counts: `{n['trainable_response_tokens_by_rollout']}`.",
            "",
            f"Replay: {n['replay_checks']} checks; {n['replay_nonzero_checks']} checks with mismatches; {n['replay_mismatched_token_comparisons']} mismatched token comparisons. These repeat across layers/stages/ranks. Per-rollout reward/loss/KL/logprob, version, and debug statistics remain in BENCHMARK_RESULTS.json.",
            "",
        ]
        if arm["mode"] == "disk-delta":
            lines += [
                f"Exact steady delta counters: every update changed bytes = **{n['all_steady_delta_updates_have_changed_bytes']}**; every update zero changed bytes = **{n['all_steady_delta_updates_zero_changed_bytes']}**. Per-update density: `{n['delta_density_by_update']}`.",
                "",
            ]
            if (
                arm["reward_mode"] == "synthetic-balanced"
                and not n["all_steady_delta_updates_have_changed_bytes"]
            ):
                lines += [
                    "**Changed-weight validation is incomplete:** synthetic reward did not establish nonzero delta bytes for every steady update; do not call this a material changed-weight benchmark.",
                    "",
                ]
        warnings = arm["warnings"]
        lines += [
            f"Warning/error inventory by log stage: `{warnings['stage_counts']}`. Startup readiness retries: `{warnings['readiness_retry_counts']}`. `freeze_gc` failure lines: `{warnings['freeze_gc_failure_lines']}`. Full line-numbered messages are retained in the JSON; successful completion does not mean an error-free log.",
            "",
            f"Local evidence: `{arm['run_dir']}`. All source manifests, observations, final logs/exits, and 14 debug dumps are SHA256-indexed in the JSON.",
            "",
        ]
    lines += [
        "## Steady phase timings",
        "",
        "Each cell is the per-update maximum over ranks or engines, then summarized across five steady updates. Phases and RPCs can overlap/nest; do not add these medians to reconstruct total time.",
        "",
    ]
    lines += table(
        ["Run", "Phase (s)", "N", "Median", "Min", "Max"],
        [
            [
                arm["name"],
                metric,
                stats["count"],
                stats["median"],
                stats["min"],
                stats["max"],
            ]
            for arm in report["arms"].values()
            for metric in PHASE_METRICS
            if (stats := arm["steady"][metric])["count"]
        ],
    )
    lines += ["", "## Scope and limitations", ""] + [
        f"- {item}" for item in report["boundaries"]
    ]
    lines += [
        "",
        "## Reproduction and identity",
        "",
        "Run each retained manifest's exact train_argv, launch_kwargs and environment; full records and normalized paired configurations are embedded in BENCHMARK_RESULTS.json. The source test is tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py; HARNESS.md documents project observation hooks and preparation.",
        "",
        "Report generation:",
        "",
        "```bash",
        "python build_benchmark_report.py \\",
        "  --campaign-metadata CAMPAIGN.json --environment environment-image.json \\",
        "  artifacts/native-broadcast-01 artifacts/native-disk-delta-01 \\",
        "  artifacts/synthetic-balanced-broadcast-01 artifacts/synthetic-balanced-disk-delta-01",
        "```",
        "",
    ]
    return "\n".join(lines)


def build(run_dirs, campaign_path, environment_path):
    require(
        len(run_dirs) == 4 and len({path.resolve() for path in run_dirs}) == 4,
        "Exactly four distinct run directories are required",
    )
    campaign = read_json(campaign_path)
    environment = read_json(environment_path)
    for field in (
        "cluster",
        "host",
        "devbox",
        "image",
        "image_index_digest",
        "image_amd64_digest",
        "miles_head",
        "sglang_head",
        "megatron_head",
        "hf_source",
        "hf_revision",
        "topology",
    ):
        require(campaign.get(field), f"Campaign metadata missing {field}")
    require(campaign["topology"] == TOPOLOGY, "Campaign topology mismatch")
    require(
        re.fullmatch(r"radixark/miles:dev-\d+", campaign["image"]),
        "Explicit timestamped Miles dev image required",
    )
    for field in ("image_index_digest", "image_amd64_digest"):
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", campaign[field]), f"Invalid {field}"
        )
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
    arms = [load_arm(path, campaign) for path in run_dirs]
    by_key = {(arm["reward_mode"], arm["mode"]): arm for arm in arms}
    require(
        set(by_key) == ARM_KEYS,
        "Require native/synthetic-balanced × broadcast/disk-delta exactly once",
    )
    comparisons = {
        reward: compare_pair(
            by_key[(reward, "broadcast")], by_key[(reward, "disk-delta")]
        )
        for reward in ("native", "synthetic-balanced")
    }
    stable_campaign = {
        key: value
        for key, value in campaign.items()
        if key
        not in {
            "active_run",
            "active_run_log",
            "active_run_exit",
            "active_run_status",
            "pending",
            "smoke",
            "native_broadcast",
        }
    }
    ordered_arms = [
        by_key[(reward, mode)]
        for reward in ("native", "synthetic-balanced")
        for mode in ("broadcast", "disk-delta")
    ]
    return {
        "schema_version": 1,
        "validated_complete_four_arm_campaign": True,
        "campaign": stable_campaign,
        "environment": environment,
        "metadata_sha256": {
            str(path.resolve()): sha256(path)
            for path in (campaign_path, environment_path)
        },
        "generator_sha256": sha256(Path(__file__)),
        "arms": {arm["name"]: arm for arm in ordered_arms},
        "comparisons": comparisons,
        "boundaries": [
            "Single C2 node, local /hai-workspace storage, two rollout engines. This is not a cross-node or cross-cluster bandwidth result; runs are sequential in the recorded campaign order, without randomized order or replicate campaigns.",
            "Native deepscaler and synthetic-balanced reward are separate workloads. Synthetic reward assigns sample-index parity 0/1 and uses ordinary GRPO/optimizer updates; it is transport validation, not learned-task quality. Actual nonzero changes require exact delta counters, not gradient norm alone.",
            "Driver wall time includes actor dispatch and rollout-version handoff. Built-in actor time is the raw rank-0 perf/update_weights_time. Trainer max is the maximum rank-local updater wall time. These are distinct boundaries and include no additional global CUDA fence.",
            "Observation writes inside timed updater phases add overhead; delta emits more observations than broadcast. No estimated observation overhead is subtracted.",
            "Generation pause spans the first pause dispatch to the last resume response across engines; it is a conservative interval, not a production latency measurement. Phase/RPC timings overlap or nest.",
            "Initial synchronization and the first post-training synchronization are excluded from five-sample steady statistics. The initial full checkpoint baseline copy and model/engine startup are not steady delta payload cost.",
            "Changed/total bytes are exact encoder tensor-byte counters summed over four trainer ranks. Delta wire bytes sum serialized safetensor lengths, excluding index JSON and the initial full baseline copy; they are not measured physical network bytes. Broadcast changed/wire bytes are unavailable, not zero.",
            "Version acknowledgements, checksums, routing replay, and subsequent successful training do not prove active GPU weight byte equality or exact numerical equivalence. The recipe disables weight-update/logprob/KL CI checkers. Replay counts repeat per layer/stage/rank.",
            "Startup and teardown warnings/errors are retained with line numbers. Missing/incomplete arms, stale logs, train retries/non-normal steps, mixed versions, missing delta counters, or pair identity/configuration mismatches prevent report generation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--campaign-metadata", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    options = parser.parse_args()
    try:
        report = build(options.runs, options.campaign_metadata, options.environment)
        rendered = markdown(report)
    except (InvalidEvidence, KeyError, ValueError, OSError, TypeError) as error:
        print(
            json.dumps(
                {
                    "validated_complete_four_arm_campaign": False,
                    "error": str(error),
                    "output_written": False,
                }
            ),
            file=sys.stderr,
        )
        return 2
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (
        (
            "BENCHMARK_RESULTS.json",
            json.dumps(report, indent=2, allow_nan=False) + "\n",
        ),
        ("BENCHMARK_RESULTS.md", rendered),
    ):
        target = options.output_dir / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(target)
    print(
        json.dumps(
            {
                "validated_complete_four_arm_campaign": True,
                "output_dir": str(options.output_dir.resolve()),
                "runs": list(report["arms"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
