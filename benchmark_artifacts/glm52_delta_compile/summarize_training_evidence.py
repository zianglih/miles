#!/usr/bin/env python3
"""Read retained Miles logs, exit status and trusted debug dumps without touching jobs.

Usage: python summarize_training_evidence.py artifacts/native-broadcast-01
Outputs training-evidence.json and training-steps.csv beneath each run directory.
Torch is optional unless --debug-dumps=require is selected.
"""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RANK = re.compile(r"actor_cell\d+_rank(\d+)")
TRAIN_STEP = re.compile(
    r"train op=train_step rollout=(\d+) step=(\d+) attempt=(\d+) outcome=(\S+) valid_step=(true|false)"
)
METRIC = re.compile(r" - (rollout|step|perf) (\d+): (\{.*\})\s*$")
REPLAY = re.compile(
    r"Replay check \(rank (\d+), stage ([^)]+)\): mismatch (\d+)/(\d+) tokens, threshold ([\d.]+)"
)
TRAIN_START = re.compile(r"ft op=train phase=start rollout=(\d+) attempt=(\d+)")
RAY_TERMINAL = re.compile(r"\bJob ['\"]([^'\"]+)['\"] (succeeded|failed|stopped)\b")


class NumericNames(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id in ("nan", "inf"):
            return ast.copy_location(ast.Constant(float(node.id)), node)
        return node


def parse_metric_dict(text):
    return ast.literal_eval(NumericNames().visit(ast.parse(text, mode="eval")))


def distribution(values):
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "mean": statistics.mean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
    }


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parse_log(path):
    result = {
        key: []
        for key in (
            "train_events",
            "rank0_rollout_metrics",
            "rank0_step_metrics",
            "native_perf_metrics",
            "rollout_service_metrics",
            "ray_terminal",
            "metric_parse_errors",
            "fatal_signature_lines",
        )
    }
    replay = defaultdict(
        lambda: {
            "checks": 0,
            "mismatched_token_comparisons": 0,
            "token_comparisons": 0,
            "max_mismatches_in_check": 0,
            "nonzero_checks": 0,
            "line_numbers": [],
        }
    )
    active_rollout = None
    rank0_step = None
    seen_steps = {}
    result["duplicate_step_metric_lines"] = []
    result["conflicting_step_metric_lines"] = []
    for line_number, raw in enumerate(path.read_text(errors="replace").split("\n"), 1):
        line = ANSI.sub("", raw)
        rank_match = RANK.search(line)
        rank = int(rank_match.group(1)) if rank_match else None
        if match := TRAIN_START.search(line):
            active_rollout = int(match.group(1))
        if match := TRAIN_STEP.search(line):
            rollout, step, attempt = map(int, match.group(1, 2, 3))
            result["train_events"].append(
                dict(
                    line=line_number,
                    rank=rank,
                    rollout=rollout,
                    step=step,
                    attempt=attempt,
                    outcome=match.group(4),
                    valid_step=match.group(5) == "true",
                )
            )
            if rank == 0:
                rank0_step = (rollout, step, attempt)
        if match := METRIC.search(line):
            kind, index, payload = match.groups()
            index = int(index)
            relevant = rank == 0 or (kind == "perf" and "rollout_executor]" in line)
            if relevant:
                try:
                    metrics = parse_metric_dict(payload)
                except (SyntaxError, ValueError) as error:
                    result["metric_parse_errors"].append(
                        dict(line=line_number, error=str(error))
                    )
                    continue
                record = dict(line=line_number, index=index, metrics=metrics)
                if kind == "rollout" and rank == 0:
                    result["rank0_rollout_metrics"].append(record)
                elif kind == "step" and rank == 0:
                    record.update(
                        rollout=rank0_step[0] if rank0_step else None,
                        step_within_rollout=rank0_step[1] if rank0_step else None,
                        attempt=rank0_step[2] if rank0_step else None,
                    )
                    if index in seen_steps:
                        key = (
                            "duplicate_step_metric_lines"
                            if metrics == seen_steps[index]
                            else "conflicting_step_metric_lines"
                        )
                        result[key].append(line_number)
                    else:
                        seen_steps[index] = metrics
                        result["rank0_step_metrics"].append(record)
                elif kind == "perf" and rank == 0 and "train_metric_utils.py:" in line:
                    # This step log drains the preceding update: perf 0 is startup;
                    # perf 1 is post-rollout 0, and so on.
                    record.update(
                        initial=index == 0,
                        after_training_rollout=None if index == 0 else index - 1,
                    )
                    result["native_perf_metrics"].append(record)
                elif kind == "perf" and "rollout_executor]" in line:
                    result["rollout_service_metrics"].append(record)
        if match := REPLAY.search(line):
            replay_rank, stage, mismatches, total, threshold = match.groups()
            item = replay[(active_rollout, int(replay_rank), stage)]
            item["checks"] += 1
            item["mismatched_token_comparisons"] += int(mismatches)
            item["token_comparisons"] += int(total)
            item["max_mismatches_in_check"] = max(
                item["max_mismatches_in_check"], int(mismatches)
            )
            item["nonzero_checks"] += int(int(mismatches) != 0)
            item["logged_rounded_threshold"] = float(threshold)
            item["line_numbers"].append(line_number)
        if match := RAY_TERMINAL.search(line):
            result["ray_terminal"].append(
                dict(line=line_number, job_id=match.group(1), status=match.group(2))
            )
        if re.search(
            r"Traceback \(most recent call last\)|(?:RuntimeError|AssertionError|CUDA error):",
            line,
        ):
            result["fatal_signature_lines"].append(
                dict(line=line_number, text=line[:700])
            )
    result["replay"] = [
        dict(rollout=key[0], rank=key[1], stage=key[2], **value)
        for key, value in sorted(
            replay.items(),
            key=lambda pair: (
                pair[0][0] if pair[0][0] is not None else -1,
                pair[0][1],
                pair[0][2],
            ),
        )
    ]
    return result


def engine_metadata(run_dir):
    path = run_dir / "trainer-rank0.jsonl"
    if not path.exists():
        return {"available": False}
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        event = json.loads(line)
        if event.get("kind") == "engine_rpc":
            records.append(
                {
                    "line": line_number,
                    **{
                        key: event.get(key)
                        for key in (
                            "update_index",
                            "engine",
                            "endpoint",
                            "success",
                            "request",
                            "response",
                        )
                    },
                }
            )
    grouped = defaultdict(
        lambda: {"calls": 0, "failed_calls": 0, "requested_versions": []}
    )
    for record in records:
        item = grouped[(record["engine"], record["endpoint"])]
        item["calls"] += 1
        item["failed_calls"] += int(record["success"] is not True)
        payload = record["request"] or {}
        for key in ("target_version", "weight_version", "new_version"):
            if key in payload:
                item["requested_versions"].append(str(payload[key]))
    return {
        "available": True,
        "source": str(path),
        "summary": [
            dict(engine=engine, endpoint=endpoint, **value)
            for (engine, endpoint), value in sorted(grouped.items())
        ],
        "records": records,
        "boundary": "RPC acknowledgement and version metadata are not active-GPU byte equality checks.",
    }


def debug_evidence(run_dir, mode):
    root = run_dir / "dump_details"
    inventory = {
        kind: sorted((root / kind).glob("*.pt"))
        for kind in ("train_data", "rollout_data")
    }
    result = {
        "files": {
            key: [str(path) for path in paths] for key, paths in inventory.items()
        },
        "train": [],
        "rollout": [],
        "errors": [],
        "note": "Only TP-rank0 on the last PP stage writes train dumps; four-rank completion comes from train_step log events.",
    }
    if mode == "skip" or not any(inventory.values()):
        result["status"] = "skipped" if mode == "skip" else "no_dumps"
        return result
    try:
        import torch
    except ImportError:
        if mode == "require":
            raise
        result["status"] = "torch_unavailable"
        return result

    def flatten(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1).to(torch.float64)
        if isinstance(value, (int, float)):
            return torch.tensor([value], dtype=torch.float64)
        tensors = [flatten(item) for item in value]
        return torch.cat(tensors) if tensors else torch.empty(0, dtype=torch.float64)

    def stats(tensor):
        finite = tensor[torch.isfinite(tensor)]
        return {
            "count": tensor.numel(),
            "finite_count": finite.numel(),
            "nonzero_count": int((finite != 0).sum()),
            "min": finite.min().item() if finite.numel() else None,
            "max": finite.max().item() if finite.numel() else None,
            "mean": finite.mean().item() if finite.numel() else None,
            "mean_abs": finite.abs().mean().item() if finite.numel() else None,
            "rms": finite.square().mean().sqrt().item() if finite.numel() else None,
        }

    def versions(value):
        if isinstance(value, list):
            return [version for item in value for version in versions(item)]
        if isinstance(value, dict):
            return [str(value["version"])] if "version" in value else []
        return []

    for kind, paths in inventory.items():
        for path in paths:
            try:
                # Only load trusted artifacts produced by this campaign; always CPU.
                data = torch.load(path, map_location="cpu", weights_only=False)
                if kind == "train_data":
                    values = data["rollout_data"]
                    mask = flatten(values["loss_masks"]) != 0
                    item = dict(
                        path=str(path),
                        rollout=data["rollout_id"],
                        rank=data["rank"],
                        response_tokens=mask.numel(),
                        trainable_response_tokens=int(mask.sum()),
                        fields={},
                    )
                    for key in (
                        "raw_reward",
                        "rewards",
                        "advantages",
                        "returns",
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                    ):
                        if key in values:
                            tensor = flatten(values[key])
                            item["fields"][key] = stats(tensor)
                            if tensor.numel() == mask.numel():
                                item["fields"][key]["unmasked"] = stats(tensor[mask])
                    if "log_probs" in values and "rollout_log_probs" in values:
                        diff = flatten(values["log_probs"]) - flatten(
                            values["rollout_log_probs"]
                        )
                        item["train_rollout_logprob_abs_diff"] = stats(diff[mask].abs())
                    result["train"].append(item)
                else:
                    samples = data["samples"]
                    groups = defaultdict(list)
                    all_versions = []
                    for sample in samples:
                        if isinstance(sample.get("reward"), (int, float)):
                            groups[str(sample.get("group_index"))].append(
                                float(sample["reward"])
                            )
                        all_versions.extend(versions(sample.get("weight_versions", [])))
                    result["rollout"].append(
                        dict(
                            path=str(path),
                            rollout=data["rollout_id"],
                            samples=len(samples),
                            statuses=dict(
                                Counter(str(sample.get("status")) for sample in samples)
                            ),
                            response_tokens=sum(
                                sample.get("response_length", 0) for sample in samples
                            ),
                            weight_versions=dict(Counter(all_versions)),
                            reward_groups=[
                                dict(
                                    group=group,
                                    count=len(rewards),
                                    rewards=rewards,
                                    zero_variance=len(set(rewards)) <= 1,
                                )
                                for group, rewards in sorted(groups.items())
                            ],
                        )
                    )
            except Exception as error:
                result["errors"].append(
                    dict(path=str(path), error=f"{type(error).__name__}: {error}")
                )
                if mode == "require":
                    raise
    result["status"] = "parsed" if not result["errors"] else "partial"
    return result


def summarize(run_dir, dump_mode):
    run_dir = run_dir.resolve()
    log_path = run_dir.parent / f"{run_dir.name}.log"
    exit_path = run_dir.parent / f"{run_dir.name}.exit"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    expected_rollouts = manifest["num_rollout"]
    ranks = list(range(manifest.get("topology", {}).get("training_gpus", 4)))
    parsed = parse_log(log_path)
    exits = int(exit_path.read_text().strip()) if exit_path.exists() else None
    terminal = parsed["ray_terminal"][-1] if parsed["ray_terminal"] else None
    expected_pairs = {
        (rollout, rank) for rollout in range(expected_rollouts) for rank in ranks
    }
    observed_pairs = {
        (event["rollout"], event["rank"])
        for event in parsed["train_events"]
        if event["valid_step"] and event["outcome"] == "NORMAL"
    }
    train_coverage = observed_pairs == expected_pairs and all(
        event["valid_step"] and event["outcome"] == "NORMAL"
        for event in parsed["train_events"]
    )
    row_by_rollout = {
        record["index"]: record for record in parsed["rank0_rollout_metrics"]
    }
    rows = []
    for step in parsed["rank0_step_metrics"]:
        rollout = row_by_rollout.get(step["rollout"], {})
        rows.append(
            dict(
                rollout=step["rollout"],
                step=step["index"],
                step_within_rollout=step["step_within_rollout"],
                train_log_line=step["line"],
                rollout_log_line=rollout.get("line"),
                **rollout.get("metrics", {}),
                **step["metrics"],
            )
        )
    native_times = [
        {
            "perf_index": record["index"],
            "line": record["line"],
            "initial": record["initial"],
            "after_training_rollout": record["after_training_rollout"],
            "seconds": record["metrics"].get("perf/update_weights_time"),
        }
        for record in parsed["native_perf_metrics"]
        if "perf/update_weights_time" in record["metrics"]
    ]
    result = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "mode": manifest["mode"],
        "reward_mode": manifest.get("reward_mode", "native"),
        "log": str(log_path),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "launcher_exit_code": exits,
        "ray_terminal": terminal,
        "run_succeeded": exits == 0
        and terminal is not None
        and terminal["status"] == "succeeded",
        "expected_rollouts": expected_rollouts,
        "expected_ranks": ranks,
        "all_expected_rank_rollouts_completed": train_coverage,
        "missing_rank_rollouts": sorted(expected_pairs - observed_pairs),
        "unexpected_rank_rollouts": sorted(observed_pairs - expected_pairs),
        "rank0_step_metric_count": len(rows),
        "rank0_steps": rows,
        "native_update_times": native_times,
        "native_update_time_distributions": {
            "initial": distribution(
                [row["seconds"] for row in native_times if row["initial"]]
            ),
            "post_training": distribution(
                [row["seconds"] for row in native_times if row["perf_index"] >= 1]
            ),
            "steady": distribution(
                [row["seconds"] for row in native_times if row["perf_index"] >= 2]
            ),
        },
        "replay_total_checks": sum(record["checks"] for record in parsed["replay"]),
        "replay_nonzero_checks": sum(
            record["nonzero_checks"] for record in parsed["replay"]
        ),
        "replay_mismatched_token_comparisons": sum(
            record["mismatched_token_comparisons"] for record in parsed["replay"]
        ),
        "engines": engine_metadata(run_dir),
        "debug_dumps": debug_evidence(run_dir, dump_mode),
        "parsed_log_evidence": parsed,
        "boundaries": [
            "Replay totals count repeated per-layer/stage/rank checks, not unique generated tokens; rollout grouping follows driver train-start log order.",
            "Logged reward/advantage values are means: zero mean is not proof that every element is zero. Debug dumps report mean absolute value and nonzero counts.",
            "Native perf 0 is initial sync, perf 1..6 are six post-training syncs, perf 2..6 are five steady samples for seven-rollout runs.",
            "Exact changed-weight bytes come from timing JSONL delta_bytes events, not gradient norm or printed rounded density.",
            "Successful RPCs, versions and training completion do not establish active GPU weight byte equality or learned-task quality.",
        ],
    }
    result = json_safe(result)
    (run_dir / "training-evidence.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (run_dir / "training-steps.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(json_safe(rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", type=Path, help="Run directories or their sibling .log files"
    )
    parser.add_argument(
        "--debug-dumps", choices=("auto", "skip", "require"), default="auto"
    )
    options = parser.parse_args()
    for path in options.runs:
        run_dir = path.with_suffix("") if path.suffix == ".log" else path
        result = summarize(run_dir, options.debug_dumps)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "run",
                        "run_succeeded",
                        "all_expected_rank_rollouts_completed",
                        "rank0_step_metric_count",
                        "replay_total_checks",
                        "replay_nonzero_checks",
                        "native_update_time_distributions",
                    )
                },
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
