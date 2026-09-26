"""Summarize CPU-v2 artifacts and compare matching transport or source pairs.

python summarize_weight_sync_cpu_v2.py artifacts/RUN [artifacts/OTHER_RUN]
First generate training-evidence.json with summarize_training_evidence.py.
This writes cpu-summary.json/cpu-updates.csv in each selected v2 run. CPU errors
preserve training but return exit 2 and prevent comparison ratios. A transport
pair needs the same implementation/source; an explicitly labeled baseline vs
candidate pair permits recorded source differences. Native/synthetic never mix.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

from build_benchmark_report import normalized_config
from summarize_training_evidence import parse_log
from summarize_weight_sync import summarize as wall_summary


FIELDS = (
    "driver_wall_s",
    "trainer_cpu_sum_s",
    "trainer_cpu_max_s",
    "driver_cpu_s",
    "receiver_scheduler_cpu_s",
    "receiver_auxiliary_cpu_s",
    "receiver_total_cpu_s",
    "trainer_user_cpu_sum_s",
    "trainer_system_cpu_sum_s",
    "trainer_minor_faults",
    "trainer_major_faults",
    "trainer_voluntary_switches",
    "trainer_involuntary_switches",
    "receiver_snapshot_cost_s",
    "observer_outside_trainer_wall_max_s",
)


def read_events(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.is_file()
        else []
    )


def aggregate(manifest, wall, driver_events, trainer_events):
    """Pure aggregation entrypoint, also exercised by CPU-only fixtures."""
    rows = []
    expected = manifest["num_rollout"]
    driver = [event for event in driver_events if event["kind"] == "cpu_driver_update"]
    trainers = [
        event for event in trainer_events if event["kind"] == "cpu_trainer_update"
    ]
    for index, original in enumerate(wall["rows"]):
        row = {**original}
        d = [event for event in driver if event["update_index"] == index]
        t = [event for event in trainers if event["update_index"] == index]
        errors = []
        if (
            len(d) != 1
            or len(t) != 4
            or {event["rank"] for event in t} != set(range(4))
        ):
            errors.append("missing/duplicate CPU driver or trainer records")
        elif not all(event["cpu_valid"] and event["success"] for event in d + t):
            errors += [
                event.get("error") or "invalid/failed CPU observation"
                for event in d + t
                if not event["cpu_valid"] or not event["success"]
            ]
        elif d[0]["wall_s"] != original["driver_wall_s"]:
            errors.append("CPU driver wall differs from original wall observation")
        receiver = next(
            (event.get("receiver") for event in t if event["rank"] == 0), None
        )
        if receiver is None:
            errors.append("receiver counters unavailable")
        elif (
            len(
                [
                    process
                    for process in receiver["processes"]
                    if process["role"] == "scheduler"
                ]
            )
            != 4
        ):
            errors.append("receiver does not contain four schedulers")
        row["cpu_errors"] = errors
        row["cpu_valid"] = not errors
        if not errors:
            row.update(
                trainer_cpu_sum_s=sum(event["cpu_s"] for event in t),
                trainer_cpu_max_s=max(event["cpu_s"] for event in t),
                driver_cpu_s=d[0]["cpu_s"],
                receiver_scheduler_cpu_s=receiver["scheduler_cpu_ns"] / 1e9,
                receiver_auxiliary_cpu_s=receiver["auxiliary_cpu_ns"] / 1e9,
                receiver_total_cpu_s=receiver["total_cpu_ns"] / 1e9,
                receiver_snapshot_cost_s=receiver["snapshot_cost_ns"] / 1e9,
                observer_outside_trainer_wall_max_s=max(
                    (
                        event.get("observer_before_wall_ns", 0)
                        + event.get("observer_after_wall_ns", 0)
                    )
                    / 1e9
                    for event in t
                ),
                receiver_process_count=len(receiver["processes"]),
            )
            for output, field in (
                ("trainer_user_cpu_sum_s", "ru_utime"),
                ("trainer_system_cpu_sum_s", "ru_stime"),
                ("trainer_minor_faults", "ru_minflt"),
                ("trainer_major_faults", "ru_majflt"),
                ("trainer_voluntary_switches", "ru_nvcsw"),
                ("trainer_involuntary_switches", "ru_nivcsw"),
            ):
                row[output] = sum(event["resource_usage"][field] for event in t)
            row["trainer_average_cores"] = (
                row["trainer_cpu_sum_s"] / row["driver_wall_s"]
            )
            row["receiver_average_cores"] = (
                row["receiver_total_cpu_s"] / row["driver_wall_s"]
            )
        rows.append(row)
    steady = rows[2:]
    valid = (
        len(rows) == expected
        and len(driver) == expected
        and len(trainers) == 4 * expected
        and wall["complete"]
        and all(row["cpu_valid"] for row in steady)
        and len(steady) >= 1
    )
    stats = {}
    for field in FIELDS:
        values = [row[field] for row in steady if row["cpu_valid"]]
        stats[field] = {
            "count": len(values),
            "raw": values,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return {
        "complete_cpu_steady": valid,
        "rows": rows,
        "steady": stats,
        "invalid_initial_or_first_post": [
            row["update_index"] for row in rows[:2] if not row["cpu_valid"]
        ],
        "boundaries": [
            "Trainer CPU includes all threads during its updater; receiver CPU brackets the rank0 updater and includes snapshot skew.",
            "Receiver auxiliary includes data-parallel controllers and stable compile workers. Process changes invalidate a sample.",
            "Driver wall includes observer discovery/validation and CPU journal flushing; trainer v1 wall excludes outer discovery/validation/flush but includes counter snapshots. Overhead is recorded, never subtracted.",
            "Native and synthetic-balanced are distinct workloads. v1 CPU is unavailable; no comparison against unmeasured historical CPU.",
        ],
    }


def summarize(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("cpu_observer_version") != 2:
        raise ValueError(f"{run}: not CPU v2")
    wall = wall_summary(run)
    driver = read_events(run / "cpu-driver.jsonl")
    trainer = [
        event
        for rank in range(4)
        for event in read_events(run / f"cpu-trainer-rank{rank}.jsonl")
    ]
    result = aggregate(manifest, wall, driver, trainer)
    evidence_path = run / "training-evidence.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    exit_path = run.parent / f"{run.name}.exit"
    log_path = run.parent / f"{run.name}.log"
    log = parse_log(log_path) if log_path.exists() else {}
    final_ray = log.get("ray_terminal", [])
    train = log.get("train_events", [])
    expected_pairs = {
        (rollout, rank)
        for rollout in range(manifest["num_rollout"])
        for rank in range(4)
    }
    e2e = (
        exit_path.is_file()
        and exit_path.read_text().strip() == "0"
        and final_ray
        and final_ray[-1]["status"] == "succeeded"
        and len(train) == len(expected_pairs)
        and {(event["rollout"], event["rank"]) for event in train} == expected_pairs
        and all(
            event["attempt"] == 0
            and event["valid_step"]
            and event["outcome"] == "NORMAL"
            for event in train
        )
    )
    result.update(
        run=run.name,
        manifest=manifest,
        e2e_complete=bool(e2e),
        training_evidence=evidence,
        setup_records=[
            event
            for event in driver + trainer
            if event["kind"] not in ("cpu_driver_update", "cpu_trainer_update")
        ],
    )
    result["comparison_eligible"] = bool(e2e) and result["complete_cpu_steady"]
    (run / "cpu-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["rows"]:
        with (run / "cpu-updates.csv").open("w", newline="") as file:
            fields = list(dict.fromkeys(key for row in result["rows"] for key in row))
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["rows"])
    return result


def compare(a, b):
    x, y = a["manifest"], b["manifest"]
    same_source = (
        x["git_head"],
        x["git_diff"],
        x["sglang_source"]["git_head"],
        x["sglang_source"]["git_diff"],
    ) == (
        y["git_head"],
        y["git_diff"],
        y["sglang_source"]["git_head"],
        y["sglang_source"]["git_diff"],
    )
    transport = (
        x["implementation_label"] == y["implementation_label"]
        and {x["mode"], y["mode"]} == {"broadcast", "disk-delta"}
        and same_source
    )
    source = x["mode"] == y["mode"] and {
        x["implementation_label"],
        y["implementation_label"],
    } == {"baseline", "candidate"}
    matches = (
        normalized_config(x) == normalized_config(y)
        and x["script_sha256"] == y["script_sha256"]
        and x["cpu_observer"] == y["cpu_observer"]
    )
    eligible = (
        (transport or source)
        and matches
        and a["comparison_eligible"]
        and b["comparison_eligible"]
    )
    if transport:
        baseline, candidate = (a, b) if x["mode"] == "broadcast" else (b, a)
        kind = "disk-delta / broadcast"
    else:
        baseline, candidate = (
            (a, b) if x["implementation_label"] == "baseline" else (b, a)
        )
        kind = "candidate / baseline"
    return {
        "kind": kind,
        "eligible": bool(eligible),
        "matching_configuration_and_helpers": matches,
        "same_source": same_source,
        "numerator": candidate["run"],
        "denominator": baseline["run"],
        "ratios": {
            key: candidate["steady"][key]["median"] / baseline["steady"][key]["median"]
            for key in FIELDS
            if baseline["steady"][key]["median"]
        }
        if eligible
        else {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    results = [summarize(run.resolve()) for run in args.runs]
    output = {
        "runs": [
            {
                "run": r["run"],
                "comparison_eligible": r["comparison_eligible"],
                "steady": r["steady"],
            }
            for r in results
        ]
    }
    if len(results) == 2:
        output["comparison"] = compare(*results)
    print(json.dumps(output, indent=2))
    return (
        0
        if all(r["comparison_eligible"] for r in results)
        and (len(results) != 2 or output["comparison"]["eligible"])
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
