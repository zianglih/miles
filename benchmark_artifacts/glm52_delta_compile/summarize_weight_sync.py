#!/usr/bin/env python3
"""Preserve every raw update and summarize initial, all post-train, and steady times."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def distribution(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "median_s": statistics.median(values),
        "mean_s": statistics.mean(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def summarize(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = []
    for path in sorted(run_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            events.append(json.loads(line))
    driver = sorted(
        (e for e in events if e["kind"] == "driver_update"), key=lambda e: e["start_ns"]
    )
    rows = []
    for update_index, outer in enumerate(driver):
        matching = [
            event for event in events if event.get("update_index") == update_index
        ]
        trainer = [event for event in matching if event["kind"] == "trainer_update"]
        byte_events = [event for event in matching if event["kind"] == "delta_bytes"]
        rpcs = [event for event in matching if event["kind"] == "engine_rpc"]
        pauses = [event for event in rpcs if event["endpoint"] == "pause_generation"]
        resumes = [
            event for event in rpcs if event["endpoint"] == "continue_generation"
        ]
        row = {
            "mode": manifest["mode"],
            "update_index": update_index,
            "rollout_id": outer["rollout_id"],
            "initial": outer["initial"],
            "success": outer["success"] and all(e["success"] for e in trainer + rpcs),
            "driver_wall_s": outer["elapsed_s"],
            "trainer_max_wall_s": max(
                (event["elapsed_s"] for event in trainer), default=None
            ),
            "trainer_rank_count": len(trainer),
            "paused_engine_count": len({event["engine"] for event in pauses}),
            "resumed_engine_count": len({event["engine"] for event in resumes}),
            "generation_pause_upper_bound_s": (
                (
                    max(event["end_ns"] for event in resumes)
                    - min(event["start_ns"] for event in pauses)
                )
                / 1e9
                if pauses and resumes
                else None
            ),
            "changed_bytes": sum(event["changed_bytes"] for event in byte_events)
            if byte_events
            else None,
            "total_bytes": sum(event["total_bytes"] for event in byte_events)
            if byte_events
            else None,
            "wire_bytes": sum(event["wire_bytes"] for event in byte_events)
            if byte_events
            else None,
            "delta_rank_count": len(byte_events),
        }
        for endpoint in (
            "pull_weights",
            "update_weights_from_disk",
            "begin_weight_update",
            "end_weight_update",
        ):
            selected = [event for event in rpcs if event["endpoint"] == endpoint]
            row[f"{endpoint}_max_rpc_s"] = max(
                (event["elapsed_s"] for event in selected), default=None
            )
        for phase in (
            "_capture_baseline",
            "_begin_encode",
            "after_base_weights",
            "_write_delta_files",
            "_reload_engines",
        ):
            selected = [
                event
                for event in matching
                if event["kind"] == "delta_phase" and event["phase"] == phase
            ]
            row[f"delta{phase}_max_s"] = max(
                (event["elapsed_s"] for event in selected), default=None
            )
        rows.append(row)
    post = [row for row in rows if not row["initial"] and row["success"]]
    steady = post[1:]
    byte_rows = [row for row in steady if row["changed_bytes"] is not None]
    summary = {
        "run_dir": str(run_dir.resolve()),
        "mode": manifest["mode"],
        "expected_post_training_updates": manifest["expected_post_training_updates"],
        "observed_successful_post_training_updates": len(post),
        "complete": len(post) == manifest["expected_post_training_updates"]
        and all(row["success"] and row["trainer_rank_count"] == 4 for row in rows)
        and all(
            row["paused_engine_count"] == 2 and row["resumed_engine_count"] == 2
            for row in post
        )
        and (
            manifest["mode"] != "disk-delta"
            or all(row["delta_rank_count"] == 4 for row in post)
        ),
        "metric_definitions": {
            "driver_wall_s": "wall time for driver await update_weights including actor RPC dispatch and rollout version handoff",
            "trainer_max_wall_s": "max rank-local WeightUpdater.update_weights wall time",
            "generation_pause_upper_bound_s": "first pause RPC dispatch to last continue RPC completion across both engines; conservative pause interval, not a production latency measurement",
            "steady": "post-training updates after excluding the first post-training sync; first sync may include canonical checkpoint conversion differences",
            "wire_bytes": "sum of serialized delta safetensor file lengths; excludes index JSON and full checkpoint baseline copy",
        },
        "initial_driver": distribution(
            [row["driver_wall_s"] for row in rows if row["initial"]]
        ),
        "post_training_driver": distribution([row["driver_wall_s"] for row in post]),
        "steady_driver": distribution([row["driver_wall_s"] for row in steady]),
        "steady_trainer": distribution(
            [
                row["trainer_max_wall_s"]
                for row in steady
                if row["trainer_max_wall_s"] is not None
            ]
        ),
        "steady_generation_pause": distribution(
            [
                row["generation_pause_upper_bound_s"]
                for row in steady
                if row["generation_pause_upper_bound_s"] is not None
            ]
        ),
        "all_steady_delta_updates_have_changed_bytes": all(
            row["changed_bytes"] > 0 for row in byte_rows
        )
        if byte_rows
        else None,
        "rows": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (run_dir / "updates.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    options = parser.parse_args()
    summaries = [summarize(path) for path in options.run_dirs]
    output = {"runs": summaries}
    if len(summaries) == 2:
        manifests = [
            json.loads((path / "manifest.json").read_text())
            for path in options.run_dirs
        ]
        ignored = {
            "--update-weight-transfer-mode",
            "--update-weight-disk-dir",
            "--update-weight-local-checkpoint-dir",
            "--save-debug-rollout-data",
            "--save-debug-train-data",
            "--save-debug-trajectory-data",
        }

        def comparable(manifest):
            argv = iter(manifest["train_argv"])
            normalized = []
            for item in argv:
                if item in ignored:
                    next(argv)
                else:
                    normalized.append(item)
            environment = {
                key: value
                for key, value in manifest["launch_kwargs"]["extra_env_vars"].items()
                if key not in ("WEIGHT_SYNC_MODE", "WEIGHT_SYNC_RUN_DIR")
            }
            return normalized, environment, manifest["git_head"], manifest["git_diff"]

        trainer_config_matches = comparable(manifests[0]) == comparable(manifests[1])
        receiver_sources = [manifest.get("sglang_source") for manifest in manifests]
        receiver_sources_match = (
            (
                all(
                    receiver is not None
                    and "git_head" in receiver
                    and "git_diff" in receiver
                    for receiver in receiver_sources
                )
                and (receiver_sources[0]["git_head"], receiver_sources[0]["git_diff"])
                == (receiver_sources[1]["git_head"], receiver_sources[1]["git_diff"])
            )
            if all(receiver is not None for receiver in receiver_sources)
            else None
        )
        script_hashes = [manifest.get("script_sha256") for manifest in manifests]
        scripts_match = (
            script_hashes[0] == script_hashes[1] if all(script_hashes) else None
        )
        identity_checks = (
            trainer_config_matches,
            receiver_sources_match,
            scripts_match,
        )
        configs_match = (
            False
            if any(check is False for check in identity_checks)
            else None
            if any(check is None for check in identity_checks)
            else True
        )
        arms = {summary["mode"]: summary for summary in summaries}
        comparison = {
            "configuration_matches_except_transport_and_output_paths": configs_match,
            "trainer_configuration_matches": trainer_config_matches,
            "sglang_receiver_source_matches": receiver_sources_match,
            "observation_and_reward_scripts_match": scripts_match,
            "source_identity_note": "Legacy manifests without receiver or helper identity remain readable, but no fully verified paired ratio is emitted.",
        }
        if (
            configs_match
            and set(arms) == {"broadcast", "disk-delta"}
            and all(s["complete"] for s in summaries)
        ):
            for metric in (
                "post_training_driver",
                "steady_driver",
                "steady_generation_pause",
            ):
                broadcast = arms["broadcast"][metric].get("median_s")
                delta = arms["disk-delta"][metric].get("median_s")
                if broadcast and delta:
                    comparison[f"{metric}_delta_over_broadcast"] = delta / broadcast
        output["comparison"] = comparison
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
