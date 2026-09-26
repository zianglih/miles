#!/usr/bin/env python3
"""Run the frozen NumPy/compiled comparison on the existing dedicated C2 devbox.

The plan names exact source commits and unique runs. No source switching, model
preparation, or automatic retry happens here. A failed run stops the campaign.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from compile_campaign_metadata import metadata_record, resolve_metadata
from preserve_compile_campaign import preserve_arm


def backup_after_arm(root, plan, name, *, complete):
    if plan.get("durable_backup_dir") is None:
        return
    result = preserve_arm(root, name, Path(plan["durable_backup_dir"]), require_complete=complete)
    print(json.dumps({"event": "durable_backup", "run": name, "complete_arm": complete,
                      "time": time.time(), **result}), flush=True)
    if result["errors"]:
        raise RuntimeError(f"Arm backup reported errors: {name}: {result['errors']}")


def summarize_and_preserve(root, plan, name, exit_code, env):
    """Only called once the arm process has exited; failure never starts another arm."""
    if exit_code:
        backup_after_arm(root, plan, name, complete=False)
        return exit_code
    try:
        for script, extra, suffix in (
            ("summarize_weight_sync.py", [], "summary"),
            ("summarize_training_evidence.py", ["--debug-dumps", "require"], "training-summary"),
            ("summarize_weight_sync_cpu_v2.py", [], "cpu-summary"),
        ):
            with (root / "artifacts" / f"{name}-{suffix}.json").open("w") as out:
                subprocess.run([sys.executable, str(root / script), str(root / "artifacts" / name), *extra],
                               cwd=root, env=env, check=True, stdout=out)
    except BaseException:
        backup_after_arm(root, plan, name, complete=False)
        raise
    backup_after_arm(root, plan, name, complete=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    options = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan_bytes = options.plan.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    resolve_metadata(options.plan, plan)
    if plan.get("durable_backup_dir") is not None:
        assert options.plan.resolve() == root / "COMPILE_VALIDATION_PLAN.json", "Durable archives require the canonical plan"
    assert plan.get("frozen") is True and plan.get("schema_version") == 1
    assert plan.get("implementation_label", "candidate") == "candidate", "All arms require candidate labeling"
    repositories = {"miles": root / "miles-torch-compile-validation", "sglang": root / "sglang"}
    artifacts = root / "artifacts"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = ":".join(
        [str(root / "sglang/python"), str(repositories["miles"]), "/root/Megatron-LM"]
    )
    for run in plan["runs"]:
        assert options.plan.read_bytes() == plan_bytes, "Frozen plan changed during campaign"
        selected_metadata = resolve_metadata(options.plan, plan)
        for helper, expected in plan["script_sha256"].items():
            assert hashlib.sha256((root / helper).read_bytes()).hexdigest() == expected, helper
        name = run["name"]
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"Invalid run name: {name!r}")
        for repo, expected in plan["sources"].items():
            actual = subprocess.check_output(
                ["git", "-C", str(repositories[repo]), "rev-parse", "HEAD"], text=True
            ).strip()
            assert actual == expected, (repo, actual, expected)
            assert not subprocess.check_output(
                ["git", "-C", str(repositories[repo]), "status", "--porcelain", "--untracked-files=normal"], text=True
            ).strip(), f"Dirty {repo} source"
        assert not (artifacts / name).exists(), f"Run already exists: {name}"
        assert not (artifacts / f"{name}.log").exists(), f"Log already exists: {name}"
        cache = Path(run["compile_cache_dir"])
        assert cache.is_relative_to(root) and not cache.exists(), f"Expected fresh task-owned cache: {cache}"
        assert run["compile_cache_state_before"] == {"exists": False, "empty": True}
        command = [
            sys.executable,
            str(root / "launch_weight_sync_compile.py"),
            "--implementation-label", "candidate",
            "--mode", run["mode"],
            "--repo", str(repositories["miles"]),
            "--sglang-repo", str(repositories["sglang"]),
            "--delta-cpu-backend", run["delta_cpu_backend"],
            "--compile-cache-dir", str(cache),
            "--reward-mode", run["reward_mode"],
            "--num-rollout", str(run.get("num_rollout", 7)),
            "--run-name", name,
            "--model-dir", str(root / "models"),
            "--data-dir", str(root / "datasets"),
            "--output-dir", str(artifacts),
        ]
        metadata_selection = metadata_record(selected_metadata, relative_to=options.plan.resolve().parent)
        with (artifacts / f"{name}-campaign-metadata.json").open("x") as marker:
            marker.write(json.dumps({"plan_sha256": plan_sha256,
                                     "metadata_selection": metadata_selection}, indent=2) + "\n")
        print(json.dumps({"event": "start", "run": name, "time": time.time(), "argv": command,
                          "metadata_selection": metadata_selection}), flush=True)
        with (artifacts / f"{name}.log").open("x") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        (artifacts / f"{name}.exit").write_text(f"{result.returncode}\n")
        print(json.dumps({"event": "end", "run": name, "time": time.time(), "exit": result.returncode}), flush=True)
        status = summarize_and_preserve(root, plan, name, result.returncode, env)
        if status:
            return status
    print("COMPILE_CAMPAIGN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        (Path(__file__).resolve().parent / "artifacts/compile-campaign.exit").write_text("1\n")
        raise
    (Path(__file__).resolve().parent / "artifacts/compile-campaign.exit").write_text(f"{status}\n")
    raise SystemExit(status)
