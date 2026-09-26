#!/usr/bin/env python3
"""Run a pinned-source CPU/wall campaign on the existing dedicated C2 devbox.

The plan names exact source commits and unique runs. No source switching, model
preparation, or automatic retry happens here. A failed run stops the campaign.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    options = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads(options.plan.read_text())
    artifacts = root / "artifacts"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = ":".join(
        [str(root / "sglang/python"), str(root / "miles"), "/root/Megatron-LM"]
    )
    for run in plan["runs"]:
        name = run["name"]
        if Path(name).name != name:
            raise ValueError(f"Invalid run name: {name!r}")
        for repo, expected in plan["sources"].items():
            actual = subprocess.check_output(
                ["git", "-C", str(root / repo), "rev-parse", "HEAD"], text=True
            ).strip()
            assert actual == expected, (repo, actual, expected)
            assert not subprocess.check_output(
                ["git", "-C", str(root / repo), "diff", "HEAD"], text=True
            ), f"Dirty {repo} source"
        assert not (artifacts / name).exists(), f"Run already exists: {name}"
        assert not (artifacts / f"{name}.log").exists(), f"Log already exists: {name}"
        command = [
            sys.executable,
            str(root / "launch_weight_sync_cpu_v2.py"),
            "--implementation-label", plan.get("implementation_label", "candidate"),
            "--mode", run["mode"],
            "--reward-mode", run["reward_mode"],
            "--num-rollout", str(run.get("num_rollout", 7)),
            "--run-name", name,
            "--model-dir", str(root / "models"),
            "--data-dir", str(root / "datasets"),
            "--output-dir", str(artifacts),
        ]
        print(json.dumps({"event": "start", "run": name, "time": time.time(), "argv": command}), flush=True)
        with (artifacts / f"{name}.log").open("x") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        (artifacts / f"{name}.exit").write_text(f"{result.returncode}\n")
        print(json.dumps({"event": "end", "run": name, "time": time.time(), "exit": result.returncode}), flush=True)
        if result.returncode:
            return result.returncode
        for script, extra, suffix in (
            ("summarize_weight_sync.py", [], "summary"),
            ("summarize_training_evidence.py", ["--debug-dumps", "require"], "training-summary"),
            ("summarize_weight_sync_cpu_v2.py", [], "cpu-summary"),
        ):
            with (artifacts / f"{name}-{suffix}.json").open("w") as out:
                subprocess.run(
                    [sys.executable, str(root / script), str(artifacts / name), *extra],
                    cwd=root, env=env, check=True, stdout=out,
                )
    print("CPU_CAMPAIGN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        (Path(__file__).resolve().parent / "artifacts/cpu-campaign.exit").write_text("1\n")
        raise
    (Path(__file__).resolve().parent / "artifacts/cpu-campaign.exit").write_text(f"{status}\n")
    raise SystemExit(status)
