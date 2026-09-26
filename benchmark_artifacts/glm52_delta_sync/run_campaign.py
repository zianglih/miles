#!/usr/bin/env python3
"""Run the paired native and changed-weight arms sequentially on the dedicated devbox."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
artifacts = root / "artifacts"
sources = {}
for name in ("miles", "sglang"):
    sources[name] = subprocess.check_output(
        ["git", "-C", str(root / name), "rev-parse", "HEAD"], text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "-C", str(root / name), "diff", "HEAD"], text=True
    ), f"Dirty {name} source"
(artifacts / "campaign-sources.json").write_text(json.dumps(sources, indent=2) + "\n")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONPATH"] = ":".join(
    [str(root / "sglang/python"), str(root / "miles"), "/root/Megatron-LM"]
)
for reward in ("native", "synthetic-balanced"):
    for mode in ("broadcast", "disk-delta"):
        run = f"{reward}-{mode}-01"
        for name, head in sources.items():
            assert subprocess.check_output(
                ["git", "-C", str(root / name), "rev-parse", "HEAD"], text=True
            ).strip() == head, f"{name} source changed during campaign"
            assert not subprocess.check_output(
                ["git", "-C", str(root / name), "diff", "HEAD"], text=True
            ), f"Dirty {name} source during campaign"
        assert not (artifacts / run).exists(), f"Run already exists: {run}"
        command = [
            sys.executable, str(root / "launch_weight_sync.py"),
            "--mode", mode, "--reward-mode", reward,
            "--num-rollout", "7", "--run-name", run,
            "--model-dir", str(root / "models"),
            "--data-dir", str(root / "datasets"),
            "--output-dir", str(artifacts),
        ]
        print(json.dumps({"event": "start", "run": run, "time": time.time(), "argv": command}), flush=True)
        with (artifacts / f"{run}.log").open("w") as log:
            completed = subprocess.run(command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        (artifacts / f"{run}.exit").write_text(f"{completed.returncode}\n")
        print(json.dumps({"event": "end", "run": run, "time": time.time(), "exit": completed.returncode}), flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        with (artifacts / f"{run}-summary.json").open("w") as summary:
            subprocess.run(
                [sys.executable, str(root / "summarize_weight_sync.py"), str(artifacts / run)],
                check=True, stdout=summary,
            )
    with (artifacts / f"{reward}-comparison.json").open("w") as summary:
        subprocess.run(
            [sys.executable, str(root / "summarize_weight_sync.py"),
             str(artifacts / f"{reward}-broadcast-01"), str(artifacts / f"{reward}-disk-delta-01")],
            check=True, stdout=summary,
        )
print("CAMPAIGN_COMPLETE", flush=True)
