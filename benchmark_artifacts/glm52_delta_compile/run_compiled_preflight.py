#!/usr/bin/env python3
"""Validate the pinned combined source before the two GPU arms are frozen."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    options = parser.parse_args()
    root = Path(__file__).resolve().parent
    assert Path(options.run_name).name == options.run_name
    plan = json.loads(options.plan.read_text())
    repo = root / "miles-torch-compile-validation"
    repositories = {"miles": repo, "sglang": root / "sglang"}
    for name, checkout in repositories.items():
        head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        assert head == plan["sources"][name], (name, head)
        assert not subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=normal"], text=True
        ).strip(), name
    output = root / "artifacts" / options.run_name
    output.mkdir(exist_ok=False)
    cache = root / f"{options.run_name}-cache"
    assert not cache.exists(), cache
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "99", "OMP_NUM_THREADS": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(cache),
        "PYTHONPATH": os.pathsep.join((str(root / "sglang/python"), str(repo), "/root/Megatron-LM")),
    }
    tests = [
        "tests/fast/utils/test_delta_preparation.py",
        "tests/fast/backends/training_utils/weight_update/test_delta_compiled_cpu.py",
        "tests/fast/backends/training_utils/weight_update/test_delta.py",
        "tests/fast/backends/training_utils/weight_update/test_disk_delta_weight_version.py",
        "tests/fast/backends/training_utils/weight_update/test_disk_delta_engine_calls.py",
    ]
    commands = {
        "focused-tests": [sys.executable, "-m", "pytest", *tests, "-q"],
        "gloo-failures": [sys.executable, str(root / "validate_compiled_sender_gloo.py"), "--miles-path", str(repo)],
    }
    record = {
        "sources": plan["sources"], "standalone_source": plan["standalone_source"],
        "environment": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "TORCHINDUCTOR_CACHE_DIR", "PYTHONPATH")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gloo_script_sha256": hashlib.sha256((root / "validate_compiled_sender_gloo.py").read_bytes()).hexdigest(),
        "checks": {},
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    for name, command in commands.items():
        argv = ["timeout", "--kill-after=10", "360", *command]
        start = time.time()
        with (output / f"{name}.log").open("x") as log:
            result = subprocess.run(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        record["checks"][name] = {"argv": argv, "started_unix_s": start, "elapsed_s": time.time() - start, "exit": result.returncode}
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"check": name, **record["checks"][name]}), flush=True)
        if result.returncode:
            return result.returncode
    print("COMPILED_PREFLIGHT_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
