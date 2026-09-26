#!/usr/bin/env python3
"""Run the upstream W4A16 E2E recipe with a selected transport and raw timing.

Checkpoints are prepared separately and are never removed by this launcher.
Use six rollouts for five post-training updates; the final rollout has no sync.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("broadcast", "disk-delta"), required=True)
    parser.add_argument("--num-rollout", type=int, default=6)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--repo", type=Path, default=ROOT / "miles")
    parser.add_argument(
        "--sglang-repo",
        type=Path,
        help="Receiver source checkout; defaults to the sibling sglang checkout.",
    )
    parser.add_argument("--model-dir", default="/root/models")
    parser.add_argument("--data-dir", default="/root/datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rollout-seed", type=int, default=42)
    parser.add_argument(
        "--reward-mode",
        choices=("native", "synthetic-balanced"),
        default="native",
        help="Keep native deepscaler reward, or explicitly use balanced synthetic rewards for changed-weight validation.",
    )
    parser.add_argument("--render-only", action="store_true")
    options = parser.parse_args()
    options.repo = options.repo.resolve()
    options.sglang_repo = (
        options.sglang_repo or options.repo.parent / "sglang"
    ).resolve()
    if Path(options.run_name).name != options.run_name:
        parser.error("run-name must be a single path component")
    run_dir = (options.output_dir / options.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(options.repo)
    sys.path.insert(0, str(options.repo))
    recipe = importlib.import_module(
        "tests.e2e.megatron.test_glm5_2_744b_a40b_5layer_nvfp4_w4a16"
    )
    recipe.MODEL_DIR = options.model_dir
    recipe.DATA_DIR = options.data_dir
    recipe.RUN_ID = options.run_name
    original_execute = recipe.U.execute_train
    recipe.U.get_default_wandb_args = lambda *args, **kwargs: ""

    def execute_with_probe(**kwargs):
        train_args = shlex.split(kwargs["train_args"])
        for index, value in enumerate(train_args):
            if value.startswith(f"/root/shared_data/{options.run_name}/"):
                train_args[index] = value.replace(
                    f"/root/shared_data/{options.run_name}", str(run_dir), 1
                )
        train_args += [
            "--seed",
            str(options.seed),
            "--rollout-seed",
            str(options.rollout_seed),
            "--custom-megatron-init-path",
            "weight_sync_probe.install",
        ]
        if options.reward_mode == "synthetic-balanced":
            train_args += [
                "--custom-rm-path",
                "weight_sync_reward.balanced_index_reward",
            ]
        kwargs["train_args"] = shlex.join(train_args)
        kwargs["train_script"] = str(ROOT / "train_weight_sync_profiled.py")
        kwargs["job_lifetime"] = "launcher"
        kwargs["extra_env_vars"].update(
            {
                "PYTHONPATH": str(ROOT),
                "WEIGHT_SYNC_RUN_DIR": str(run_dir),
                "WEIGHT_SYNC_MODE": options.mode,
                "RAY_DEDUP_LOGS": "0",
            }
        )
        manifest = {
            "mode": options.mode,
            "num_rollout": options.num_rollout,
            "expected_post_training_updates": options.num_rollout - 1,
            "seed": options.seed,
            "rollout_seed": options.rollout_seed,
            "reward_mode": options.reward_mode,
            "topology": {
                "training_gpus": 4,
                "rollout_gpus": 4,
                "rollout_engines": 2,
                "gpus_per_engine": 2,
            },
            "repo": str(options.repo.resolve()),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
            "source_manifest_version": 2,
            "sglang_source": {
                "repo": str(options.sglang_repo),
                "git_head": subprocess.check_output(
                    ["git", "-C", str(options.sglang_repo), "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                "git_diff": subprocess.check_output(
                    ["git", "-C", str(options.sglang_repo), "diff", "HEAD"], text=True
                ),
            },
            "script_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "launch_weight_sync.py",
                    "weight_sync_probe.py",
                    "train_weight_sync_profiled.py",
                    "weight_sync_reward.py",
                )
            },
            "train_argv": train_args,
            "launch_kwargs": kwargs,
            "measurement": "driver wall time includes actor dispatch and rollout version handoff; updater wall time includes nested observation writes but excludes its own final journal append; initial sync reported separately",
            "workload": (
                "unmodified deepscaler GRPO reward; no synthetic gradients or weight mutations"
                if options.reward_mode == "native"
                else "synthetic balanced rewards sample.index % 2 override deepscaler via custom-rm-path; ordinary GRPO and optimizer; no direct gradient or weight mutation"
            ),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "mode": options.mode,
                    "render_only": options.render_only,
                }
            ),
            flush=True,
        )
        if not options.render_only:
            original_execute(**kwargs)

    recipe.U.execute_train = execute_with_probe
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    recipe.execute(
        num_rollout=options.num_rollout,
        update_weight_transfer_mode=options.mode,
        update_weight_disk_dir=str(run_dir / "delta-publication")
        if options.mode == "disk-delta"
        else None,
        update_weight_local_checkpoint_dir=str(run_dir / "rollout-checkpoint")
        if options.mode == "disk-delta"
        else None,
    )


if __name__ == "__main__":
    main()
