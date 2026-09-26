"""Prepare the unchanged W4A16 test checkpoints once inside this effort."""

import importlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root / "miles"
sys.path.insert(0, str(repo))
os.chdir(repo)
recipe = importlib.import_module("tests.e2e.megatron.test_glm5_2_744b_a40b_5layer_nvfp4_w4a16")
recipe.MODEL_DIR = str(root / "models")
recipe.DATA_DIR = str(root / "datasets")
os.environ.update(recipe.NVFP4_ENV)
os.environ["PYTHONPATH"] = str(repo) + ":/root/Megatron-LM:" + os.environ.get("PYTHONPATH", "")
models = Path(recipe.MODEL_DIR)
models.mkdir(exist_ok=True)
Path(recipe.DATA_DIR).mkdir(exist_ok=True)
baseline = models / recipe.MODEL_NAME
subprocess.run([
    "hf", "download", f"{recipe.MODEL_ORG}/{recipe.MODEL_NAME}",
    "--revision", "1c749139f70e158e4420ba67f342bef1de2e650d",
    "--local-dir", str(baseline),
], check=True)
recipe.U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=recipe.DATA_DIR)
recipe._validate_glm_checkpoint()
quantized = models / f"{recipe.MODEL_NAME}-NVFP4"
marker = models / "nvfp4-conversion-complete.json"
if not marker.exists():
    if quantized.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quantized.rename(quantized.with_name(quantized.name + ".incomplete-" + stamp))
    subprocess.run([
        sys.executable, "tools/convert_hf_to_nvfp4.py", "--model-dir", str(baseline),
        "--save-dir", str(quantized), "--extra-high-precision-layers-hf", *recipe.EXTRA_HIGH_PRECISION_LAYERS_HF,
    ], check=True)
    marker.write_text(json.dumps({"source": str(baseline), "target": str(quantized), "env": recipe.NVFP4_ENV}, indent=2))
distributed = models / f"{recipe.MODEL_NAME}_torch_dist"
tracker = distributed / "latest_checkpointed_iteration.txt"
if not tracker.exists() or tracker.read_text().strip() != "release":
    if distributed.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        distributed.rename(distributed.with_name(distributed.name + ".incomplete-" + stamp))
    # This bare devbox's node hostname is not DNS-resolvable inside the pod.
    # A static single-process rendezvous avoids torchrun advertising that name.
    subprocess.run([
        "torchrun", "--rdzv-backend=static", "--master-addr=127.0.0.1", "--master-port=29507",
        "--nproc-per-node=1", str(repo / "tools/convert_hf_to_torch_dist.py"),
        *shlex.split(recipe.U.shell_safe_model_args(recipe.MODEL_TYPE)),
        "--hf-checkpoint", str(baseline), "--save", str(distributed),
        "--tensor-model-parallel-size", "1", "--expert-tensor-parallel-size", "1",
        "--pipeline-model-parallel-size", "1", "--expert-model-parallel-size", "1",
    ], check=True)
print("PREPARE_COMPLETE", flush=True)
