#!/usr/bin/env python3
"""Resume exact pinned inputs/conversions after preemption; never delete evidence.

Execute only with the recovery model-preparation slot. A completed stage is reused
only after its recorded file hashes verify. Unknown/partial conversion directories
are renamed and retained before retrying. This is separate from the frozen launcher.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

MODEL_REV = "1c749139f70e158e4420ba67f342bef1de2e650d"
DATA_REV = "2e65612930298bde4c5d58fd97b3f23a483aaff9"
DATA_SHA = "cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6"
MILES_SHA = "f17ba4bce13bf7d357e7560182dc859c41a7cb37"
MEGATRON_SHA = "f148a32b4385b758b66a77c9c3ad1641f1295d4b"


def sha(path, algorithm="sha256", git_blob=False):
    value = hashlib.new(algorithm)
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while data := stream.read(2 * 1024 * 1024):
            value.update(data)
    return value.hexdigest()


def files(directory):
    return {str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(directory.rglob("*")) if path.is_file() and ".cache" not in path.relative_to(directory).parts}


def check_files(directory, manifest):
    for name, item in manifest.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"Prepared file changed or missing: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--backup-dir", type=Path, help="Optional task-owned durable backup; never a publication directory")
    args = parser.parse_args()
    root = args.root.resolve()
    repo, models, datasets = root / "miles", root / "models", root / "datasets"
    for path in (models, datasets, root / "artifacts"):
        path.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != MILES_SHA or subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Recovery conversion requires clean retained f17 Miles source")
    sys.path.insert(0, str(repo))
    recipe = importlib.import_module("tests.e2e.megatron.test_glm5_2_744b_a40b_5layer_nvfp4_w4a16")
    recipe.MODEL_DIR, recipe.DATA_DIR = str(models), str(datasets)
    os.environ.update(recipe.NVFP4_ENV)
    os.environ["PYTHONPATH"] = f"{repo}:/root/Megatron-LM:" + os.environ.get("PYTHONPATH", "")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    baseline = models / recipe.MODEL_NAME
    data_dir = datasets / "dapo-math-17k"
    for path in (baseline, data_dir):
        if path.is_symlink():
            raise ValueError(f"Do not modify a shared symlink target: {path}; copy into this effort first")

    def run(stage, command):
        log = root / "artifacts" / f"recovery-{stamp}-{stage}.log"
        with log.open("xb") as output:
            result = subprocess.run(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
        log.with_suffix(".exit").write_text(str(result.returncode) + "\n")
        if result.returncode:
            raise RuntimeError(f"{stage} failed ({result.returncode}); retain {log} and all partial files")

    if not args.verify_only:
        run("download-model", ["hf", "download", "Pinaster/GLM-5.2_5layer", "--revision", MODEL_REV, "--local-dir", str(baseline)])
        run("download-data", ["hf", "download", "zhuzilin/dapo-math-17k", "--repo-type", "dataset",
                              "--revision", DATA_REV, "--local-dir", str(data_dir)])
    recipe._validate_glm_checkpoint()
    dataset = data_dir / "dapo-math-17k.jsonl"
    if dataset.stat().st_size != 10490834 or sha(dataset) != DATA_SHA:
        raise ValueError("Dataset bytes differ from the retained original input")
    # Verify cached/reused model bytes against the exact immutable Hub revision,
    # not merely local download timestamps or the old conversion-complete marker.
    from huggingface_hub import HfApi
    info = HfApi().model_info("Pinaster/GLM-5.2_5layer", revision=MODEL_REV, files_metadata=True)
    if info.sha != MODEL_REV:
        raise ValueError("Hub returned an unexpected model revision")
    source_files = {}
    for item in info.siblings:
        path = baseline / item.rfilename
        if not path.is_file() or path.stat().st_size != item.size:
            raise ValueError(f"Pinned source file missing or truncated: {path}")
        expected = item.lfs.sha256 if item.lfs else item.blob_id
        actual = sha(path) if item.lfs else sha(path, "sha1", git_blob=True)
        if actual != expected:
            raise ValueError(f"Pinned source hash mismatch: {path}")
        source_files[item.rfilename] = {"bytes": item.size, "sha256": actual if item.lfs else sha(path)}
    identity = {"model_revision": MODEL_REV, "dataset_revision": DATA_REV, "dataset_sha256": DATA_SHA,
                "miles": head, "nvfp4_env": recipe.NVFP4_ENV, "source_files": source_files}
    (root / "artifacts" / f"recovery-{stamp}-input-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    if args.download_only:
        print("PINNED_INPUTS_VERIFIED", flush=True)
        return
    megatron = subprocess.check_output(["git", "-C", "/root/Megatron-LM", "rev-parse", "HEAD"], text=True).strip()
    if megatron != MEGATRON_SHA:
        raise ValueError(f"Image Megatron changed ({megatron}); root must resolve the retained source overlay before conversion")
    identity["megatron"] = megatron
    identity["runtime"] = {name: importlib.metadata.version(name) for name in ("torch", "transformer-engine", "safetensors")}
    converted = []
    for stage, target, command in (
        ("nvfp4", models / f"{recipe.MODEL_NAME}-NVFP4",
         [sys.executable, "tools/convert_hf_to_nvfp4.py", "--model-dir", str(baseline), "--save-dir",
          str(models / f"{recipe.MODEL_NAME}-NVFP4"), "--extra-high-precision-layers-hf", *recipe.EXTRA_HIGH_PRECISION_LAYERS_HF]),
        ("torch-dist", models / f"{recipe.MODEL_NAME}_torch_dist",
         ["torchrun", "--rdzv-backend=static", "--master-addr=127.0.0.1", "--master-port=29507", "--nproc-per-node=1",
          str(repo / "tools/convert_hf_to_torch_dist.py"), *shlex.split(recipe.U.shell_safe_model_args(recipe.MODEL_TYPE)),
          "--hf-checkpoint", str(baseline), "--save", str(models / f"{recipe.MODEL_NAME}_torch_dist"),
          "--tensor-model-parallel-size", "1", "--expert-tensor-parallel-size", "1", "--pipeline-model-parallel-size", "1",
          "--expert-model-parallel-size", "1"]),
    ):
        marker = models / f"recovery-{stage}-verified.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous["identity"] != identity:
                raise ValueError(f"Existing completed stage has a different identity: {marker}; preserve and resolve explicitly")
            check_files(target, previous["files"])
            converted.append(target)
            continue
        if args.verify_only:
            raise ValueError(f"Missing hash-verified completion record: {marker}")
        if target.exists() or target.is_symlink():
            target.rename(target.with_name(target.name + ".unverified-" + stamp))
        run(stage, command)
        if stage == "torch-dist" and (target / "latest_checkpointed_iteration.txt").read_text().strip() != "release":
            raise ValueError("Converted distributed checkpoint lacks its release tracker")
        output_files = files(target)
        if not output_files:
            raise ValueError(f"Conversion produced no files: {target}")
        marker.write_text(json.dumps({"identity": identity, "files": output_files}, indent=2) + "\n")
        converted.append(target)
    if args.backup_dir:
        backup = args.backup_dir.resolve()
        if not backup.is_relative_to(Path("/data")) or backup == Path("/data"):
            raise ValueError("Use a task-owned /data backup subdirectory")
        sources = [baseline, *converted, data_dir]
        index = {str(path.relative_to(root)): files(path) for path in sources}
        total = sum(item["bytes"] for manifest in index.values() for item in manifest.values())
        backup.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(backup).free < total * 1.1:
            raise ValueError(f"Durable backup needs {total} bytes plus reserve; prepared local inputs are preserved")
        for relative, manifest in index.items():
            for name, item in manifest.items():
                source, target = root / relative / name, backup / relative / name
                if target.exists():
                    check_files(backup / relative, {name: item})
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial-" + stamp)
                shutil.copyfile(source, partial)
                check_files(partial.parent, {partial.name: item})
                partial.rename(target)
        for marker in models.glob("recovery-*-verified.json"):
            target = backup / "models" / marker.name
            if target.exists() and target.read_bytes() != marker.read_bytes():
                raise ValueError(f"Different durable verification marker already exists: {target}")
            if not target.exists():
                shutil.copyfile(marker, target)
        (backup / f"input-backup-{stamp}.json").write_text(json.dumps({"identity": identity, "files": index}, indent=2) + "\n")
        print(f"DURABLE_INPUT_BACKUP_VERIFIED {backup}", flush=True)
    print("RECOVERY_PREPARE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
