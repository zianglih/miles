#!/usr/bin/env python3
"""Copy stopped campaign evidence without deleting or modifying source artifacts.

Run beside the campaign files after all writers have stopped. Failed/incomplete
attempts are retained; incomplete expected evidence returns 2 after preservation.
No remote commands, repository edits, or publication are performed.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile
import uuid


CHUNK = 2 * 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            value.update(chunk)
    return value.hexdigest()


def preserve_snapshot(root, destination, *, run_names=None, include_other_attempts=True, require_complete=True):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    artifacts = root / "artifacts"
    plan_path = root / "COMPILE_VALIDATION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    planned = [run["name"] for run in plan["runs"]] if run_names is None else list(run_names)
    if any(Path(name).name != name or name in {"", ".", ".."} for name in planned):
        raise ValueError("Run names must be single path components")
    if destination == root or root.is_relative_to(destination) or destination.exists():
        raise ValueError("Destination must be a new separate snapshot directory")
    names = sorted(set(planned) | ({path.name for path in artifacts.glob("compile-v*-*") if path.is_dir()}
                                  if include_other_attempts else set()))
    selected = {plan_path, root / "CAMPAIGN.json", root / "environment-image.json", root / "preserve_compile_campaign.py"}
    excluded, errors, pinned_hashes = [], [], {}
    archived_plan = root / "COMPILE_VALIDATION_PLAN_V1.json"
    if archived_plan.is_file():
        selected.add(archived_plan)
    for field in ("campaign_metadata", "environment_metadata", "cpu_calibration_metadata"):
        if field not in plan:
            continue
        metadata = plan[field]
        relative = Path(metadata["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe pinned metadata path: {field}")
        source = root / relative
        selected.add(source)
        pinned_hashes[str(relative)] = metadata["sha256"]
        if not source.is_file() or digest(source) != metadata["sha256"]:
            errors.append(f"Pinned metadata missing or changed: {field}")
    for path in artifacts.iterdir():
        if path.is_file() and (path.name.startswith("compile-campaign") or
                              (include_other_attempts and path.name.startswith("compile-v")) or
                              any(path.name == name or path.name.startswith(name + "-") or
                                  path.name.startswith(name + ".") for name in names)):
            selected.add(path)
    for name in names:
        run_dir = artifacts / name
        if run_dir.is_symlink():
            errors.append(f"Symlink not followed: {run_dir.relative_to(root)}")
            continue
        for path in run_dir.rglob("*"):
            relative = path.relative_to(run_dir)
            if relative.parts[0] == "rollout-checkpoint":
                if len(relative.parts) == 1:
                    excluded.append(str(path.relative_to(root)))
                continue
            if path.is_symlink():
                errors.append(f"Symlink not followed: {path.relative_to(root)}")
            elif path.is_file():
                selected.add(path)
    destination.mkdir(parents=True, exist_ok=False)
    copied = {}
    for source in sorted(selected):
        relative = str(source.relative_to(root))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if source.is_symlink():
                raise ValueError("Symlink not followed")
            before = source.stat()
            value = hashlib.sha256()
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                while chunk := incoming.read(CHUNK):
                    value.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source changed during copy; stop all campaign writers and create a new snapshot")
            if target.stat().st_size != before.st_size or digest(target) != value.hexdigest():
                raise ValueError("Copied bytes failed SHA256 verification")
            copied[relative] = {"bytes": before.st_size, "sha256": value.hexdigest(), "verified": True}
        except Exception as error:
            errors.append(f"{relative}: {type(error).__name__}: {error}")
            if target.is_file():
                copied[relative] = {"bytes": target.stat().st_size, "sha256": digest(target), "verified": False}
    for relative, expected_hash in pinned_hashes.items():
        if copied.get(relative, {}).get("sha256") != expected_hash:
            errors.append(f"Copied pinned metadata hash differs: {relative}")
    coverage = {}
    total_shards = 0
    for name in planned:
        prefix = f"artifacts/{name}/"
        files = [relative[len(prefix):] for relative in copied if relative.startswith(prefix)]
        missing = [item for item in ("manifest.json", "driver.jsonl", *(f"trainer-rank{rank}.jsonl" for rank in range(4)))
                   if item not in files]
        missing += [f"../{name}{suffix}" for suffix in (".log", ".exit") if f"artifacts/{name}{suffix}" not in copied]
        debug = [item for item in files if item.startswith("dump_details/") and item.endswith(".pt")]
        shards = [item for item in files if item.startswith("delta-publication/") and item.endswith(".safetensors")]
        indexes = [f"delta-publication/weight_v{version:06d}/model.safetensors.index.json" for version in range(1, 7)]
        missing.extend(item for item in indexes if item not in files)
        total_shards += len(shards)
        coverage[name] = {"missing": missing, "debug_dump_count": len(debug), "shard_count": len(shards)}
        if require_complete and (missing or len(debug) < 14 or len(shards) != 6):
            errors.append(f"{name}: incomplete expected evidence; preserved all available files")
    if require_complete and total_shards != 6 * len(planned):
        errors.append(f"Expected {6 * len(planned)} delta shards, observed {total_shards}")
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source_root": str(root), "planned_runs": planned, "preserved_run_directories": names,
                "excluded_reconstructible_baselines": excluded, "coverage": coverage, "errors": errors,
                "all_copied_files_verified": all(item["verified"] for item in copied.values()), "files": copied,
                "require_complete": require_complete,
                "note": "Preservation is not validation admission. Nonzero exits, incomplete runs and failures remain raw."}
    (destination / "PRESERVATION.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = destination.with_name(destination.name + ".tar")
    partial = archive.with_name(archive.name + ".partial")
    if archive.exists():
        raise ValueError(f"Archive already exists: {archive}")
    expected = {name: item["sha256"] for name, item in copied.items()}
    expected["PRESERVATION.json"] = digest(destination / "PRESERVATION.json")
    with tarfile.open(partial, "x") as stream:
        for name in sorted(expected):
            stream.add(destination / name, arcname=name, recursive=False)
    with tarfile.open(partial, "r") as stream:
        members = stream.getmembers()
        if {member.name for member in members} != set(expected) or len(members) != len(expected):
            raise ValueError("Archive inventory mismatch; source and snapshot retained")
        for member in members:
            value = hashlib.sha256()
            with stream.extractfile(member) as incoming:
                while chunk := incoming.read(CHUNK):
                    value.update(chunk)
            if value.hexdigest() != expected[member.name]:
                raise ValueError(f"Archive hash mismatch: {member.name}; source and snapshot retained")
    archive_hash = digest(partial)
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    sidecar = archive.with_name(archive.name + ".sha256")
    with sidecar.open("x") as stream:
        stream.write(f"{archive_hash}  {archive.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.rename(archive)
    return {"snapshot": str(destination), "archive": str(archive), "sha256": archive_hash,
            "files": len(copied), "delta_shards": total_shards, "errors": errors}


def preserve_arm(root, run_name, backup_dir, *, require_complete=True):
    """Called after one arm stops; return errors without discarding partial evidence."""
    root, backup_dir = Path(root).resolve(), Path(backup_dir).resolve()
    if not backup_dir.is_relative_to(Path("/data")) or backup_dir == Path("/data"):
        raise ValueError("Per-arm durable backup must use a task-owned /data subdirectory")
    if Path(run_name).name != run_name or run_name not in {
        run["name"] for run in json.loads((root / "COMPILE_VALIDATION_PLAN.json").read_text())["runs"]
    }:
        raise ValueError("Per-arm backup requires an exact planned run name")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = backup_dir / f"{run_name}-{stamp}-{uuid.uuid4().hex[:8]}"
    return preserve_snapshot(root, destination, run_names=[run_name], include_other_attempts=False,
                             require_complete=require_complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--destination", type=Path, required=True, help="New snapshot directory; must not exist")
    args = parser.parse_args()
    result = preserve_snapshot(args.root, args.destination)
    print(json.dumps(result, indent=2))
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
