"""Resolve per-campaign metadata without rewriting historical environment files.

Plans may pin campaign_metadata/environment_metadata (together), and optionally
cpu_calibration_metadata, as {path: task-relative JSON, sha256: 64 hex digits}.
Legacy plans omitting these fields retain their original filename defaults.
"""

import hashlib
from pathlib import Path
import re

DEFAULTS = {
    "campaign_metadata": "CAMPAIGN.json",
    "environment_metadata": "environment-image.json",
    "cpu_calibration_metadata": "artifacts/cpu-observer-fast-final-calibration.json",
}


def validate_specs(plan):
    pair = {field for field in ("campaign_metadata", "environment_metadata") if field in plan}
    if pair not in (set(), {"campaign_metadata", "environment_metadata"}):
        raise ValueError("Pin campaign_metadata and environment_metadata together")
    for field in DEFAULTS:
        if field not in plan:
            continue
        value = plan[field]
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError(f"{field} must contain exactly path and sha256")
        name = value["path"]
        if not isinstance(name, str):
            raise ValueError(f"{field} path must be a relative JSON filename")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
            raise ValueError(f"{field} path must stay under the plan directory and name JSON")
        if not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
            raise ValueError(f"{field} requires a full SHA256")
    backup = plan.get("durable_backup_dir")
    if backup is not None:
        if not isinstance(backup, str):
            raise ValueError("durable_backup_dir must be a string")
        path = Path(backup)
        if not path.is_absolute() or ".." in path.parts or path.parts[1:2] != ("data",) or len(path.parts) < 4:
            raise ValueError("durable_backup_dir must be an absolute task-owned /data/<owner>/<effort> path")


def resolve_metadata(plan_path, plan, *, campaign=None, environment=None, calibration=None):
    """Explicit legacy overrides remain valid; pinned overrides must agree exactly."""
    validate_specs(plan)
    root = Path(plan_path).resolve().parent
    overrides = {"campaign_metadata": campaign, "environment_metadata": environment,
                 "cpu_calibration_metadata": calibration}
    resolved = {}
    for field, default in DEFAULTS.items():
        override = overrides[field]
        if field in plan:
            path = root / plan[field]["path"]
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"{field} escapes the plan directory")
            if any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)):
                raise ValueError(f"{field} cannot use symlinks")
            if override is not None and Path(override).resolve() != path.resolve():
                raise ValueError(f"Explicit {field} override differs from frozen plan")
            if not path.is_file():
                raise ValueError(f"Missing {field}: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != plan[field]["sha256"]:
                raise ValueError(f"{field} bytes differ from frozen SHA256")
        else:
            path = Path(override) if override is not None else root / default
        if not path.is_file():
            raise ValueError(f"Missing {field}: {path}")
        resolved[field] = path.resolve()
    return resolved


def metadata_record(paths, *, relative_to=None):
    return {field: {"path": str(path.relative_to(Path(relative_to).resolve()) if relative_to else path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for field, path in paths.items()}
