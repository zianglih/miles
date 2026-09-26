"""Keep earlier CPU evidence distinct from a verified test-only descendant.

This never treats old test logs as a new-head run. The sole admitted intervening
change is the known Torch compatibility test file; every other tracked path must
be identical. Production hashes are recorded explicitly as additional evidence.
"""

import hashlib
import json
from pathlib import Path
import re
import subprocess


COMPATIBILITY_TEST = "tests/fast/utils/test_delta_preparation.py"
PRODUCTION_FILES = (
    "miles/utils/delta_preparation.py", "miles/utils/arguments.py",
    "miles/backends/training_utils/weight_update/packed_delta.py",
    "miles/backends/training_utils/weight_update/protocols/delta.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def prove_test_only_descendant(repo, measured_head, selected_head):
    for head in (measured_head, selected_head):
        require(isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head), "Attribution needs full source commits")
    require(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", measured_head, selected_head],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
            "Selected head is not a descendant of the measured source")
    changed = git_bytes(repo, "diff", "--name-only", "--no-renames", "-z", measured_head, selected_head).decode().split("\0")
    changed = [name for name in changed if name]
    require(set(changed) <= {COMPATIBILITY_TEST}, "Intervening changes are not confined to the compatibility test")
    hashes = {}
    for name in PRODUCTION_FILES:
        before = git_bytes(repo, "show", f"{measured_head}:{name}")
        after = git_bytes(repo, "show", f"{selected_head}:{name}")
        require(before == after, f"Production bytes differ from measured source: {name}")
        hashes[name] = sha256(after)
    return {"measured_source": measured_head, "selected_source": selected_head, "changed_paths": changed,
            "all_other_tracked_paths_identical": True, "production_sha256": hashes,
            "selected_compatibility_test_sha256": sha256(git_bytes(repo, "show", f"{selected_head}:{COMPATIBILITY_TEST}"))}


def validate_attribution(root, plan):
    root = Path(root)
    scope = plan.get("evidence_source_commits", {
        "cpu_replay": plan["standalone_source"], "combined_preflight": plan["sources"]["miles"]})
    require(isinstance(scope, dict) and set(scope) == {"cpu_replay", "combined_preflight"},
            "evidence_source_commits requires cpu_replay and combined_preflight")
    tested_head = (root / "artifacts/torch-compile-final-production-tested-head.txt").read_text().strip()
    cpu = json.loads((root / "TORCH_COMPILE_FINAL_CPU_RESULTS.json").read_bytes())
    require(scope["cpu_replay"] == tested_head == cpu["source_commit"],
            "Historical CPU replay/test attribution does not match retained records")
    preflight = json.loads((root / "artifacts/compile-final-preflight-01/manifest.json").read_bytes())
    require(preflight["sources"] == {"miles": scope["combined_preflight"], "sglang": plan["sources"]["sglang"]}
            and preflight["standalone_source"] == scope["cpu_replay"],
            "Historical preflight attribution does not match retained records")
    return {
        "standalone_cpu_replay": prove_test_only_descendant(root / "miles-torch-compile", scope["cpu_replay"], plan["standalone_source"]),
        "combined_historical_preflight": prove_test_only_descendant(root / "miles-torch-compile-validation", scope["combined_preflight"], plan["sources"]["miles"]),
        "scope": "Earlier CPU timings/tests retain their measured commits; production identity does not make them new-head tests.",
    }


def selected_preflight(root, plan):
    """Admit a new selected-head preflight independently of historical checks."""
    name = plan.get("preflight_evidence_directory")
    if name is None:
        return None
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts and len(relative.parts) == 2
            and relative.parts[0] == "artifacts", "Preflight evidence must be an artifacts/<run> directory")
    root = Path(root).resolve()
    directory = root / relative
    require(directory.resolve().is_relative_to(root) and not directory.is_symlink(), "Unsafe preflight directory")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    require(manifest["sources"] == plan["sources"] and manifest["standalone_source"] == plan["standalone_source"],
            "New preflight source pins do not match selected source")
    require(set(manifest["checks"]) == {"focused-tests", "gloo-failures"}
            and all(check["exit"] == 0 for check in manifest["checks"].values()),
            "Selected-head preflight did not complete both checks successfully")
    for filename, key in (("run_compiled_preflight.py", "script_sha256"),
                          ("validate_compiled_sender_gloo.py", "gloo_script_sha256")):
        require(sha256((root / filename).read_bytes()) == manifest[key], "Selected preflight helper hash drift")
    focused = (directory / "focused-tests.log").read_bytes().decode()
    gloo = (directory / "gloo-failures.log").read_bytes().decode()
    require(re.search(r"\b\d+ passed(?:, \d+ \w+)* in [\d.]+s", focused), "Selected preflight has no passing pytest summary")
    require("PASS: both failure modes drained all four collectives on both ranks without publication" in gloo,
            "Selected preflight has no completed Gloo result")
    return {"directory": str(relative), "manifest": manifest,
            "log_sha256": {"focused-tests.log": sha256(focused.encode()), "gloo-failures.log": sha256(gloo.encode())}}


def validate_compatibility(root, attribution):
    root = Path(root).resolve()
    path = root / "COMPILE_CI_COMPATIBILITY.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_bytes())
    require(record["schema_version"] == 1 and record["test_path"] == COMPATIBILITY_TEST,
            "Unexpected compatibility evidence schema/test")
    measured = attribution["standalone_cpu_replay"]
    require(record["historical_head"] == measured["measured_source"]
            and record["tested_test_sha256"] == measured["selected_compatibility_test_sha256"],
            "Compatibility diagnostic test bytes do not match selected source")
    require(record["upstream_pr"] == "https://github.com/pytorch/pytorch/pull/178950"
            and record["upstream_commit"] == "https://github.com/pytorch/pytorch/commit/d7b75b8250f43949bb2609f406ac942f935cab46",
            "Compatibility upstream attribution changed; review it")
    for name, info in record["files"].items():
        relative = Path(name)
        source = root / relative
        require(not relative.is_absolute() and ".." not in relative.parts
                and source.resolve().is_relative_to(root) and not source.is_symlink(), "Unsafe compatibility file")
        data = source.read_bytes()
        require(sha256(data) == info["sha256"] and len(data) == info["bytes"], f"Compatibility evidence drift: {name}")
    for key in ("focused", "full_shard"):
        result = record[key]
        if result.get("state") == "pending":
            require(key == "full_shard", "Focused compatibility result cannot be pending")
            continue
        require(result["log"] in record["files"], f"Compatibility {key} log is not hashed")
        text = (root / result["log"]).read_bytes().decode()
        require(result["summary"] in text and re.fullmatch(r"\d+ passed(?:, \d+ \w+)* in [\d.]+s", result["summary"]),
                f"Compatibility {key} has no matching pass summary")
        if "phases" in result:
            require(result["phases"] in record["files"], "Compatibility phase journal is not hashed")
            phases = [json.loads(line) for line in (root / result["phases"]).read_bytes().splitlines() if line]
            finishes = [event for event in phases if event.get("event") == "finish" and event.get("phase") == result["phase"]]
            require(len(finishes) == 1 and finishes[0]["returncode"] == 0 and finishes[0]["timed_out"] is False,
                    f"Compatibility {key} process did not complete successfully")
    return record


def current_hosted_ci(root, head):
    """Read the new head's compact capture, never substitute historical checks."""
    name = f"artifacts/compile-pr-ci-3720-{head[:7]}"
    directory = Path(root) / name
    if not (directory / "summary.json").is_file():
        return None
    summary = json.loads((directory / "summary.json").read_bytes())
    latest = json.loads((directory / "latest.json").read_bytes())
    require(summary["head"] == head == latest["pr"]["headRefOid"], "Hosted CI capture does not match selected head")
    require(all(run["headSha"] == head for run in latest["runs"]), "Hosted workflow run source differs from selected head")
    expected = [{key: run[key] for key in ("databaseId", "name", "status", "conclusion", "url")}
                | {"jobs": [{key: job[key] for key in ("name", "status", "conclusion", "url")}
                            for job in run["jobs"]]} for run in latest["runs"]]
    require(summary["runs"] == expected and summary["all_runs_completed"] ==
            (bool(expected) and all(run["status"] == "completed" for run in expected)),
            "Hosted CI summary differs from complete captured run/job records")
    for run in expected:
        require(re.fullmatch(r"https://github\.com/radixark/miles/actions/runs/[0-9]+", run["url"]),
                "Unexpected hosted workflow URL")
        for job in run["jobs"]:
            require(re.fullmatch(r"https://github\.com/radixark/miles/actions/runs/[0-9]+/job/[0-9]+", job["url"]),
                    "Unexpected hosted job URL")
    return {"directory": name, "summary": summary, "files": [f"{name}/summary.json", f"{name}/latest.json"]}
