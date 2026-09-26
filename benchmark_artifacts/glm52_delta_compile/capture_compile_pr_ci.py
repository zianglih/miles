#!/usr/bin/env python3
"""Capture one read-only hosted-CI snapshot for Miles draft PR 3720."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "artifacts" / "compile-pr-ci-3720-6fb0e2a"
HEAD = "6fb0e2a9d81fc343863ba6015512fe93992892e4"


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default=HEAD, help="Exact 40-character PR head SHA")
    parser.add_argument("--destination", type=Path, default=DEST, help="Separate evidence directory for this head")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.head):
        parser.error("--head must be a full lowercase 40-character Git SHA")
    args.destination = args.destination.expanduser().resolve()
    if args.head != HEAD and args.destination == DEST.resolve():
        parser.error("a new head requires a separate --destination; preserve the original CI evidence")
    validate_destination(args.destination, args.head)
    return args


def validate_destination(destination, head):
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ValueError(f"CI destination is not a directory: {destination}")
    entries = list(destination.iterdir())
    records = [path for path in entries if path.name in ("summary.json", "latest.json")
               or (path.name.startswith("snapshot-") and path.suffix == ".json")]
    if entries and not records:
        raise ValueError(f"Refuse nonempty destination without CI head records: {destination}")
    for path in records:
        value = json.loads(path.read_text())
        recorded_head = value.get("head") if path.name == "summary.json" else value.get("pr", {}).get("headRefOid")
        if recorded_head != head:
            raise ValueError(f"Refuse to mix CI heads in {path}: {recorded_head} != {head}")


def main(argv=None):
    args = options(argv)
    destination, head = args.destination, args.head
    gh("auth", "switch", "--hostname", "github.com", "--user", "zianglih")
    if gh("api", "user", "--jq", ".login").strip() != "zianglih":
        raise RuntimeError("Wrong GitHub CLI identity")
    pr = json.loads(gh("pr", "view", "3720", "--repo", "radixark/miles", "--json",
                       "number,url,isDraft,headRefOid,headRefName,baseRefName,statusCheckRollup"))
    if pr["headRefOid"] != head:
        raise RuntimeError(f"PR head changed: {pr['headRefOid']}")
    validate_destination(destination, head)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    runs = []
    for run_id in sorted({re.search(r"/actions/runs/(\d+)", check["detailsUrl"])[1]
                          for check in pr["statusCheckRollup"] if "/actions/runs/" in check.get("detailsUrl", "")}):
        run = json.loads(gh("run", "view", run_id, "--repo", "radixark/miles", "--json",
                            "databaseId,name,event,status,conclusion,headSha,headBranch,createdAt,updatedAt,url,jobs"))
        runs.append(run)
        log_path = destination / f"run-{run_id}.log"
        if run["status"] == "completed" and not log_path.exists():
            result = subprocess.run(["gh", "run", "view", run_id, "--repo", "radixark/miles", "--log"],
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                with log_path.open("x") as stream:
                    stream.write(result.stdout)
            else:
                with (destination / f"run-{run_id}-{stamp}-log-unavailable.txt").open("x") as stream:
                    stream.write(result.stderr)
    snapshot = {"captured_at_utc": stamp, "identity": "zianglih", "pr": pr, "runs": runs}
    raw = json.dumps(snapshot, indent=2) + "\n"
    with (destination / f"snapshot-{stamp}.json").open("x") as stream:
        stream.write(raw)
    (destination / "latest.json").write_text(raw)
    summary = {"captured_at_utc": stamp, "pr_url": pr["url"], "head": head,
               "all_runs_completed": bool(runs) and all(run["status"] == "completed" for run in runs),
               "runs": [{key: run[key] for key in ("databaseId", "name", "status", "conclusion", "url")}
                        | {"jobs": [{key: job[key] for key in ("name", "status", "conclusion", "url")}
                                    for job in run["jobs"]]} for run in runs],
               "files": {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in sorted(destination.iterdir()) if path.is_file() and path.suffix == ".log"}}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
