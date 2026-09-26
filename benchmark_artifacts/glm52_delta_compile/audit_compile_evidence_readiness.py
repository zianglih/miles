#!/usr/bin/env python3
"""Audit completed prerequisites without admitting or fabricating GPU results."""

import json
from pathlib import Path

import publish_compile_evidence as publisher
from compile_campaign_metadata import metadata_record, resolve_metadata
from compile_evidence_attribution import current_hosted_ci, selected_preflight, validate_attribution, validate_compatibility

ROOT = Path(__file__).resolve().parent
FINAL_REPLAY = "artifacts/torch-compile-word-two-phase-32x1-replay.jsonl"


def main():
    plan_path = ROOT / "COMPILE_VALIDATION_PLAN.json"
    plan = json.loads(plan_path.read_bytes())
    publisher.reporter.validate_plan(plan)
    metadata = resolve_metadata(plan_path, plan)
    attribution = validate_attribution(ROOT, plan)
    publisher.require(json.loads((ROOT / "COMPILE_SOURCE_ATTRIBUTION.json").read_bytes()) == attribution,
                      "Stored source attribution differs")
    preflight = selected_preflight(ROOT, plan)
    compatibility = validate_compatibility(ROOT, attribution)
    ci = current_hosted_ci(ROOT, plan["standalone_source"])
    frozen = json.loads((ROOT / "artifacts/compile-campaign-frozen-inputs.json").read_bytes())
    for name, expected in frozen.items():
        publisher.require(publisher.digest(publisher.read_local(ROOT, name)) == expected, f"Frozen input drift: {name}")
    replays = {}
    replay_files = set()
    for name, source in ((publisher.NEGATIVE_REPLAY, publisher.NEGATIVE_SOURCE),
                         (publisher.MATERIALIZE_REPLAY, publisher.MATERIALIZE_SOURCE),
                         (FINAL_REPLAY, attribution["standalone_cpu_replay"]["measured_source"])):
        files, record = publisher.validate_replay(ROOT, name, source, ROOT / "miles-torch-compile",
                                                   allow_uncommitted=name == publisher.MATERIALIZE_REPLAY)
        replay_files.update(files)
        replays[name] = record
    names = set(publisher.FILES) - {"COMPILED_VALIDATION_RESULTS.json", "COMPILED_VALIDATION_RESULTS.md"}
    names |= replay_files | set(compatibility["files"]) | set(ci["files"])
    names |= {str(path.relative_to(ROOT)) for path in metadata.values()}
    names |= {f"{preflight['directory']}/{file}" for file in ("manifest.json", "focused-tests.log", "gloo-failures.log")}
    scanned = {name: publisher.digest(publisher.read_local(ROOT, name)) for name in sorted(names)}
    campaign = json.loads(metadata["campaign_metadata"].read_bytes())
    prepared = campaign["prepared_inputs"]
    publisher.require(scanned[prepared["path"]] == prepared["sha256"], "Prepared input seal changed")
    output = {
        "scope": "Completed local prerequisites only; GPU admission/publication intentionally remain pending.",
        "gpu_admitted": False, "plan_sha256": publisher.digest(plan_path.read_bytes()),
        "source_attribution": attribution, "selected_preflight": preflight,
        "linux_compatibility_state": compatibility["state"],
        "hosted_ci": ci["summary"], "metadata_selection": metadata_record(metadata, relative_to=ROOT),
        "cpu_replays": replays, "frozen_inputs_verified": len(frozen), "scanned_file_sha256": scanned,
        "scan_scope": "No unapproved matches; exact public synthetic-URL exceptions retain raw bytes and recorded source proof.",
        "codegen_scope": "Retained files scanned/hashed; generated-code conclusions retain their separate manual review scope.",
        "remaining": ["Both complete GPU arms and strict paired report", "Exact public evidence commit and link-byte verification"],
    }
    destination = ROOT / "artifacts/compile-evidence-readiness.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"prerequisites_validated": True, "gpu_admitted": False,
                      "files_scanned": len(scanned), "cpu_samples": sum(row["samples"] for row in replays.values()),
                      "frozen_inputs_verified": len(frozen), "output": str(destination)}))


if __name__ == "__main__":
    main()
