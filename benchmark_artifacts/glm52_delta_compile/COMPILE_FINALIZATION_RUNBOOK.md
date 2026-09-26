# Complete the matched compile campaign and publish its evidence

Work from `/Users/ziangli/playground/projects/glm52-w4a16-delta-sync-c2`.
Use the project's `.venv-delta-cpu/bin/python` for all evidence commands: strict
admission loads retained binary training dumps with PyTorch; the system Python
has no PyTorch installation. Root owns harvesting, Git commits/pushes and PR edits. This runbook performs no
automatic remote work. Keep source, plan, runner, preflight and measurement helpers
fixed while the campaign runs.

The completed prerequisite audit covers **216 text files**, **84 CPU samples**
across the selected and two historical replays, and **27 frozen runtime inputs**.
It is recorded in `artifacts/compile-evidence-readiness.json` with
`gpu_admitted: false`. The 35 local helper fixtures passed in 1.998 s. These facts
do not admit the running GPU pair.

Frozen identities:

- Plan: `d7483ae580a0e73b4f2f5f0d355a96d5adc7d05b834006acfe7d388b7332be9d`.
- Standalone: `d34a55113647315cc5b5eb5a3532dc563c72fa6c`.
- Combined: `767439e95bca5203211ba374b4e04b702c2b7ed2`.
- Receiver: `2f5fb2a09f08eb9a44c0fed5a28be0bda2942b0f`.
- Runtime file seal: `artifacts/compile-campaign-frozen-inputs.json`, SHA256
  `febdbe0e0cc4af55d034ef0d77c51bdb2787f68162cf9c0d3de9d487e8a58c68`.

## 1. Harvest both complete arms before report generation

Required runs are `compile-v2-synthetic-numpy-01` and
`compile-v2-synthetic-torch-01`. Preserve the complete run directories under local
`artifacts/`, their sibling `.log`, `.exit`, summary files and
`-campaign-metadata.json` sidecars. Retain all driver/trainer/CPU journals, full
manifests, training evidence, **14 binary debug dumps per arm**, and all six delta
publication indexes and payload shards. Binary evidence stays private but is
necessary for strict local admission. Do not substitute v1 summaries or remove
slow samples.

The runner produces summaries after each arm exits, then verifies its durable
archive before starting the next arm. Harvest the campaign exit/backup records
and retain their successful archive SHA256 checks. A failed/incomplete campaign
or backup needs explicit diagnosis; do not bypass an admission failure.

If the three summaries were not generated despite complete stopped raw evidence,
run the unchanged summarizers for each arm after harvesting:

```bash
for compile_run in compile-v2-synthetic-numpy-01 compile-v2-synthetic-torch-01; do
  .venv-delta-cpu/bin/python summarize_weight_sync.py "artifacts/$compile_run"
  .venv-delta-cpu/bin/python summarize_training_evidence.py "artifacts/$compile_run" --debug-dumps require
  .venv-delta-cpu/bin/python summarize_weight_sync_cpu_v2.py "artifacts/$compile_run"
done
```

This is post-run evidence processing, not a new benchmark. The strict reporter
will independently reject missing raw evidence, invalid CPU windows, stale
summaries or timed compiler activity.

## 2. Build and admit the paired GPU report

```bash
.venv-delta-cpu/bin/python audit_compile_evidence_readiness.py
.venv-delta-cpu/bin/python build_compiled_validation_report.py \
  --plan COMPILE_VALIDATION_PLAN.json \
  --output-dir . \
  artifacts/compile-v2-synthetic-numpy-01 \
  artifacts/compile-v2-synthetic-torch-01
```

The metadata arguments are intentionally omitted: the frozen plan selects and
hash-pins the replacement campaign/environment and historical external observer
calibration. Success writes `COMPILED_VALIDATION_RESULTS.json` and `.md`.
Read the two reports before staging. Require all **14 update rows**, **56 per-rank
compiler observations**, and zero new measured u2–u6 graph/cache/kernel activity.
Initial/u1 records remain visible. Compare only this matched v2 NumPy/compiled
pair; do not compute a ratio against the original broadcast/delta campaign.

The original CPU replay remains measured at `6fb0e2a9`; original combined
preflight remains `283a1cb4`. `COMPILE_SOURCE_ATTRIBUTION.json` proves that only the
compatibility test changed in their selected descendants. The actual selected-head
replacement preflight is `artifacts/compile-v2-preflight-01`: 64 passed / 28 warnings
/ 36.26 s, plus completed two-rank Gloo checks. Complete logs are retained.

The compatibility diagnostic has 3,172 passed / 27 skipped / 44 warnings / 288.97 s.
Selected-head hosted CI is already terminal in
`artifacts/compile-pr-ci-3720-d34a551/{summary,latest}.json`: pre-commit, all four
stage-a CPU shards and stage-b CPU passed; GPU stages were policy-skipped. No new
capture is needed unless the selected head changes. The finalizer verifies each
workflow's `headSha`; embedded old PR-body output is not new test evidence.

## 3. Audit, then stage the dedicated public text bundle

Use these exact proof/log arguments. Generated-code evidence is from the original
CPU replay's byte-identical production implementation. The two explicit validation
logs are from the new selected-head preflight. All historical logs, CPU samples,
CI diagnostic logs and compact recovery/frozen metadata are in the helper's input
list; binary weights/debug data are excluded from publication.

```bash
.venv-delta-cpu/bin/python publish_compile_evidence.py \
  --final-replay artifacts/torch-compile-word-two-phase-32x1-replay.jsonl \
  --codegen-proof artifacts/torch-compile-final-codegen-complete/CODEGEN.md \
  --codegen-proof artifacts/torch-compile-final-codegen-complete/proof.json \
  --validation-log artifacts/compile-v2-preflight-01/focused-tests.log \
  --validation-log artifacts/compile-v2-preflight-01/gloo-failures.log \
  --include COMPILE_FINALIZATION_RUNBOOK.md
```

Only after this audit succeeds, repeat the same command with `--stage`:

```bash
.venv-delta-cpu/bin/python publish_compile_evidence.py \
  --final-replay artifacts/torch-compile-word-two-phase-32x1-replay.jsonl \
  --codegen-proof artifacts/torch-compile-final-codegen-complete/CODEGEN.md \
  --codegen-proof artifacts/torch-compile-final-codegen-complete/proof.json \
  --validation-log artifacts/compile-v2-preflight-01/focused-tests.log \
  --validation-log artifacts/compile-v2-preflight-01/gloo-failures.log \
  --include COMPILE_FINALIZATION_RUNBOOK.md \
  --stage
```

The only destination is
`miles-evidence/benchmark_artifacts/glm52_delta_compile`. Audit reruns strict GPU
admission, checks all frozen inputs and source attribution, and scans every text
file. The three documented synthetic malformed-URL fixtures are allowed only by
exact file/value hashes and test-node context; there is no general secret bypass
or redaction. Staging preserves raw bytes. An existing differing destination is
rejected and requires review rather than silent replacement.

## 4. Root commits and pushes only the fork evidence branch

The publisher verifies the task-local Git store, branch
`glm52-delta-sync-evidence`, and `fork` remote `zianglih/miles`. Inspect the working
tree and staged paths, then commit only the new evidence folder:

```bash
git -C miles-evidence status --short --untracked-files=all
git -C miles-evidence add -- benchmark_artifacts/glm52_delta_compile
git -C miles-evidence diff --cached --name-only
git -C miles-evidence commit -m "Preserve GLM-5.2 compiled delta comparison evidence"
compile_evidence_commit=$(git -C miles-evidence rev-parse HEAD)
git -C miles-evidence push fork HEAD:glm52-delta-sync-evidence
git -C miles-evidence ls-remote fork refs/heads/glm52-delta-sync-evidence
```

The remote head must equal `$compile_evidence_commit`. Every staged path must be
under `benchmark_artifacts/glm52_delta_compile`; the original
`benchmark_artifacts/glm52_delta_sync` evidence remains unchanged at its existing
published commit. No upstream evidence commit, force push or Git credential
configuration change is needed.

Verify every committed public file against the exact committed manifest:

```bash
.venv-delta-cpu/bin/python - "$compile_evidence_commit" <<'PY'
import hashlib, json, subprocess, sys
commit = sys.argv[1]
prefix = "benchmark_artifacts/glm52_delta_compile/"
def blob(name):
    return subprocess.check_output(["git", "-C", "miles-evidence", "show", f"{commit}:{prefix}{name}"])
manifest = json.loads(blob("SHA256.json"))
for name, info in manifest.items():
    content = blob(name)
    assert len(content) == info["bytes"], name
    assert hashlib.sha256(content).hexdigest() == info["sha256"], name
print(f"Verified {len(manifest)} exact committed files at {commit}")
PY
```

## 5. Assemble the final PR body using the exact public commit

```bash
compile_evidence_url="https://github.com/zianglih/miles/tree/${compile_evidence_commit}/benchmark_artifacts/glm52_delta_compile"
.venv-delta-cpu/bin/python finalize_compile_pr_body.py --evidence-url "$compile_evidence_url"
```

This writes `miles-torch-compile-pr-body-final.md` only after rereading strict raw
GPU admission and checking exact published Git bytes for every linked or embedded
log. It retains all 28 selected CPU samples, all 14 GPU update rows, complete new
preflight/Gloo output, actual new-head CI statuses and the historical boundaries.
It enforces a body smaller than 65,000 UTF-8 bytes. The root-owned remote push and
reachability check above are separate from the finalizer's local Git checks.

## 6. Root updates the existing draft and verifies raw body bytes

Inspect the final body and conclusions first. Select the public GitHub identity
before the write; do not change Git transport credentials:

```bash
gh auth switch --hostname github.com --user zianglih
gh api user --jq .login
gh pr edit 3720 --repo radixark/miles --body-file miles-torch-compile-pr-body-final.md
gh api repos/radixark/miles/pulls/3720 > artifacts/compile-pr3720-final-readback.json
.venv-delta-cpu/bin/python - <<'PY'
import json
from pathlib import Path
plan = json.loads(Path("COMPILE_VALIDATION_PLAN.json").read_bytes())
pr = json.loads(Path("artifacts/compile-pr3720-final-readback.json").read_bytes())
assert pr["draft"] is True
assert pr["head"]["sha"] == plan["standalone_source"]
assert pr["body"].encode() == Path("miles-torch-compile-pr-body-final.md").read_bytes()
print("Draft/head/body raw-byte readback verified")
PY
```

Use `read_bytes().decode()` or JSON decoding for body/log comparisons. Ordinary
text reading can normalize carriage-return progress records and invalidate a
byte-exact comparison. Do not post a follow-up PR comment; this workflow updates
the requested draft body.
