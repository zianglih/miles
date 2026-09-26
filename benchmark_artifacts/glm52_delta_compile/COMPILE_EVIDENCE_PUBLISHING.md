# Separate compile evidence bundle

`publish_compile_evidence.py` is a local audit/staging helper. It never commits or
pushes and never writes the original `benchmark_artifacts/glm52_delta_sync` folder.
Its sole staging target is
`miles-evidence/benchmark_artifacts/glm52_delta_compile` on the existing
`glm52-delta-sync-evidence` branch. The project-local Git store and `fork` remote
identity (`zianglih/miles`) are verified before an explicitly requested stage.

The selected source and CPU replay are complete. The replacement GPU pair and
public evidence commit remain pending until strict admission succeeds. No
illustrative command below asserts that those pending inputs already exist.

## Campaign metadata and interruption boundary

The original `CAMPAIGN.json`, `environment-image.json` and archived frozen
`COMPILE_VALIDATION_PLAN_V1.json` retain their original bytes. V1 was interrupted
when the ephemeral devbox disappeared; only three NumPy summaries and its startup
manifest remain from that new GPU campaign. See `COMPILE_GPU_V1_INTERRUPTION.md`.
Those summaries cannot be used as an arm in the replacement comparison.

Plans can pin `campaign_metadata` and `environment_metadata` together, each as
`{"path": "task-relative-file.json", "sha256": "full 64-character digest"}`.
Optional `cpu_calibration_metadata` uses the same contract. The resolver rejects
path traversal, symlinks, hash drift and explicit CLI overrides of a pinned path.
Plans without pins keep the original defaults. Reporter, publisher and finalizer
use the selected metadata; the original files remain separate historical inputs.
The existing preflight helper has no image metadata input and remains byte-stable;
its completed original-image logs are labeled as such, not as replacement tests.

The runner resolves and verifies metadata before each arm and writes
`artifacts/<run>-campaign-metadata.json` with the exact plan SHA256 and relative
metadata/hash bindings before launch. Strict admission requires that sidecar for
plans with pinned campaign metadata. Measurement helpers are unchanged.

An optional `durable_backup_dir` must name an absolute task-owned
`/data/<owner>/<effort>` directory. The runner calls `preserve_arm` only after the
child exits and its summaries finish, before launching the next arm. Successful
arms require complete debug/shard coverage and zero backup errors; failed runs or
summarizers preserve a partial archive and stop. The preserver verifies copied
bytes and tar entries before atomically publishing the final tar and its SHA256
sidecar. This archive includes training debug data and all six delta shards, but
excludes the reconstructible baseline. It is a durable private backup, not the
public text bundle. Primary measurement/publication files stay on local disk.

V2 selects `CAMPAIGN_COMPILE_V2.json` and `environment-image-compile-v2.json` for
`glm52-delta-0926` on `hu-pdx-130`, image `radixark/miles:dev-202609252029`.
The external calibration still belongs to original `hu-pdx-90` and is explicitly
historical; per-run actual costs/inventories are retained without cost subtraction.
Completed CPU replay/preflight remains scoped to `dev-202609251434`.

## Test-only compatibility descendants

`evidence_source_commits` keeps the CPU replay at standalone `6fb0e2a9` and the
original combined preflight at `283a1cb4`. Selected standalone `d34a5511` and
combined `767439e9` add the same 12-line test fixture for the Torch 2.11 fake CUDA
guard lifetime issue. `compile_evidence_attribution.py` verifies Git ancestry,
rejects every changed path except `tests/fast/utils/test_delta_preparation.py`,
and records matching production SHA256 values. Replay admission still compares
its archived module to the measured commit. It never relabels old timing or
test logs as new-head execution.

The separate `preflight_evidence_directory=artifacts/compile-v2-preflight-01`
must contain completed tests and Gloo checks on the actual selected heads; helper
hashes, source pins and retained complete logs are checked. The final PR includes
these new logs inline and links all historical logs from the exact evidence commit.
The source snapshots include both measured and selected versions.

`COMPILE_CI_COMPATIBILITY.json`/`.md` retain the Linux reproduction, main-thread
fixture, exact tested file hash, upstream PyTorch citation and explicit pending or
completed outcomes. The finalizer matches committed test bytes before discussing
that fix. Automatic new-head CI is captured in a separate directory:

```bash
python3 capture_compile_pr_ci.py \
  --head d34a55113647315cc5b5eb5a3532dc563c72fa6c \
  --destination artifacts/compile-pr-ci-3720-d34a551
```

The publisher/finalizer read its compact `summary.json` and `latest.json`, verify
the full selected head and every captured workflow/job, and report failed, pending
and skipped outcomes explicitly. Earlier hosted failures remain preserved. No
new hosted pass is inferred from the Linux focused result or C2 GPU campaign.

The replacement input seal and compact recovery/header/backup manifests are also
published. New prepared inputs have full hashes; their original comparison is
limited to matching 14 shard sizes/headers and 6,226 canonical tensors, because
original full converted payload hashes were not retained. No cross-campaign
performance ratio uses the replacement runtime.

## Complete admission before publication

- Freeze `COMPILE_VALIDATION_PLAN.json` with exact standalone and combined source
  pins and actual per-run cache preconditions. Complete both named GPU arms, all
  seven updates per arm, and the original wall/training/CPU summaries.
- Generate `COMPILED_VALIDATION_RESULTS.json` and `.md` with the strict compiled
  report helper. The publisher reruns that helper in memory and requires both
  stored reports to match current evidence exactly. Retained local delta shards
  and binary training dumps remain necessary for this admission.
- Retain the final CPU replay JSONL, adjacent `.log` and `.exit`, and the adjacent
  `-sources` folder. Admission requires all 28 verified samples, two serial
  actual-layout warmups, zero sample compiler activity, complete raw medians,
  matching archived helper/layout hashes, and a module identical to the frozen
  standalone source. The exact f17 control uses 32 workers; the stage uses a total
  worker × kernel-thread budget of 32, preserving the actual configuration.
- Supply exact final codegen proof and final-source test/Gloo logs. Add additional
  diagnostic text files explicitly. Completed historical replays with their own
  committed modules may be added as `JSONL=FULL_COMMIT`. An uncommitted historical
  experiment uses `JSONL=sha256:MODULE_SHA256`, must match the exact archived module
  bytes, and is explicitly labeled uncommitted with no invented commit. All 28
  samples and the same baseline/helper/archive checks remain required. This option
  cannot relax final replay admission against its actual standalone Git commit.
  The completed 6eca and materialize-first 8×4 negative replays are already required
  by the publisher; their historical flags need not be repeated.

## Invocation

From the project root, substitute actual completed artifact paths:

```bash
.venv-delta-cpu/bin/python publish_compile_evidence.py \
  --final-replay artifacts/FINAL-REPLAY.jsonl \
  --codegen-proof artifacts/FINAL-CODEGEN-PROOF.cpp \
  --validation-log artifacts/FINAL-CPU-TESTS.log \
  --validation-log artifacts/FINAL-GLOO-TESTS.log \
  --include FINAL-CPU-REPORT.md
```

Audit is the default and creates no files. Add `--stage` only when ready to copy
the audited bundle. Repeating a stage is allowed only if all destination files
already match exactly; a changed or extra file requires explicit review and is
never silently overwritten or deleted. Use `--include` once per additional text
file; directories and globs are not accepted. Optional extra completed replays:

```text
--historical-replay artifacts/EARLIER-REPLAY.jsonl=FULL_40_CHARACTER_COMMIT
--historical-replay artifacts/UNCOMMITTED-EXPERIMENT.jsonl=sha256:MODULE_SHA256
```

The helper rejects missing files, invalid reports, source drift, symlinks,
unexpected binary content, unapproved credential-pattern matches, text files larger than
16 MiB, and bundles larger than 128 MiB. A secret rejection reports only the
path, line number, and pattern class, never the matched value. Raw files are not
redacted or normalized. Generated source snapshots use exact `git show` bytes.
Every staged file is read back; `SHA256.json` records the byte count and hash of
every other file, including bundle metadata and its README.

The sole scanner exception is the public malformed-URL parameter from
`tests/ci/test/test_ci_failure_analysis.py`, printed in three complete diagnostic
logs. `COMPILE_SECRET_SCAN_FIXTURE_ALLOWLIST.json` binds the exception to exact
whole-file SHA256 values, exact matched-value SHA256 and test-node context, and
records the committed fixture source/hash/line. Altering a raw log or moving it
to another path rejects the exception. All other matches still fail without
printing their values, and no raw log is redacted.

`artifacts/compile-campaign-frozen-inputs.json` seals the 27 runtime inputs,
including plan, metadata, preflight manifest, runner, preservation helper and all
measurement helpers. Publication checks every frozen hash before admission.
Evidence/reporting helper changes do not rewrite this runtime seal.

Run `python3 audit_compile_evidence_readiness.py` for a local prerequisite audit
while the GPU pair is running. It checks source attribution, complete CPU replay
history, selected preflight, compatibility/hosted records, frozen inputs and text
scanning, and writes `artifacts/compile-evidence-readiness.json` explicitly with
`gpu_admitted: false`. It cannot substitute for the strict final paired report or
exact public-commit link verification.

## Published scope

The bundle retains final helper sources, tests and logs, the initial negative
6eca replay, the final 28-sample replay, bounded probe evidence, explicit codegen
proof, all 14 GPU updates and 56 per-rank compiler records, raw journals, command
manifests, training evidence, publication indexes, and recorded payload hashes
and per-tensor checksum metadata. Exact production/test source snapshots are
separate for the standalone PR and combined validation commit.

Explicit codegen proof/test logs are preserved, scanned, and hashed. The publisher
does not infer their pass status or authenticate their final-source attribution;
the publishing reviewer must check that context before selecting those files.

No model or delta weight tensor, binary training dump, compiled library, object
file, binary compiler cache, or archive is published. This text bundle cannot by
itself rerun strict payload admission, reconstruct checkpoints, recompute payload
checksums, or establish GPU equality. Successful strict local admission and
receiver checksum validation have their stated scope in the GPU report.

Local helper verification:

```bash
.venv-delta-cpu/bin/python -m unittest -v test_publish_compile_evidence.py
```

Tests use temporary isolated staging fixtures and read the already retained
historical replay. They do not touch the real evidence checkout or C2.

## Final PR body after both GPU arms pass

The initial upstream draft is Miles #3720. Its running-state body is preserved
separately; `finalize_compile_pr_body.py` requires a complete strict GPU report and
an exact published fork commit before writing `miles-torch-compile-pr-body-final.md`.
It never edits GitHub. It rereads raw logs as bytes to preserve carriage-return
progress output, keeps all final CPU rows and complete final tests/Gloo inline,
adds all 14 GPU update rows, and links complete historical reports from the exact
evidence commit. Every linked file is checked against that commit's manifest and
actual Git blob, including a check that local bytes still match. The output is
limited to 65,000 UTF-8 bytes to remain below GitHub's body limit.

```bash
.venv-delta-cpu/bin/python finalize_compile_pr_body.py \
  --evidence-url https://github.com/zianglih/miles/tree/EXACT_40_HEX_COMMIT/benchmark_artifacts/glm52_delta_compile
```

The illustrative URL is not a published evidence link. Replace it with the
actual new evidence commit after root-owned commit/push and verification. A
moving branch, the original delta-sync folder, missing GPU rows or incomplete
reports are rejected. The assembler performs no C2 operations.

Hosted CI currently has a separate unresolved native termination at `6fb0e2a`.
The input list includes `HOSTED_CI_REVIEW.md` and the compact
`artifacts/compile-pr-ci-3720-6fb0e2a/` subset: terminal review/status JSON, README,
failed-job metadata, the full 809,714-byte failed shard, its pytest argv, and
targeted local Torch 2.11 diagnostic logs. These files passed text/secret scanning.
The 8,355,212-byte whole PR workflow log remains local and is not embedded in the
body or selected bundle. The finalizer adds a direct exact GitHub failed-job link,
explains skipped hosted GPU stages, and preserves the unresolved status. If a
later CI/source record changes, it stops for wording review instead of inferring
a pass from the successful C2 or macOS checks.
