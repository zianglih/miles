# Hosted CI: unresolved failure at 6fb0e2a

Miles #3720 hosted CI did **not** pass at source
`6fb0e2a9d81fc343863ba6015512fe93992892e4`.
The [failed stage-a CPU shard 1](https://github.com/radixark/miles/actions/runs/36209271034/job/108312293487)
terminated with status 250 while running
`test_actor_inference_allocations_match_warmed_worker_dispatch[entries1-4]`.
The preceding `[entries0-64]` case passed. No pytest failure traceback, raw
signal/core artifact, or workflow-uploaded artifact establishes the cause.
The wrapper's exit-code conversion could map a child SIGABRT to 250, but that
remains an inference, not a confirmed diagnosis.

The failed environment was Ubuntu 24.04 image `20260920.314.1`, Python 3.11.16,
Torch 2.11.0, NumPy 2.4.6 and uv 0.12.19, without an explicit workflow OMP/MKL
limit. The selected C2 CPU/preflight results use a different Torch build and
do not resolve this hosted failure. A targeted macOS ARM Torch 2.11 run passed
18 tests with 15 warnings in 7.68 seconds after correcting a relative-cache
setup error; it does not reproduce the hosted Linux runtime or full shard order.

Pre-commit, stage-a CPU shards 0/2/3, and stage-b CPU passed. All seven GPU stages
were explicitly skipped by resolved regular policy. ROCm workflow success only
covers setup/policy jobs; it is not hosted GPU validation. Exact statuses, passing
test summaries and skips are retained in `final-review.json`, `summary.json`, and
`latest.json` under `artifacts/compile-pr-ci-3720-6fb0e2a/`.

The publishable subset contains the compact review and README, job/check metadata,
annotations, the complete 809,714-byte `failed-shard-1.log`, the recorded shard
pytest argv, and targeted local diagnostic logs. It excludes the 8,355,212-byte
complete PR workflow log and other whole-workflow logs. Those raw logs remain in
the durable local project and are available through the linked GitHub job/run.
The original local CI `SHA256.json` catalogs the full retained collection; the
outer compile bundle's `SHA256.json` catalogs the actual published subset.

Hosted workflow logs include the PR body as an environment value, with earlier
C2 test output embedded without hosted timestamps. Those embedded results are
not counted as hosted passes. Raw bytes are preserved and scanned before
publication. Investigation is ongoing; source pins and C2 benchmark evidence
remain unchanged, and no hosted pass or compatibility fix is claimed.
