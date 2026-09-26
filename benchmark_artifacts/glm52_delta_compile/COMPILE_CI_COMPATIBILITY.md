# Torch 2.11 compatibility investigation

The original hosted job at standalone `6fb0e2a9d81fc343863ba6015512fe93992892e4`
failed with wrapper status 250 during the compact-scalar allocation test. Its
retained log did not identify the raw signal; that historical limitation remains.
See `HOSTED_CI_REVIEW.md` and the original job records.

A separate C2 Linux diagnostic reproduced a SIGSEGV (`returncode=-11`) at the
same scalar-layout test with a native stack containing
`c10::impl::ensureCUDADeviceGuardSet`. It used Python 3.11.16, Torch 2.11.0+cu130
git `70d99e998b4955e0049d13a98d77ae1b14db1f45`, NumPy 2.4.6, no visible CUDA
device, and no explicit OMP/MKL thread limit. This reproduces the relevant
software path, not the hosted hardware, memory limit, or original raw signal.

[PyTorch PR #178950](https://github.com/pytorch/pytorch/pull/178950), landed as
[d7b75b8250f43949bb2609f406ac942f935cab46](https://github.com/pytorch/pytorch/commit/d7b75b8250f43949bb2609f406ac942f935cab46),
fixes the fake guard lifetime: a global registry retained a pointer owned by the
initializing thread, which became invalid when that worker exited. The upstream
fix gives the guard process lifetime. This supports the reproduced failure's
diagnosis; it does not retroactively recover the original hosted job's signal.

The Miles compatibility change adds a 12-line module-scoped autouse test fixture
in `tests/fast/utils/test_delta_preparation.py`. On a CUDA-enabled Torch build
without an available GPU, it initializes `FakeTensorMode` on the main pytest
thread before short-lived compilation workers. Production code and benchmark
module bytes remain unchanged. The diagnostic tested file SHA256 is
`62622e15c9d61a7aa319eff2a8c7502a695bd36b1241bcb686aae5625933b631`.

The fixed Linux focused run passed **18 tests, 18 warnings in 45.59 s**. Its raw
log is `artifacts/compile-pr-ci-3720-6fb0e2a/linux-compatibility-focused.log`.
The final Linux full-shard command passed **3,172 tests, 27 skipped, 44 warnings in
288.97 s**; the process exited 0 after 304.060041052 s including startup. It used
the hosted shard's exact file list on diagnostic merge `5093563d01e81e6a71a1425470f32e359338f37f`
plus the byte-identical compatibility test fixture. This is separate from the
new selected-head C2 preflight and automatic hosted CI.

Earlier full-shard attempts hit environment-only logical-GPU assumptions: empty
`CUDA_VISIBLE_DEVICES` failed SGLang parsing; one invalid ID did not supply eight
logical Ray slots; eight invalid IDs failed hardcoded logical indices 10–13 after
3,079 tests. The final diagnostic used 16 invalid physical IDs (99 through 84),
preserving those logical indices while Torch reported no available GPU and zero
real devices. No test was removed or changed for those retries. Every failed
attempt, command, phase journal and complete final log is retained under
`artifacts/compile-pr-ci-3720-6fb0e2a/ci-compatibility-complete/`. Cleanup records
confirm the diagnostic processes stopped before the GPU comparison launched.

The source fix is committed as standalone
`d34a55113647315cc5b5eb5a3532dc563c72fa6c` and combined
`767439e95bca5203211ba374b4e04b702c2b7ed2`. Automatic hosted CI belongs to its own
per-head capture; no hosted pass is inferred from the Linux diagnostic pass.

The automatic `d34a5511` capture at `20260926T024506Z` completed successfully for
pre-commit, all four stage-a CPU shards and stage-b CPU
([PR Test workflow](https://github.com/radixark/miles/actions/runs/36212183697)).
Every captured workflow's `headSha` matches the full selected commit. GPU stages
were skipped by policy; ROCm setup/policy success is not GPU validation. Compact
records live under `artifacts/compile-pr-ci-3720-d34a551/`; the 10.4 MB whole
workflow log remains local. Embedded older PR-body logs in that workflow are not
counted as newly executed tests.

Three retained diagnostic full-shard logs contain one public malformed-URL test
parameter, repeated as a test ID and timing rows. The credential-URL scanner's
exception is bound to each complete raw file hash, the exact matched-value hash,
and the exact test-node context. `COMPILE_SECRET_SCAN_FIXTURE_ALLOWLIST.json`
records committed public source provenance. No real credential value was exposed
or removed; all raw logs remain unchanged and all other matches are rejected.

`COMPILE_CI_COMPATIBILITY.json` records exact raw file hashes and explicit outcome
states. The publisher and PR finalizer verify that selected committed test bytes
match these diagnostic bytes. Earlier CPU replay stays attributed to `6fb0e2a`;
old combined preflight stays attributed to `283a1cb4`. New source ancestry and
production-byte identity are checked separately. A new-image/new-head preflight
and matched GPU pair are separate validation records, not retroactive reruns of
the earlier timings.
