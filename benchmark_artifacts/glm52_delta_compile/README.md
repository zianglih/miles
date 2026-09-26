# GLM-5.2 packed CPU delta evidence

This directory belongs to the separate torch.compile Miles PR. Original delta-sync
evidence remains in `../glm52_delta_sync/` at its original published commit.

- [Matched GPU validation](COMPILED_VALIDATION_RESULTS.md) retains all 14 updates.
  Its JSON retains all 56 per-rank compiler records, source/helper identities,
  runtime settings, publication shard SHA256 and per-tensor checksum metadata.
- [Final CPU replay](artifacts/torch-compile-word-two-phase-32x1-replay.jsonl) retains all 28 raw warmup/measured samples.
- [Historical negative replay](TORCH_COMPILE_STAGE_REPLAY_V1.md) and its raw files
  remain separate from the final candidate. Fixtures/probes are diagnostic evidence,
  not substitutes for the full replay or combined GPU run.
- [Materialize-first experiment](TORCH_COMPILE_MATERIALIZE_FIRST_HISTORY.md) retains
  its separate 28-sample replay and explicitly uncommitted archived module identity.
- `source_snapshots/` contains exact committed production/test file bytes.
  `BUNDLE_METADATA.json` identifies explicit final codegen proof and test logs.
- [Hosted CI review](HOSTED_CI_REVIEW.md) retains the historical failure separately
  from the [compatibility investigation](COMPILE_CI_COMPATIBILITY.md), selected-head
  hosted snapshots and C2 validation. The compact subset includes the raw failed shard; complete
  workflow logs remain local and available through the linked GitHub jobs.
- `SHA256.json` covers every other file, including generated metadata and this page.
  Original raw evidence was neither normalized nor redacted.

The two GPU arms are `compile-v2-synthetic-numpy-01` and `compile-v2-synthetic-torch-01`. Sources are pinned by
`COMPILE_VALIDATION_PLAN.json`; selected campaign/environment files are
`CAMPAIGN_COMPILE_V2.json` and
`environment-image-compile-v2.json`. Replacement plans pin their
metadata hashes; the original `CAMPAIGN.json` / `environment-image.json` and
`COMPILE_VALIDATION_PLAN_V1.json` remain historical evidence for the original host.
`COMPILE_GPU_V1_INTERRUPTION.md` records the lost ephemeral devbox and v1's
summary-only boundary. Those summaries are never substituted into the matched pair.
CPU replay/preflight evidence uses the original image and is labeled separately.
External calibration scope is explicit in the GPU report; per-run observed costs
and inventories belong to the selected campaign.
Only u2-u6 enter steady GPU medians; initial/u1 observations are still retained.
Compiler subprocess CPU is excluded from trainer-process CPU. Graph/kernel counts
alone do not prove fusion, acceleration, or any multi-node benefit.

The local publisher reruns strict admission before copying. Weight payloads,
compressed delta shards, binary debug tensors, and compiled binaries are deliberately
excluded. Publication index bytes and the report's recorded shard hashes/checksums
are retained. Rerunning full strict admission requires the durable local payloads
or repeating the workload; this text bundle alone cannot reconstruct model state,
prove GPU equality, or independently recompute payload checksums.

No commit or push is performed by the publishing helper. Publication targets only
the `fork` remote (`zianglih/miles`); it is not an upstream source change.
