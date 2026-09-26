# Compiled GPU campaign v1: interrupted and not comparison-eligible

At approximately 2026-09-26 01:59 UTC, the C2 devbox `glm52-delta-0925` was deleted
externally. Its pod is gone and `/hai-workspace` used ephemeral `emptyDir` storage.
The compiled arm was interrupted during startup. The NumPy arm reported completion,
but only three summary files and its startup manifest had been copied locally.

The original frozen plan is preserved byte-for-byte as
`COMPILE_VALIDATION_PLAN_V1.json`, SHA256
`b2517283b974f46e43f88458b53d14a4a70d70fe9bdcb2c99f56f261340a559d`.
It selected standalone Miles `6fb0e2a9d81fc343863ba6015512fe93992892e4`, combined
Miles `283a1cb4e6ed4a3712a39bc5dbd26060465bdcca`, receiver
`2f5fb2a09f08eb9a44c0fed5a28be0bda2942b0f`, and the original runtime recorded in
`CAMPAIGN.json` / `environment-image.json`: image `radixark/miles:dev-202609251434`,
host `hu-pdx-90`. Those metadata files remain unchanged.

Retained new-campaign files are limited to:

- `artifacts/compile-v1-synthetic-numpy-01-summary.json`
- `artifacts/compile-v1-synthetic-numpy-01-cpu-summary.json`
- `artifacts/compile-v1-synthetic-numpy-01-training-summary.json`
- `artifacts/compile-v1-synthetic-numpy-01-manifest-startup.json`

The new campaign's full driver/trainer/CPU journals, complete final manifests,
binary training evidence and publication indexes/shards were not preserved locally
before deletion. The NumPy summaries cannot replace those raw inputs for strict
admission. No complete compiled arm or valid matched GPU comparison exists from
v1; no speedup, full runtime correctness, or GPU byte-equality result is claimed.
These summary-only records must not be mixed with a replacement compiled run.

The original completed delta-sync bring-up evidence and the selected complete CPU
replay/preflight evidence were already preserved locally and remain valid within
their recorded scope. They are not evidence for a different GPU runtime.

A replacement C2 devbox `glm52-delta-0926` was created on `hu-pdx-130` with the
freshly selected explicit image `radixark/miles:dev-202609252029`. Actual runtime
metadata is retained in `environment-image-compile-v2.json`, SHA256
`ee2e29f7ccdc3bbbb9a0f0c20e432ab25211a40b282551778b134d584ad8c96d`. The replacement
campaign record is `CAMPAIGN_COMPILE_V2.json`; its exact hash is pinned by the
active plan after final source selection and prepared-input sealing.
Both NumPy and compiled arms must run again as a new matched v2 pair on that
runtime. Its plan is kept unfrozen during preparation. Capturing the environment
does not establish a GPU validation result.

The external CPU observer calibration remains the original `hu-pdx-90` measurement;
it is not a new-host calibration. Per-run actual snapshot costs, CPU-window validity
and inventories remain required and authoritative. The replacement plan includes
an owned `/data/ziangli/glm52-delta-sync-c2/compile-v2-backups` destination: completed
arms are archived after summaries and before the next arm starts, with verified
file/archive SHA256. No backup copying occurs inside timed weight-update windows.
