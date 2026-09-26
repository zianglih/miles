## Summary

@humansand

<!-- Local publication draft. Replace every PENDING field from retained evidence before publishing. Final source pins are selected; complete replay evidence, C2 preflight, and combined GPU validation remain pending. -->

Add an opt-in PyTorch CPU preparation stage for disk-delta updates. A compiled count phase derives exact per-tensor byte changes from a stable packed layout and completed incoming CPU bytes. An unchanged group reuses its previous snapshot; a changed group runs a second compiled phase that returns independently owned XOR/snapshot storage for the existing compression and publication path.

- **Selection:** `--update-weight-delta-cpu-backend {numpy,torch-compile}` defaults to `numpy`. The compiled backend requires `--update-weight-delta-encoding xor`; overwrite remains on NumPy. The selector supports controlled performance studies. A default replacement requires representative whole-stage and end-to-end evidence; this draft claims no such result yet.
- **PyTorch implementation:** `torch.compile(backend="inductor", fullgraph=True, dynamic=True)` covers word-lane byte counting, row reduction, exact per-name aggregation, and conditional owned XOR plus snapshot materialization. The two phases preserve the unchanged-group fast path. Python retains layout/queue/error control; native zstd and checksum run after preparation. No handwritten C++ source, custom extension, forced ISA, or process-wide Torch thread/affinity policy is added. Inductor generates host code and requires a working host compiler.
- **Wire compatibility:** retain per-name XOR bytes, zstd payloads, selected checksums, exact changed-byte counters, and omission of unchanged tensors. Tiny FP32 scales retain these same rules; this does not introduce unconditional scale transmission or receiver-format changes.
- **Dependency boundary:** the standalone source change is based on upstream Miles `41c5e38b94ea23677de93b01a4a77d55677a8f09`. [Miles #3711](https://github.com/radixark/miles/pull/3711) is needed for the combined GLM-5.2 canonical NVFP4 layout/recipe validation, not for the CPU module itself. That separate integration also uses the receiver in [SGLang #41274](https://github.com/sgl-project/sglang/pull/41274); these dependencies are not folded into this PR's source diff.

## Design and ownership

- **Stable subdivisions:** subdivide each existing gathered bucket in its existing order into weight groups of at most 128 MiB padded storage, except a single larger tensor. Weight entries use 4,096-byte rows. Collect scalar FP32 `.weight_scale_2` values into a separate compact four-byte-row group per gathered bucket, so a scalar change cannot copy an otherwise unchanged large weight group. Capture subgroup names/order, dtype, and shape once; reject changed membership/order, duplicates, omissions, or dtype/shape changes rather than reinterpreting an old layout.
- **Bounded transfer staging:** copy weights directly into named slices of a shared CPU lease, without a recurring CPU concatenation. Compact scalars use one `torch.stack` and transfer. Synchronize the CUDA stream once after a group's copies. Per sending rank/encoder, pool capacity is at most 64 leases and 32 GiB total, with a bounded pageable fallback if pinned allocation fails. Pending work is limited to 64 groups and 8 GiB of packed input, with one oversized tensor allowed up to the 32 GiB staging cap. These are staging/in-flight-input bounds, not total RSS bounds: changed-group XOR/replacement slabs add up to twice the admitted input bytes (including the oversized exception); persistent snapshots, compression, compiler state, and setup scratch are additional storage. A tensor larger than the staging cap is unsupported; lowering the iterator bucket limit cannot split that tensor.
- **Padding invariant:** baseline storage starts with zero padding. On assignment of a shared lease to a different layout, reset only its padding gaps; payload copies write only valid named slices. Equal padding is required because the count stage scans complete padded rows. Padding is not transmitted.
- **Meaningful compiled count stage:** view aligned packed bytes as int32 words, XOR corresponding words, and test four byte-lane masks, including the sign byte. Sum the lane counts into int32 row totals, then convert explicitly to int64 for cumulative reduction and boundary gathers that yield exact per-name byte counts. Layout rows must be positive multiples of four bounded by int32; buffer storage offsets are checked for word alignment, and zero padding covers odd tensor tails. Total padded length is guarded by int64. Native ATen `cumsum` remains part of the workload; this is not claimed to be one fused machine kernel. Counting is dtype-blind and uses no floating-point accumulation.
- **No-change/changed ownership:** an all-zero group preserves the old snapshot object and avoids owned XOR/replacement allocation. A changed group performs word-view XOR and an owned clone together, returns uint8 views, then releases its staging lease exactly once before native per-name compression/checksum. A changed group also copies valid unchanged tensors and padding within that group; the benchmark reports those bytes explicitly. The count phase and conditional materialization phase are separate, so changed groups perform XOR work in both phases; this is the cost of retaining the unchanged-group fast path.
- **Snapshot commit:** promote a changed group's snapshot only after all of its native handoff succeeds, replacing every name's view so unchanged names in that changed group cannot retain obsolete packed slabs. An entirely unchanged group keeps its snapshot identity and skips per-name view reinstallation. Inputs remain immutable. Earlier successful groups may already have committed if a later group fails; this retains fail-stop behavior rather than adding whole-update rollback or automatic retry.
- **Host bookkeeping:** cache each layout's valid payload-byte total during capture instead of summing entries on every update. Remove unused encoder copy/padding diagnostic accumulation; the wire changed-byte and total-byte counters remain. Replay-only copy and materialization accounting is retained and runs inside its timed work, so replay timings still include diagnostic overhead that production no longer pays.
- **Errors and initialization:** initialize after draining baseline iterator collectives; compile both graphs serially in an ordinary worker using every actual layout, owned snapshot, and shared scratch before concurrent submissions. Ordinary allocations also remain ordinary under an enclosing `inference_mode`. Updates drain work, return leases, and coordinate detected packed-preparation errors before publication. This is not a general recovery protocol for process death, filesystem failure, or receiver failure.
- **CPU concurrency:** on the C2 host, the existing worker limit is 32. Generated kernels use `cpp_wrapper=True`, `cpp.threads=1`, and `cpp.dynamic_threads=False`, with a GIL-releasing generated wrapper. Native external operations retain the runtime's settings; the generated-kernel budget is not a claim that every operation uses one thread. Runtime thread settings, affinity, and per-sample compiler counters are retained in the evidence.

## Validation

### Source and environment

| Item | Evidence |
| --- | --- |
| Selected standalone PR commit | `6fb0e2a9d81fc343863ba6015512fe93992892e4`; parent `6eca79a001e2e2c9b62385b1397f1ad0fc800e81`, based on upstream `41c5e38b94ea23677de93b01a4a77d55677a8f09` |
| Selected preparation module SHA256 | `281534dfa6b8fef9840f17af2522c3d8100097a68baf0b30bf0c2e1af69a7788` |
| Selected combined GLM validation commit | `283a1cb4e6ed4a3712a39bc5dbd26060465bdcca`; parent `c5ea5a83e31cc3a16aa1b185ec3f8fb0c3a2f59e`, which contains the f17 GLM integration and canonical-storage matching |
| Earlier candidate with completed C2 CPU test/replay evidence | `6eca79a001e2e2c9b62385b1397f1ad0fc800e81`; historical evidence only |
| GLM compatibility dependency | Miles `f17ba4bce13bf7d357e7560182dc859c41a7cb37` |
| Receiver dependency | SGLang `2f5fb2a09f08eb9a44c0fed5a28be0bda2942b0f`, based on `sglang-miles` |
| Public evidence commit and files | **PENDING_PUBLIC_EVIDENCE_URL** |
| C2 CPU | `hu-pdx-90`; 2 × Intel Xeon 6776P, 64 cores/socket, SMT2, 256 logical CPUs, four NUMA nodes; AVX-512 available |
| Explicit image | `radixark/miles:dev-202609251434`; amd64 digest `sha256:7c4c6c8cc9e76be893941e064f0e43924ef9258b04da365c8009709592ff0984` |
| CPU software | Python 3.12.3; PyTorch `2.13.0+cu130`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`; NumPy 2.3.5; zstandard 0.25.0; xxhash 3.7.1 |
| GPU integration environment | 8 × B300 SXM6 AC, driver 590.48.01, CUDA toolkit 13.0.88; 4 trainer + 4 rollout GPUs; exact final run manifests **PENDING** |
| Integration stack | Megatron `f148a32b4385b758b66a77c9c3ad1641f1295d4b`; TE 2.17.0; FlashInfer 0.6.18; CuTe DSL 4.6.2; Triton 3.7.1; Ray 2.58.0 |

### Historical exact-image CPU checks on 6eca

The remote checkout was upstream `41c5e38` plus the exact v3 source archive bytes subsequently committed as `6eca79a0`. The compiler cache was fresh. From `/hai-workspace/glm52-delta/miles-torch-compile`:

```bash
CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 \
  TORCHINDUCTOR_CACHE_DIR=/hai-workspace/glm52-delta/torch-compile-stage-c2-v3-cache \
  PYTHONPATH=. timeout --kill-after=5 240 /opt/sglang/bin/python -m pytest \
  tests/fast/utils/test_delta_preparation.py \
  tests/fast/backends/training_utils/weight_update/test_delta_compiled_cpu.py \
  tests/fast/backends/training_utils/weight_update/test_delta.py \
  tests/fast/backends/training_utils/weight_update/test_disk_delta_weight_version.py \
  tests/fast/backends/training_utils/weight_update/test_disk_delta_engine_calls.py -q
```

```text
42 passed, 24 warnings in 23.20s
```

Raw log: `artifacts/torch-compile-stage-c2-v3-tests.log` (**public link pending**). Coverage includes actual compiled counts/ownership, counts above 2^24, odd/empty entries, compact scalar layout, mixed changed/unchanged groups, padding reset after lease reassignment, actual-layout warmup, concurrent warmed dispatch, baseline compile failure coordination, bounded in-flight work, failed handoff without premature snapshot promotion, all-view replacement, and existing delta version/engine behavior. Encoder lifecycle tests use eager preparation and mocked GPU/distributed boundaries; they do not establish live CUDA transfer or distributed runtime behavior.

The corrected benchmark helper also passed local macOS ARM/PyTorch 2.14 eager and actual-compile fixtures with four distinct layouts, empty/oversized entries, and one compact three-scalar group. Its compiled fixture used **zero warmup iterations after explicit actual-layout initialization**; all eight timed samples verified bytes/counts/checksums with no new compiler activity. An injected compiler counter proved measured compilation is recorded and then rejected before a successful summary. This validates the harness, not C2 performance.

### Selected-source preflight — PENDING

`run_compiled_preflight.py` verifies the selected source pins and clean combined checkout, then runs the five focused CPU test files and the distributed Gloo failure harness with GPUs hidden and a fresh task-specific compiler cache. Its manifest records exact commands, source/helper hashes, environment, exit codes and elapsed times. The selected tests include exhaustive byte-pair coverage in all four word lanes, odd tails, large exact counts, storage alignment, unchanged identity, changed ownership, concurrent dispatch, and encoder lifecycle/error cases.

Selected-source C2 preflight output, hosted CI, and combined GPU checks: **PENDING**. The earlier 42-test result above does not stand in for final-source validation. Retain warnings and failed/invalid runs; do not describe this as an error-free campaign.

## Whole-stage CPU replay — PENDING

The comparison uses the complete CPU stage, including task submission/drain, exact counts, owned snapshot/XOR allocations, lease return, zstd, and checksums. The primary control extracts the exact `f17ba4b` `_diff_and_compress` **and** `_diff_and_compress_batch` ASTs: ordinary weights use single-name work, and the same compact FP32 scalar groups use the original batch method. This avoids charging the control one Python task per scalar while batching the candidate. Both sides use 32 workers; generated candidate kernels use one thread. Timed candidate replay work also constructs diagnostic per-name views and copy/materialized-bucket accounting; production omits unused counters and skips view rebuilding for unchanged snapshots. These replay costs are retained rather than subtracted.

- **Identical inputs:** all 4,690 emitted canonical tensors, totaling 17,900,804,608 bytes; omit conversion-only `.input_scale`. Separate unchanged canonical bytes from reconstructed published synthetic v1→v2 bytes. The retained changed input has 79,500,803 differing byte positions, but only 465,303,040 bytes (2.5993%) reside in wholly unchanged tensors. Changed-byte density and unchanged-tensor-byte fraction are different metrics.
- **Layouts:** sorted-name representative groups with 128 MiB maximum padded storage except an oversized tensor, 4,096-byte weight rows, and compact four-byte scalar rows. These are reproducible CPU replay layouts, **not a capture of live iterator subgroup boundaries**. Retain the complete layout manifest, padding, copied unchanged bytes, copied snapshot bytes, and bucket-size distribution.
- **Warm/cold boundary:** serially invoke both graphs with each actual bucket's old/incoming storage before ending initialization timing or starting fanout. Record cold/cache state and compiler counters separately. Two alternating-order warmup pairs precede five measured pairs. Keep every raw sample and reject any compilation in a measured sample. Calling-process CPU excludes compiler subprocess CPU; cold initialization is not complete host CPU accounting.
- **Excluded work:** checkpoint reads/reconstruction, initial packing, pool startup, and verification are outside timing. Inputs are ordinary completed-D2H CPU views: no GPU copies, actual pinned pool/backpressure, live gather, baseline repacking, snapshot installation, filesystem publication, or receiver reload is measured. No end-to-end speedup follows from this replay alone.

Reproduction inside the pinned image, using the frozen project helper/evidence files (**exact public evidence URL and final run paths pending**); module SHA256 `281534dfa6b8fef9840f17af2522c3d8100097a68baf0b30bf0c2e1af69a7788`:

```bash
cd /hai-workspace/glm52-delta
CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 /opt/sglang/bin/python \
  benchmark_delta_preparation.py \
  --miles-path /hai-workspace/glm52-delta/miles \
  --baseline-ref f17ba4bce13bf7d357e7560182dc859c41a7cb37 \
  --preparation-module /hai-workspace/glm52-delta/miles-torch-compile/miles/utils/delta_preparation.py \
  --checkpoint /hai-workspace/glm52-delta/models/GLM-5.2_5layer-NVFP4 \
  --synthetic-source /hai-workspace/glm52-delta/artifacts/synthetic-balanced-disk-delta-01/delta-publication \
  --expected-tensors 4690 --baseline-workers 32 --stage-workers 32 \
  --kernel-threads 1 --block-bytes 4096 --bucket-mib 128 \
  --warmups 2 --repeats 5 \
  --cache-dir /hai-workspace/glm52-delta/PENDING_FRESH_REPLAY_CACHE \
  --output /hai-workspace/glm52-delta/artifacts/PENDING_FINAL_REPLAY.jsonl
```

The recorded helper source SHA256 before the final run is `90fe76dccd02b4f9881177d6ace7b49912b54f7002c8622d63a896f3c73e5ec9`; replace it only with the exact validated final helper. Model revision: `Pinaster/GLM-5.2_5layer` at `1c749139f70e158e4420ba67f342bef1de2e650d`, using the same prepared NVFP4 checkpoint as #3711. Reconstruct the synthetic input from the retained published v1/v2 artifacts; newly generated training does not guarantee the identical byte workload.

| Workload | Exact f17 wall median | Packed stage wall median | Exact f17 CPU median | Packed stage CPU median | All bytes/counts/checksums verified | Measured compiler activity |
| --- | --- | --- | --- | --- | --- | --- |
| Unchanged canonical; completed phase, full replay still pending | 1.119358066 s | 0.555282267 s | 15.653191363 s | 8.054724406 s | Final raw transfer pending | Final raw transfer pending |
| Published synthetic v1→v2 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

**PENDING_RAW_REPLAY_OUTPUT:** insert every warmup/measured pair, separate initialization wall/process CPU and cache state, complete byte/copy/layout statistics, and the verified final completion record. Do not substitute the earlier tiny-kernel probe or f17 before/after worker replay for this whole-stage result.

The completed unchanged phase reports 50.39% less wall time and 48.54% less process CPU for the selected module. This is a partial campaign observation until the complete raw replay is retained and admitted. Changed-weight results remain pending; no changed-weight speedup or default switch is inferred from the unchanged phase.

## Completed 6eca CPU replay — negative result, separate from final candidate

The complete replay at `6eca79a001e2e2c9b62385b1397f1ad0fc800e81` exited with code 0 and verified all 28 warmup/measured samples. It did **not** justify replacing NumPy: unchanged wall time increased 58.10% despite 1.37% less process CPU; changed-workload wall increased 70.96% and process CPU increased 53.02%. This is retained negative evidence for that exact int32-row-count stage. Final-candidate replay and GPU validation remain separate pending results.

Source module SHA256 `fd87a4a382909add1d9053233d92ed2cd4ab66a15332851ed84c7bd96158be8a`; helper SHA256 `90fe76dccd02b4f9881177d6ace7b49912b54f7002c8622d63a896f3c73e5ec9`; raw JSONL SHA256 `bc44ce8b9a878e914086a2caabab9279c889da9db19e1ac7ae820fea2c00479d`. Local raw files: `artifacts/torch-compile-stage-replay-v1.jsonl`, `.log`, `.exit`, and `artifacts/torch-compile-stage-replay-v1-sources/` (**public evidence links pending**).

The replay ran on C2 host `hu-pdx-90`: 2 × Intel Xeon 6776P, 64 cores per socket, SMT2, 256 logical CPUs, and four NUMA nodes, with AVX-512 available. The image was `radixark/miles:dev-202609251434`, amd64 digest `sha256:7c4c6c8cc9e76be893941e064f0e43924ef9258b04da365c8009709592ff0984`. Software: Python 3.12.3; PyTorch `2.13.0+cu130`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`; NumPy 2.3.5; zstandard 0.25.0; xxhash 3.7.1.

Runtime used 32 workers for both arms, one generated-kernel/intra-op thread, 128 inter-op threads, and recorded affinity `0–1,11–129,138–255`. The standalone module matches the committed source; the exact f17 per-tensor and scalar-batch ASTs are preserved.

The 4,690 named tensors contain 17,900,804,608 valid bytes. The 112 representative sorted-name groups add 44,544 padding bytes; the two preallocated input states own 35,801,698,304 bytes. The exact control submits 3,155 tasks, including one compact 1,536-scalar group. Production subdivisions/staging bounds are not reproduced by this CPU-only replay.

| Workload | f17 wall median s | 6eca wall median s | Wall change | f17 CPU median s | 6eca CPU median s | CPU change |
| --- | --- | --- | --- | --- | --- | --- |
| native-unchanged | 1.125101647 | 1.778736112 | +58.10% | 16.075276682 | 15.855253043 | -1.37% |
| synthetic-v1-v2 | 3.529938037 | 6.034834150 | +70.96% | 36.298767904 | 55.543541675 | +53.02% |

**Initialization is outside the replay samples:** both graphs ran serially against every actual layout before fanout. The first workload started with an empty task-specific disk cache; the second reused the initialized compiler. Calling-process CPU excludes compiler subprocess CPU. Warmup includes full-layout execution/materialization and is not pure compiler latency.

| Workload | All-layout warmup wall s | Process CPU s | New graphs | New generated kernels |
| --- | --- | --- | --- | --- |
| native-unchanged | 58.232279793 | 43.329225336 | 3 | 17 |
| synthetic-v1-v2 | 38.065546597 | 38.068060900 | 0 | 0 |

All 28 timed records have empty compiler counter deltas. Graph/kernel counts are observability data and do not establish fusion or performance.

<details>
<summary>All 28 raw CPU replay samples, including two warmup pairs per workload</summary>

B = exact f17; C = packed stage 6eca. Order alternates by pair; only pairs 2–6 enter medians, with no samples removed. Wall and process CPU are seconds.

**native-unchanged**

| Pair | Phase | Order | B wall s | C wall s | B CPU s | C CPU s |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | warmup | B→C | 1.130001772 | 1.760311283 | 15.957901287 | 15.500207543 |
| 1 | warmup | C→B | 1.139313117 | 1.759618964 | 16.141250380 | 15.305593954 |
| 2 | measured | B→C | 1.125101647 | 1.866264602 | 16.314899115 | 16.107030080 |
| 3 | measured | C→B | 1.066849843 | 1.770045894 | 16.214130711 | 15.709281407 |
| 4 | measured | B→C | 1.606760446 | 1.778736112 | 16.075276682 | 15.267684626 |
| 5 | measured | C→B | 1.622636632 | 1.772907419 | 15.898084304 | 15.855253043 |
| 6 | measured | B→C | 1.120254939 | 1.780867644 | 16.011328341 | 16.003224925 |

**synthetic-v1-v2**

| Pair | Phase | Order | B wall s | C wall s | B CPU s | C CPU s |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | warmup | B→C | 3.794184700 | 6.090968547 | 40.053432131 | 60.122372519 |
| 1 | warmup | C→B | 3.750945677 | 5.926031381 | 38.689728287 | 60.020253004 |
| 2 | measured | B→C | 3.794779886 | 6.794410720 | 39.062372466 | 61.434463655 |
| 3 | measured | C→B | 2.996733481 | 6.034834150 | 37.727198153 | 60.461662628 |
| 4 | measured | B→C | 3.569558824 | 4.605608953 | 36.125778705 | 54.512039273 |
| 5 | measured | C→B | 3.529938037 | 6.064806106 | 34.611736899 | 55.543541675 |
| 6 | measured | B→C | 2.998813627 | 5.938902053 | 36.298767904 | 54.718316252 |

</details>

Copy and wire-work input counters are constant across samples of each case:

| Case | Variant | Copied unchanged bytes | Copied padding bytes | Snapshot copy bytes | Compressed worker payload bytes |
| --- | --- | --- | --- | --- | --- |
| native-unchanged | exact_f17 | 6144 | 0 | 6144 | 0 |
| native-unchanged | packed_stage | 0 | 0 | 0 | 0 |
| synthetic-v1-v2 | exact_f17 | 6144 | 0 | 17435507712 | 182769068 |
| synthetic-v1-v2 | packed_stage | 465284608 | 44544 | 17900830720 | 182769068 |

Synthetic v1→v2 changes 79,500,803 byte positions across 2,646 tensors; 465,303,040 valid bytes reside in entirely unchanged tensors. The stage materializes 110 of 112 groups and copies 465,284,608 unchanged valid bytes within changed groups. Both arms generate 182,769,068 compressed payload bytes, excluding safetensors headers/index. f17 still copies the 6,144-byte compact scale group even when every scalar is unchanged.

Checks verify every snapshot, decoded XOR, per-name count/checksum, unchanged input, and lease return. Timed work includes submission/drain, preparation allocations/counts/materialization, native zstd/checksums and lease return. Reads/reconstruction/packing, initial allocations, worker startup and verification are outside timing. GPU gather/D2H, actual pinned-pool backpressure, live layout selection, snapshot installation, publication and receiver reload remain unmeasured.

## Materialize-first 8×4 replay — separate rejected experiment

An uncommitted variant materialized owned XOR/snapshot buffers before counting every group, including unchanged groups. Its complete 28-sample replay verified exact bytes/counts/checksums without timed compilation, but changed weights took 11.63% more wall time and 30.70% more CPU than the matched f17 control; unchanged weights took 265.04% more wall time and 184.46% more CPU. It used eight preparation/compression workers with four generated-kernel threads each, so compression concurrency also differed from the 32-worker control.

The module SHA256 is `09fc7f41579c5e517a63263bb6a5763c08ad769a16e34dd658f3255ec900dc5d`; it has no source commit. `TORCH_COMPILE_MATERIALIZE_FIRST_HISTORY.md` retains all 28 raw rows, exact launch/environment, initialization counters, copy costs and limits, alongside `artifacts/torch-compile-materialize-first-8x4-replay.{jsonl,log,exit}` and its archived sources (**public evidence links pending**). It is not the selected two-phase implementation.

## Combined GLM-5.2 GPU validation — PENDING

Validate a separately identified integration with #3711 and receiver #41274 using the existing five-layer NVFP4 W4A16 recipe: one 8-B300 C2 node, trainer TP4/EP4 with PP1/CP1/ETP1, two rollout engines each TP2/DP2/EP2, four training and four rollout GPUs. Keep native precision/shared-expert exclusions and the existing disabled logprob/KL/weight-equality CI checks unchanged.

Seven rollouts provide startup `u0`, first post-training update `u1`, and all five steady `u2..u6` samples. Native deepscaler is separate zero-change validation. Synthetic balanced `sample.index % 2` rewards use ordinary GRPO and optimizer updates; require nonzero gradients/advantages and changed-byte counts. This transport workload does not establish task quality or equal training trajectories.

Use frozen `launch_weight_sync_compile.py` / `weight_sync_probe_compile.py` with the existing v2 CPU/wall observers. Counter observations belong to each trainer rank and cover the **entire updater**, not exclusively the CPU preparation graph. They lie outside updater CPU/wall windows and inside outer driver wall. Retain all four ranks' successful update/counter records; validate source/configuration matching and steady compilation behavior before comparing backends.

```bash
cd /hai-workspace/glm52-delta
python launch_weight_sync_compile.py \
  --mode disk-delta --reward-mode synthetic-balanced \
  --implementation-label candidate --num-rollout 7 \
  --run-name PENDING_COMPILED_SYNTHETIC_RUN \
  --delta-cpu-backend torch-compile \
  --compile-cache-dir /hai-workspace/glm52-delta/PENDING_FRESH_INTEGRATION_CACHE \
  --repo /hai-workspace/glm52-delta/miles-torch-compile-validation \
  --sglang-repo /hai-workspace/glm52-delta/sglang \
  --model-dir /hai-workspace/glm52-delta/models \
  --data-dir /hai-workspace/glm52-delta/datasets \
  --output-dir /hai-workspace/glm52-delta/artifacts
```

For a paired backend comparison, run the same combined source and helpers sequentially with `--delta-cpu-backend numpy`, its own fresh output/cache paths, and otherwise matching configuration. Record actual commands, launcher/helper hashes, clean source commits, cache state before launch, CPU affinity/thread settings, model/data pins, training completion, and raw counters. Dataset: `zhuzilin/dapo-math-17k` revision `2e65612930298bde4c5d58fd97b3f23a483aaff9`, 10,490,834-byte JSONL with SHA256 `cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6`; 8 prompts × 8 samples, response limit 100, temperature 1, LR 1e-6, trainer/rollout seeds 1234/42.

**PENDING_FINAL_GPU_RAW_OUTPUT:** insert all seven wall/CPU/changed-byte/wire-byte rows per completed arm, training/routing-replay evidence, warning inventory, exact final source/runtime manifests, compiler counters, and cold versus steady classification. Original #3711 NumPy/broadcast results remain a separate completed campaign; do not silently treat them as a matching compiled-backend run.

CPU scopes retain the v2 observer limits: trainer all-thread process CPU across updater windows; inventoried receiver schedulers/auxiliaries over rank-0's update; driver CPU separately. Membership refresh/journaling and compiler observation add driver overhead; no guessed cost is subtracted. Process membership is non-atomic, transient children can escape, and receiver module loading is not independently attested. These components are not simultaneous whole-host CPU. Checksums and successful reload/training do not prove active GPU byte equality. Multi-node bandwidth/scaling and amortization are unmeasured.

## Earlier kernel probes and prior art

The initial XOR plus int64-count prototype was slower despite generated AVX-512 and a GIL-releasing C++ wrapper. On this exact C2 image, each thread made 16 calls on the same 4 MiB + 3 byte inputs:

| Workers | NumPy wall s | Compiled wall s | NumPy process CPU s | Compiled process CPU s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.015673769 | 0.121165356 | 0.015697823 | 0.121190109 |
| 2 | 0.015600622 | 0.121077570 | 0.030319941 | 0.241121471 |
| 32 | 0.035654672 | 0.140478621 | 0.907117173 | 3.565812516 |

Its cold compile wall was 20.843314670 s; calling-process CPU was 2.117614195 s, excluding compiler children. This short probe ran during engine startup and is diagnostic evidence about that superseded primitive, not controlled final-stage performance. Raw JSONL and generated source/disassembly are retained under `artifacts/torch-compile-c2-smoke/` (**public link pending**). A bounded int16-count probe was rejected. Later probes motivated the selected word-lane count plus conditional word-view materialization design, which retains exact int32 row/int64 per-name counts and the unchanged-group fast path. Those bounded probes do not substitute for the final complete replay and GPU comparison.

[Megatron-LM #103](https://github.com/radixark/Megatron-LM/pull/103) supplied historical CPU implementation context: earlier commit `203df6f92236eaa54170c9c9b9ca6f5f1c1a736a` explored CPU Inductor, while the inspected final PR head `483a7ae7cce92c262b1ecce4df6cbdbfe82a4fd9` uses stock fused AdamW and gradient aliasing. Its final AdamW performance numbers are not evidence for compiled delta performance.
