# Selected word-two-phase 32×1 CPU replay

All 28 warmup/measured samples completed with exact byte/count/checksum verification and no timed compiler activity. The unchanged case used 50.39% less wall time and 48.54% less process CPU than the exact f17 control. The changed case used 46.44% more wall time and 27.70% more process CPU. NumPy remains the default; these results support an opt-in experiment, not replacement of the existing backend or a claim of end-to-end speedup.

## Exact source, environment and input

Selected standalone commit `6fb0e2a9d81fc343863ba6015512fe93992892e4`; module SHA256 `281534dfa6b8fef9840f17af2522c3d8100097a68baf0b30bf0c2e1af69a7788`. The replay loaded the frozen `delta_preparation_word_two_phase_variant.py`; its archived module is byte-identical to that committed production module. The exact control is `f17ba4bce13bf7d357e7560182dc859c41a7cb37`, including both `_diff_and_compress` and `_diff_and_compress_batch` ASTs. Raw JSONL SHA256 `61c6dd06df9be03609b64c2c1bab3c3937e48a4e7f8e4f1041b64e516bb5feec`; full source/layout/control hashes remain in its provenance.

C2 host `hu-pdx-90`: 2 × Intel Xeon 6776P, 64 cores/socket, SMT2, 256 logical CPUs, four NUMA nodes, AVX-512 available. Image `radixark/miles:dev-202609251434`, amd64 digest `sha256:7c4c6c8cc9e76be893941e064f0e43924ef9258b04da365c8009709592ff0984`. Python 3.12.3; PyTorch `2.13.0+cu130`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`; NumPy 2.3.5, zstandard 0.25.0, xxhash 3.7.1. Both arms use 32 workers; generated candidate kernels use one thread, `cpp_wrapper=True`, and `cpp.dynamic_threads=False`. Runtime reports Torch intra/inter-op threads 1/128 and CPU affinity `0–1,11–129,138–255`.

The input has 4,690 emitted canonical tensors and 17,900,804,608 valid bytes; `.input_scale` is excluded from 6,226 checkpoint tensors. Model source: `Pinaster/GLM-5.2_5layer` revision `1c749139f70e158e4420ba67f342bef1de2e650d`, prepared as NVFP4 using the #3711 workflow. The unchanged case uses identical canonical bytes. The changed case reconstructs the retained synthetic published v1→v2 data: 79,500,803 differing byte positions across 2,646 tensors. Wholly unchanged tensors hold 465,303,040 bytes (2.5993%); changed-byte density is 0.4441%, a different measure.

Sorted-name layouts form 112 representative groups, limited to 128 MiB padded storage except a single larger tensor. Weight rows are 4,096 bytes; FP32 scalar scales have a separate compact four-byte-row group. Padding is 44,544 bytes, and the two preallocated old/new states own 35,801,698,304 bytes. The f17 control submits 3,155 tasks, including one exact compact batch for 1,536 scalars. These are reproducible CPU layouts, not a capture of live iterator subgroup boundaries.

## Method and complete medians

Each case starts with serial warmup of both graphs against all actual old/incoming layouts, followed by two alternating-order warmup pairs and five measured pairs. Medians use every measured sample, with no sample removal. CPU is process CPU-seconds, not utilization; compiler children are excluded.

Timed work includes submission/drain, exact counts, owned XOR/snapshot materialization when changed, lease return, zstd and checksum. Candidate replay diagnostics also build per-name views and compute copy/materialized-bucket counters; production caches payload-byte totals, omits unused encoder diagnostic accumulation, and skips unchanged snapshot-view reinstallation. These replay costs are retained in the numbers rather than subtracted. The CPU replay does not measure those production encoder savings.

| Case | f17 wall s | Compiled wall s | Wall change | f17 CPU s | Compiled CPU s | CPU change |
| --- | --- | --- | --- | --- | --- | --- |
| native-unchanged | 1.119358066 | 0.555282267 | -50.39% | 15.653191363 | 8.054724406 | -48.54% |
| synthetic-v1-v2 | 3.724775137 | 5.454653862 | +46.44% | 36.574509748 | 46.706875297 | +27.70% |

## Initialization and compiler activity

The first workload started with an empty task-specific disk cache; the second reused the initialized compiler. Layout warmup includes full execution/materialization and is not pure compilation latency. An empty disk cache does not establish absence of every external/runtime cache.

| Case | Actual layouts warmed | Wall s | Process CPU s | New graphs | New generated kernels |
| --- | --- | --- | --- | --- | --- |
| native-unchanged | 112 | 48.067237528 | 28.769855296 | 4 | 26 |
| synthetic-v1-v2 | 112 | 25.336171547 | 25.338994096 | 0 | 0 |

All 28 timed counter deltas are empty. The first warmup records four captured graphs and 26 generated kernels across layout specializations; these counts do not prove one fused loop or performance. The count phase uses word-lane XOR/masks, int32 row reductions, and an int64 prefix/boundary gather. Only groups with nonzero counts run the second compiled word-view XOR/snapshot phase.

## All 28 raw samples

Rows preserve observed execution order. `exact_f17` is the control; `packed_stage` is the selected module. Every row passed exact byte/count/checksum verification and had zero new compiler counters.

| Sequence | Case | Pair | Phase | Variant | Wall s | Process CPU s |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | native-unchanged | 0 | warmup | exact_f17 | 1.089844939 | 15.425387115 |
| 1 | native-unchanged | 0 | warmup | packed_stage | 0.496756019 | 8.142640718 |
| 2 | native-unchanged | 1 | warmup | packed_stage | 0.552796554 | 8.231005017 |
| 3 | native-unchanged | 1 | warmup | exact_f17 | 1.608770286 | 15.756873427 |
| 4 | native-unchanged | 2 | measured | exact_f17 | 1.119358066 | 15.855158649 |
| 5 | native-unchanged | 2 | measured | packed_stage | 0.555282267 | 8.096353478 |
| 6 | native-unchanged | 3 | measured | packed_stage | 0.799841530 | 8.031209716 |
| 7 | native-unchanged | 3 | measured | exact_f17 | 1.100333746 | 15.653191363 |
| 8 | native-unchanged | 4 | measured | exact_f17 | 1.124822663 | 15.734905227 |
| 9 | native-unchanged | 4 | measured | packed_stage | 0.879706997 | 8.429512676 |
| 10 | native-unchanged | 5 | measured | packed_stage | 0.553948698 | 8.054724406 |
| 11 | native-unchanged | 5 | measured | exact_f17 | 1.098065036 | 15.589114154 |
| 12 | native-unchanged | 6 | measured | exact_f17 | 1.594989287 | 15.467863072 |
| 13 | native-unchanged | 6 | measured | packed_stage | 0.554174144 | 7.996540909 |
| 14 | synthetic-v1-v2 | 0 | warmup | exact_f17 | 2.951852575 | 39.365542680 |
| 15 | synthetic-v1-v2 | 0 | warmup | packed_stage | 5.227535310 | 51.553978847 |
| 16 | synthetic-v1-v2 | 1 | warmup | packed_stage | 5.537208967 | 50.414514636 |
| 17 | synthetic-v1-v2 | 1 | warmup | exact_f17 | 4.222081284 | 39.006424625 |
| 18 | synthetic-v1-v2 | 2 | measured | exact_f17 | 3.724775137 | 36.574509748 |
| 19 | synthetic-v1-v2 | 2 | measured | packed_stage | 5.508079992 | 48.592996747 |
| 20 | synthetic-v1-v2 | 3 | measured | packed_stage | 3.779247727 | 46.706875297 |
| 21 | synthetic-v1-v2 | 3 | measured | exact_f17 | 3.788397153 | 37.534630554 |
| 22 | synthetic-v1-v2 | 4 | measured | exact_f17 | 3.825567818 | 36.044874570 |
| 23 | synthetic-v1-v2 | 4 | measured | packed_stage | 5.516496644 | 46.688594173 |
| 24 | synthetic-v1-v2 | 5 | measured | packed_stage | 5.454653862 | 46.535586093 |
| 25 | synthetic-v1-v2 | 5 | measured | exact_f17 | 3.716017536 | 37.878641596 |
| 26 | synthetic-v1-v2 | 6 | measured | exact_f17 | 2.727061980 | 33.498578357 |
| 27 | synthetic-v1-v2 | 6 | measured | packed_stage | 5.052016316 | 46.805521089 |

## Copy and wire-work counters

Values are constant across the seven samples in each case/variant.

| Case | Variant | Copied unchanged bytes | Copied padding bytes | Snapshot copy bytes | Compressed payload bytes |
| --- | --- | --- | --- | --- | --- |
| native-unchanged | exact_f17 | 6144 | 0 | 6144 | 0 |
| native-unchanged | packed_stage | 0 | 0 | 0 | 0 |
| synthetic-v1-v2 | exact_f17 | 6144 | 0 | 17435507712 | 182769068 |
| synthetic-v1-v2 | packed_stage | 465284608 | 44544 | 17900830720 | 182769068 |

The unchanged candidate materializes zero groups, preserves old snapshot identity and emits no payload. The changed candidate materializes 110 of 112 groups, including 465,284,608 unchanged valid bytes inside those groups. Both arms produce 182,769,068 compressed worker payload bytes; safetensors headers/index are excluded. f17 still copies the 6,144-byte compact scalar batch when it is unchanged. Snapshot-copy counters exclude input reads, XOR output writes, count buffers and allocator costs, so they are not total memory traffic. Each candidate sample returns 112 leases exactly once before compression; the control returns 3,155 tensor/scalar-group leases.

## Exact replay command and retained inputs

Run from `/hai-workspace/glm52-delta` inside the recorded image. The frozen variant file is identical to the selected production module. The full launch script, including preceding exactness checks, is `run_word_two_phase_cpu.sh`.

```bash
CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 timeout --kill-after=10 1200 \
  /opt/sglang/bin/python benchmark_delta_preparation.py \
  --miles-path miles --preparation-module delta_preparation_word_two_phase_variant.py \
  --checkpoint models/GLM-5.2_5layer-NVFP4 \
  --synthetic-source artifacts/synthetic-balanced-disk-delta-01/delta-publication \
  --expected-tensors 4690 --block-bytes 4096 --bucket-mib 128 \
  --baseline-workers 32 --stage-workers 32 --kernel-threads 1 \
  --warmups 2 --repeats 5 \
  --output artifacts/torch-compile-word-two-phase-32x1-replay.jsonl \
  > artifacts/torch-compile-word-two-phase-32x1-replay.log 2>&1
```

No explicit cache argument was supplied; the helper used `artifacts/torch-compile-word-two-phase-32x1-replay-inductor-cache`. Replay helper SHA256 `90fe76dccd02b4f9881177d6ace7b49912b54f7002c8622d63a896f3c73e5ec9`; reconstruction helper SHA256 `96bb0f3df1b5be1709c19dbeb91975eacea6a6f3a62b6d4c84c4064599fe3689`; layout SHA256 `c1b72468d9a164abaabf6689208d01971f964a9adfe10e85aeeb6bbdc8244fe9`.

Retained raw files: `artifacts/torch-compile-word-two-phase-32x1-replay.jsonl`, adjacent `.log`/`.exit`, and the adjacent `-sources/` folder. Their complete provenance includes source hashes, compiler counters, tensor/layout distributions and old/new checksum manifests. Recreating training does not guarantee the same byte workload; use the retained published v1/v2 artifacts.

## Limits

Checkpoint reads/reconstruction, initial packing and allocations, worker startup, warmup and verification are outside sample timing. No GPU gather/D2H, pinned-pool backpressure, production layout selection, snapshot installation, filesystem publication or receiver reload is measured. This comparison supports neither GPU equality nor multi-node bandwidth/scaling or amortization claims. CPU replay includes its own diagnostics and does not substitute for the separately matched NumPy/compiled GPU campaign.
