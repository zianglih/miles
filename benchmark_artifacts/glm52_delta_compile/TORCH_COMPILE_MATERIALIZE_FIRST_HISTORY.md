# Materialize-first 8×4 CPU replay: retained negative result

This uncommitted experiment completed with exit code 0 and verified all 28 warmup/measured samples. It was rejected for the host CPU objective: changed weights took 30.70% more process CPU and 11.63% more wall time than the exact f17 control. Unchanged weights took 184.46% more process CPU and 265.04% more wall time. This report is historical evidence; it does not describe the final PR implementation or its pending validation.

## Exact source and environment

The experimental module is `delta_preparation_materialize_first_variant.py`, SHA256 `09fc7f41579c5e517a63263bb6a5763c08ad769a16e34dd658f3255ec900dc5d`. It has no source commit. The archived `-sources/delta_preparation.py` matches the project file byte for byte. The raw JSONL SHA256 is `738580f077e3cf2fd812599e1a06e7e83b830bfbd4e9bd9654fa13dc12830b01`. Replay helper SHA256 `90fe76dccd02b4f9881177d6ace7b49912b54f7002c8622d63a896f3c73e5ec9`; reconstruction helper SHA256 `96bb0f3df1b5be1709c19dbeb91975eacea6a6f3a62b6d4c84c4064599fe3689`.

C2 host `hu-pdx-90`: 2 × Intel Xeon 6776P, 64 cores per socket, SMT2, 256 logical CPUs, four NUMA nodes, AVX-512 available. Image `radixark/miles:dev-202609251434`, amd64 digest `sha256:7c4c6c8cc9e76be893941e064f0e43924ef9258b04da365c8009709592ff0984`. Python 3.12.3; PyTorch `2.13.0+cu130`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`; NumPy 2.3.5; zstandard 0.25.0; xxhash 3.7.1. Runtime reported one Torch intra-op thread, 128 inter-op threads, and CPU affinity `0–1,11–129,138–255`.

The control uses the exact committed f17 per-tensor and scalar-batch workers with a 32-worker pool. The experiment uses eight preparation/compression workers and four generated-kernel threads per worker. The configured generated-kernel budget is 32 threads, but compression concurrency differs (8 versus 32); this is a comparison of those complete configurations, not an isolated kernel comparison.

Exact C2 launch from `/hai-workspace/glm52-delta`:

```bash
CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 timeout --kill-after=10 1200 \
  /opt/sglang/bin/python benchmark_delta_preparation.py \
  --miles-path miles \
  --preparation-module delta_preparation_materialize_first_variant.py \
  --checkpoint models/GLM-5.2_5layer-NVFP4 \
  --synthetic-source artifacts/synthetic-balanced-disk-delta-01/delta-publication \
  --expected-tensors 4690 --block-bytes 4096 --bucket-mib 128 \
  --baseline-workers 32 --stage-workers 8 --kernel-threads 4 \
  --warmups 2 --repeats 5 \
  --output artifacts/torch-compile-materialize-first-8x4-replay.jsonl \
  > artifacts/torch-compile-materialize-first-8x4-replay.log 2>&1
```

No explicit cache argument was supplied. The helper selected `artifacts/torch-compile-materialize-first-8x4-replay-inductor-cache` and recorded `cache_initially_empty=true`.

## Workload and design tradeoff

The replay selects 4,690 named tensors (17,900,804,608 valid bytes) from 6,226 canonical tensors, excluding `.input_scale`. Stable sorted names produce 112 representative groups with at most 128 MiB of padded storage unless a single tensor is larger. Weight rows use 4,096 bytes; the compact FP32 scalar group uses 4-byte rows. Padding totals 44,544 bytes and the two preallocated input states own 35,801,698,304 bytes. The control submits 3,155 tasks, including one exact batch for 1,536 FP32 scalars (6,144 bytes). These layouts are representative, not captured live production subdivisions.

The first compiled phase materializes an owned XOR and snapshot using int32 word views. The second phase counts nonzero byte lanes from that owned XOR, reduces rows, and derives per-name int64 counts using a prefix sum and boundary gathers. The phase boundary avoids recomputing `new ^ old` in a separate reduction loop, but it always materializes every nonempty group, including completely unchanged groups. Only names with nonzero counts proceed to native compression/checksum. Graph/kernel totals alone do not prove fusion or acceleration.

The unchanged case compares identical bytes. Synthetic v1→v2 changes 79,500,803 byte positions across 2,646 tensors; entirely unchanged tensors contain 465,303,040 valid bytes. Both arms produce 182,769,068 compressed worker payload bytes for the changed case, excluding safetensors headers/index.

## Complete medians

Medians use all five measured pairs (iterations 2–6), with no sample removal. CPU means process CPU-seconds, not CPU utilization. Variant order alternates per pair; both warmup pairs remain in the raw table below.

| Case | f17 wall s | 8×4 wall s | Wall change | f17 CPU s | 8×4 CPU s | CPU change |
| --- | --- | --- | --- | --- | --- | --- |
| native-unchanged | 1.153190670 | 4.209630460 | +265.04% | 16.407209183 | 46.672686736 | +184.46% |
| synthetic-v1-v2 | 3.348511277 | 3.737940219 | +11.63% | 36.209462476 | 47.324183514 | +30.70% |

## Cold and already initialized layout warmup

Before fanout, both compiled phases run serially against every actual old/incoming layout. The first case starts with an empty task-specific disk cache; the second reuses the initialized compiler. These times include full-layout execution/materialization and are not pure compilation latency. Calling-process CPU excludes compiler-child CPU. The runtime may still have other caches; the empty disk cache is a recorded condition.

| Case | Layouts warmed | Wall s | Process CPU s | New graphs | New generated kernels |
| --- | --- | --- | --- | --- | --- |
| native-unchanged | 112 | 34.808798097 | 35.807375023 | 4 | 26 |
| synthetic-v1-v2 | 112 | 9.601313450 | 31.676338740 | 0 | 0 |

Every timed sample has empty compiler counter deltas. Full warmup counter dictionaries remain in the raw JSONL.

## All 28 raw samples

Rows preserve their observed execution order. `exact_f17` is the control; `packed_stage` is this uncommitted materialize-first experiment. All rows passed exact byte/count/checksum verification and had no timed compiler activity.

| Sequence | Case | Pair | Phase | Variant | Wall s | Process CPU s |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | native-unchanged | 0 | warmup | exact_f17 | 1.066240562 | 16.159637382 |
| 1 | native-unchanged | 0 | warmup | packed_stage | 3.482836896 | 43.923641453 |
| 2 | native-unchanged | 1 | warmup | packed_stage | 3.822864562 | 44.686293106 |
| 3 | native-unchanged | 1 | warmup | exact_f17 | 1.111578839 | 16.694607641 |
| 4 | native-unchanged | 2 | measured | exact_f17 | 1.090380819 | 17.104434510 |
| 5 | native-unchanged | 2 | measured | packed_stage | 4.414925840 | 50.046197366 |
| 6 | native-unchanged | 3 | measured | packed_stage | 4.535154984 | 47.631452992 |
| 7 | native-unchanged | 3 | measured | exact_f17 | 1.153190670 | 16.407209183 |
| 8 | native-unchanged | 4 | measured | exact_f17 | 1.620442025 | 16.097449749 |
| 9 | native-unchanged | 4 | measured | packed_stage | 4.158176309 | 45.550653279 |
| 10 | native-unchanged | 5 | measured | packed_stage | 4.209630460 | 46.672686736 |
| 11 | native-unchanged | 5 | measured | exact_f17 | 1.726885599 | 16.443235294 |
| 12 | native-unchanged | 6 | measured | exact_f17 | 1.118493165 | 16.058917747 |
| 13 | native-unchanged | 6 | measured | packed_stage | 4.009350852 | 44.940521497 |
| 14 | synthetic-v1-v2 | 0 | warmup | exact_f17 | 3.802222350 | 40.310767555 |
| 15 | synthetic-v1-v2 | 0 | warmup | packed_stage | 3.656788837 | 47.599307944 |
| 16 | synthetic-v1-v2 | 1 | warmup | packed_stage | 3.794409368 | 49.315498187 |
| 17 | synthetic-v1-v2 | 1 | warmup | exact_f17 | 3.775578207 | 39.786166130 |
| 18 | synthetic-v1-v2 | 2 | measured | exact_f17 | 3.348511277 | 36.583928973 |
| 19 | synthetic-v1-v2 | 2 | measured | packed_stage | 3.583165166 | 46.775607558 |
| 20 | synthetic-v1-v2 | 3 | measured | packed_stage | 3.600636177 | 46.607146266 |
| 21 | synthetic-v1-v2 | 3 | measured | exact_f17 | 3.771694406 | 38.229441894 |
| 22 | synthetic-v1-v2 | 4 | measured | exact_f17 | 2.873773966 | 36.209462476 |
| 23 | synthetic-v1-v2 | 4 | measured | packed_stage | 3.830219448 | 51.210018372 |
| 24 | synthetic-v1-v2 | 5 | measured | packed_stage | 3.782262895 | 47.533083226 |
| 25 | synthetic-v1-v2 | 5 | measured | exact_f17 | 4.176866774 | 34.860766168 |
| 26 | synthetic-v1-v2 | 6 | measured | exact_f17 | 2.894729759 | 33.874551642 |
| 27 | synthetic-v1-v2 | 6 | measured | packed_stage | 3.737940219 | 47.324183514 |

## Copy and ownership accounting

These raw counters are constant across all seven samples of each case/variant.

| Case | Variant | Copied unchanged bytes | Copied padding bytes | Snapshot copy bytes | Compressed payload bytes | Lease returns |
| --- | --- | --- | --- | --- | --- | --- |
| native-unchanged | exact_f17 | 6144 | 0 | 6144 | 0 | 3155 |
| native-unchanged | packed_stage | 17900804608 | 44544 | 17900849152 | 0 | 112 |
| synthetic-v1-v2 | exact_f17 | 6144 | 0 | 17435507712 | 182769068 | 3155 |
| synthetic-v1-v2 | packed_stage | 465303040 | 44544 | 17900849152 | 182769068 | 112 |

The experiment materializes all 112 groups in both workloads: each call copies 17,900,849,152 bytes into owned snapshot storage and also allocates/materializes an XOR buffer of the same padded size. The table's snapshot-copy counter is not total memory traffic; it excludes reads, XOR output writes, count buffers and allocator effects. In the unchanged case, f17 only snapshots the 6,144-byte compact scalar batch. This extra full-model materialization explains a clear design cost, though these measurements do not attribute all timing differences to one operation.

The experimental stage returns each of its 112 input leases once before native compression. The exact control returns 3,155 tensor/scalar-group leases. Inputs remain unchanged; owned output snapshots, decoded XOR payloads, per-name counts and checksums are verified outside timing.

## Retained evidence and validation limits

- Raw replay: `artifacts/torch-compile-materialize-first-8x4-replay.jsonl`, adjacent `.log` and `.exit`, and the adjacent `-sources/` folder.
- Layout SHA256: `c1b72468d9a164abaabf6689208d01971f964a9adfe10e85aeeb6bbdc8244fe9`.
- Exact control: `f17ba4bce13bf7d357e7560182dc859c41a7cb37`; archived per-tensor/scalar-batch ASTs and full source hashes are retained.
- C2 exactness probe: `artifacts/torch-compile-materialize-first-exactness.log`. Its complete output is:

```text
W0926 00:52:50.772000 396982 torch/utils/cpp_extension.py:178] [0/0] No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
3 layouts x 4 concurrent preparations + 3 no-change preparations passed; exact byte pairs, tails, >2^24 counts, ownership, and no timed recompilation
```

Timed work includes worker submission/drain, native preparation/count/materialization allocations, zstd/checksum work and lease return. Reads/reconstruction/packing, initial allocations, worker startup, layout warmup and verification are outside timing. GPU gather/D2H, actual pinned-pool backpressure, production layout selection, snapshot installation, file publication and receiver reload are unmeasured. This CPU-only replay establishes neither GPU equality nor multi-node benefit. It remains separate from the earlier 6eca replay and the final candidate's CPU/GPU validation.

Publisher admission for this experiment:

```text
--historical-replay artifacts/torch-compile-materialize-first-8x4-replay.jsonl=sha256:09fc7f41579c5e517a63263bb6a5763c08ad769a16e34dd658f3255ec900dc5d
```

The source is explicitly labeled uncommitted; no source commit is inferred from the archive hash.
