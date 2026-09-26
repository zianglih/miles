# Packed CPU stage: 6eca replay

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
