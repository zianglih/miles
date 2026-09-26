# Sender CPU component replay

`benchmark_sender_replay.py` compares the exact baseline and candidate `_diff_and_compress` method ASTs without importing Miles training or changing either implementation. It records the Git commits, SHA-256 hashes of both complete source files and extracted methods, helper-source hashes, script hash, Python/package versions, CPU affinity and canonical shard-header hashes. Full source snapshots are retained beside each output file. A working-tree candidate is labeled explicitly; the file hashes identify its actual contents.

This is a CPU component measurement. No GPU APIs are called. Each input is a regular CPU tensor standing in for an already-completed D2H view; pages are not pinned and there is no finite pinned-buffer pool or DMA. It cannot measure producer backpressure, D2H overlap, distributed trainer ranks, weight gathering, file publication, receiver apply, NVFP4 reload or end-to-end sync latency.

The later scalar-batching implementation does not change the extracted per-tensor method. This replay therefore isolates deferred snapshot copying; it does not measure GPU scale packing, fewer D2H copies/synchronizations/futures, the group wrapper, or the coordinated error-reduction optimization. The follow-up C2 sync run measures the combined sender change.

## Workloads and timing

- **Native unchanged:** all sender-emitted canonical checkpoint tensors, copied into separate old/incoming arrays with identical bytes. Conversion-only `.input_scale` entries are excluded by the explicit `--exclude-suffix` default because the Miles exporter never emits them.
- **Synthetic v1→v2:** materialize the canonical bytes in memory, apply and checksum-verify the published v1 deltas, copy all tensors, then apply and checksum-verify v2. Checkpoint and publication files are read only. Published v2 names must exactly equal the tensors with actual changed bytes.
- **Primary concurrency:** one persistent, warmed 32-thread pool; every checkpoint tensor submits one exact worker call. Inputs are prepared in sorted tensor-name order. `--workers 32 1` additionally permits a direct sequential comparison, separately labeled.
- **Sampling:** two warmup pairs and five measured pairs by default. Within successive pairs the execution order alternates baseline/candidate and candidate/baseline. Raw process CPU and wall times are retained, with separate medians.
- **Timed:** task submission/drain, exact worker diff/mask allocation, changed-byte count, snapshot allocation/copy, compression, checksum and queue return. Worker allocations remain timed because they are the optimization being evaluated.
- **Untimed:** canonical reading, delta reconstruction, input allocation/copy, thread creation, verification and result disposal. Every call checks full snapshot bytes, changed-byte counts, decompressed/reapplied bytes and digests, and confirms the old snapshots did not change.
- **Workload metadata:** tensor counts and byte-size distribution; unchanged tensor byte fraction; actual changed-byte density; compressed payload bytes; input-state checksum-manifest hashes. Compressed bytes exclude safetensors headers and filesystem publication.

The canonical checkpoint contains 6,226 tensors and 17,900,810,752 bytes. Excluding 1,536 conversion-only FP32 scalar `.input_scale` entries gives the actual sender scope of 4,690 tensors and 17,900,804,608 bytes. The script reports canonical and selected counts separately. The old and incoming arrays require twice the selected payload; baseline output snapshots and concurrent temporary arrays require additional RAM. The default 64 GiB cap applies only to old+incoming inputs. Full-model allocation is suitable for the recorded large-memory C2 host once the active GPU campaign is finished.

## Reproduction command

Copy the script into the effort root and make the candidate source or committed candidate revision available in its Miles checkout. Use a new output filename for every run. The paths below follow the current campaign manifest; the synthetic delta run must first have completed and published both v1 and v2.

```bash
cd /hai-workspace/glm52-delta
python benchmark_sender_replay.py \
  --miles-path ./miles \
  --baseline-ref 81d43908021deaf916808675d7ac78e4fbccfa2f \
  --candidate-ref f17ba4bce13bf7d357e7560182dc859c41a7cb37 \
  --checkpoint ./models/GLM-5.2_5layer-NVFP4 \
  --synthetic-source ./artifacts/synthetic-balanced-disk-delta-01/delta-publication \
  --expected-tensors 4690 \
  --workers 32 --warmups 2 --repeats 5 \
  --output ./artifacts/sender-cpu-replay-01.jsonl
```

The completed run used the exact candidate commit shown above. The script rejects identical worker sources and differing helper files to catch incorrect source selection. It performs no checkout, commit, remote operation, source edit or GPU action. Do not run it alongside measured training updates: its CPU and memory-bandwidth demand would contaminate them.

## Measured C2 replay

On C2 `hu-pdx-90` in `radixark/miles:dev-202609251434`, replayed the exact `_diff_and_compress` methods from Miles `81d43908021deaf916808675d7ac78e4fbccfa2f` and `f17ba4bce13bf7d357e7560182dc859c41a7cb37`. One warmed 32-thread pool processed 4,690 sender tensors totaling **17,900,804,608 bytes**, using XOR/zstd/`xxh3-128`. Python 3.12.3, PyTorch 2.13.0+cu130, NumPy 2.3.5, zstandard 0.25.0, xxhash 3.7.1; recorded CPU affinity was `0–1,11–129,138–255` on `hu-pdx-90`.

This isolates the **deferred snapshot-copy worker change**. Regular CPU tensors stand in for completed D2H buffers; no GPU/DMA, actual pinned pool/backpressure, gathering, scalar batching, publication, receiver apply, or reload is measured. Submission/drain, worker allocations, diff/count, snapshot copy, compression, checksum, and queue return are timed. Input preparation, pool creation, verification, and result disposal are outside timing. Process CPU counts all threads in this one process, not four trainer ranks.

Two warmup pairs precede five measured pairs; B→C/C→B order alternates. All **28 samples** passed full snapshot/decode/checksum/count verification, including unchanged old inputs. The unchanged case uses identical canonical bytes. The changed case reconstructs the published synthetic v1→v2 states: **79,500,803 changed bytes**, but only **465,303,040 bytes (2.5993%)** belong to wholly unchanged tensors. Its compressed worker payload is **182,769,068 bytes** for both implementations; this excludes safetensors headers/index and is distinct from the end-to-end shard-byte counter.

| Case | Baseline CPU median s | Candidate CPU median s | CPU change | Baseline wall median s | Candidate wall median s | Wall change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged canonical | 26.897635228 | 16.222345924 | -39.69% | 1.847023436 | 1.136617184 | -38.46% |
| Synthetic v1→v2 | 37.302611630 | 38.633915017 | +3.57% | 3.640603384 | 3.750314473 | +3.01% |

The unchanged case benefits; the changed case regresses **3.57% CPU / 3.01% wall** in this replay. Avoiding whole-tensor snapshot copies has limited opportunity when only 2.5993% of tensor bytes are in entirely unchanged tensors. These component measurements establish neither an end-to-end CPU improvement nor GPU weight equality.

<details>
<summary>All raw replay times, including warmups (seconds)</summary>

B = baseline `81d4390`; C = candidate `f17ba4b`. Only pairs 2–6 enter the medians; no measured sample is removed.

**Unchanged canonical**

| Pair | Phase | Order | B CPU s | C CPU s | B wall s | C wall s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | warmup | B→C | 27.979748897 | 16.185486650 | 1.805648297 | 1.116833735 |
| 1 | warmup | C→B | 27.086273169 | 16.196714541 | 1.858525590 | 1.132617468 |
| 2 | measured | B→C | 27.213451787 | 16.121260930 | 1.847023436 | 1.183669705 |
| 3 | measured | C→B | 26.882458437 | 16.038373422 | 1.855969505 | 1.136617184 |
| 4 | measured | B→C | 27.039739858 | 16.235907799 | 1.842206435 | 1.127102712 |
| 5 | measured | C→B | 26.897635228 | 16.284115492 | 1.848244950 | 1.142407622 |
| 6 | measured | B→C | 26.646809783 | 16.222345924 | 1.836968459 | 1.135969413 |

**Synthetic v1→v2**

| Pair | Phase | Order | B CPU s | C CPU s | B wall s | C wall s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | warmup | B→C | 39.355023081 | 39.802861279 | 3.859005058 | 2.954202173 |
| 1 | warmup | C→B | 37.124392072 | 39.361680575 | 3.622423058 | 3.760052274 |
| 2 | measured | B→C | 37.397130149 | 38.558384635 | 3.640603384 | 3.738598617 |
| 3 | measured | C→B | 37.302611630 | 38.688703524 | 3.633185970 | 3.750314473 |
| 4 | measured | B→C | 37.199620585 | 38.570953075 | 3.678468989 | 3.795329645 |
| 5 | measured | C→B | 37.057762144 | 38.637409051 | 3.570194925 | 3.756599146 |
| 6 | measured | B→C | 37.380946638 | 38.633915017 | 3.674611326 | 3.734263952 |

</details>

<!-- Retained local evidence: artifacts/sender-cpu-replay-01.jsonl and artifacts/sender-cpu-replay-01-sources/. Replay script SHA256 96bb0f3df1b5be1709c19dbeb91975eacea6a6f3a62b6d4c84c4064599fe3689. Add a real public evidence URL before describing these files or the script as attached/published. -->

## Local functional validation

The script passed small local fixtures with eight tensors and approximately 332 KiB of payload, both XOR and overwrite publications, both native and reconstructed v1→v2 cases, two-worker and sequential execution, one warmup pair and one measured pair. All 32 total samples passed snapshot/decode/checksum verification. Raw logs and provenance are under `validation/sender-replay-local/`. These fixture timings are not performance evidence. Python compilation and Ruff pass. The measured C2 replay is recorded above.

## Receiver runtime review

Independent read-only review against SGLang `e3ec949a06323adf4032d35df288c5658c3984bb` found no blockers in the two current runtime changes:

- `local_checkpoint.py`: empty `weight_map` advances only the atomic applied-version marker after retaining base-version, compression, required checksum-field and encoding validation. A failed marker replacement preserves the predecessor marker and can safely retry. The nonempty mutation, mmap, buffer-ownership and checksum path stays unchanged.
- `model_loader/loader.py`: CUDA memory probes run only for DEBUG logging. Explicit synchronization and cache release for online quantization remain in their original branch. Normal log levels stop triggering incidental cache eviction through diagnostic probes.

The review inspected the accompanying empty-delta replacement/retry/metadata and DEBUG × online-quantization tests. It did not run those tests, change receiver source or perform remote validation.
