# Frozen CPU observer v2 and final report

The implementation supersedes the initial design in `CPU_OBSERVATION_PLAN.md`.
The final matrix is `CPU_CAMPAIGN.json`: candidate native disk-delta, candidate
synthetic-balanced broadcast, and candidate synthetic-balanced disk-delta.
Each run has seven rollouts and seven weight updates, including initial sync;
indices 2–6 provide five steady post-training samples. Native is the final
zero-gradient/zero-change integration check. Synthetic provides explicit
nonzero gradients, absolute advantages and changed weight bytes.

This C2 4+4 run measures correctness and local CPU/wall overhead for future
large multi-node use. Local broadcast need not lose for the experiment to be
useful. There is no cross-node speedup or amortization projection.

## Observation contract

- `process_cpu_clocks_cpu_v2.py` reads all-thread process clocks. Four trainer
  processes also record `getrusage(RUSAGE_SELF)` user/system CPU, page faults and
  context-switch deltas. Driver CPU is a separate scope.
- Four receiver schedulers and their engine process-tree auxiliaries use cached
  Linux process clock IDs. Auxiliary includes the two data-parallel controllers
  and inventoried compile workers. Receiver snapshots bracket the rank0 updater,
  **not** the whole driver dispatch. The process CPU windows differ slightly.
- Receiver membership is refreshed before and after each update, outside the
  trainer CPU and original trainer wall window. These refreshes, and buffered
  CPU journal flushing, remain inside driver/builtin actor wall time. Snapshot
  reads are outside trainer CPU but inside original trainer wall time.
- Membership refresh traverses only the receiver trees through Linux
  `/proc/PID/task/TID/children`, including children created by nonleader threads.
  It reads each reached process's start time and parent, reuses stable clock
  handles, and fails diagnostically if the scan is incomplete or an identity
  changes. It does not call psutil's per-root global process scans.
  The [Linux children-file interface](https://man7.org/linux/man-pages/man5/proc_tid_children.5.html)
  can omit surviving siblings during concurrent exits. A previously inventoried
  PID that vanishes from traversal but still has the same start time invalidates
  the scan. Live process discovery remains non-atomic; entirely transient or
  newly born children during concurrent exits may be unobserved.
- PID/start-time replacement, membership changes during an update or inaccessible
  clocks invalidate CPU evidence while training continues. Children born and
  exited entirely between the two membership checks are not counted.
- Setup records include runtime affinity, visible cgroup configuration and only
  selected thread environment variables plus PYTHONPATH. Per-role signatures
  ignore PIDs and must match within the synthetic pair. Some container quota
  files may be unavailable; that limitation is reported rather than inferred.
- Launcher/driver/trainers record actual loaded Miles/SGLang module paths when
  present. Receiver source is checked through its selected launch command and
  PYTHONPATH. Actual loaded receiver module paths are not independently attested.
  No server hooks, plugins, import hooks, sitecustomize or monitoring threads are
  installed.
- Measurement files are the four original v1 scripts and four v2 files listed
  in each manifest. The summarizer/report generator are outside the workload.
  v1 scripts stay unchanged. Source heads, tracked diffs and helper SHA256 values
  must match the candidate plan and each other before paired results are admitted.

## CPU-only calibration

Run on C2 with the two actual engine URLs, after startup, outside active update
measurement. The command reads counters/process metadata; it sends no engine
requests and starts no GPU work.

```bash
python process_cpu_clocks_cpu_v2.py \
  --engine-url http://HOST:PORT_A --engine-url http://HOST:PORT_B \
  --iterations 1000 --membership-iterations 20 \
  > artifacts/cpu-observer-calibration-v2.json
```

`self` and receiver `calibration` measure two counter/snapshot reads.
`membership_refresh_pair` measures two cached-membership discovery refreshes;
the first clock-only calibration cannot bound this cost. Per-update snapshots
and outside-trainer-wall bookkeeping costs are also retained. Journal flush
duration is available as `previous_flush_ns` on a subsequent record; the final
flush is not measured until another write. No observer cost is subtracted.

The initial psutil-recursive implementation was rejected before the candidate
campaign: two refreshes cost median 281.46 ms on C2. The final direct `/proc`
traversal passed an independent psutil inventory comparison: **144 processes,
four schedulers and 140 auxiliaries**. Final C2 observer tests: **15 passed in
0.26 s** ([raw output](artifacts/cpu-observer-v2-fast-final-tests-c2.log)).

Final idle-engine calibration ([raw JSON](artifacts/cpu-observer-fast-final-calibration.json)):

| Two-read operation | Iterations | Median ms | Min ms | Max ms |
|---|---:|---:|---:|---:|
| Trainer process clock | 1000 | 0.001222 | 0.001166 | 0.006597 |
| Receiver snapshot | 1000 | 0.3270025 | 0.323423 | 0.437857 |
| Membership refresh | 20 | 60.510559 | 60.267367 | 60.765224 |

Calibration SHA256: `ce8386c75e01125a7e3754844389a82f2e889200df0ce233cd030aff87216707`.
The remaining 60.51 ms refresh cost is material: approximately **4.8% of a
1.27 s historical local broadcast update**. That is a scale comparison, not an
observer-on/off result or a correction. Refresh remains inside driver/builtin
actor wall; actual per-update overhead is retained and nothing is subtracted.
The final campaign accepts and discloses this cost. No campaign uses the
rejected discovery implementation.

## Measurement freeze

Final bundle: [cpu-observation-v2-final-frozen.tar.gz](artifacts/cpu-observation-v2-final-frozen.tar.gz),
SHA256 `aad0e8e0f9689c4f14f17ffa0c9788be0c81dbf3adc7f4661c81851e70969247`.
Older bundles remain historical. The report requires these eight manifest
helper SHA256 values; report/document changes do not modify measurement code.

| File | SHA256 |
|---|---|
| launch_weight_sync.py | `1e5c4a50a7f17fdcd61f3dc6875349ca7399667a4eb153c6525e193811ae5815` |
| weight_sync_probe.py | `1e3f80082af5821993c57b6efc9624e9b60418b2264513fb0b54442fb8bc7ed1` |
| train_weight_sync_profiled.py | `243b880d31a606ee2684635076040bd5c773eb2c6feb5c137fb7871514739051` |
| weight_sync_reward.py | `2f785003f111cee264f4aa880cdf14127f6240cc8e3971765d4c7567b782e49c` |
| launch_weight_sync_cpu_v2.py | `46f5b845dea53a59d6582f12f0346352adf8ed022900c6770b69d7be4eb6a2ab` |
| weight_sync_probe_cpu_v2.py | `760cc0a1645d5a0a4dd48eaf8a4f26eede18015b21d82cd87eee72e819593e55` |
| train_weight_sync_profiled_cpu_v2.py | `f8fb68508f7b3dd86979b3e66eb57dc265d11e726fdf34abf41f10270145f8ef` |
| process_cpu_clocks_cpu_v2.py | `4a3b3ec5cf3fd1c27096d9979098a80008f69f88306527b7a75dd7b0b36aab81` |

## Summaries and strict final report

Root's campaign runner owns launches. After each run, in order:

```bash
python summarize_weight_sync.py artifacts/RUN
python summarize_training_evidence.py artifacts/RUN --debug-dumps require
python summarize_weight_sync_cpu_v2.py artifacts/RUN
```

The final report reads retained local artifacts and never launches jobs:

```bash
python build_cpu_benchmark_report.py \
  --campaign-metadata CAMPAIGN.json \
  --candidate-plan CPU_CAMPAIGN.json \
  --environment environment-image.json \
  artifacts/cpu-v2-native-disk-delta-01 \
  artifacts/cpu-v2-synthetic-balanced-broadcast-01 \
  artifacts/cpu-v2-synthetic-balanced-disk-delta-01
```

This writes `CPU_BENCHMARK_RESULTS.md` and `.json` only after all three arms pass
exit0/final Ray success, seven rollouts × four ranks, attempt0 normal returns,
fresh training/debug evidence, exact configs/sources/helpers, raw clock
arithmetic, matched synthetic runtime settings and five valid steady samples.
It retains all seven rows, scheduler/auxiliary/trainer CPU separately, builtin
actor/driver/trainer wall, pause, byte counts, overhead, replay and warnings.
Missing arms or invalid evidence produce exit2 and no new report.

The original four-arm v1 campaign did not record CPU counters. Final synthetic
delta-versus-broadcast CPU ratios are candidate transport comparisons, not a
before/after E2E CPU reduction claim. Baseline/candidate CPU-only sender/receiver
replay is separate component evidence and excludes gathering/GPU reload.

CPU-only validation:

```bash
python -m unittest -v test_weight_sync_cpu_v2.py test_build_cpu_benchmark_report.py
ruff check *_cpu_v2.py build_cpu_benchmark_report.py test_build_cpu_benchmark_report.py
```
