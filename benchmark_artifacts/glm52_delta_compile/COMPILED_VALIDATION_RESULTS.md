# Compiled disk-delta validation

All planned runs passed strict source, recipe, seven-update, 28-rank training/CPU, publication, and per-rank compiler admission.

Sources: `{'miles': '767439e95bca5203211ba374b4e04b702c2b7ed2', 'sglang': '2f5fb2a09f08eb9a44c0fed5a28be0bda2942b0f'}`. Image `radixark/miles:dev-202609252029`; amd64 `sha256:0c2e77991235b6592dea7fe1238a74d60aa94aa3cc82543fbfcb40f3f064cbb8`.

Historical observer calibration on original hu-pdx-90/dev-202609251434; not remeasured on this campaign host. Actual per-run snapshot costs and inventories remain in each arm.

Steady = all u2–u6; no sample removal. Full raw compiler dictionaries, runtime settings, source/helper hashes, publication checksums, and warnings remain in JSON.

## synthetic-balanced: matched CPU backends

Ratios are compiled / NumPy median. Values are wall seconds or component process CPU-seconds.

| Metric | NumPy median | Compiled median | Ratio |
| --- | --- | --- | --- |
| driver_wall_s | 13.016042927 | 16.608324024 | 1.2759887253866 |
| builtin_actor_s | 13.010178089141846 | 16.60264825820923 | 1.2761276705401612 |
| trainer_max_wall_s | 12.979594213 | 16.571093566 | 1.276703515846655 |
| generation_pause_upper_bound_s | 3.598636055 | 3.556330823 | 0.9882440926636022 |
| trainer_cpu_sum_s | 45.949220765 | 66.442446955 | 1.4459972519405573 |
| trainer_cpu_max_s | 29.549733189 | 49.042508819 | 1.6596599537912666 |
| receiver_total_cpu_s | 142.961725486 | 179.270005116 | 1.2539720299721453 |
| receiver_scheduler_cpu_s | 117.55558826 | 145.385337324 | 1.2367369299573272 |
| receiver_auxiliary_cpu_s | 25.992464158 | 33.329556479 | 1.2822776738827117 |
| driver_cpu_s | 0.065991766 | 0.083442258 | 1.2644343841321055 |

## compile-v2-synthetic-numpy-01

Backend `numpy`; reward `synthetic-balanced`; cache before `{'exists': False, 'empty': True}`.

| Update | Driver s | Actor s | Trainer CPU sum s | Trainer CPU max s | Receiver CPU s | Driver CPU s | Changed bytes | Wire bytes | CPU valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 16.742245542 | 16.658995389938354 | 58.406388244 | 16.440242345 | 56.349841456 | 0.070774058 | — | — | True |
| 1 | 24.795834247 | 24.790311813354492 | 66.041242387 | 50.056152136 | 224.528279431 | 0.095813561 | 90333140 | 204822174 | True |
| 2 | 13.016042927 | 13.010178089141846 | 47.696585554 | 30.913979251 | 141.896610505 | 0.064325299 | 79870924 | 184032627 | True |
| 3 | 13.894306085 | 13.888745546340942 | 46.360371321 | 29.576285113 | 146.450593912 | 0.066565558 | 76062156 | 176839130 | True |
| 4 | 13.217150716 | 13.21164321899414 | 45.949220765 | 29.549733189 | 143.961696074 | 0.070227723 | 64277222 | 154060113 | True |
| 5 | 12.403634974 | 12.397961378097534 | 42.490361242 | 28.10574968 | 139.771697328 | 0.065991766 | 72428518 | 169092076 | True |
| 6 | 12.71725352 | 12.711289405822754 | 44.223776098 | 28.081929343 | 142.961725486 | 0.06453142 | 63942934 | 154006978 | True |

Raw per-rank compiler activity (complete counter deltas retained in JSON):

| Update | Rank | PID | Success | Steady | New graph/cache/kernel counters |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 51963 | True | False | {} |
| 0 | 1 | 51960 | True | False | {} |
| 0 | 2 | 51959 | True | False | {} |
| 0 | 3 | 51961 | True | False | {} |
| 1 | 0 | 51963 | True | False | {} |
| 1 | 1 | 51960 | True | False | {} |
| 1 | 2 | 51959 | True | False | {} |
| 1 | 3 | 51961 | True | False | {} |
| 2 | 0 | 51963 | True | True | {} |
| 2 | 1 | 51960 | True | True | {} |
| 2 | 2 | 51959 | True | True | {} |
| 2 | 3 | 51961 | True | True | {} |
| 3 | 0 | 51963 | True | True | {} |
| 3 | 1 | 51960 | True | True | {} |
| 3 | 2 | 51959 | True | True | {} |
| 3 | 3 | 51961 | True | True | {} |
| 4 | 0 | 51963 | True | True | {} |
| 4 | 1 | 51960 | True | True | {} |
| 4 | 2 | 51959 | True | True | {} |
| 4 | 3 | 51961 | True | True | {} |
| 5 | 0 | 51963 | True | True | {} |
| 5 | 1 | 51960 | True | True | {} |
| 5 | 2 | 51959 | True | True | {} |
| 5 | 3 | 51961 | True | True | {} |
| 6 | 0 | 51963 | True | True | {} |
| 6 | 1 | 51960 | True | True | {} |
| 6 | 2 | 51959 | True | True | {} |
| 6 | 3 | 51961 | True | True | {} |

Gradient norms: `[2.7865278720855713, 2.70878267288208, 2.3267931938171387, 2.0322420597076416, 2.3288302421569824, 1.9762156009674072, 2.3722891807556152]`. Routing replay checks: `1464`, mismatching checks: `0`.

Observer snapshot/driver ratios: `[4.0696469961788105e-05, 3.6395196485985576e-05, 4.047392751241136e-05, 4.416028052648819e-05, 4.007563419243686e-05]`; outer bookkeeping/driver ratios: `[0.003756996137325354, 0.0034728606599571687, 0.003525465813414123, 0.003847188594313434, 0.0037703706955745267]`. Invalid early CPU indices: `[]`.

Warning inventory: `{'startup': 159, 'training': 128, 'teardown': 3}`. Readiness retries: `{'retry_until_deadline': 21, 'wait_expected_num_cells': 49}`.

## compile-v2-synthetic-torch-01

Backend `torch-compile`; reward `synthetic-balanced`; cache before `{'exists': False, 'empty': True}`.

| Update | Driver s | Actor s | Trainer CPU sum s | Trainer CPU max s | Receiver CPU s | Driver CPU s | Changed bytes | Wire bytes | CPU valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 114.796290212 | 114.71379065513611 | 206.071009724 | 93.098896254 | 642.029459215 | 0.383578416 | — | — | True |
| 1 | 14.495778291 | 14.489706039428711 | 65.9057045 | 49.534372211 | 161.384023909 | 0.073088928 | 90347400 | 204933741 | True |
| 2 | 14.732254393 | 14.72694993019104 | 66.442446955 | 49.042508819 | 154.633183589 | 0.076927188 | 79488087 | 183444567 | True |
| 3 | 16.969739729 | 16.964051723480225 | 68.023237368 | 49.465876805 | 179.270005116 | 0.088471392 | 75066868 | 175201540 | True |
| 4 | 15.487924229 | 15.482511281967163 | 64.012483953 | 46.717864434 | 171.45362866 | 0.069908515 | 63568600 | 152594023 | True |
| 5 | 17.29287018 | 17.28737473487854 | 69.88039563299999 | 49.107584397 | 180.027104819 | 0.083442258 | 71634072 | 167624110 | True |
| 6 | 16.608324024 | 16.60264825820923 | 65.671509528 | 48.262450866 | 179.375451422 | 0.085111221 | 63639449 | 153124525 | True |

Raw per-rank compiler activity (complete counter deltas retained in JSON):

| Update | Rank | PID | Success | Steady | New graph/cache/kernel counters |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 107438 | True | False | {"aot_autograd.total": 4, "inductor.fxgraph_cache_miss": 4, "inductor.generated_kernel_count": 26, "stats.calls_captured": 90, "stats.unique_graphs": 4} |
| 0 | 1 | 107439 | True | False | {} |
| 0 | 2 | 107440 | True | False | {} |
| 0 | 3 | 107441 | True | False | {} |
| 1 | 0 | 107438 | True | False | {} |
| 1 | 1 | 107439 | True | False | {} |
| 1 | 2 | 107440 | True | False | {} |
| 1 | 3 | 107441 | True | False | {} |
| 2 | 0 | 107438 | True | True | {} |
| 2 | 1 | 107439 | True | True | {} |
| 2 | 2 | 107440 | True | True | {} |
| 2 | 3 | 107441 | True | True | {} |
| 3 | 0 | 107438 | True | True | {} |
| 3 | 1 | 107439 | True | True | {} |
| 3 | 2 | 107440 | True | True | {} |
| 3 | 3 | 107441 | True | True | {} |
| 4 | 0 | 107438 | True | True | {} |
| 4 | 1 | 107439 | True | True | {} |
| 4 | 2 | 107440 | True | True | {} |
| 4 | 3 | 107441 | True | True | {} |
| 5 | 0 | 107438 | True | True | {} |
| 5 | 1 | 107439 | True | True | {} |
| 5 | 2 | 107440 | True | True | {} |
| 5 | 3 | 107441 | True | True | {} |
| 6 | 0 | 107438 | True | True | {} |
| 6 | 1 | 107439 | True | True | {} |
| 6 | 2 | 107440 | True | True | {} |
| 6 | 3 | 107441 | True | True | {} |

Gradient norms: `[2.709739923477173, 2.6820340156555176, 2.7607762813568115, 2.4329144954681396, 2.3261475563049316, 2.027017593383789, 2.0719356536865234]`. Routing replay checks: `1464`, mismatching checks: `0`.

Observer snapshot/driver ratios: `[3.4126684659944975e-05, 3.219854922502094e-05, 3.1389487242564425e-05, 3.0045966608881344e-05, 2.5975288016815727e-05]`; outer bookkeeping/driver ratios: `[0.0033772447632753827, 0.002826452660203908, 0.003068446377792516, 0.0027420813032437855, 0.002938667317031627]`. Invalid early CPU indices: `[]`.

Warning inventory: `{'startup': 134, 'training': 115}`. Readiness retries: `{'retry_until_deadline': 18, 'wait_expected_num_cells': 28}`.

## Boundaries

- Single-node C2 4+4 correctness and local overhead; no multi-node speedup or amortization inference.
- All seven updates are retained; only u2-u6 enter steady medians. Any invalid steady CPU or compiler activity rejects admission.
- Compiler counters cover each entire trainer updater, not only packed CPU graphs. Their reads/write lie outside updater timing and inside driver wall.
- Initial setup wall includes initialization/compilation; an empty on-disk cache is recorded but cannot prove absence of all external/compiler caches. Trainer process CPU excludes compiler child CPU.
- CPU windows cover trainer processes and inventoried receiver trees, not simultaneous whole-host CPU. Non-atomic membership can omit transient children. No observer cost is subtracted.
- Publication indexes, shard SHA256, mapping/ranges, and checksum presence are checked. Successful receiver apply supplies runtime checksum checks; this report does not independently reconstruct checkpoint bytes or attest GPU equality.
- Synthetic balanced rewards test changed-weight transport, not task quality or equal training trajectories. Disabled original logprob/KL/weight-equality checkers remain disabled.
- A single backend arm supports correctness only. Backend performance ratios appear only for matched same-reward pairs in this frozen plan; earlier campaigns are not substituted.
