# Candidate CPU and wall benchmark

Three candidate runs passed completion, source/helper/configuration, raw CPU counter, and five-steady-sample checks. Synthetic delta is compared with synthetic broadcast. Native delta is standalone zero-change path validation.

**Earlier v1 CPU measurements are unavailable. This report makes no before/after E2E CPU reduction claim.**

Environment: c2 / hu-pdx-90 / glm52-delta-0925; 8 B300, 4 trainer + 4 rollout GPUs, two TP2 engines. Image `radixark/miles:dev-202609251434`; index `sha256:abfd1b58539bef057717928a7ce115970fe6854bea82c958c2fe8e18e3014ced`; amd64 `sha256:7c4c6c8cc9e76be893941e064f0e43924ef9258b04da365c8009709592ff0984`.

Candidate source pins: `{'miles': 'f17ba4bce13bf7d357e7560182dc859c41a7cb37', 'sglang': '2f5fb2a09f08eb9a44c0fed5a28be0bda2942b0f'}`; Megatron `f148a32b4385b758b66a77c9c3ad1641f1295d4b`. Full configs, helper hashes, package versions, input SHA256 values and process inventories are retained in the JSON.

## CPU observer calibration

[Final calibration](artifacts/cpu-observer-fast-final-calibration.json) independently matched 144 processes: four schedulers and 140 auxiliary processes. SHA256 `ce8386c75e01125a7e3754844389a82f2e889200df0ce233cd030aff87216707`. All run helper hashes must match the calibrated final freeze.

| Two-read operation | N | Median ms | Min ms | Max ms |
| --- | --- | --- | --- | --- |
| self | 1000 | 0.001222 | 0.001166 | 0.006597 |
| receiver_snapshot_pair | 1000 | 0.3270025 | 0.323423 | 0.437857 |
| membership_refresh_pair | 20 | 60.510559 | 60.267367 | 60.765224 |

The 60.51 ms refresh-pair median is approximately 4.8% of a 1.27 s historical broadcast update. This is a scale comparison, not an observer correction or an observer-on/off result; actual v2 per-update bookkeeping costs remain authoritative and no cost is subtracted.

## Synthetic transport comparison

Ratios are delta / broadcast median. Wall values are seconds; process CPU values are CPU-seconds summed over the named processes/threads.

| Metric | Broadcast median | Broadcast min–max | Delta median | Delta min–max | Delta / broadcast |
| --- | --- | --- | --- | --- | --- |
| driver_wall_s | 1.357924333 | 1.340206148–1.787202798 | 12.948734415 | 12.859471272–13.197179936 | 9.53568184936561 |
| builtin_actor_s | 1.3526790142059326 | 1.3348963260650635–1.7822566032409668 | 12.943434238433838 | 12.853264808654785–13.191802024841309 | 9.568740331224893 |
| trainer_max_wall_s | 1.309907198 | 1.289739435–1.737288781 | 12.898158838 | 12.805195889–13.163580658 | 9.846620323709374 |
| generation_pause_upper_bound_s | 1.267533657 | 1.24804142–1.697737651 | 3.420274695 | 2.953697278–3.565661697 | 2.6983699218647255 |
| trainer_cpu_sum_s | 4.026750893 | 4.011894221–5.309132592 | 45.646671813 | 44.063447301–48.104858048 | 11.335856879637376 |
| trainer_cpu_max_s | 1.076006927 | 1.071213831–1.5022826 | 29.822337512 | 28.877551331–31.328232497 | 27.715748629190752 |
| receiver_total_cpu_s | 7.651091828 | 7.567369506–10.549256524 | 146.672678635 | 144.859401781–154.145447389 | 19.170163152170705 |
| receiver_scheduler_cpu_s | 5.051480995 | 5.003141864–6.659390511 | 120.941810622 | 118.937348513–128.37677129 | 23.941852051251754 |
| receiver_auxiliary_cpu_s | 2.599610833 | 2.551502988–3.889866013 | 25.836985533 | 25.730868013–26.481371007 | 9.938789762305934 |
| driver_cpu_s | 0.017463761 | 0.015825876–0.023349701 | 0.073974408 | 0.065658256–0.090932981 | 4.235880690304912 |

## cpu-v2-native-disk-delta-01

Exit0 and final Ray success; seven rollouts, 28 normal attempt0 trainer returns; seven raw updates including initial sync. Steady samples are indices2–6.

| Update | Driver s | Builtin actor s | Trainer CPU s | Receiver CPU s | Scheduler CPU s | Auxiliary CPU s | Pause s | Changed bytes | Wire bytes | CPU valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 33.223720826 | 33.137890577316284 | 123.33591663600001 | 158.847461965 | 92.748210919 | 66.099251046 | — | — | — | True |
| 1 | 18.550976809 | 18.545242309570312 | 36.130356025 | 118.336787219 | 81.275402886 | 37.061384333 | 3.20283569 | 0 | 0 | True |
| 2 | 6.032622859 | 6.028110027313232 | 21.445608288 | 42.767788931 | 30.779228786 | 11.988560145 | 3.255587207 | 0 | 0 | True |
| 3 | 5.95771775 | 5.9517247676849365 | 15.581085792 | 43.041944528 | 31.199378855 | 11.842565673 | 3.668791513 | 0 | 0 | True |
| 4 | 5.197755373 | 5.192755222320557 | 15.469122472999999 | 38.791682486 | 28.442788097 | 10.348894389 | 2.981856777 | 0 | 0 | True |
| 5 | 5.920635012 | 5.9155261516571045 | 21.002613882000002 | 43.180810873 | 31.380767017 | 11.800043856 | 3.142344457 | 0 | 0 | True |
| 6 | 6.137999254 | 6.132843971252441 | 15.766396908 | 44.845692641 | 32.699917637 | 12.145775004 | 3.812172841 | 0 | 0 | True |

Steady medians and ranges:

| Metric | N | Median | Min | Max |
| --- | --- | --- | --- | --- |
| driver_wall_s | 5 | 5.95771775 | 5.197755373 | 6.137999254 |
| builtin_actor_s | 5 | 5.9517247676849365 | 5.192755222320557 | 6.132843971252441 |
| trainer_max_wall_s | 5 | 5.894836332 | 5.164198162 | 6.084372924 |
| generation_pause_upper_bound_s | 5 | 3.255587207 | 2.981856777 | 3.812172841 |
| trainer_cpu_sum_s | 5 | 15.766396908 | 15.469122472999999 | 21.445608288 |
| trainer_cpu_max_s | 5 | 9.385173152 | 9.320973243 | 13.623246557 |
| receiver_total_cpu_s | 5 | 43.041944528 | 38.791682486 | 44.845692641 |
| receiver_scheduler_cpu_s | 5 | 31.199378855 | 28.442788097 | 32.699917637 |
| receiver_auxiliary_cpu_s | 5 | 11.842565673 | 10.348894389 | 12.145775004 |
| driver_cpu_s | 5 | 0.043579739 | 0.03231688 | 0.047804055 |
| trainer_user_cpu_sum_s | 5 | 13.56858299999999 | 13.347278999999958 | 15.602979000000005 |
| trainer_system_cpu_sum_s | 5 | 2.1978060000000426 | 2.112934999999993 | 5.842624000000029 |
| trainer_minor_faults | 5 | 34402 | 13897 | 2932338 |
| trainer_major_faults | 5 | 0 | 0 | 0 |
| trainer_voluntary_switches | 5 | 53475 | 51261 | 54161 |
| trainer_involuntary_switches | 5 | 34 | 12 | 46 |
| receiver_snapshot_cost_s | 5 | 0.001411671 | 0.00129769 | 0.003276566 |
| observer_outside_trainer_wall_max_s | 5 | 0.061537434 | 0.059277825 | 0.079683064 |

Gradient norms: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`. Unmasked advantage mean absolute values: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`. Replay: 1464 checks, 0 with mismatches; counts repeat across ranks/stages/layers.

Snapshot cost / driver wall, raw steady ratios: `[0.00023400617492504847, 0.0005499699948021203, 0.0002496635387538466, 0.00024092365043765004, 0.0002127088886739705]`. Outside-trainer-wall bookkeeping max / driver wall: `[0.00982621098409361, 0.01337476318007848, 0.01170485135103337, 0.010393721936122619, 0.012822960176918913]`. These are observation costs, not subtracted corrections. Invalid initial/first-post CPU indices: `[]`.

Warning/error counts by log stage: `{'startup': 128, 'training': 76}`. Readiness retries: `{'retry_until_deadline': 17, 'wait_expected_num_cells': 32}`. Full line-numbered warnings/errors, CPU setup calibration, per-process clocks, and raw resource usage counters remain in the retained artifacts and JSON.

Receiver source check: Configured SGLang checkout present in process PYTHONPATH; commands retained in inventory. Actual receiver loaded-module files were not independently sampled. CPU quota observation: CPU quota files not exposed at probed cgroup paths; cgroup membership and affinity retained, quota not independently verified.


## cpu-v2-synthetic-balanced-broadcast-01

Exit0 and final Ray success; seven rollouts, 28 normal attempt0 trainer returns; seven raw updates including initial sync. Steady samples are indices2–6.

| Update | Driver s | Builtin actor s | Trainer CPU s | Receiver CPU s | Scheduler CPU s | Auxiliary CPU s | Pause s | Changed bytes | Wire bytes | CPU valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 17.637189671 | 17.555196046829224 | 68.329116753 | 100.715674424 | 65.993567285 | 34.722107139 | 17.249660386 | — | — | True |
| 1 | 1.409174562 | 1.4038469791412354 | 4.0561591 | 7.646010845 | 5.03361538 | 2.612395465 | 1.267226297 | — | — | True |
| 2 | 1.39087803 | 1.3857004642486572 | 4.017955261 | 7.869144568 | 5.192342946 | 2.676801622 | 1.301213026 | — | — | True |
| 3 | 1.787202798 | 1.7822566032409668 | 5.309132592 | 10.549256524 | 6.659390511 | 3.889866013 | 1.697737651 | — | — | True |
| 4 | 1.346844507 | 1.3419957160949707 | 4.026750893 | 7.567903032 | 5.003141864 | 2.564761168 | 1.256781167 | — | — | True |
| 5 | 1.340206148 | 1.3348963260650635 | 4.011894221 | 7.567369506 | 5.015866518 | 2.551502988 | 1.24804142 | — | — | True |
| 6 | 1.357924333 | 1.3526790142059326 | 4.084551345 | 7.651091828 | 5.051480995 | 2.599610833 | 1.267533657 | — | — | True |

Steady medians and ranges:

| Metric | N | Median | Min | Max |
| --- | --- | --- | --- | --- |
| driver_wall_s | 5 | 1.357924333 | 1.340206148 | 1.787202798 |
| builtin_actor_s | 5 | 1.3526790142059326 | 1.3348963260650635 | 1.7822566032409668 |
| trainer_max_wall_s | 5 | 1.309907198 | 1.289739435 | 1.737288781 |
| generation_pause_upper_bound_s | 5 | 1.267533657 | 1.24804142 | 1.697737651 |
| trainer_cpu_sum_s | 5 | 4.026750893 | 4.011894221 | 5.309132592 |
| trainer_cpu_max_s | 5 | 1.076006927 | 1.071213831 | 1.5022826 |
| receiver_total_cpu_s | 5 | 7.651091828 | 7.567369506 | 10.549256524 |
| receiver_scheduler_cpu_s | 5 | 5.051480995 | 5.003141864 | 6.659390511 |
| receiver_auxiliary_cpu_s | 5 | 2.599610833 | 2.551502988 | 3.889866013 |
| driver_cpu_s | 5 | 0.017463761 | 0.015825876 | 0.023349701 |
| trainer_user_cpu_sum_s | 5 | 3.8964670000000012 | 3.84251299999994 | 5.123585999999932 |
| trainer_system_cpu_sum_s | 5 | 0.16938500000003387 | 0.12149700000000507 | 0.18555300000002717 |
| trainer_minor_faults | 5 | 6528 | 2433 | 20211 |
| trainer_major_faults | 5 | 0 | 0 | 0 |
| trainer_voluntary_switches | 5 | 5719 | 5571 | 7175 |
| trainer_involuntary_switches | 5 | 13 | 11 | 18 |
| receiver_snapshot_cost_s | 5 | 0.000989111 | 0.000916586 | 0.00100516 |
| observer_outside_trainer_wall_max_s | 5 | 0.078231261 | 0.078126576 | 0.080459363 |

Gradient norms: `[2.7134013175964355, 2.423283576965332, 2.027893304824829, 2.0274155139923096, 2.3340871334075928, 2.0304176807403564, 2.3436272144317627]`. Unmasked advantage mean absolute values: `[0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321]`. Replay: 1464 checks, 0 with mismatches; counts repeat across ranks/stages/layers.

Snapshot cost / driver wall, raw steady ratios: `[0.0007160059893964965, 0.0005624207846612827, 0.0007343913828651046, 0.0006840014884038571, 0.0006749904819622966]`. Outside-trainer-wall bookkeeping max / driver wall: `[0.05617068809405236, 0.04374207509493839, 0.058611174185083575, 0.06003506484436751, 0.05761091328790557]`. These are observation costs, not subtracted corrections. Invalid initial/first-post CPU indices: `[]`.

Warning/error counts by log stage: `{'startup': 124, 'training': 70}`. Readiness retries: `{'retry_until_deadline': 17, 'wait_expected_num_cells': 28}`. Full line-numbered warnings/errors, CPU setup calibration, per-process clocks, and raw resource usage counters remain in the retained artifacts and JSON.

Receiver source check: Configured SGLang checkout present in process PYTHONPATH; commands retained in inventory. Actual receiver loaded-module files were not independently sampled. CPU quota observation: CPU quota files not exposed at probed cgroup paths; cgroup membership and affinity retained, quota not independently verified.


## cpu-v2-synthetic-balanced-disk-delta-01

Exit0 and final Ray success; seven rollouts, 28 normal attempt0 trainer returns; seven raw updates including initial sync. Steady samples are indices2–6.

| Update | Driver s | Builtin actor s | Trainer CPU s | Receiver CPU s | Scheduler CPU s | Auxiliary CPU s | Pause s | Changed bytes | Wire bytes | CPU valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 21.467089816 | 21.39485812187195 | 79.188084678 | 83.695668378 | 41.017119996 | 42.678548382 | — | — | — | True |
| 1 | 31.784991194 | 31.77933359146118 | 76.570439584 | 260.640006505 | 197.056753051 | 63.583253454 | 3.163173266 | 90305369 | 204772440 | True |
| 2 | 12.948734415 | 12.943434238433838 | 48.104858048 | 145.571528214 | 119.734542681 | 25.836985533 | 3.067691025 | 79961858 | 184276719 | True |
| 3 | 12.982007072 | 12.976326942443848 | 44.063447301 | 144.859401781 | 118.937348513 | 25.922053268 | 3.565661697 | 75947006 | 176770775 | True |
| 4 | 12.920405952 | 12.91453766822815 | 45.266450567 | 154.145447389 | 128.37677129 | 25.768676099 | 2.953697278 | 64444569 | 154332617 | True |
| 5 | 13.197179936 | 13.191802024841309 | 46.902058744 | 148.161714366 | 121.680343359 | 26.481371007 | 3.462969117 | 72426351 | 169199936 | True |
| 6 | 12.859471272 | 12.853264808654785 | 45.646671813 | 146.672678635 | 120.941810622 | 25.730868013 | 3.420274695 | 64661832 | 155250402 | True |

Steady medians and ranges:

| Metric | N | Median | Min | Max |
| --- | --- | --- | --- | --- |
| driver_wall_s | 5 | 12.948734415 | 12.859471272 | 13.197179936 |
| builtin_actor_s | 5 | 12.943434238433838 | 12.853264808654785 | 13.191802024841309 |
| trainer_max_wall_s | 5 | 12.898158838 | 12.805195889 | 13.163580658 |
| generation_pause_upper_bound_s | 5 | 3.420274695 | 2.953697278 | 3.565661697 |
| trainer_cpu_sum_s | 5 | 45.646671813 | 44.063447301 | 48.104858048 |
| trainer_cpu_max_s | 5 | 29.822337512 | 28.877551331 | 31.328232497 |
| receiver_total_cpu_s | 5 | 146.672678635 | 144.859401781 | 154.145447389 |
| receiver_scheduler_cpu_s | 5 | 120.941810622 | 118.937348513 | 128.37677129 |
| receiver_auxiliary_cpu_s | 5 | 25.836985533 | 25.730868013 | 26.481371007 |
| driver_cpu_s | 5 | 0.073974408 | 0.065658256 | 0.090932981 |
| trainer_user_cpu_sum_s | 5 | 40.516272999999956 | 37.614146999999946 | 40.80324300000001 |
| trainer_system_cpu_sum_s | 5 | 6.1232160000000135 | 5.130381999999997 | 7.301607999999987 |
| trainer_minor_faults | 5 | 740497 | 407477 | 1056309 |
| trainer_major_faults | 5 | 0 | 0 | 0 |
| trainer_voluntary_switches | 5 | 100946 | 97959 | 102795 |
| trainer_involuntary_switches | 5 | 59 | 46 | 72 |
| receiver_snapshot_cost_s | 5 | 0.001506363 | 0.001414894 | 0.001577283 |
| observer_outside_trainer_wall_max_s | 5 | 0.074887187 | 0.059512918 | 0.079820628 |

Gradient norms: `[2.770756244659424, 2.4277002811431885, 2.3706765174865723, 2.0085737705230713, 2.0201711654663086, 2.2982523441314697, 2.5479576587677]`. Unmasked advantage mean absolute values: `[0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321]`. Replay: 1464 checks, 0 with mismatches; counts repeat across ranks/stages/layers.

Snapshot cost / driver wall, raw steady ratios: `[0.00011101154398030026, 0.00012149762292164618, 0.00012031905234056226, 0.00010721184426230136, 0.00011714035267374715]`. Outside-trainer-wall bookkeeping max / driver wall: `[0.006164357491766503, 0.0057685369130262626, 0.005845854323819314, 0.0045095178127910005, 0.004753256623628934]`. These are observation costs, not subtracted corrections. Invalid initial/first-post CPU indices: `[]`.

Warning/error counts by log stage: `{'startup': 129, 'training': 74}`. Readiness retries: `{'retry_until_deadline': 16, 'wait_expected_num_cells': 34}`. Full line-numbered warnings/errors, CPU setup calibration, per-process clocks, and raw resource usage counters remain in the retained artifacts and JSON.

Receiver source check: Configured SGLang checkout present in process PYTHONPATH; commands retained in inventory. Actual receiver loaded-module files were not independently sampled. CPU quota observation: CPU quota files not exposed at probed cgroup paths; cgroup membership and affinity retained, quota not independently verified.

## Boundaries

- One C2 node, eight B300 GPUs, 4+4 split with two TP2 rollout engines; local shared storage. No cross-node or cross-cluster claim.
- This is a correctness and local overhead benchmark for future large multi-node use. No cross-node speedup or amortization projection is made; beating local broadcast is not an admission requirement.
- Synthetic balanced reward is an explicit transport-validation workload, not learned-task quality. Every synthetic rollout must have nonzero gradient norm and nonzero absolute unmasked advantages; every post-training delta must contain nonzero changed bytes.
- Trainer CPU uses all-thread process clocks inside each updater. Receiver CPU brackets trainer-rank0 updater; auxiliary includes data-parallel controllers and inventoried compile workers. Driver CPU is separate. These windows differ slightly and should not be treated as an exact simultaneous whole-host total.
- Driver wall and builtin actor time include observer discovery/validation and trainer CPU journal flushing. Trainer wall excludes outer discovery/validation/flush but includes CPU snapshots. Actual refresh/snapshot costs are shown; no guessed overhead is subtracted. Other waiting ranks can experience observer-induced synchronization delay.
- Cached clock-read calibration does not bound membership-refresh cost. Runtime bookkeeping ratios are reported separately; a large ratio limits latency interpretation. Previous journal flush cost is lagged, not an exact same-update correction.
- Final idle-engine calibration costs are retained above. Discovery is material for short local broadcast updates; it is included in driver wall, not called negligible or subtracted.
- Receiver trees are checked before and after each update using live, non-atomic /proc child lists. A missing previously known PID still alive with the same start time invalidates the scan. Entirely transient children, or newly born children omitted during concurrent exits, may remain unobserved. Receiver source is verified as the configured launch/PYTHONPATH checkout, not independently attested loaded-module files. CPU quota may be unavailable in the container; observed cgroup metadata and per-role affinity/thread settings are compared.
- Five steady samples are update indices2..6. Initial sync and first post-training update remain in raw tables but not steady medians. Invalid early CPU observations are disclosed; any invalid steady CPU observation blocks this report.
- Delta wire bytes are serialized safetensor lengths, excluding index JSON and baseline checkpoint copy; they are not measured physical network bytes. Broadcast changed/wire bytes are unavailable, not zero.
- Version acknowledgements, byte checksums, replay, and subsequent successful training do not establish active GPU byte equality or exact numerical equivalence. Original recipe weight/logprob/KL CI checkers remain disabled.
- The earlier v1 campaign lacks CPU counters. This report compares candidate synthetic delta against candidate synthetic broadcast; it does not establish before/after E2E CPU savings. CPU-only component replay remains separately labeled.

CPU-only sender/receiver component replay is separate evidence and is not combined into these E2E ratios.
