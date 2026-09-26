# GLM-5.2 NVFP4 W4A16 weight synchronization benchmark

All four retained arms passed the report's completion and source/configuration checks. Native and synthetic-balanced reward arms are separate workloads. Ratios are delta / broadcast median time: below 1 means delta took less time.

Environment: c2 / hu-pdx-90 / glm52-delta-0925; one node, eight NVIDIA B300 GPUs, 4 trainer + 4 rollout GPUs (two 2-GPU engines). Image `radixark/miles:dev-202609251434`; image-index digest `sha256:abfd1b58539bef057717928a7ce115970fe6854bea82c958c2fe8e18e3014ced`; amd64 digest `sha256:7c4c6c8cc9e76be893941e064f0e43924ef9258b04da365c8009709592ff0984`.

Tested sources: Miles `81d43908021deaf916808675d7ac78e4fbccfa2f`; SGLang `e3ec949a06323adf4032d35df288c5658c3984bb`; Megatron `f148a32b4385b758b66a77c9c3ad1641f1295d4b`. Both tested Miles/SGLang tracked diffs are empty. HF source `Pinaster/GLM-5.2_5layer` at `1c749139f70e158e4420ba67f342bef1de2e650d`.

Container package metadata: torch `2.13.0+cu130`; triton `3.7.1`; transformer-engine `2.17.0`; flashinfer-python `0.6.18`; flashinfer-cubin `0.6.18`; nvidia-cutlass-dsl `4.6.2`; sglang `0.5.21.dev60+g880e3d2`; ray `2.58.0`; safetensors `0.8.0`; zstandard `0.25.0`; xxhash `3.7.1`. Miles/SGLang run from the tested source overrides above, rather than the container's baseline source pins.

Each arm has seven rollouts, six post-training updates, and five steady samples (update indices 2–6). Initial sync and the first post-training update are excluded from steady statistics. All samples, including slow ones, are retained. Seeds, checkpoint paths, prompts, precision settings, batch sizes, and observation helpers match within each pair; only transport/output options differ.

## Steady timing comparison

| Reward | Metric (s) | Broadcast median | Broadcast min–max | Delta median | Delta min–max | Delta / broadcast |
| --- | --- | --- | --- | --- | --- | --- |
| native | driver_wall_s | 1.279045859 | 1.278445526–1.367027581 | 6.615738117 | 6.454698969–6.937701596 | 5.172401028820343 |
| native | builtin_actor_s | 1.271005630493164 | 1.2707023620605469–1.3588969707489014 | 6.609894752502441 | 6.447613477706909–6.931683778762817 | 5.200523580637271 |
| native | trainer_max_wall_s | 1.267740132 | 1.26737209–1.355028753 | 6.603545004 | 6.443049675–6.925990865 | 5.208910593989148 |
| native | generation_pause_upper_bound_s | 1.266733841 | 1.266324899–1.353951303 | 3.408622779 | 3.119624202–3.554552211 | 2.690875279931832 |
| synthetic-balanced | driver_wall_s | 1.271644968 | 1.260217968–1.737577933 | 13.924335042 | 13.389641909–14.268226971 | 10.949860528996329 |
| synthetic-balanced | builtin_actor_s | 1.2649214267730713 | 1.2517056465148926–1.7282910346984863 | 13.919214725494385 | 13.384368658065796–14.262575626373291 | 11.004015293664175 |
| synthetic-balanced | trainer_max_wall_s | 1.261573798 | 1.248620204–1.725515947 | 13.915731814 | 13.379620114–14.258443852 | 11.030454053548757 |
| synthetic-balanced | generation_pause_upper_bound_s | 1.260436728 | 1.247560357–1.724485113 | 3.525939937 | 3.131973534–3.663615843 | 2.7973954254687508 |

## Raw updates and training evidence

### native-broadcast-01

Exit 0; final Ray success; 28 normal train-step returns across seven rollouts × four ranks, all attempt 0; seven successful updater returns on every rank; six post-training and five steady samples.

| Update | After train rollout | Driver s | Built-in actor s | Trainer max s | Pause upper bound s | Pull RPC max s | Reload RPC max s | Changed bytes | Total bytes | Delta wire bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | — | 17.318400992 | 17.210155248641968 | 17.136784499 | 17.135682224 | — | — | — | — | — |
| 1 | 0 | 1.266493022 | 1.2585723400115967 | 1.249748768 | 1.248563038 | — | — | — | — | — |
| 2 | 1 | 1.278445526 | 1.2708542346954346 | 1.26737209 | 1.266324899 | — | — | — | — | — |
| 3 | 2 | 1.279045859 | 1.2707023620605469 | 1.267376831 | 1.266393668 | — | — | — | — | — |
| 4 | 3 | 1.281121685 | 1.273667573928833 | 1.270216997 | 1.269116702 | — | — | — | — | — |
| 5 | 4 | 1.278696248 | 1.271005630493164 | 1.267740132 | 1.266733841 | — | — | — | — | — |
| 6 | 5 | 1.367027581 | 1.3588969707489014 | 1.355028753 | 1.353951303 | — | — | — | — | — |

Gradient norms: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`. Unmasked advantage mean absolute values: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`; nonzero counts: `[0, 0, 0, 0, 0, 0, 0]`. Trainable response-token counts: `[6400, 6400, 6400, 6400, 6400, 6400, 6400]`.

Replay: 1464 checks; 0 checks with mismatches; 0 mismatched token comparisons. These repeat across layers/stages/ranks. Per-rollout reward/loss/KL/logprob, version, and debug statistics remain in BENCHMARK_RESULTS.json.

Warning/error inventory by log stage: `{'startup': 120, 'training': 74, 'teardown': 3}`. Startup readiness retries: `{'retry_until_deadline': 15, 'wait_expected_num_cells': 26}`. `freeze_gc` failure lines: `[2500, 2691]`. Full line-numbered messages are retained in the JSON; successful completion does not mean an error-free log.

Local evidence: `/Users/ziangli/playground/projects/glm52-w4a16-delta-sync-c2/artifacts/native-broadcast-01`. All source manifests, observations, final logs/exits, and 14 debug dumps are SHA256-indexed in the JSON.

### native-disk-delta-01

Exit 0; final Ray success; 28 normal train-step returns across seven rollouts × four ranks, all attempt 0; seven successful updater returns on every rank; six post-training and five steady samples.

| Update | After train rollout | Driver s | Built-in actor s | Trainer max s | Pause upper bound s | Pull RPC max s | Reload RPC max s | Changed bytes | Total bytes | Delta wire bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | — | 18.173466532 | 18.06361722946167 | 18.055196701 | — | 10.107072226 | — | — | — | — |
| 1 | 0 | 24.984287577 | 24.975753784179688 | 24.972286338 | 3.393843204 | 0.021358044 | 3.3743455 | 0 | 17900804608 | 0 |
| 2 | 1 | 6.454698969 | 6.447613477706909 | 6.443049675 | 3.119624202 | 0.020893752 | 3.104569118 | 0 | 17900804608 | 0 |
| 3 | 2 | 6.937701596 | 6.931683778762817 | 6.925990865 | 3.554552211 | 0.02178868 | 3.538891406 | 0 | 17900804608 | 0 |
| 4 | 3 | 6.615738117 | 6.609894752502441 | 6.603545004 | 3.276342037 | 0.021305979 | 3.261083966 | 0 | 17900804608 | 0 |
| 5 | 4 | 6.535530022 | 6.5295984745025635 | 6.524371476 | 3.408622779 | 0.0155214 | 3.393141174 | 0 | 17900804608 | 0 |
| 6 | 5 | 6.90629906 | 6.897026538848877 | 6.893651117 | 3.545309097 | 0.022675187 | 3.529202385 | 0 | 17900804608 | 0 |

Gradient norms: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`. Unmasked advantage mean absolute values: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`; nonzero counts: `[0, 0, 0, 0, 0, 0, 0]`. Trainable response-token counts: `[6400, 6400, 6400, 6400, 6400, 6400, 6400]`.

Replay: 1464 checks; 0 checks with mismatches; 0 mismatched token comparisons. These repeat across layers/stages/ranks. Per-rollout reward/loss/KL/logprob, version, and debug statistics remain in BENCHMARK_RESULTS.json.

Exact steady delta counters: every update changed bytes = **False**; every update zero changed bytes = **True**. Per-update density: `[None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`.

Warning/error inventory by log stage: `{'startup': 122, 'training': 76}`. Startup readiness retries: `{'retry_until_deadline': 15, 'wait_expected_num_cells': 28}`. `freeze_gc` failure lines: `[2633, 2703]`. Full line-numbered messages are retained in the JSON; successful completion does not mean an error-free log.

Local evidence: `/Users/ziangli/playground/projects/glm52-w4a16-delta-sync-c2/artifacts/native-disk-delta-01`. All source manifests, observations, final logs/exits, and 14 debug dumps are SHA256-indexed in the JSON.

### synthetic-balanced-broadcast-01

Exit 0; final Ray success; 28 normal train-step returns across seven rollouts × four ranks, all attempt 0; seven successful updater returns on every rank; six post-training and five steady samples.

| Update | After train rollout | Driver s | Built-in actor s | Trainer max s | Pause upper bound s | Pull RPC max s | Reload RPC max s | Changed bytes | Total bytes | Delta wire bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | — | 17.368027898 | 17.252811193466187 | 17.1710864 | 17.169655978 | — | — | — | — | — |
| 1 | 0 | 1.278332511 | 1.270350456237793 | 1.266076565 | 1.264881299 | — | — | — | — | — |
| 2 | 1 | 1.271644968 | 1.2649214267730713 | 1.261573798 | 1.260436728 | — | — | — | — | — |
| 3 | 2 | 1.31585837 | 1.308382511138916 | 1.302805633 | 1.301777017 | — | — | — | — | — |
| 4 | 3 | 1.737577933 | 1.7282910346984863 | 1.725515947 | 1.724485113 | — | — | — | — | — |
| 5 | 4 | 1.260217968 | 1.2517056465148926 | 1.248620204 | 1.247560357 | — | — | — | — | — |
| 6 | 5 | 1.262003201 | 1.2528681755065918 | 1.249683967 | 1.248643007 | — | — | — | — | — |

Gradient norms: `[2.4921531677246094, 2.3972816467285156, 2.451223611831665, 2.412675380706787, 1.9836899042129517, 1.9878116846084595, 2.0134193897247314]`. Unmasked advantage mean absolute values: `[0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321]`; nonzero counts: `[6400, 6400, 6400, 6400, 6400, 6400, 6400]`. Trainable response-token counts: `[6400, 6400, 6400, 6400, 6400, 6400, 6400]`.

Replay: 1464 checks; 0 checks with mismatches; 0 mismatched token comparisons. These repeat across layers/stages/ranks. Per-rollout reward/loss/KL/logprob, version, and debug statistics remain in BENCHMARK_RESULTS.json.

Warning/error inventory by log stage: `{'startup': 122, 'training': 76, 'teardown': 3}`. Startup readiness retries: `{'retry_until_deadline': 16, 'wait_expected_num_cells': 27}`. `freeze_gc` failure lines: `[2622, 2703]`. Full line-numbered messages are retained in the JSON; successful completion does not mean an error-free log.

Local evidence: `/Users/ziangli/playground/projects/glm52-w4a16-delta-sync-c2/artifacts/synthetic-balanced-broadcast-01`. All source manifests, observations, final logs/exits, and 14 debug dumps are SHA256-indexed in the JSON.

### synthetic-balanced-disk-delta-01

Exit 0; final Ray success; 28 normal train-step returns across seven rollouts × four ranks, all attempt 0; seven successful updater returns on every rank; six post-training and five steady samples.

| Update | After train rollout | Driver s | Built-in actor s | Trainer max s | Pause upper bound s | Pull RPC max s | Reload RPC max s | Changed bytes | Total bytes | Delta wire bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | — | 16.249907501 | 16.146458625793457 | 16.140010815 | — | 10.931100061 | — | — | — | — |
| 1 | 0 | 31.42165691 | 31.415995836257935 | 31.412394865 | 3.225430794 | 3.398678787 | 3.208752235 | 90329011 | 17900804608 | 204800625 |
| 2 | 1 | 13.389641909 | 13.384368658065796 | 13.379620114 | 3.131973534 | 4.063482558 | 3.117006402 | 79500803 | 17900804608 | 183308084 |
| 3 | 2 | 13.924335042 | 13.919214725494385 | 13.915731814 | 3.619206647 | 3.925403147 | 3.60460498 | 75413963 | 17900804608 | 175793256 |
| 4 | 3 | 13.489594715 | 13.483909130096436 | 13.480548346 | 3.169787832 | 4.144839014 | 3.154461589 | 63946640 | 17900804608 | 153285636 |
| 5 | 4 | 14.268226971 | 14.262575626373291 | 14.258443852 | 3.525939937 | 4.670654627 | 3.510168071 | 71351139 | 17900804608 | 167146269 |
| 6 | 5 | 14.001918526 | 13.996295928955078 | 13.99208762 | 3.663615843 | 4.189341177 | 3.646960398 | 63599301 | 17900804608 | 153284995 |

Gradient norms: `[2.7867448329925537, 2.7648022174835205, 2.454990863800049, 2.377612590789795, 2.3417630195617676, 2.344681739807129, 2.0527284145355225]`. Unmasked advantage mean absolute values: `[0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321, 0.935412585735321]`; nonzero counts: `[6400, 6400, 6400, 6400, 6400, 6400, 6400]`. Trainable response-token counts: `[6400, 6400, 6400, 6400, 6400, 6400, 6400]`.

Replay: 1464 checks; 0 checks with mismatches; 0 mismatched token comparisons. These repeat across layers/stages/ranks. Per-rollout reward/loss/KL/logprob, version, and debug statistics remain in BENCHMARK_RESULTS.json.

Exact steady delta counters: every update changed bytes = **True**; every update zero changed bytes = **False**. Per-update density: `[None, 0.005046086641246914, 0.0044411860104026, 0.004212881188943705, 0.0035722774143583345, 0.0039859179831566146, 0.0035528738731429427]`.

Warning/error inventory by log stage: `{'startup': 123, 'training': 74}`. Startup readiness retries: `{'retry_until_deadline': 15, 'wait_expected_num_cells': 29}`. `freeze_gc` failure lines: `[2614, 2703]`. Full line-numbered messages are retained in the JSON; successful completion does not mean an error-free log.

Local evidence: `/Users/ziangli/playground/projects/glm52-w4a16-delta-sync-c2/artifacts/synthetic-balanced-disk-delta-01`. All source manifests, observations, final logs/exits, and 14 debug dumps are SHA256-indexed in the JSON.

## Steady phase timings

Each cell is the per-update maximum over ranks or engines, then summarized across five steady updates. Phases and RPCs can overlap/nest; do not add these medians to reconstruct total time.

| Run | Phase (s) | N | Median | Min | Max |
| --- | --- | --- | --- | --- | --- |
| native-broadcast-01 | begin_weight_update_max_rpc_s | 5 | 0.003661558 | 0.003177301 | 0.004168376 |
| native-broadcast-01 | end_weight_update_max_rpc_s | 5 | 0.016880681 | 0.016770785 | 0.017506842 |
| native-disk-delta-01 | pull_weights_max_rpc_s | 5 | 0.021305979 | 0.0155214 | 0.022675187 |
| native-disk-delta-01 | update_weights_from_disk_max_rpc_s | 5 | 3.393141174 | 3.104569118 | 3.538891406 |
| native-disk-delta-01 | delta_begin_encode_max_s | 5 | 0.003165917 | 0.002064847 | 0.003557771 |
| native-disk-delta-01 | deltaafter_base_weights_max_s | 5 | 0.191340453 | 0.16151158 | 0.202576137 |
| native-disk-delta-01 | delta_write_delta_files_max_s | 5 | 0.005018598 | 0.004654761 | 0.009196634 |
| native-disk-delta-01 | delta_reload_engines_max_s | 5 | 3.42560259 | 3.141934835 | 3.577914128 |
| synthetic-balanced-broadcast-01 | begin_weight_update_max_rpc_s | 5 | 0.003616575 | 0.003287793 | 0.003983547 |
| synthetic-balanced-broadcast-01 | end_weight_update_max_rpc_s | 5 | 0.016857649 | 0.016509074 | 0.016995992 |
| synthetic-balanced-disk-delta-01 | pull_weights_max_rpc_s | 5 | 4.144839014 | 3.925403147 | 4.670654627 |
| synthetic-balanced-disk-delta-01 | update_weights_from_disk_max_rpc_s | 5 | 3.510168071 | 3.117006402 | 3.646960398 |
| synthetic-balanced-disk-delta-01 | delta_begin_encode_max_s | 5 | 0.017259642 | 0.014583777 | 0.024863809 |
| synthetic-balanced-disk-delta-01 | deltaafter_base_weights_max_s | 5 | 0.234907515 | 0.213977245 | 0.261273737 |
| synthetic-balanced-disk-delta-01 | delta_write_delta_files_max_s | 5 | 0.433624682 | 0.38925682 | 0.468738918 |
| synthetic-balanced-disk-delta-01 | delta_reload_engines_max_s | 5 | 7.553640626 | 7.206047198 | 8.205136592 |

## Scope and limitations

- Single C2 node, local /hai-workspace storage, two rollout engines. This is not a cross-node or cross-cluster bandwidth result; runs are sequential in the recorded campaign order, without randomized order or replicate campaigns.
- Native deepscaler and synthetic-balanced reward are separate workloads. Synthetic reward assigns sample-index parity 0/1 and uses ordinary GRPO/optimizer updates; it is transport validation, not learned-task quality. Actual nonzero changes require exact delta counters, not gradient norm alone.
- Driver wall time includes actor dispatch and rollout-version handoff. Built-in actor time is the raw rank-0 perf/update_weights_time. Trainer max is the maximum rank-local updater wall time. These are distinct boundaries and include no additional global CUDA fence.
- Observation writes inside timed updater phases add overhead; delta emits more observations than broadcast. No estimated observation overhead is subtracted.
- Generation pause spans the first pause dispatch to the last resume response across engines; it is a conservative interval, not a production latency measurement. Phase/RPC timings overlap or nest.
- Initial synchronization and the first post-training synchronization are excluded from five-sample steady statistics. The initial full checkpoint baseline copy and model/engine startup are not steady delta payload cost.
- Changed/total bytes are exact encoder tensor-byte counters summed over four trainer ranks. Delta wire bytes sum serialized safetensor lengths, excluding index JSON and the initial full baseline copy; they are not measured physical network bytes. Broadcast changed/wire bytes are unavailable, not zero.
- Version acknowledgements, checksums, routing replay, and subsequent successful training do not prove active GPU weight byte equality or exact numerical equivalence. The recipe disables weight-update/logprob/KL CI checkers. Replay counts repeat per layer/stage/rank.
- Startup and teardown warnings/errors are retained with line numbers. Missing/incomplete arms, stale logs, train retries/non-normal steps, mixed versions, missing delta counters, or pair identity/configuration mismatches prevent report generation.

## Reproduction and identity

Run each retained manifest's exact train_argv, launch_kwargs and environment; full records and normalized paired configurations are embedded in BENCHMARK_RESULTS.json. The source test is tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py; HARNESS.md documents project observation hooks and preparation.

Report generation:

```bash
python build_benchmark_report.py \
  --campaign-metadata CAMPAIGN.json --environment environment-image.json \
  artifacts/native-broadcast-01 artifacts/native-disk-delta-01 \
  artifacts/synthetic-balanced-broadcast-01 artifacts/synthetic-balanced-disk-delta-01
```
