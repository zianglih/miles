# GLM-5.2 W4A16 paired weight synchronization harness

`launch_weight_sync.py` reuses the upstream E2E recipe, with its native 4 training
GPUs and 4 rollout GPUs (two engines, each TP/DP/EP2). It keeps the original
checkpoint, DAPO math dataset, deepscaler reward, GRPO settings, fake NVFP4 QAT,
BF16 shared experts and W4A16 rollout kernels. No synthetic gradient or model
mutation is introduced. Trainer seed is 1234 and rollout seed is 42 in both arms.

Prepare the BF16, NVFP4 and torch_dist checkpoints and dataset once. This helper
never calls the test's `prepare()`, which deletes its converted checkpoints.
Each run must use a new name; existing run directories are rejected. Use separate
delta publication and rollout checkpoint directories per run, automatically
placed beneath the run directory.

On the dedicated C2 devbox, after copying the helper Python files into
`/hai-workspace/glm52-delta`:

```bash
python /hai-workspace/glm52-delta/launch_weight_sync.py \
  --repo /hai-workspace/glm52-delta/miles \
  --model-dir /hai-workspace/glm52-delta/models \
  --data-dir /hai-workspace/glm52-delta/datasets \
  --output-dir /hai-workspace/glm52-delta/artifacts \
  --mode broadcast --num-rollout 2 --run-name smoke-broadcast

python /hai-workspace/glm52-delta/launch_weight_sync.py \
  --repo /hai-workspace/glm52-delta/miles \
  --model-dir /hai-workspace/glm52-delta/models \
  --data-dir /hai-workspace/glm52-delta/datasets \
  --output-dir /hai-workspace/glm52-delta/artifacts \
  --mode disk-delta --num-rollout 2 --run-name smoke-delta
```

Repeat with six rollouts and new names, such as `measure-broadcast` and
`measure-delta`, for five post-training updates each. The final rollout does not
perform a weight update. For five steady updates after excluding the first
post-training sync, use seven rollouts instead. Capture complete stdout/stderr
logs in the artifact directory. `--render-only` writes a manifest without
starting Ray; it still requires an unused run name.

```bash
python /hai-workspace/glm52-delta/summarize_weight_sync.py \
  /hai-workspace/glm52-delta/artifacts/measure-broadcast \
  /hai-workspace/glm52-delta/artifacts/measure-delta
```

Each run retains:

- `manifest.json`: complete trainer argv, NVFP4/GLM environment, source commit
  and tracked diff, model paths, seed and topology.
- `driver.jsonl`: per-update outer wall time including trainer dispatch and
  rollout version handoff. Initial checkpoint synchronization is identified
  separately and excluded from all post-training distributions.
- `trainer-rankN.jsonl`: per-rank updater wall times, native Miles timers,
  delta phases, exact local changed/total/wire byte counts, and rank-0
  per-engine RPC timings and results.
- `updates.csv` and `summary.json`: every raw update, medians/min/max/means,
  separate initial and post-training/steady distributions, and completeness.

Generation pause is an upper bound from the first pause RPC dispatch to the
last continue RPC completion across the two engines. These synchronous runs
do not exercise overlap with concurrent generation, so this is not a serving
latency or throughput benchmark. Delta wire bytes exclude the JSON index and
one-time full checkpoint baseline materialization. Both arms still perform
their ordinary preprocessing and finalization, which are included in total
weight-update time.

Inspect `changed_bytes` on delta versions 2 and later before making claims
about trained-weight compression. The first delta can include conversion
differences between the canonical HF checkpoint and Megatron's round trip.
The five-layer model may receive only zero deepscaler rewards at response
length 100; zero GRPO advantages and BF16 rounding can then produce no actual
parameter changes. In that case report the original test as an unchanged-weight
case and run a separately labeled material-change validation; do not claim its
tiny deltas represent a learned model update.

## Optional synthetic reward case

`--reward-mode native` is the default and retains deepscaler unchanged. If
native later updates change no bytes, copy `weight_sync_reward.py` beside the
other helper files and explicitly select `--reward-mode synthetic-balanced`
for a separate pair. It assigns `float(sample.index % 2)` through the supported
custom reward hook: four zeros and four ones in each eight-sample prompt group.
This callback overrides deepscaler, while ordinary GRPO, optimizer, learning
rate, checkpoint, precision, seeds and topology remain the same. There are no
direct gradient or weight edits.

```bash
python /hai-workspace/glm52-delta/launch_weight_sync.py \
  --repo /hai-workspace/glm52-delta/miles \
  --model-dir /hai-workspace/glm52-delta/models \
  --data-dir /hai-workspace/glm52-delta/datasets \
  --output-dir /hai-workspace/glm52-delta/artifacts \
  --mode broadcast --num-rollout 7 --run-name synthetic-broadcast-01 \
  --reward-mode synthetic-balanced

python /hai-workspace/glm52-delta/launch_weight_sync.py \
  --repo /hai-workspace/glm52-delta/miles \
  --model-dir /hai-workspace/glm52-delta/models \
  --data-dir /hai-workspace/glm52-delta/datasets \
  --output-dir /hai-workspace/glm52-delta/artifacts \
  --mode disk-delta --num-rollout 7 --run-name synthetic-delta-01 \
  --reward-mode synthetic-balanced
```

The manifest labels this reward mode. Compare only matching reward modes; the
pair validator detects the custom-RM argv difference if a native run is mixed
with a synthetic one. Verify actual gradient norm and version-2+ changed bytes:
nonconstant rewards do not by themselves prove a representable parameter
update. Synthetic compression density is not native math-task training evidence.

## Measurement audit

The trainer wall timer excludes its final record append but includes nested
phase/RPC/native-timer journal writes. Delta writes more observations, so this
is a small, uncalibrated measurement overhead rather than an overhead-free
timing. Do not subtract an assumed cost. Individual RPC/phase end timestamps
are captured before their own record is appended.

`summary.json`'s `complete` field means all expected update samples, four
trainer ranks and both engines were observed successfully. It does not prove
that the final rollout completed: that rollout performs no final sync. Retain
the launcher exit status and complete E2E log as separate run-success evidence.
The paired manifest comparator verifies trainer argv, precision environment,
Miles source, SGLang source, and observation/reward script hashes. Image identity
must also be checked against the campaign/environment records. Older smoke
manifests remain readable; missing receiver/helper identity suppresses a fully
verified paired ratio rather than silently treating missing identity as equal.
