# GLM-5.2 W4A16 delta synchronization evidence

This fork-only directory contains the observation scripts, source/configuration
manifests, raw timing results, and focused test output used for Miles #3711 and
SGLang #41274. It is separate from their upstream source diffs. The exact source
commits and container image are recorded in each report and campaign manifest;
the fork branch's source tree is not the tested candidate.

- `BENCHMARK_RESULTS.md` / `.json`: completed original four-arm C2 campaign.
- `CPU_BENCHMARK_RESULTS.md` / `.json`, when present: final candidate three-arm
  campaign with process CPU counters. Earlier runs did not measure CPU time.
- `SENDER_REPLAY.md` and `artifacts/sender-cpu-replay-01.jsonl`: separate exact
  worker comparison on identical checkpoint bytes. This excludes GPU/D2H,
  scalar batching, filesystem publication and receiver work.
- `CPU_OBSERVATION_V2.md`: counter boundaries, calibration, limitations and
  strict report-generation commands.
- `SHA256.json`: hashes of the retained files. Model/delta weights and binary
  training debug dumps are retained in the original project, not redistributed
  in this branch. Reports retain their hashes and validated statistics.

## Reproduction layout

Extract this directory into a fresh effort directory. Clone Miles into its
`miles/` child and SGLang into its `sglang/` child; check out the exact revisions
in `CPU_CAMPAIGN.json`. Keep all helper files at the effort root, next to those
clones. Use the explicit image, GPU layout, model/data revisions, preparation
steps and environment in the PR body and `HARNESS.md`. The recipe manages
Ray/SGLang process lifecycle, so run on a dedicated node.

Run the arms sequentially with fresh output directories. `run_cpu_campaign.py`
verifies pinned sources and refuses existing run directories; it does not
perform automatic retries or model preparation. Its absolute default paths
match the recorded C2 workspace `/hai-workspace/glm52-delta`.

These are single-node correctness and local-overhead measurements for an
intended multi-node deployment. They do not measure cross-node performance,
GPU-resident bitwise equality, or learning equivalence. Native zero-gradient
and explicitly synthetic changed-weight runs are separate workloads.
