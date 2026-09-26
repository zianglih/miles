# Miles #3720 hosted CI at 6fb0e2a

Hosted CI did **not** pass. Stage-a CPU shard 1 terminated with exit 250 during
`test_actor_inference_allocations_match_warmed_worker_dispatch[entries1-4]`.
The preceding `[entries0-64]` case passed. There is no pytest failure traceback or
summary. The runner wrapper could translate a child SIGABRT (-6) to status 250,
but no raw signal/core artifact confirms that cause. Treat it as unresolved.

Pre-commit, stage-a shards 0/2/3 and stage-b CPU passed. All GPU stages were
explicitly skipped by resolved regular policy; no hosted GPU validation occurred.
ROCm's workflow-level success covers its setup/policy jobs, not a GPU test.

The failed job used Ubuntu 24.04 image 20260920.314.1, Python 3.11.16, Torch 2.11.0,
NumPy 2.4.6 and uv 0.12.19. The C2 focused tests used a different Torch build.
The workflow uploaded no artifacts; check annotations only repeat exit 250.

`latest.json` retains complete final run/job statuses; `summary.json` is compact.
`final-review.json` records real failures, skips, actual test summaries and limits.
Full `gh run view --log` output is retained, plus the failed shard subset and job
metadata/annotations. Logs contain the PR body as an environment value, including
historical C2 test output; those untimestamped embedded results must not be counted
as hosted test passes. `SHA256.json` indexes the retained evidence.

No manual reruns, comments, production edits or C2 calls were performed.
