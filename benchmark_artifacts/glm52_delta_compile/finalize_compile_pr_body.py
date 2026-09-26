#!/usr/bin/env python3
"""Build a final PR body only after both GPU arms and evidence publication exist.

No GitHub writes or Git mutations. Historical full reports are linked from an
exact evidence commit; final CPU rows and complete final tests/Gloo remain inline.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import build_compiled_validation_report as gpu_report
from build_materialize_first_history import table
from compile_evidence_attribution import current_hosted_ci, selected_preflight, validate_attribution, validate_compatibility

ROOT = Path(__file__).resolve().parent
PREFIX = "benchmark_artifacts/glm52_delta_compile/"


def raw(path):
    # Preserve CR progress bytes in complete captured test/Gloo output.
    return path.read_bytes().decode("utf-8")


def parse_url(url):
    matched = re.fullmatch(r"https://github\.com/zianglih/miles/tree/([0-9a-f]{40})/benchmark_artifacts/glm52_delta_compile/?", url)
    if not matched:
        raise ValueError("Require the exact 40-hex fork commit tree URL for glm52_delta_compile")
    return matched.group(1), url.rstrip("/")


def publication_links(root, url):
    commit, base = parse_url(url)
    checkout = root / "miles-evidence"

    def committed(name):
        return subprocess.check_output(["git", "-C", str(checkout), "show", f"{commit}:{PREFIX}{name}"])

    manifest = json.loads(committed("SHA256.json"))

    def link(name, label):
        if Path(name).is_absolute() or ".." in Path(name).parts or name not in manifest:
            raise ValueError(f"Evidence file missing from exact published manifest: {name}")
        content = committed(name)
        digest = hashlib.sha256(content).hexdigest()
        if digest != manifest[name]["sha256"] or len(content) != manifest[name]["bytes"]:
            raise ValueError(f"Published manifest/bytes disagree: {name}")
        if (root / name).is_file() and (root / name).read_bytes() != content:
            raise ValueError(f"Local evidence differs from the published bytes: {name}")
        return f"[{label}]({base}/{name})"

    return link


def gpu_section(report, link):
    arms = report["arms"]
    if len(arms) != 2 or {arm["manifest"]["delta_cpu_backend"] for arm in arms.values()} != {"numpy", "torch-compile"}:
        raise ValueError("Final body requires a complete NumPy/compiled pair")
    if sum(len(arm["rows"]) for arm in arms.values()) != 14 or any(
        [row["update_index"] for row in arm["rows"]] != list(range(7)) for arm in arms.values()
    ):
        raise ValueError("Final body requires all 14 update rows")
    lines = ["## Completed paired GLM-5.2 GPU validation", "",
             "Both planned arms passed the strict source/recipe, seven-update, training/routing-replay, CPU-window, "
             "publication and per-rank compiler checks. The paired source is "
             f"`{report['plan']['sources']['miles']}` with receiver `{report['plan']['sources']['sglang']}`. "
             "Both arms use one C2 8-B300 node with a 4 trainer + 4 rollout split: trainer TP4/EP4, PP1/CP1/ETP1; "
             "two rollout engines each TP2/DP2/EP2. This is local single-node validation; no multi-node benefit is inferred.", "",
             "The two synthetic-balanced arms share source, frozen helpers, prepared model/data, seeds and recipe; "
             "only the CPU backend and admitted output/cache paths differ. Each run has an independent empty task-specific "
             "disk cache before launch. Startup u0 and first post-training u1 remain visible; every u2–u6 sample enters "
             "steady medians without removal. NumPy remains the default.", "",
             "Comparison values are steady medians. Ratios are compiled / NumPy. Wall metrics are seconds; CPU metrics "
             "are process CPU-seconds, not utilization or simultaneous whole-host CPU.", ""]
    campaign = report["campaign"]
    metadata_names = {field: report["plan"].get(field, {"path": default})["path"]
                      for field, default in (("campaign_metadata", "CAMPAIGN.json"),
                                             ("environment_metadata", "environment-image.json"))}
    lines += [f"GPU campaign environment: `{campaign['devbox']}` on `{campaign['host']}`, "
              f"image `{campaign['image']}`, index `{campaign['image_index_digest']}`, "
              f"amd64 `{campaign['image_amd64_digest']}`. Exact captured "
              + link(metadata_names["campaign_metadata"], "campaign metadata") + " and "
              + link(metadata_names["environment_metadata"], "runtime metadata")
              + " are pinned by the plan. The CPU replay and historical preflight used the original `hu-pdx-90` "
              "image `radixark/miles:dev-202609251434`; they are not new-image reruns. The selected-head "
              "replacement preflight is labeled separately above.", "",
              report["external_calibration_scope"] + " No historical calibration cost is subtracted from the GPU rows.", "",
              "The original v1 campaign was interrupted when its devbox disappeared at approximately 01:59 UTC "
              "on 2026-09-26. Only three NumPy summaries and its startup manifest were saved; the new campaign "
              "reruns both arms and uses none of those summary-only results. "
              + link("COMPILE_GPU_V1_INTERRUPTION.md", "The interruption record") + " and "
              + link("COMPILE_VALIDATION_PLAN_V1.json", "the original frozen plan") + " preserve that boundary.", ""]
    if "campaign_metadata" in report["plan"]:
        lines += ["Both replacement arms use the same newly prepared inputs, whose full hashes and durable-copy checks "
                  "are retained in " + link("artifacts/recovery-inputs-ready.json", "the input seal") + ". "
                  "All 14 converted shard sizes and serialized headers match the retained original 6,226-tensor "
                  "checkpoint evidence, but the original full converted payload checksums were not retained. "
                  + link("artifacts/recovery-nvfp4-header-verification.json", "The header comparison") + " records that limit. "
                  "This matched v2 NumPy/compiled comparison is separate from the original broadcast/delta campaign "
                  "on a different host/image; no cross-campaign ratio is computed.", ""]
    metric_names = (
        "driver_wall_s", "builtin_actor_s", "trainer_max_wall_s", "generation_pause_upper_bound_s",
        "trainer_cpu_sum_s", "trainer_cpu_max_s", "receiver_total_cpu_s", "driver_cpu_s",
    )
    for reward, pair in report["backend_comparisons"].items():
        lines += [f"**{reward}: `{pair['numpy']}` versus `{pair['torch_compile']}`**", ""]
        lines += table(["Metric", "NumPy median", "Compiled median", "Ratio"], [
            [metric, pair["metrics"][metric]["numpy"]["median"], pair["metrics"][metric]["torch_compile"]["median"],
             pair["metrics"][metric]["compiled_over_numpy"]] for metric in metric_names])
        wall_change = 100 * (pair["metrics"]["driver_wall_s"]["compiled_over_numpy"] - 1)
        cpu_change = 100 * (pair["metrics"]["trainer_cpu_sum_s"]["compiled_over_numpy"] - 1)
        lines += ["", f"The observed driver wall change is **{wall_change:+.2f}%** and trainer CPU-sum change is "
                  f"**{cpu_change:+.2f}%**. Receiver CPU is accumulated over rank-0's full update window, which "
                  "differs between arms; it is not an isolated measurement of receiver apply work or evidence "
                  "that the unchanged receiver algorithm became slower. The generation-pause upper bound is "
                  "reported separately. Changed/wire bytes and gradient trajectories differ between arms, so "
                  "this is a matched recipe/source comparison, not an identical byte-workload replay or a pure "
                  "causal estimate of kernel overhead. The results do not support replacing the NumPy default.", ""]
    if not report["backend_comparisons"]:
        raise ValueError("Final body requires a strictly matched backend comparison")
    lines += ["", "All 14 raw update rows follow. Changed bytes and serialized shard wire bytes are integers. "
              "Complete trainer-rank observations and checksum metadata remain in the exact-commit report.", ""]
    for name, arm in arms.items():
        lines += [f"**{name} — `{arm['manifest']['delta_cpu_backend']}`**", ""]
        lines += table(["Update", "Driver s", "Actor s", "Trainer CPU sum s", "Trainer CPU max s", "Receiver CPU s", "Driver CPU s", "Changed bytes", "Wire bytes", "CPU valid"], [
            [row[key] for key in ("update_index", "driver_wall_s", "builtin_actor_s", "trainer_cpu_sum_s", "trainer_cpu_max_s",
                                  "receiver_total_cpu_s", "driver_cpu_s", "changed_bytes", "wire_bytes", "cpu_valid")]
            for row in arm["rows"]])
        steady = [row for row in arm["compiler_observations"] if row["measured_steady"]]
        if len(arm["compiler_observations"]) != 28 or len(steady) != 20 or any(row["graph_or_kernel_activity"] for row in steady):
            raise ValueError("Missing compiler observations or measured compiler activity")
        lines += ["", f"All 20 steady per-rank compiler observations have zero new graph/cache/kernel activity. "
                  f"Gradient norms: `{arm['training']['gradient_norms']}`. Routing replay checks: "
                  f"`{arm['training']['replay_checks']}`, mismatches: `{arm['training']['replay_nonzero_checks']}`. "
                  f"Early CPU-invalid indices (retained): `{arm['cpu_observer']['initial_or_first_post_cpu_invalid']}`.", "",
                  "Full " + link(f"artifacts/{name}/manifest.json", "run manifest") + ", "
                  + link(f"artifacts/{name}.log", "runtime log") + ", "
                  + link(f"artifacts/{name}/cpu-summary.json", "raw CPU summary") + ", and "
                  + link(f"artifacts/{name}/training-evidence.json", "training evidence") + ".", ""]
    lines += ["Complete " + link("COMPILED_VALIDATION_RESULTS.md", "validated GPU report") + " and "
              + link("COMPILED_VALIDATION_RESULTS.json", "raw report JSON")
              + " retain all 56 rank observations, including initial/u1 compiler activity, exact runtime/source/helper "
              "hashes, process inventories, observed warnings and observer-cost ratios. No observer cost is subtracted.", "",
              "Reproduction uses " + link("COMPILE_VALIDATION_PLAN.json", "the frozen plan") + " with "
              + link("run_compile_campaign.py", "the campaign runner") + " from `/hai-workspace/glm52-delta`:", "",
              "```bash", "/opt/sglang/bin/python run_compile_campaign.py COMPILE_VALIDATION_PLAN.json", "```", "",
              "The workload uses ordinary GRPO updates with `sample.index % 2` rewards, 8 prompts × 8 samples, response "
              "limit 100, temperature 1, LR 1e-6, trainer/rollout seeds 1234/42. Model/data revisions and prepared input "
              "hashes are retained in the manifests. This synthetic transport workload does not establish task quality "
              "or equal training trajectories. Existing disabled logprob/KL/GPU-weight-equality CI checks remain disabled.", "",
              "Compiler records cover the entire updater and sit outside updater CPU/wall windows but inside driver wall. "
              "CPU scopes are trainer processes, inventoried receiver trees over rank-0's update, and driver CPU separately; "
              "membership is non-atomic and transient children can escape. Initial process CPU excludes compiler children. "
              "Publication admission checks indexes, shard hashes, name/range/header contracts and checksum presence; "
              "runtime receiver apply supplies checksum checks. This does not independently reconstruct checkpoint state "
              "or attest GPU equality. Binary weights/debug dumps are excluded from the published text bundle."]
    return "\n".join(lines) + "\n\n"


def hosted_ci_section(review, link, head, local_test_output, compatibility=None, current_ci=None):
    """Keep the observed failure explicit; changed CI records need fresh review."""
    if review["head"] != head or review.get("terminal") is not True or review.get("hosted_ci_passed") is not False:
        raise ValueError("Hosted CI record/source changed; review the final wording before claiming resolution")
    failed = review["failed_job"]
    if not re.fullmatch(r"https://github\.com/radixark/miles/actions/runs/[0-9]+/job/[0-9]+", failed["url"]):
        raise ValueError("Hosted failed-job URL is not an exact Miles Actions job")
    if failed["exit_code"] != 250 or failed["traceback_or_pytest_failure_summary"] is not False:
        raise ValueError("Hosted CI failure evidence changed; review the diagnosis")
    if "18 passed, 15 warnings in 7.68s" not in local_test_output:
        raise ValueError("Targeted Torch 2.11 local diagnostic evidence changed")
    env = review["failed_job_environment"]
    prefix = "artifacts/compile-pr-ci-3720-6fb0e2a/"
    section = (
        "## Hosted CI — unresolved failure\n\n"
        f"Hosted CI did **not** pass at `{head}`. The [{failed['name']}]({failed['url']}) job terminated "
        f"with exit status 250 during `{failed['last_test']}` after the preceding `[entries0-64]` case passed. "
        "There is no pytest failure traceback or raw signal/core artifact establishing the cause. A wrapper mapping "
        "from SIGABRT to 250 is only an inference, not a confirmed signal or diagnosis. Investigation remains open.\n\n"
        f"The hosted job used {env['os']} image `{env['image']}`, Python `{env['python']}`, "
        f"Torch `{env['torch']}`, NumPy `{env['numpy']}` and uv `{env['uv']}`, without an explicit workflow OMP/MKL limit. "
        "The passing C2 tests use a different Torch build. A targeted macOS ARM Torch 2.11 run passed 18 tests "
        "with 15 warnings in 7.68 s after a relative-cache setup error was corrected; it does not establish "
        "compatibility with the hosted Linux runtime or full shard order.\n\n"
        "Pre-commit, stage-a CPU shards 0/2/3 and stage-b CPU passed. All seven GPU stages were explicitly skipped "
        "by policy; ROCm workflow success covers setup/policy jobs, not GPU tests. No hosted GPU validation is claimed. "
        "The failed job uploaded no artifacts. Workflow logs contain embedded historical C2 output from the PR body; "
        "that untimestamped embedded output is not counted as hosted test success.\n\n"
        "Exact-commit " + link("HOSTED_CI_REVIEW.md", "compact CI review") + ", "
        + link(prefix + "README.md", "CI evidence README") + ", "
        + link(prefix + "final-review.json", "terminal job review") + ", "
        + link(prefix + "failed-shard-1.log", "complete failed shard log") + ", and "
        + link(prefix + "local-torch211-delta-tests.log", "targeted local Torch 2.11 log")
        + " retain the failure and diagnostic limits. Complete workflow logs are kept outside the PR body. "
        "This unresolved hosted result is separate from the selected CPU replay and C2 GPU campaign.\n\n"
    )
    if compatibility is not None:
        section = section.replace("## Hosted CI — unresolved failure", "## Historical hosted CI failure and compatibility fix")
        section = section.replace("Investigation remains open.", "A later Linux reproduction is retained separately below.")
        section = section.replace("This unresolved hosted result", "This historical hosted result")
        section += (
            "The separate Linux Python 3.11 / CUDA-wheel Torch 2.11 diagnostic reproduced SIGSEGV at the same scalar test "
            "with `ensureCUDADeviceGuardSet` in its native stack. "
            "[PyTorch #178950](https://github.com/pytorch/pytorch/pull/178950) fixes the matching fake-guard lifetime defect "
            "([upstream commit](https://github.com/pytorch/pytorch/commit/d7b75b8250f43949bb2609f406ac942f935cab46)). "
            "The test-only fixture initializes FakeTensorMode on the main pytest thread when Torch has CUDA support "
            "but no visible GPU. The selected committed test bytes match the diagnosed fixture; production bytes are unchanged. "
            "This reproduction does not recover the original hosted signal or reproduce hosted hardware/memory limits.\n\n"
            f"Linux focused result: **{compatibility['focused']['summary']}**. "
            + link(compatibility["focused"]["log"], "Complete focused log") + ". "
        )
        if compatibility["full_shard"].get("state") == "pending":
            section += "The full hosted-shard command retry is still pending; no full-shard pass is claimed.\n\n"
        else:
            section += f"Linux full-shard result: **{compatibility['full_shard']['summary']}**. "
            section += link(compatibility["full_shard"]["log"], "Complete full-shard log") + ".\n\n"
            section += ("This diagnostic used the hosted shard file list on merge `5093563d` plus the identical test fixture; "
                        "16 invalid physical CUDA IDs preserved logical placement indices while Torch reported zero real GPUs. "
                        "All earlier environment-only attempts are retained. This is separate from selected-head hosted CI.\n\n")
        section += link("COMPILE_CI_COMPATIBILITY.md", "Compatibility diagnosis and limits") + ", "
        section += link("COMPILE_CI_COMPATIBILITY.json", "exact diagnostic source/raw hashes") + ".\n\n"
    section += "### Selected-head hosted CI\n\n"
    if current_ci is None:
        section += "A selected-head hosted snapshot has not been captured yet; no new hosted pass is claimed.\n\n"
    else:
        summary = current_ci["summary"]
        section += f"Snapshot `{summary['captured_at_utc']}` for `{summary['head']}`; "
        section += f"all captured workflows terminal: `{summary['all_runs_completed']}`. "
        section += "Skipped GPU jobs are not GPU validation. Every captured job is shown below; pending or failed jobs remain explicit.\n\n"
        section += "\n".join(table(["Workflow", "Job", "Status", "Conclusion"], [
            [run["name"], f"[{job['name']}]({job['url']})", job["status"], job["conclusion"]]
            for run in summary["runs"] for job in run["jobs"]])) + "\n\n"
        section += link(current_ci["files"][0], "Complete selected-head CI summary") + " and "
        section += link(current_ci["files"][1], "raw workflow/job capture") + ".\n\n"
    return section


def source_preflight_section(root, plan, attribution, preflight, link):
    old_standalone = attribution["standalone_cpu_replay"]["measured_source"]
    old_combined = attribution["combined_historical_preflight"]["measured_source"]
    text = "### Source attribution and selected-head preflight\n\n"
    text += (f"The CPU replay remains measured at `{old_standalone}`. Its original-image standalone checks were "
             f"45 passed / 24 warnings / 30.89 s; combined `{old_combined}` passed 64 tests / 28 warnings / 30.02 s "
             "and the two-rank Gloo failure harness. These are historical source/image results, not reruns on the new heads. ")
    text += link("artifacts/torch-compile-final-production-tests.log", "Complete original standalone output") + ", "
    text += link("artifacts/compile-final-preflight-01/focused-tests.log", "original combined output") + ", and "
    text += link("artifacts/compile-final-preflight-01/gloo-failures.log", "original Gloo output") + ".\n\n"
    text += (f"Selected standalone `{plan['standalone_source']}` and combined `{plan['sources']['miles']}` are verified "
             "descendants whose only permitted intervening path is `tests/fast/utils/test_delta_preparation.py`. "
             "Every other tracked path is identical; core production SHA256 values and both measured/current snapshots "
             "are retained in " + link("COMPILE_SOURCE_ATTRIBUTION.json", "the exact source-attribution proof") + ". "
             "This establishes implementation identity for the old CPU timings, not new test execution.\n\n")
    directory = preflight["directory"]
    text += ("The following preflight independently tests the selected heads in the replacement GPU campaign image. "
             "Both subprocesses exited 0. CUDA is hidden; the real two-rank Gloo failure checks use compiled CPU "
             "preparation and a mocked finalizer, so they do not establish receiver RPC or GPU equality. "
             + link(f"{directory}/manifest.json", "Its manifest") + " retains exact commands, environment and helper hashes.\n\n")
    for title, name in (("Complete selected-head focused tests", "focused-tests.log"),
                        ("Complete selected-head two-rank Gloo checks", "gloo-failures.log")):
        # Check exact published Git bytes before embedding the local raw log.
        text += link(f"{directory}/{name}", title) + ".\n\n"
        text += f"<details>\n<summary>{title}</summary>\n\n```text\n{raw(root / directory / name).rstrip()}\n```\n\n</details>\n\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-url", required=True)
    args = parser.parse_args()
    link = publication_links(ROOT, args.evidence_url)
    plan_path = ROOT / "COMPILE_VALIDATION_PLAN.json"
    plan = json.loads(raw(plan_path))
    attribution = validate_attribution(ROOT, plan)
    preflight = selected_preflight(ROOT, plan)
    compatibility = validate_compatibility(ROOT, attribution)
    current_ci = current_hosted_ci(ROOT, plan["standalone_source"])
    fresh = gpu_report.build([ROOT / "artifacts" / run["name"] for run in plan["runs"]], plan_path)
    stored = json.loads(raw(ROOT / "COMPILED_VALIDATION_RESULTS.json"))
    if json.loads(json.dumps(fresh)) != stored or raw(ROOT / "COMPILED_VALIDATION_RESULTS.md") != gpu_report.markdown(fresh):
        raise ValueError("Stored GPU report differs from strict current raw admission")
    body = raw(ROOT / "miles-torch-compile-pr-body-initial.md")
    inline_logs = ["artifacts/torch-compile-final-production-tests.log", "artifacts/compile-final-preflight-01/focused-tests.log",
                   "artifacts/compile-final-preflight-01/gloo-failures.log"]
    if preflight is not None:
        start, stop = body.index("### Completed selected-source CPU and distributed preflight"), body.index("### Generated code and AVX-512")
        body = body[:start] + source_preflight_section(ROOT, plan, attribution, preflight, link) + body[stop:]
        inline_logs = [f"{preflight['directory']}/focused-tests.log", f"{preflight['directory']}/gloo-failures.log"]
        body = re.sub(r"^\| Selected standalone PR commit \|.*$",
                      f"| Selected standalone PR commit | `{plan['standalone_source']}`; CPU replay source `{attribution['standalone_cpu_replay']['measured_source']}` |", body, flags=re.M)
        body = re.sub(r"^\| Selected combined GLM validation commit \|.*$",
                      f"| Selected combined GLM validation commit | `{plan['sources']['miles']}`; original preflight source `{attribution['combined_historical_preflight']['measured_source']}` |", body, flags=re.M)
    start, stop = body.index("## Completed 6eca CPU replay"), body.index("## Combined GLM-5.2 GPU comparison")
    historical = """## Historical CPU experiments

Earlier source results remain separate from the selected implementation:

| Experiment | Unchanged wall/CPU change vs its matched f17 control | Changed wall/CPU change vs its matched f17 control |
| --- | --- | --- |
| Committed 6eca uint8 comparison/count stage | +58.10% / -1.37% | +70.96% / +53.02% |
| Uncommitted materialize-first 8×4 variant | +265.04% / +184.46% | +11.63% / +30.70% |

"""
    historical += link("TORCH_COMPILE_STAGE_REPLAY_V1.md", "Complete 6eca report and all 28 samples") + "; "
    historical += link("artifacts/torch-compile-stage-replay-v1.jsonl", "original raw replay") + ". "
    historical += link("TORCH_COMPILE_MATERIALIZE_FIRST_HISTORY.md", "Complete materialize-first report and all 28 samples") + "; "
    historical += link("artifacts/torch-compile-materialize-first-8x4-replay.jsonl", "original raw replay") + ".\n\n"
    historical += "The 8×4 module is explicitly uncommitted, SHA256 `09fc7f41579c5e517a63263bb6a5763c08ad769a16e34dd658f3255ec900dc5d`; "
    historical += "its eight compression workers differ from the 32-worker control. Complete cold/warm, copy, source and "
    historical += "command evidence is retained. Neither history is substituted for final-source results.\n\n"
    body = body[:start] + historical + body[stop:]
    start, stop = body.index("## Combined GLM-5.2 GPU comparison"), body.index("## Earlier kernel probes and prior art")
    body = body[:start] + gpu_section(stored, link) + body[stop:]
    ci = json.loads(raw(ROOT / "artifacts/compile-pr-ci-3720-6fb0e2a/final-review.json"))
    ci_section = hosted_ci_section(ci, link, attribution["standalone_cpu_replay"]["measured_source"],
                                   raw(ROOT / "artifacts/compile-pr-ci-3720-6fb0e2a/local-torch211-delta-tests.log"),
                                   compatibility, current_ci)
    body = body.replace("## Earlier kernel probes and prior art", ci_section + "## Earlier kernel probes and prior art", 1)
    body = body.replace("Hosted CI is separate and not represented as completed here.",
                        "Historical CI failure, compatibility diagnostics and selected-head CI results are recorded separately below.")
    body = body.replace("the GPU pair has no result yet.", "the completed paired GPU result is reported below.")
    body = body.replace("Replay module bytes match selected commit", "Replay module bytes match measured commit")
    body = body.replace("| C2 CPU |", "| Original C2 CPU replay/preflight host |")
    body = body.replace("| Explicit image |", "| CPU replay/preflight image |")
    body = body.replace("| GPU integration environment |", "| Original C2 hardware |")
    body = body.replace("4 trainer + 4 rollout GPUs; paired campaign running, no result yet",
                        "the replacement GPU campaign environment is recorded separately below")
    body = body.replace("| Integration stack |", "| Original preflight stack |")
    body = body.replace("Recorded source/library hashes and full generated source/disassembly remain in `artifacts/torch-compile-final-codegen-complete/`;",
                        "Recorded source/library hashes and full generated source/disassembly are in "
                        + link("artifacts/torch-compile-final-codegen-complete/CODEGEN.md", "the exact generated-code evidence") + ";")
    evidence_paragraph = "\n\nExact-commit evidence: " + link("TORCH_COMPILE_FINAL_CPU_RESULTS.md", "complete selected CPU report")
    evidence_paragraph += ", " + link("artifacts/torch-compile-word-two-phase-32x1-replay.jsonl", "all 28 raw CPU samples")
    evidence_paragraph += ", " + link("artifacts/torch-compile-final-production-tests.log", "standalone test log")
    evidence_paragraph += ", " + link("artifacts/compile-final-preflight-01/focused-tests.log", "combined test log")
    evidence_paragraph += ", and " + link("artifacts/compile-final-preflight-01/gloo-failures.log", "Gloo log") + ".\n"
    body += evidence_paragraph
    for name in inline_logs:
        if raw(ROOT / name).rstrip() not in body:
            raise ValueError(f"Complete final log was not retained inline: {name}")
    if body.splitlines()[:3] != ["## Summary", "", "@humansand"] or "PENDING" in body or "running, no result" in body:
        raise ValueError("Final PR body contains a stale status or malformed first content")
    if len(body.encode()) >= 65000:
        raise ValueError(f"Final body exceeds safe GitHub size budget: {len(body.encode())}")
    output = ROOT / "miles-torch-compile-pr-body-final.md"
    output.write_bytes(body.encode())
    print(json.dumps({"output": str(output), "bytes": len(body.encode()), "gpu_updates": 14,
                      "evidence_url": args.evidence_url, "sha256": hashlib.sha256(body.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
