"""Real local Git fixtures for source attribution; no benchmark measurements."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import compile_evidence_attribution as subject

ROOT = Path(__file__).resolve().parent


class AttributionTests(unittest.TestCase):
    def git(self, repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()

    def commit(self, repo, message):
        self.git(repo, "add", ".")
        self.git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", message)
        return self.git(repo, "rev-parse", "HEAD")

    def test_descendant_requires_only_test_change_and_production_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="attribution-fixture-") as directory:
            repo = Path(directory)
            self.git(repo, "init", "-q")
            for name in (*subject.PRODUCTION_FILES, subject.COMPATIBILITY_TEST):
                target = repo / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("original\n")
            base = self.commit(repo, "Original fixture")
            (repo / subject.COMPATIBILITY_TEST).write_text("test-only change\n")
            test_head = self.commit(repo, "Compatibility fixture")
            record = subject.prove_test_only_descendant(repo, base, test_head)
            self.assertEqual(record["measured_source"], base)
            self.assertEqual(record["selected_source"], test_head)
            self.assertEqual(record["changed_paths"], [subject.COMPATIBILITY_TEST])
            self.assertEqual(len(record["production_sha256"]), 4)
            with self.assertRaisesRegex(ValueError, "not a descendant"):
                subject.prove_test_only_descendant(repo, test_head, base)
            (repo / subject.PRODUCTION_FILES[0]).write_text("production changed\n")
            production_head = self.commit(repo, "Forbidden production change")
            with self.assertRaisesRegex(ValueError, "not confined"):
                subject.prove_test_only_descendant(repo, base, production_head)
            self.git(repo, "checkout", "--detach", test_head)
            (repo / "unrelated.py").write_text("unrelated change\n")
            other = self.commit(repo, "Forbidden extra path")
            with self.assertRaisesRegex(ValueError, "not confined"):
                subject.prove_test_only_descendant(repo, base, other)

    def test_retained_cpu_attribution_is_not_relabelled(self):
        plan = json.loads((ROOT / "COMPILE_VALIDATION_PLAN_V1.json").read_bytes())
        actual = subject.validate_attribution(ROOT, plan)
        self.assertEqual(actual["standalone_cpu_replay"]["measured_source"], "6fb0e2a9d81fc343863ba6015512fe93992892e4")
        plan["evidence_source_commits"] = {"cpu_replay": "a" * 40, "combined_preflight": plan["sources"]["miles"]}
        with self.assertRaisesRegex(ValueError, "Historical CPU replay/test attribution"):
            subject.validate_attribution(ROOT, plan)

    def test_new_preflight_must_match_new_heads_and_complete_logs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="preflight-attribution-fixture-") as directory:
            root = Path(directory)
            plan = {"sources": {"miles": "a" * 40, "sglang": "b" * 40}, "standalone_source": "c" * 40,
                    "preflight_evidence_directory": "artifacts/preflight-fixture"}
            output = root / plan["preflight_evidence_directory"]
            output.mkdir(parents=True)
            for name in ("run_compiled_preflight.py", "validate_compiled_sender_gloo.py"):
                (root / name).write_bytes(b"fixture script\n")
            manifest = {"sources": plan["sources"], "standalone_source": plan["standalone_source"],
                        "checks": {"focused-tests": {"exit": 0}, "gloo-failures": {"exit": 0}},
                        "script_sha256": subject.sha256(b"fixture script\n"),
                        "gloo_script_sha256": subject.sha256(b"fixture script\n")}
            path = output / "manifest.json"
            path.write_text(json.dumps(manifest))
            (output / "focused-tests.log").write_bytes(b"progress\r64 passed, 28 warnings in 30.02s\n")
            (output / "gloo-failures.log").write_text("PASS: both failure modes drained all four collectives on both ranks without publication\n")
            result = subject.selected_preflight(root, plan)
            self.assertEqual(result["manifest"]["standalone_source"], plan["standalone_source"])
            manifest["standalone_source"] = "d" * 40
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source pins"):
                subject.selected_preflight(root, plan)
            manifest["standalone_source"] = plan["standalone_source"]
            manifest["checks"]["focused-tests"]["exit"] = 1
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "both checks"):
                subject.selected_preflight(root, plan)

    def test_compatibility_requires_selected_test_bytes_and_raw_log_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="ci-attribution-fixture-") as directory:
            root = Path(directory)
            data = b"18 passed, 18 warnings in 45.59s\n"
            (root / "focused.log").write_bytes(data)
            record = {"schema_version": 1, "test_path": subject.COMPATIBILITY_TEST,
                      "historical_head": "a" * 40, "tested_test_sha256": "b" * 64,
                      "upstream_pr": "https://github.com/pytorch/pytorch/pull/178950",
                      "upstream_commit": "https://github.com/pytorch/pytorch/commit/d7b75b8250f43949bb2609f406ac942f935cab46",
                      "files": {"focused.log": {"sha256": subject.sha256(data), "bytes": len(data)}},
                      "focused": {"log": "focused.log", "summary": data.decode().strip()}, "full_shard": {"state": "pending"}}
            (root / "COMPILE_CI_COMPATIBILITY.json").write_text(json.dumps(record))
            attribution = {"standalone_cpu_replay": {"measured_source": "a" * 40,
                           "selected_compatibility_test_sha256": "b" * 64}}
            self.assertEqual(subject.validate_compatibility(root, attribution)["full_shard"]["state"], "pending")
            attribution["standalone_cpu_replay"]["selected_compatibility_test_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "test bytes"):
                subject.validate_compatibility(root, attribution)
            attribution["standalone_cpu_replay"]["selected_compatibility_test_sha256"] = "b" * 64
            (root / "focused.log").write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "evidence drift"):
                subject.validate_compatibility(root, attribution)

    def test_current_ci_requires_current_head_and_complete_job_snapshot(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="ci-snapshot-fixture-") as directory:
            root = Path(directory)
            head = "a" * 40
            output = root / f"artifacts/compile-pr-ci-3720-{head[:7]}"
            output.mkdir(parents=True)
            run = {"databaseId": 1, "name": "fixture", "status": "completed", "conclusion": "success",
                   "url": "https://github.com/radixark/miles/actions/runs/1", "jobs": []}
            summary = {"head": head, "all_runs_completed": True, "runs": [run]}
            latest = {"pr": {"headRefOid": head}, "runs": [{**run, "headSha": head}]}
            (output / "summary.json").write_text(json.dumps(summary))
            (output / "latest.json").write_text(json.dumps(latest))
            self.assertTrue(subject.current_hosted_ci(root, head)["summary"]["all_runs_completed"])
            latest["runs"][0]["headSha"] = "b" * 40
            (output / "latest.json").write_text(json.dumps(latest))
            with self.assertRaisesRegex(ValueError, "workflow run source"):
                subject.current_hosted_ci(root, head)
            latest["runs"][0]["headSha"] = head
            (output / "latest.json").write_text(json.dumps(latest))
            summary["runs"] = []
            (output / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "complete captured"):
                subject.current_hosted_ci(root, head)
            summary["head"] = "b" * 40
            (output / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "selected head"):
                subject.current_hosted_ci(root, head)


if __name__ == "__main__":
    unittest.main()
