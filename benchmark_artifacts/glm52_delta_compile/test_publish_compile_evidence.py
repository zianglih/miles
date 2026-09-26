"""Publisher fixture tests; these are not performance measurements."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import publish_compile_evidence as subject


class BundleTests(unittest.TestCase):
    attribution = {"standalone_cpu_replay": {"measured_source": "a" * 40},
                   "combined_historical_preflight": {"measured_source": "b" * 40}}
    def test_text_boundary_and_non_disclosing_secret_scan(self):
        subject.check_text("ok.py", b'key_name = "WANDB_API_KEY"\n')
        secret = "ghp_" + "x" * 36
        with self.assertRaises(ValueError) as raised:
            subject.check_text("private.log", ("first\n" + secret).encode())
        self.assertIn("private.log:2", str(raised.exception))
        self.assertNotIn(secret, str(raised.exception))
        for content in ("WANDB_API_KEY=" + "a" * 40,
                        json.dumps({"AWS_SECRET_ACCESS_KEY": "b" * 40})):
            with self.subTest(content_kind=content[:12]), self.assertRaises(ValueError):
                subject.check_text("credentials.log", content.encode())
        for name in ("../outside.py", "/tmp/outside.py", "weights.safetensors", "cache.so", "models/config.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                subject.relative_name(name)
        with self.assertRaises(ValueError):
            subject.check_text("binary.log", b"abc\0def")
        with mock.patch.object(subject, "MAX_FILE_BYTES", 3), self.assertRaises(ValueError):
            subject.check_text("huge.log", b"four")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "real.py").write_bytes(b"preserved\r\n")
            self.assertEqual(subject.read_local(root, "real.py"), b"preserved\r\n")
            (root / "linked.py").symlink_to(root / "real.py")
            with self.assertRaises(ValueError):
                subject.read_local(root, "linked.py")

    def test_only_exact_public_test_url_fixture_is_allowed(self):
        fixture = json.loads((subject.ROOT / "COMPILE_SECRET_SCAN_FIXTURE_ALLOWLIST.json").read_bytes())
        name = next(iter(fixture["files"]))
        data = (subject.ROOT / name).read_bytes()
        subject.check_text(name, data)
        with self.assertRaisesRegex(ValueError, "credential-url"):
            subject.check_text("different.log", data)
        with self.assertRaisesRegex(ValueError, "credential-url"):
            subject.check_text(name, data + b"changed raw file\n")

    def test_real_historical_replay_and_failed_sample(self):
        files, record = subject.validate_replay(subject.ROOT, subject.NEGATIVE_REPLAY,
                                                 subject.NEGATIVE_SOURCE, subject.ROOT / "miles-torch-compile")
        self.assertEqual(record["samples"], 28)
        self.assertEqual(len(files), 9)
        reference = "sha256:" + record["module_sha256"]
        _, experiment = subject.validate_replay(subject.ROOT, subject.NEGATIVE_REPLAY,
                                                  reference, subject.ROOT / "miles-torch-compile", allow_uncommitted=True)
        self.assertIsNone(experiment["source_commit"])
        self.assertEqual(experiment["source_kind"], "uncommitted historical experiment")
        with self.assertRaisesRegex(ValueError, "Only an explicitly historical"):
            subject.validate_replay(subject.ROOT, subject.NEGATIVE_REPLAY,
                                    reference, subject.ROOT / "miles-torch-compile")
        original = subject.read_local
        bad = [json.loads(line) for line in original(subject.ROOT, subject.NEGATIVE_REPLAY).splitlines()]
        next(row for row in bad if row["event"] == "sample")["new_graphs_or_kernels"] = True

        def corrupt(root, name):
            if name == subject.NEGATIVE_REPLAY:
                return ("\n".join(json.dumps(row) for row in bad) + "\n").encode()
            return original(root, name)

        with mock.patch.object(subject, "read_local", side_effect=corrupt), self.assertRaisesRegex(ValueError, "compiler admission"):
            subject.validate_replay(subject.ROOT, subject.NEGATIVE_REPLAY,
                                    subject.NEGATIVE_SOURCE, subject.ROOT / "miles-torch-compile")
        for label, mutate, error in (
            ("subset", lambda rows: rows[0].update(selected_tensor_count=20), "complete 4,690"),
            ("negative-time", lambda rows: next(row for row in rows if row["event"] == "sample").update(wall_s=-1), "invalid raw timings"),
            ("changed-bytes", lambda rows: next(row for row in rows if row["event"] == "workload").update(changed_bytes=1), "workload differs"),
        ):
            bad[:] = [json.loads(line) for line in original(subject.ROOT, subject.NEGATIVE_REPLAY).splitlines()]
            mutate(bad)
            with self.subTest(label=label), mock.patch.object(subject, "read_local", side_effect=corrupt), self.assertRaisesRegex(ValueError, error):
                subject.validate_replay(subject.ROOT, subject.NEGATIVE_REPLAY,
                                        subject.NEGATIVE_SOURCE, subject.ROOT / "miles-torch-compile")

    def fixture(self, root):
        plan = copy.deepcopy(json.loads((subject.ROOT / "COMPILE_VALIDATION_PLAN.json").read_text()))
        plan["frozen"] = True
        names = set(subject.FILES) | set(plan["script_sha256"]) | {"artifacts/proof.cpp", "artifacts/final-tests.log"}
        names.update(plan[field]["path"] for field in ("campaign_metadata", "environment_metadata", "cpu_calibration_metadata")
                     if field in plan)
        for run in plan["runs"]:
            if "campaign_metadata" in plan:
                names.add(f"artifacts/{run['name']}-campaign-metadata.json")
            names.update(f"artifacts/{run['name']}/{filename}" for filename in subject.RUN_FILES)
            names.update(f"artifacts/{run['name']}.{suffix}" for suffix in ("log", "exit"))
            names.update(f"artifacts/{run['name']}/delta-publication/weight_v{version:06d}/model.safetensors.index.json"
                         for version in range(1, 7))
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture raw bytes\r\n")
        for name in plan["script_sha256"]:
            (root / name).write_bytes((subject.ROOT / name).read_bytes())
        for field in ("campaign_metadata", "environment_metadata", "cpu_calibration_metadata"):
            if field in plan:
                name = plan[field]["path"]
                (root / name).write_bytes((subject.ROOT / name).read_bytes())
        for name in ("artifacts/recovery-inputs-ready.json", "artifacts/recovery-original-canonical-headers.json",
                     "artifacts/recovery-nvfp4-header-verification.json", "artifacts/input-backup-20260926T021916Z-4640.json",
                     "artifacts/recovery-20260926T021916Z-4640-input-identity.json"):
            (root / name).write_bytes((subject.ROOT / name).read_bytes())
        (root / "COMPILE_VALIDATION_PLAN.json").write_text(json.dumps(plan))
        (root / "artifacts/compile-campaign-frozen-inputs.json").write_text(json.dumps({
            "COMPILE_VALIDATION_PLAN.json": subject.digest((root / "COMPILE_VALIDATION_PLAN.json").read_bytes())}))
        report = {"arms": {run["name"]: {"rows": list(range(7)), "compiler_observations": list(range(28))}
                           for run in plan["runs"]}}
        (root / "COMPILED_VALIDATION_RESULTS.json").write_text(json.dumps(report))
        (root / "COMPILED_VALIDATION_RESULTS.md").write_text("fixture report\n")
        (root / "COMPILE_SOURCE_ATTRIBUTION.json").write_text(json.dumps(self.attribution))
        return plan, report

    def test_complete_bundle_preserves_bytes_and_audit_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, report = self.fixture(root)
            with mock.patch.object(subject.reporter, "build", return_value=report), \
                    mock.patch.object(subject, "validate_attribution", return_value=self.attribution), \
                    mock.patch.object(subject, "selected_preflight", return_value=None), \
                    mock.patch.object(subject, "validate_compatibility", return_value=None), \
                    mock.patch.object(subject, "current_hosted_ci", return_value=None), \
                    mock.patch.object(subject.reporter, "markdown", return_value="fixture report\n"), \
                    mock.patch.object(subject, "validate_replay", return_value=([], {"samples": 28})), \
                    mock.patch.object(subject, "git_file", return_value=b"committed source\n"):
                data = subject.assemble(root, "artifacts/final.jsonl", ["artifacts/proof.cpp"], ["artifacts/final-tests.log"], [])
                self.assertFalse((root / subject.DESTINATION).exists())
                self.assertEqual(data["artifacts/proof.cpp"], b"fixture raw bytes\r\n")
                hashes = json.loads(data["SHA256.json"])
                self.assertEqual(set(hashes), set(data) - {"SHA256.json"})
                for name, value in hashes.items():
                    self.assertEqual(value, {"bytes": len(data[name]), "sha256": subject.digest(data[name])})
                self.assertFalse(any(Path(name).suffix in {".safetensors", ".pt", ".so"} for name in data))
                (root / "COMPILED_VALIDATION_RESULTS.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "stale"):
                    subject.assemble(root, "artifacts/final.jsonl", ["artifacts/proof.cpp"], ["artifacts/final-tests.log"], [])

    def test_missing_raw_file_rejects_before_any_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, report = self.fixture(root)
            missing = root / "artifacts" / plan["runs"][0]["name"] / "trainer-rank3.jsonl"
            missing.unlink()
            with mock.patch.object(subject.reporter, "build", return_value=report), \
                    mock.patch.object(subject, "validate_attribution", return_value=self.attribution), \
                    mock.patch.object(subject, "selected_preflight", return_value=None), \
                    mock.patch.object(subject, "validate_compatibility", return_value=None), \
                    mock.patch.object(subject, "current_hosted_ci", return_value=None), \
                    mock.patch.object(subject.reporter, "markdown", return_value="fixture report\n"), \
                    mock.patch.object(subject, "validate_replay", return_value=([], {"samples": 28})), \
                    self.assertRaisesRegex(ValueError, "Required evidence is missing"):
                subject.assemble(root, "artifacts/final.jsonl", ["artifacts/proof.cpp"], ["artifacts/final-tests.log"], [])
            self.assertFalse((root / subject.DESTINATION).exists())

    def test_stage_is_dedicated_idempotent_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "miles-evidence/benchmark_artifacts/glm52_delta_sync/retained.json"
            original.parent.mkdir(parents=True)
            original.write_bytes(b"original evidence\n")
            data = {"README.md": b"new evidence\n", "artifacts/raw.jsonl": b"{\"raw\": true}\r\n"}

            def git_output(argv, text):
                if "rev-parse" in argv:
                    return str(root / "miles/.git") + "\n"
                if "branch" in argv:
                    return "glm52-delta-sync-evidence\n"
                return "https://github.com/zianglih/miles.git\n"

            with mock.patch.object(subject.subprocess, "check_output", side_effect=git_output):
                subject.stage(root, data)
                subject.stage(root, data)
                self.assertEqual(original.read_bytes(), b"original evidence\n")
                self.assertEqual((root / subject.DESTINATION / "artifacts/raw.jsonl").read_bytes(), data["artifacts/raw.jsonl"])
                with self.assertRaisesRegex(ValueError, "differs"):
                    subject.stage(root, {**data, "README.md": b"changed\n"})
                self.assertEqual((root / subject.DESTINATION / "README.md").read_bytes(), b"new evidence\n")


if __name__ == "__main__":
    unittest.main()
