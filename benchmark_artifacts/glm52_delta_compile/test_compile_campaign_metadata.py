"""Metadata selection fixtures; no runtime or benchmark measurements."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import compile_campaign_metadata as subject


class MetadataTests(unittest.TestCase):
    def fixture(self, root):
        for name in subject.DEFAULTS.values():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{"historical": true}\n')
        plan = {}
        for field, name in (("campaign_metadata", "campaign-v2.json"),
                            ("environment_metadata", "environment-v2.json"),
                            ("cpu_calibration_metadata", "calibration-v2.json")):
            (root / name).write_bytes(b'{"replacement": true}\n')
            plan[field] = {"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
        return plan

    def test_legacy_defaults_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            paths = subject.resolve_metadata(root / "plan.json", {})
            self.assertEqual(paths["campaign_metadata"], (root / "CAMPAIGN.json").resolve())
            paths = subject.resolve_metadata(root / "plan.json", {}, campaign=root / "campaign-v2.json")
            self.assertEqual(paths["campaign_metadata"], (root / "campaign-v2.json").resolve())

    def test_pinned_selection_bytes_and_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.fixture(root)
            paths = subject.resolve_metadata(root / "plan.json", plan, campaign=root / "campaign-v2.json")
            self.assertEqual(subject.metadata_record(paths, relative_to=root), plan)
            with self.assertRaisesRegex(ValueError, "override differs"):
                subject.resolve_metadata(root / "plan.json", plan, campaign=root / "CAMPAIGN.json")
            (root / "campaign-v2.json").write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, "SHA256"):
                subject.resolve_metadata(root / "plan.json", plan)

    def test_malformed_and_escaping_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.fixture(root)
            for mutate in (
                lambda p: p.pop("environment_metadata"),
                lambda p: p["campaign_metadata"].update(path="../outside.json"),
                lambda p: p["campaign_metadata"].update(path="/absolute.json"),
                lambda p: p["campaign_metadata"].update(sha256="abcd"),
                lambda p: p["campaign_metadata"].update(unexpected=True),
            ):
                plan = copy.deepcopy(original)
                mutate(plan)
                with self.subTest(plan=plan), self.assertRaises(ValueError):
                    subject.resolve_metadata(root / "plan.json", plan)
            (root / "linked.json").symlink_to(root / "campaign-v2.json")
            original["campaign_metadata"]["path"] = "linked.json"
            with self.assertRaisesRegex(ValueError, "symlink"):
                subject.resolve_metadata(root / "plan.json", original)

    def test_backup_directory_requires_owned_data_subtree(self):
        subject.validate_specs({"durable_backup_dir": "/data/ziangli/glm52-delta-sync-c2/compile-v2-backups"})
        for value in (True, 3, "/data", "/data/ziangli", "relative", "/tmp/effort", "/data/ziangli/../other"):
            with self.subTest(path=value), self.assertRaises(ValueError):
                subject.validate_specs({"durable_backup_dir": value})


if __name__ == "__main__":
    unittest.main()
