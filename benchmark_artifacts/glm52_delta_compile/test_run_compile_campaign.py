"""Coordination fixtures; never launch GPU work or create durable real archives."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_compile_campaign as subject


class CampaignRunnerTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"durable_backup_dir": "/data/ziangli/glm52-delta-sync-c2/fixture"}

    def test_completed_arm_is_summarized_before_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            events = []
            def run(argv, **kwargs):
                events.append(Path(argv[1]).name)
                kwargs["stdout"].write("{}\n")
            def preserve(*args, **kwargs):
                events.append("backup")
                self.assertTrue(kwargs["require_complete"])
                self.assertEqual(len(list((root / "artifacts").glob("*.json"))), 3)
                return {"errors": [], "archive": "fixture.tar", "sha256": "fixture", "files": 3, "delta_shards": 6}
            with patch.object(subject.subprocess, "run", side_effect=run), \
                    patch.object(subject, "preserve_arm", side_effect=preserve):
                self.assertEqual(subject.summarize_and_preserve(root, self.plan, "fixture", 0, {}), 0)
            self.assertEqual(events, ["summarize_weight_sync.py", "summarize_training_evidence.py",
                                      "summarize_weight_sync_cpu_v2.py", "backup"])

    def test_failed_arm_archives_partial_without_summarizers(self):
        with patch.object(subject.subprocess, "run") as run, \
                patch.object(subject, "preserve_arm", return_value={"errors": []}) as preserve:
            self.assertEqual(subject.summarize_and_preserve(Path("/fixture"), self.plan, "fixture", 9, {}), 9)
            run.assert_not_called()
            self.assertFalse(preserve.call_args.kwargs["require_complete"])

    def test_summary_failure_archives_partial_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            with patch.object(subject.subprocess, "run", side_effect=subprocess.CalledProcessError(3, "fixture")), \
                    patch.object(subject, "preserve_arm", return_value={"errors": []}) as preserve, \
                    self.assertRaises(subprocess.CalledProcessError):
                subject.summarize_and_preserve(root, self.plan, "fixture", 0, {})
            self.assertFalse(preserve.call_args.kwargs["require_complete"])

    def test_backup_errors_stop_and_absence_preserves_legacy_behavior(self):
        with patch.object(subject, "preserve_arm", return_value={"errors": ["incomplete"]}) as preserve:
            with self.assertRaisesRegex(RuntimeError, "backup reported errors"):
                subject.backup_after_arm(Path("/fixture"), self.plan, "fixture", complete=True)
            subject.backup_after_arm(Path("/fixture"), {}, "fixture", complete=True)
            self.assertEqual(preserve.call_count, 1)


if __name__ == "__main__":
    unittest.main()
