import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import capture_compile_pr_ci as capture


class CaptureOptionsTest(unittest.TestCase):
    def test_defaults_preserved(self):
        with patch.object(capture, 'validate_destination'):
            args = capture.options([])
        self.assertEqual(args.head, capture.HEAD)
        self.assertEqual(args.destination, capture.DEST.resolve())

    def test_head_validation_and_old_destination_protected(self):
        for argv in (['--head', 'a' * 7], ['--head', 'g' * 40], ['--head', 'a' * 40]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                capture.options(argv)

    def test_existing_records_must_all_match(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            (dest / 'summary.json').write_text(json.dumps({'head': 'a' * 40}))
            (dest / 'latest.json').write_text(json.dumps({'pr': {'headRefOid': 'a' * 40}}))
            capture.validate_destination(dest, 'a' * 40)
            (dest / 'snapshot-old.json').write_text(json.dumps({'pr': {'headRefOid': capture.HEAD}}))
            with self.assertRaises(ValueError):
                capture.validate_destination(dest, 'a' * 40)

    def test_unknown_nonempty_directory_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            (dest / 'run-1.log').write_text('preserve me')
            with self.assertRaises(ValueError):
                capture.validate_destination(dest, capture.HEAD)
            self.assertEqual((dest / 'run-1.log').read_text(), 'preserve me')

    def test_new_head_snapshot_and_repeated_capture_preserve_raw(self):
        head = 'a' * 40
        pr = {'headRefOid': head, 'url': 'https://github.com/radixark/miles/pull/3720', 'statusCheckRollup': []}

        def fake_gh(*args):
            if args[:2] == ('auth', 'switch'):
                return ''
            if args[:2] == ('api', 'user'):
                return 'zianglih\n'
            return json.dumps(pr)

        with tempfile.TemporaryDirectory() as directory, patch.object(capture, 'gh', side_effect=fake_gh), contextlib.redirect_stdout(io.StringIO()):
            dest = Path(directory) / 'new-head'
            argv = ['--head', head, '--destination', str(dest)]
            capture.main(argv)
            first = next(dest.glob('snapshot-*.json'))
            raw = first.read_bytes()
            capture.main(argv)
            self.assertEqual(first.read_bytes(), raw)
            self.assertEqual(len(list(dest.glob('snapshot-*.json'))), 2)
            self.assertEqual(json.loads((dest / 'summary.json').read_text())['head'], head)

    def test_actual_pr_mismatch_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(capture, 'gh', side_effect=['', 'zianglih', json.dumps({'headRefOid': 'b' * 40})]):
            dest = Path(directory) / 'absent'
            with self.assertRaises(RuntimeError):
                capture.main(['--head', 'a' * 40, '--destination', str(dest)])
            self.assertFalse(dest.exists())


if __name__ == '__main__':
    unittest.main()
