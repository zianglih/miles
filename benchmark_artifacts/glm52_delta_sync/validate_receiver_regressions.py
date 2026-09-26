"""Show that the receiver regression cases fail on the upstream base."""

import importlib.util
import subprocess
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "sglang"
BASE_COMMIT = "880e3d2453eb7ef1738350e8c35ba2b956cc93a9"
spec = importlib.util.spec_from_file_location(
    "receiver_tests", REPO / "test/registered/unit/rl/test_local_checkpoint.py"
)
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
baseline = types.ModuleType("receiver_before")
source = subprocess.check_output(
    [
        "git",
        "show",
        f"{BASE_COMMIT}:python/sglang/srt/weight_sync/local_checkpoint.py",
    ],
    cwd=REPO,
    text=True,
)
exec(compile(source, "local_checkpoint_before.py", "exec"), baseline.__dict__)
tests.local_checkpoint = baseline
cases = [
    (tests.test_new_trainer_stream_reseeds_reused_local_checkpoint, {}),
    (tests.test_failed_xor_apply_invalidates_state_and_retry_reseeds, {}),
    (
        tests.test_incomplete_publication_preserves_checkpoint_and_version,
        {"damage": "missing_shard"},
    ),
    (
        tests.test_incomplete_publication_preserves_checkpoint_and_version,
        {"damage": "missing_tensor"},
    ),
    (
        tests.test_incomplete_publication_preserves_checkpoint_and_version,
        {"damage": "missing_index"},
    ),
    (
        tests.test_incomplete_publication_preserves_checkpoint_and_version,
        {"damage": "empty_publication"},
    ),
]
for case, kwargs in cases:
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        stream = tests.stream.__wrapped__(Path(temporary))
        label = case.__name__ + (f"[{kwargs['damage']}]" if kwargs else "")
        try:
            case(stream, **kwargs)
        except (AssertionError, tests.pytest.fail.Exception):
            print(f"Baseline regression confirmed: {label}")
        else:
            raise AssertionError(f"Expected baseline failure: {label}")
