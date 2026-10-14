"""Post-commit evidence retains actual replies and never triggers XOR replay."""

import json
from types import SimpleNamespace

from miles.backends.training_utils.weight_update.protocols.gpu_delta.evidence import (
    record_completed_update,
)


def _protocol(tmp_path):
    identity = {"rank_id": "original", "pid": 41, "start_ticks": 99}
    return SimpleNamespace(
        _initial_sync=False,
        _completed_publication={
            "stream_id": "stream",
            "base_version": 3,
            "target_version": 4,
        },
        _cohort=SimpleNamespace(identities=[identity]),
        _completed_activation={
            "session_id": "session",
            "receipts": [{"identity": identity, "state": "APPLIED"}],
            "resumed_receipts": [{"identity": identity, "state": "RESUMED", "generation_paused": False}],
        },
        _recovery_resumed_receipts=[],
        update_weight_metrics={"perf/blocked_s": 1.25},
        _version_dir=tmp_path,
        _stream_id="stream",
        _target_version=4,
    )


def test_completion_preserves_independent_pool_training_step_and_original_replies(
    tmp_path,
):
    protocol = _protocol(tmp_path)
    record_completed_update(protocol, pool_id="eval", rollout_id=8)
    record = json.loads((tmp_path / "monitoring-success.json").read_text())
    assert record["pool_id"] == "eval"
    assert record["native_rollout_id"] == 8 and not record["initial_sync"]
    assert record["publication"]["target_version"] == 4  # Eval skipped training steps.
    assert record["activation"] == protocol._completed_activation
    assert record["expected_participants"] == list(protocol._cohort.identities)
    assert record["metrics"] == protocol.update_weight_metrics


def test_initial_evidence_is_explicit_and_existing_evidence_is_never_overwritten(tmp_path, caplog):
    protocol = _protocol(tmp_path)
    protocol._initial_sync = True
    record_completed_update(protocol, pool_id="primary", rollout_id=None)
    path = tmp_path / "monitoring-success.json"
    initial = path.read_bytes()
    assert json.loads(initial)["initial_sync"] and json.loads(initial)["native_rollout_id"] is None
    protocol._initial_sync = False
    record_completed_update(protocol, pool_id="different", rollout_id=1)
    assert path.read_bytes() == initial
    assert "update committed, do not replay" in caplog.text


def test_oversized_evidence_does_not_emit_partial_success_or_raise_after_commit(tmp_path, caplog):
    protocol = _protocol(tmp_path)
    protocol._completed_activation["oversize"] = "x" * (1 << 20)
    record_completed_update(protocol, pool_id="primary", rollout_id=1)
    assert not (tmp_path / "monitoring-success.json").exists()
    assert "1 MiB bound" in caplog.text


def test_startup_warm_transfer_is_explicitly_excluded_from_post_training(tmp_path):
    protocol = _protocol(tmp_path)
    record_completed_update(protocol, pool_id="primary", rollout_id=None)
    record = json.loads((tmp_path / "monitoring-success.json").read_text())
    assert record["update_kind"] == "startup_warm"
    assert record["initial_sync"] is False and record["native_rollout_id"] is None


def test_nonfinite_metrics_cannot_be_published_as_completion(tmp_path, caplog):
    protocol = _protocol(tmp_path)
    protocol.update_weight_metrics["perf/blocked_s"] = float("nan")
    record_completed_update(protocol, rollout_id=1)
    assert not (tmp_path / "monitoring-success.json").exists()
    assert "update committed, do not replay" in caplog.text
