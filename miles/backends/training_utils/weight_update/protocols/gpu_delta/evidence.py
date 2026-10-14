"""Bounded post-commit evidence, separate from protocol and receiver spans."""

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)
_MAX_BYTES = 1 << 20


def record_completed_update(protocol, rollout_id, pool_id=None):
    """Retain actual RPC replies without new receiver calls or collectives.

    Evidence failure must not replay an already committed XOR update. The
    monitor treats a missing/partial file as incomplete proof and stops counting.
    """
    try:
        record = {
            "schema": "miles_gpu_delta_completion_v1",
            "pool_id": pool_id,
            "native_rollout_id": rollout_id,
            "initial_sync": protocol._initial_sync,
            "update_kind": (
                "initial" if protocol._initial_sync else "startup_warm" if rollout_id is None else "post_training"
            ),
            "publication": protocol._completed_publication,
            "expected_participants": list(protocol._cohort.identities),
            "activation": protocol._completed_activation,
            "recovery_resumed_receipts": protocol._recovery_resumed_receipts,
            "metrics": protocol.update_weight_metrics,
        }
        content = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if len(content) > _MAX_BYTES:
            raise ValueError("GPU-delta completion evidence exceeds the 1 MiB bound")
        path = protocol._version_dir / "monitoring-success.json"
        with path.open("xb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        logger.info(
            "[gpu delta evidence] pool=%s initial=%s rollout_id=%s stream=%s version=%s bytes=%s sha256=%s path=%s",
            pool_id,
            protocol._initial_sync,
            rollout_id,
            protocol._stream_id,
            protocol._target_version,
            len(content),
            hashlib.sha256(content).hexdigest(),
            path,
        )
    except Exception:
        logger.exception("GPU-delta post-commit evidence failed; update committed, do not replay")
