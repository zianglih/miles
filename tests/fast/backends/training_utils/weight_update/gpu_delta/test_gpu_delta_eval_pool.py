"""Receiver pools own independent delta baselines, connections and versions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.training_utils.weight_update import receiver_pool as pools_module
from miles.backends.training_utils.weight_update import updater as module
from miles.backends.training_utils.weight_update.conn_status import ConnStatusManager
from miles.backends.training_utils.weight_update.protocols.gpu_delta.protocol import UpdateWeightFromGpuDelta


def make_updater(monkeypatch):
    updater = object.__new__(module.WeightUpdater)
    updater.args = SimpleNamespace(update_weight_transfer_mode="gpu-delta")
    updater.protocol = Mock(spec=UpdateWeightFromGpuDelta)
    updater.conn_status = ConnStatusManager()
    updater.weight_version = 4
    updater._receiver_pools = module.ReceiverPools()
    updater._hf_weight_iterator = SimpleNamespace()
    updater._registered_adapters = {"rollout-adapter"}
    protocols = []

    def create(_args, receiver_pool):
        assert receiver_pool == "eval"
        protocol = Mock(spec=UpdateWeightFromGpuDelta)
        protocols.append(protocol)
        return protocol

    monkeypatch.setattr(pools_module, "get_weight_transfer_protocol", create)
    return updater, protocols


def test_eval_and_rollout_keep_independent_committed_state(monkeypatch):
    updater, created = make_updater(monkeypatch)
    rollout_protocol, rollout_connection = updater.protocol, updater.conn_status
    rollout_connection.mark_reconnected({"rollout": "incarnation-1"})
    with updater.receiver_pool("eval"):
        assert updater.protocol is not rollout_protocol
        assert updater.weight_version == 0
        assert updater._registered_adapters == set()
        updater._registered_adapters.add("eval-adapter")
        assert updater.conn_status.needs_reconnect({"eval": "incarnation-1"})
        updater.conn_status.mark_reconnected({"eval": "incarnation-1"})
        updater.weight_version = 1
        assert updater._hf_weight_iterator.local_consumer == updater.protocol.send_bucket
    assert updater.protocol is rollout_protocol
    assert updater.conn_status is rollout_connection
    assert updater.weight_version == 4
    assert updater._registered_adapters == {"rollout-adapter"}
    assert updater._hf_weight_iterator.local_consumer == rollout_protocol.send_bucket
    with updater.receiver_pool("rollout"):
        updater.weight_version = 5
    with updater.receiver_pool("eval"):
        assert updater.protocol is created[0]
        assert updater.weight_version == 1
        assert updater._registered_adapters == {"eval-adapter"}
        assert not updater.conn_status.needs_reconnect({"eval": "incarnation-1"})
        assert updater.conn_status.needs_reconnect({"eval": "incarnation-2"})
    assert updater.weight_version == 5
    assert len(created) == 1


def test_failed_eval_restores_active_rollout_without_rolling_back_committed_eval(monkeypatch):
    updater, _ = make_updater(monkeypatch)
    original = updater.protocol
    with pytest.raises(RuntimeError, match="post-commit"):
        with updater.receiver_pool("eval"):
            updater.weight_version = 2
            raise RuntimeError("post-commit")
    assert updater.protocol is original
    assert updater.weight_version == 4
    with updater.receiver_pool("eval"):
        assert updater.weight_version == 2


def test_trainer_reconfiguration_invalidates_every_pool(monkeypatch):
    updater, _ = make_updater(monkeypatch)
    updater.conn_status.mark_reconnected({"r": "1"})
    with updater.receiver_pool("eval"):
        updater.conn_status.mark_reconnected({"e": "1"})
    updater.mark_trainer_stale()
    assert updater.conn_status.needs_reconnect({"r": "1"})
    with updater.receiver_pool("eval"):
        assert updater.conn_status.needs_reconnect({"e": "1"})


@pytest.mark.parametrize("mode", ["broadcast", "disk-delta"])
def test_unsupported_protocol_cannot_join_eval_pool(monkeypatch, mode):
    updater, _ = make_updater(monkeypatch)
    updater.protocol = object()
    updater.args.update_weight_transfer_mode = mode
    with updater.receiver_pool("rollout"):
        pass
    with pytest.raises(ValueError, match="receiver-pool support"):
        with updater.receiver_pool("eval"):
            pytest.fail("must reject before entering")


@pytest.mark.parametrize("pool_id", ["eval", "unknown"])
def test_disk_factory_rejects_unsupported_pool_before_constructing(monkeypatch, pool_id):
    from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
    from miles.backends.training_utils.weight_update.protocols import delta

    monkeypatch.setattr(
        delta, "UpdateWeightFromDiskDelta", lambda *_args: pytest.fail("must reject before construction")
    )
    with pytest.raises(ValueError, match="does not support receiver pool"):
        get_weight_transfer_protocol(SimpleNamespace(update_weight_transfer_mode="disk-delta"), pool_id)


def test_eval_metrics_stay_with_their_protocol(monkeypatch):
    updater, _ = make_updater(monkeypatch)
    rollout = updater.protocol
    rollout.pop_metrics.return_value = {"rollout": 1}
    with updater.receiver_pool("eval"):
        updater.protocol.pop_metrics.return_value = {"eval": 2}
        assert updater.pop_metrics() == {"eval": 2}
    assert updater.pop_metrics() == {"rollout": 1}
