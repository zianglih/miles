"""Explicit eval pool selection and version checks preserve the rollout controller."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.ray.rollout.eval_fleet import InferenceControllerEvalFleet
from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock


@pytest.mark.asyncio
async def test_eval_selection_uses_existing_lock_and_leaves_rollout_default():
    controller = object.__new__(InferenceController)
    controller.context_lock = ContextLock("test")
    controller.args = SimpleNamespace(update_weight_transfer_mode="gpu-delta", eval_num_gpus=4)
    controller._eval_fleet = object()
    rollout = SimpleNamespace(
        update_weights=True, api_clients=["rollout"], engine_gpu_counts=[4], engine_gpu_offsets=[0], server_cells={}
    )
    eval_server = SimpleNamespace(
        update_weights=False, api_clients=["eval"], engine_gpu_counts=[2], engine_gpu_offsets=[8], server_cells={}
    )
    controller.servers = {"default": rollout, "eval": eval_server}
    controller._health_monitoring_pause = AsyncMock()
    controller._ensure_cells_ready = AsyncMock()
    info = await controller.start_update_weights(receiver_pool="eval")
    assert info.receiver_pool == "eval"
    assert info.rollout_engines == ["eval"]
    assert info.engine_gpu_counts == [2] and info.engine_gpu_offsets == [8]
    assert controller.context_lock.locked
    controller._ensure_cells_ready.assert_awaited_once_with(model_id="eval")
    await controller.end_update_weights(info.snapshot_cell_id_to_hashes)
    assert not controller.context_lock.locked
    info = await controller.start_update_weights()
    assert info.receiver_pool == "rollout" and info.rollout_engines == ["rollout"]
    await controller.abort_update_weights()
    assert not controller.context_lock.locked
    assert eval_server.update_weights is False


@pytest.mark.parametrize("mode", ["broadcast", "disk-delta"])
@pytest.mark.asyncio
async def test_eval_admission_failure_releases_update_lock(mode):
    controller = object.__new__(InferenceController)
    controller.context_lock = ContextLock("test")
    controller.args = SimpleNamespace(update_weight_transfer_mode=mode, eval_num_gpus=4)
    controller._eval_fleet = None
    with pytest.raises(ValueError, match="requires a supported"):
        await controller.start_update_weights(receiver_pool="eval")
    assert not controller.context_lock.locked


@pytest.mark.parametrize("versions,reason", [(["2", "2"], None), (["2", "1"], "pin_violation"), ([], "pin_violation")])
@pytest.mark.asyncio
async def test_current_eval_checks_every_engine_without_loading_weights(versions, reason):
    fleet = object.__new__(InferenceControllerEvalFleet)
    clients = [
        SimpleNamespace(get_weight_version=AsyncMock(return_value=value), update_weights_from_disk=AsyncMock())
        for value in versions
    ]
    fleet._fleet_api_clients = AsyncMock(return_value=clients)
    fleet._wait_router_ready = AsyncMock()
    result = await fleet.check_version("2")
    assert result.skip_reason == reason
    for client in clients:
        client.get_weight_version.assert_awaited_once()
        client.update_weights_from_disk.assert_not_awaited()
    assert fleet._wait_router_ready.await_count == (1 if reason is None else 0)


@pytest.mark.asyncio
async def test_live_eval_uses_confirmed_pool_without_a_checkpoint(monkeypatch):
    import asyncio

    from miles.ray.rollout import rollout_executor as module

    executor = object.__new__(module.RolloutExecutor)
    executor.args = SimpleNamespace()
    executor._eval_lock = asyncio.Lock()
    executor._metric_checker = None
    executor.rollout_id = 4
    state = object()
    executor._eval_fleet = SimpleNamespace(current=AsyncMock(return_value=state), pin=AsyncMock())
    captured = []
    executor.eval_generate_rollout = object()

    def generate(function, input):
        assert function is executor.eval_generate_rollout
        captured.append(input)
        return SimpleNamespace(data={}, metrics={})

    monkeypatch.setattr(module, "call_rollout_function", generate)
    monkeypatch.setattr(module, "save_debug_rollout_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "log_eval_rollout_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "is_complete_hf_export", lambda _path: pytest.fail("must not inspect a checkpoint"))
    await executor._eval_checkpoint(4, None, None, True, weight_version="3")
    executor._eval_fleet.current.assert_awaited_once_with("3")
    executor._eval_fleet.pin.assert_not_awaited()
    assert captured[0].weight_version == "3"
    assert captured[0].generate_state is state
    assert captured[0].hf_dir is None


@pytest.mark.asyncio
async def test_unconfirmed_eval_pool_never_generates(monkeypatch):
    import asyncio

    from miles.ray.rollout import rollout_executor as module
    from miles.rollout.checkpoint_eval import EvalSkip

    executor = object.__new__(module.RolloutExecutor)
    executor.args = SimpleNamespace()
    executor._eval_lock = asyncio.Lock()
    executor._eval_fleet = SimpleNamespace(current=AsyncMock(side_effect=EvalSkip("pin_violation")))
    executor.report_eval_skip = lambda step, reason: (step, reason)
    monkeypatch.setattr(
        module, "call_rollout_function", lambda *_args: pytest.fail("unconfirmed pool must not generate")
    )
    assert await executor._eval_checkpoint(4, None, None, True, weight_version="3") == (4, "pin_violation")
