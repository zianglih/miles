"""Direct eval sync is blocking and never exports or reloads a checkpoint."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.ray.rollout.eval_dispatch import EvalDispatcher


def make_dispatcher():
    args = SimpleNamespace(update_weight_transfer_mode="gpu-delta", eval_num_gpus=4, eval_uses_snapshots=True)
    info = SimpleNamespace(receiver_pool="eval", snapshot_cell_id_to_hashes={"eval-cell": "incarnation"})
    controller = SimpleNamespace(
        start_update_weights=AsyncMock(return_value=info),
        end_update_weights=AsyncMock(),
        abort_update_weights=AsyncMock(),
    )
    actor = SimpleNamespace(update_weights=AsyncMock(return_value=2), export_hf=AsyncMock())
    executor = SimpleNamespace(eval=AsyncMock())
    return EvalDispatcher(args, actor, executor, inference_controller=controller), info


@pytest.mark.asyncio
async def test_eval_sync_runs_before_generation_without_snapshot_export():
    dispatcher, info = make_dispatcher()
    events = []

    async def update(**kwargs):
        assert kwargs == {"info": info, "rollout_id": 9}
        events.append("sync")
        return 2

    async def end(**kwargs):
        assert kwargs == {"snapshot_cell_id_to_hashes": info.snapshot_cell_id_to_hashes}
        events.append("unlock")

    async def evaluate(rollout_id, **kwargs):
        assert rollout_id == 9 and kwargs == {"weight_version": "2"}
        assert events == ["sync", "unlock"]
        events.append("generate")

    dispatcher.actor_model.update_weights.side_effect = update
    dispatcher.inference_controller.end_update_weights.side_effect = end
    dispatcher.rollout_manager.eval.side_effect = evaluate
    await dispatcher.dispatch(9, hf_dir="unused-initial-checkpoint")
    assert events == ["sync", "unlock", "generate"]
    dispatcher.inference_controller.start_update_weights.assert_awaited_once_with(receiver_pool="eval")
    dispatcher.actor_model.export_hf.assert_not_awaited()
    assert not dispatcher.pending


@pytest.mark.parametrize("error", [RuntimeError("sync failed"), asyncio.CancelledError()])
@pytest.mark.asyncio
async def test_failed_or_cancelled_sync_unlocks_once_without_eval_or_retry(error):
    dispatcher, _ = make_dispatcher()
    dispatcher.actor_model.update_weights.side_effect = error
    with pytest.raises(type(error)):
        await dispatcher.dispatch(3)
    dispatcher.actor_model.update_weights.assert_awaited_once()
    dispatcher.inference_controller.abort_update_weights.assert_awaited_once()
    dispatcher.inference_controller.end_update_weights.assert_not_awaited()
    dispatcher.rollout_manager.eval.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_synced_version_is_not_an_eval_success():
    dispatcher, _ = make_dispatcher()
    dispatcher.actor_model.update_weights.return_value = None
    with pytest.raises(RuntimeError, match="did not produce"):
        await dispatcher.dispatch(1)
    dispatcher.inference_controller.abort_update_weights.assert_awaited_once()
    dispatcher.rollout_manager.eval.assert_not_awaited()


@pytest.mark.asyncio
async def test_generation_is_awaited_before_a_next_sync_can_be_dispatched():
    dispatcher, _ = make_dispatcher()
    entered, release = asyncio.Event(), asyncio.Event()

    async def generate(*_args, **_kwargs):
        entered.set()
        await release.wait()

    dispatcher.rollout_manager.eval.side_effect = generate
    task = asyncio.create_task(dispatcher.dispatch(1))
    await entered.wait()
    assert not task.done()
    release.set()
    await task


@pytest.mark.parametrize("mode", ["disk-delta", "broadcast"])
@pytest.mark.asyncio
async def test_unsupported_protocol_keeps_existing_snapshot_eval(mode):
    dispatcher, _ = make_dispatcher()
    dispatcher.args.update_weight_transfer_mode = mode
    dispatcher.args.eval_max_in_flight = 1
    await dispatcher.dispatch(3, hf_dir="existing-hf-export")
    await dispatcher.drain()
    dispatcher.inference_controller.start_update_weights.assert_not_awaited()
    dispatcher.actor_model.update_weights.assert_not_awaited()
    dispatcher.rollout_manager.eval.assert_awaited_once_with(
        3, hf_dir="existing-hf-export", export_time_seconds=None, require_marker=False
    )
