"""Run the ordinary synchronous Miles driver with outer update wall timing."""

from pathlib import Path
import runpy
import time

from miles.ray import placement_group
import weight_sync_probe


original_update = placement_group.update_weights


async def measured_update(actor_model, rollout_executor, *, rollout_id=None):
    start = time.perf_counter_ns()
    success = False
    try:
        result = await original_update(
            actor_model, rollout_executor, rollout_id=rollout_id
        )
        success = True
        return result
    finally:
        end = time.perf_counter_ns()
        weight_sync_probe.record(
            "driver_update",
            rollout_id=rollout_id,
            initial=rollout_id is None,
            start_ns=start,
            end_ns=end,
            elapsed_s=(end - start) / 1e9,
            success=success,
        )


if __name__ == "__main__":
    placement_group.update_weights = measured_update
    runpy.run_path(
        str(Path(placement_group.__file__).resolve().parents[2] / "train.py"),
        run_name="__main__",
    )
