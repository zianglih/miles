"""Ordinary synchronous driver with frozen v1 wall timing and v2 CPU accounting."""

from pathlib import Path
import logging
import os
import runpy
import time

from miles.ray import placement_group
from process_cpu_clocks_cpu_v2 import (
    Journal,
    calibrate,
    runtime_metadata,
    self_cpu_snapshot,
    self_cpu_delta,
)
import weight_sync_probe as wall_probe

logger = logging.getLogger(__name__)
original_update = placement_group.update_weights
journal = None
update_index = -1


async def measured_update(actor_model, rollout_executor, *, rollout_id=None):
    global journal, update_index
    update_index += 1
    if update_index == 0:
        try:
            journal = Journal(
                Path(os.environ["WEIGHT_SYNC_RUN_DIR"]) / "cpu-driver.jsonl"
            )
            journal.append(
                {
                    "kind": "cpu_driver_installed",
                    "runtime": runtime_metadata(),
                    "calibration": calibrate(
                        time.process_time_ns,
                        int(
                            os.environ.get(
                                "WEIGHT_SYNC_CPU_CALIBRATION_ITERATIONS", "200"
                            )
                        ),
                    ),
                }
            )
            journal.flush()
        except Exception as error:
            logger.warning("CPU driver setup failed; training continues: %s", error)
    self_before = self_cpu_snapshot()
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
        self_after = self_cpu_snapshot()
        wall_probe.record(
            "driver_update",
            rollout_id=rollout_id,
            initial=rollout_id is None,
            start_ns=start,
            end_ns=end,
            elapsed_s=(end - start) / 1e9,
            success=success,
        )
        try:
            if journal is not None:
                journal.append(
                    {
                        "kind": "cpu_driver_update",
                        "update_index": update_index,
                        "rollout_id": rollout_id,
                        "initial": rollout_id is None,
                        "success": success,
                        "start_ns": start,
                        "end_ns": end,
                        "wall_s": (end - start) / 1e9,
                        **self_cpu_delta(self_before, self_after),
                    }
                )
                journal.flush()
        except Exception as error:
            logger.warning("CPU driver journal failed; training continues: %s", error)


if __name__ == "__main__":
    placement_group.update_weights = measured_update
    runpy.run_path(
        str(Path(placement_group.__file__).resolve().parents[2] / "train.py"),
        run_name="__main__",
    )
