"""Add process CPU accounting while retaining the frozen v1 wall observation.

Layering is deliberate: CPU measurement wraps the original updater, v1 wraps
that measurement, and a final outer wrapper flushes the CPU journal after v1's
trainer wall endpoint. Thus v1 wall includes counter reads but excludes the CPU
journal flush; driver wall includes the flush. Use identical v2 on both source
versions. Any CPU-observer error is recorded and leaves training running.
"""

from functools import wraps
import logging
import os
from pathlib import Path
import time

from process_cpu_clocks_cpu_v2 import (
    Journal,
    ReceiverClocks,
    calibrate,
    runtime_metadata,
    self_cpu_snapshot,
    self_cpu_delta,
)
import weight_sync_probe as wall_probe


logger = logging.getLogger(__name__)


def install(args):
    import torch.distributed as dist
    from miles.backends.training_utils.weight_update.updater import WeightUpdater

    rank = dist.get_rank()
    root = Path(os.environ["WEIGHT_SYNC_RUN_DIR"])
    state = {
        "receiver": None,
        "receiver_error": None,
        "initialized": False,
        "journal": None,
    }
    backend = os.environ.get("WEIGHT_SYNC_CPU_BACKEND", "process-clock")
    receiver_policy = os.environ.get("WEIGHT_SYNC_CPU_RECEIVER", "required")
    iterations = int(os.environ.get("WEIGHT_SYNC_CPU_CALIBRATION_ITERATIONS", "200"))
    try:
        state["journal"] = Journal(root / f"cpu-trainer-rank{rank}.jsonl")
        state["journal"].append(
            {
                "kind": "cpu_probe_installed",
                "rank": rank,
                "runtime": runtime_metadata(),
                "self_calibration": calibrate(time.process_time_ns, iterations),
                "receiver_policy": receiver_policy,
                "backend": backend,
                "boundary": "CPU inside v1 updater wall; receiver clocks bracket trainer-rank0 updater, not driver dispatch",
            }
        )
        state["journal"].flush()
    except Exception as error:
        logger.warning("CPU v2 observer setup failed; training continues: %s", error)

    original_update = WeightUpdater.update_weights

    @wraps(original_update)
    def cpu_measured_update(self):
        before = after = receiver_result = None
        error = state["receiver_error"]
        receiver = state["receiver"]
        if receiver is not None and error is None:
            try:
                before = receiver.snapshot()
            except Exception as exc:
                error = f"receiver start: {type(exc).__name__}: {exc}"
        self_before = self_cpu_snapshot(include_resource=True)
        start = time.perf_counter_ns()
        success = False
        try:
            result = original_update(self)
            success = True
            return result
        finally:
            end = time.perf_counter_ns()
            self_after = self_cpu_snapshot(include_resource=True)
            self_delta = self_cpu_delta(self_before, self_after)
            error = error or self_delta["error"]
            try:
                if before is not None and receiver is not None:
                    after = receiver.snapshot()
                    receiver_result = receiver.difference(before, after)
            except Exception as exc:
                error = f"receiver end: {type(exc).__name__}: {exc}"
            event = {
                "kind": "cpu_trainer_update",
                "rank": rank,
                "pid": os.getpid(),
                "update_index": wall_probe._update_index,
                "initial": wall_probe._update_index == 0,
                "success": success,
                "start_ns": start,
                "end_ns": end,
                "observer_before_wall_ns": state.get("observer_before_wall_ns"),
                "wall_s": (end - start) / 1e9,
                **self_delta,
                "receiver": receiver_result,
                "cpu_valid": self_delta["cpu_valid"] and error is None,
                "receiver_required": rank == 0 and receiver_policy == "required",
                "error": error,
            }
            try:
                if state["journal"] is not None:
                    state["journal"].append(event)
            except Exception as exc:
                logger.warning("CPU v2 record failed; training continues: %s", exc)

    WeightUpdater.update_weights = cpu_measured_update
    wall_probe.install(args)
    original_wall_wrapper = WeightUpdater.update_weights

    @wraps(original_wall_wrapper)
    def flush_outside_wall(self):
        observer_start = time.perf_counter_ns()
        state["receiver_error"] = None
        if rank == 0 and (
            not state["initialized"]
            or (receiver_policy == "required" and state["receiver"] is None)
        ):
            state["initialized"] = True
            if receiver_policy == "required":
                try:
                    state["receiver"] = ReceiverClocks(
                        [client.server_url for client in self.protocol.rollout_engines],
                        backend,
                    )
                    state["journal"].append(
                        {
                            "kind": "cpu_receiver_inventory",
                            "rank": rank,
                            "processes": state["receiver"].inventory(),
                            "calibration": calibrate(
                                state["receiver"].snapshot, iterations
                            ),
                        }
                    )
                except Exception as exc:
                    state["receiver_error"] = f"discovery: {type(exc).__name__}: {exc}"
                    logger.warning(
                        "CPU receiver discovery failed; training continues: %s",
                        state["receiver_error"],
                    )
        if state["receiver"] is not None and state["receiver_error"] is None:
            try:
                changed = state["receiver"].refresh()
                if changed and state["journal"] is not None:
                    state["journal"].append(
                        {
                            "kind": "cpu_receiver_inventory",
                            "rank": rank,
                            "before_update_index": wall_probe._update_index + 1,
                            "processes": state["receiver"].inventory(),
                        }
                    )
                state["membership_before"] = state["receiver"].membership()
            except Exception as exc:
                state["receiver_error"] = (
                    f"identity before: {type(exc).__name__}: {exc}"
                )
        state["observer_before_wall_ns"] = time.perf_counter_ns() - observer_start
        try:
            return original_wall_wrapper(self)
        finally:
            observer_end_start = time.perf_counter_ns()
            try:
                if state["receiver"] is not None:
                    state["receiver"].refresh()
                    if state["receiver"].membership() != state.get("membership_before"):
                        if state["journal"] is not None:
                            state["journal"].append(
                                {
                                    "kind": "cpu_receiver_inventory",
                                    "rank": rank,
                                    "after_update_index": wall_probe._update_index,
                                    "processes": state["receiver"].inventory(),
                                }
                            )
                        raise RuntimeError(
                            "Receiver process membership changed during update"
                        )
            except Exception as exc:
                state["receiver_error"] = f"identity after: {type(exc).__name__}: {exc}"
                if state["journal"] is not None:
                    for event in state["journal"].pending:
                        if event["kind"] == "cpu_trainer_update":
                            event.update(cpu_valid=False, error=state["receiver_error"])
            try:
                if state["journal"] is not None:
                    for event in state["journal"].pending:
                        if event["kind"] == "cpu_trainer_update":
                            event["observer_after_wall_ns"] = (
                                time.perf_counter_ns() - observer_end_start
                            )
                    state["journal"].flush()
            except Exception as exc:
                logger.warning(
                    "CPU v2 journal flush failed; training continues: %s", exc
                )

    WeightUpdater.update_weights = flush_outside_wall
