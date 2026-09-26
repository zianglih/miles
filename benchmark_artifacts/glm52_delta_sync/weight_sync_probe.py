"""Project-only observation hooks; does not change model weights or reward behavior."""

from functools import wraps
import json
import os
from pathlib import Path
import time


_update_index = -1


def record(kind, *, rank=None, **fields):
    root = Path(os.environ["WEIGHT_SYNC_RUN_DIR"])
    name = "driver" if rank is None else f"trainer-rank{rank}"
    payload = {
        "kind": kind,
        "rank": rank,
        "pid": os.getpid(),
        "update_index": _update_index,
        "mode": os.environ["WEIGHT_SYNC_MODE"],
        "time_ns": time.time_ns(),
        **fields,
    }
    with (root / f"{name}.jsonl").open("a") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")


def install(args):
    # Delay these imports until Megatron's init hook: importing them in the
    # launcher would initialize CUDA and distributed libraries unnecessarily.
    import torch.distributed as dist
    from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
    from miles.backends.training_utils.weight_update.protocols.delta import (
        UpdateWeightFromDiskDelta,
    )
    from miles.backends.training_utils.weight_update.updater import WeightUpdater
    from miles.utils.timer import Timer

    rank = dist.get_rank()
    original_update = WeightUpdater.update_weights

    @wraps(original_update)
    def measured_update(self):
        global _update_index
        _update_index += 1
        before_version = self.weight_version
        start = time.perf_counter_ns()
        success = False
        try:
            result = original_update(self)
            success = True
            return result
        finally:
            end = time.perf_counter_ns()
            record(
                "trainer_update",
                rank=rank,
                start_ns=start,
                end_ns=end,
                elapsed_s=(end - start) / 1e9,
                success=success,
                initial=_update_index == 0,
                version_before=before_version,
                version_after=self.weight_version,
            )

    WeightUpdater.update_weights = measured_update

    def timer_sink(name, start, end):
        if "update_weight" in name or name == "finalize_and_resume_engines":
            record(
                "native_timer",
                rank=rank,
                name=name,
                start_unix_s=start,
                end_unix_s=end,
                elapsed_s=end - start,
            )

    Timer().event_sinks.append(timer_sink)
    original_metrics = UpdateWeightFromDiskDelta._record_metrics

    @wraps(original_metrics)
    def measured_metrics(self, weight_version):
        original_metrics(self, weight_version)
        record(
            "delta_bytes",
            rank=rank,
            version=weight_version,
            changed_bytes=self.changed_bytes,
            total_bytes=self.total_bytes,
            wire_bytes=self.wire_bytes,
            global_metrics=self.update_weight_metrics,
        )

    UpdateWeightFromDiskDelta._record_metrics = measured_metrics

    for name in (
        "_capture_baseline",
        "_begin_encode",
        "after_base_weights",
        "_write_delta_files",
        "_reload_engines",
    ):
        original = getattr(UpdateWeightFromDiskDelta, name)

        def make_phase_wrapper(method, phase):
            @wraps(method)
            def wrapper(self, *pos, **kw):
                start = time.perf_counter_ns()
                success = False
                try:
                    result = method(self, *pos, **kw)
                    success = True
                    return result
                finally:
                    end = time.perf_counter_ns()
                    record(
                        "delta_phase",
                        rank=rank,
                        phase=phase,
                        start_ns=start,
                        end_ns=end,
                        elapsed_s=(end - start) / 1e9,
                        success=success,
                    )

            return wrapper

        setattr(UpdateWeightFromDiskDelta, name, make_phase_wrapper(original, name))

    if rank == 0:
        original_request = SGLangApiClient._make_request
        observed = {
            "pull_weights",
            "update_weights_from_disk",
            "begin_weight_update",
            "end_weight_update",
            "update_weight_version",
        }

        @wraps(original_request)
        async def measured_request(self, endpoint, payload=None):
            if endpoint not in observed:
                return await original_request(self, endpoint, payload)
            start = time.perf_counter_ns()
            success = False
            result = None
            try:
                result = await original_request(self, endpoint, payload)
                success = (
                    not isinstance(result, dict) or result.get("success") is not False
                )
                return result
            finally:
                end = time.perf_counter_ns()
                record(
                    "engine_rpc",
                    rank=rank,
                    endpoint=endpoint,
                    engine=self.server_url,
                    start_ns=start,
                    end_ns=end,
                    elapsed_s=(end - start) / 1e9,
                    success=success,
                    request=payload,
                    response=result,
                )

        SGLangApiClient._make_request = measured_request
        # These three client methods bypass _make_request and return an HTTP
        # response (or None), so record them directly without serializing it.
        for name in ("pause_generation", "continue_generation", "flush_cache"):
            method = getattr(SGLangApiClient, name)

            def make_rpc_wrapper(original, endpoint):
                @wraps(original)
                async def wrapper(self, *pos, **kw):
                    start = time.perf_counter_ns()
                    success = False
                    try:
                        result = await original(self, *pos, **kw)
                        success = True
                        return result
                    finally:
                        end = time.perf_counter_ns()
                        record(
                            "engine_rpc",
                            rank=rank,
                            endpoint=endpoint,
                            engine=self.server_url,
                            start_ns=start,
                            end_ns=end,
                            elapsed_s=(end - start) / 1e9,
                            success=success,
                            request=kw,
                        )

                return wrapper

            setattr(SGLangApiClient, name, make_rpc_wrapper(method, name))
    record("probe_installed", rank=rank, seed=args.seed, rollout_seed=args.rollout_seed)
