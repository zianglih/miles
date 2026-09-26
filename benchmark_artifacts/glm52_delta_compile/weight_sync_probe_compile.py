"""Record compiler activity outside the unchanged v2 updater CPU/wall windows.

The two dictionary snapshots and one JSON append remain inside driver wall.
Counter deltas bracket the entire updater, including initialization on update 0;
they identify graph generation, not its individual call sites or cache hits.
"""

from functools import wraps

import weight_sync_probe as wall_probe
import weight_sync_probe_cpu_v2 as cpu_probe


def compiler_counters():
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    result = {
        f"{group}.{key}": int(value)
        for group, values in counters.items()
        for key, value in values.items()
        if isinstance(value, (int, float))
    }
    result["inductor.generated_kernel_count"] = int(metrics.generated_kernel_count)
    return result


def install(args):
    import torch.distributed as dist
    from miles.backends.training_utils.weight_update.updater import WeightUpdater

    rank = dist.get_rank()
    cpu_probe.install(args)
    compiler_counters()  # Import/setup before any update timing starts.
    original_update = WeightUpdater.update_weights

    @wraps(original_update)
    def observed_update(self):
        before = compiler_counters()
        success = False
        try:
            result = original_update(self)
            success = True
            return result
        finally:
            after = compiler_counters()
            wall_probe.record(
                "compile_observation",
                rank=rank,
                success=success,
                cpu_backend=getattr(args, "update_weight_delta_cpu_backend", None),
                before=before,
                after=after,
                delta={
                    key: after.get(key, 0) - before.get(key, 0)
                    for key in sorted(set(before) | set(after))
                    if after.get(key, 0) != before.get(key, 0)
                },
            )

    WeightUpdater.update_weights = observed_update
