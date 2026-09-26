"""Project-only exactness checks for an unadopted compiled CPU-stage candidate."""
import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('fused_delta_candidate', Path(__file__).with_name(os.environ.get('DELTA_CANDIDATE_MODULE', 'delta_preparation_fused_variant.py')))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def main():
    from torch._dynamo.utils import counters
    values = np.arange(256, dtype=np.uint8)
    old_pairs = np.repeat(np.repeat(values, 256), 4)
    new_pairs = np.repeat(np.tile(values, 256), 4)
    layouts = [module.PackedDeltaLayout((("empty", 0), ("pairs", old_pairs.size), ("tail", 67))),
               module.PackedDeltaLayout((("s0", 4), ("s1", 4), ("s2", 4)), 4),
               module.PackedDeltaLayout((("large", (1 << 24) + 17),))]
    jobs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for layout in layouts:
            old, incoming = layout.allocate(), layout.allocate()
            views_old, views_new = dict(layout.views(old)), dict(layout.views(incoming))
            for name in views_old:
                views_old[name].copy_(torch.arange(views_old[name].numel(), dtype=torch.uint8))
                views_new[name].copy_(views_old[name])
            if 'pairs' in views_old:
                views_old['pairs'].copy_(torch.from_numpy(old_pairs))
                views_new['pairs'].copy_(torch.from_numpy(new_pairs))
                views_new['tail'][-1] ^= 255
            elif 'large' in views_old:
                views_new['large'].bitwise_xor_(255)
            else:
                views_new['s1'][3] ^= 128
            stage = module.CpuDeltaPreparer(layout, kernel_threads=int(os.environ.get('DELTA_KERNEL_THREADS', '1')))
            pool.submit(stage.warmup, old, incoming).result()
            jobs.append((layout, stage, old, incoming))
        before = counters['stats']['unique_graphs']
        for layout, stage, old, incoming in jobs:
            previous, expected = old.clone(), incoming.clone()
            counts = tuple(int(np.count_nonzero(a.numpy() != b.numpy()))
                           for (_, a), (_, b) in zip(layout.views(old), layout.views(incoming)))
            results = [f.result() for f in [pool.submit(stage.prepare, incoming, old) for _ in range(4)]]
            incoming.fill_(42)
            for result in results:
                assert result.changed_counts == counts
                assert result.xor is not None
                assert result.snapshot.data_ptr() not in (incoming.data_ptr(), old.data_ptr())
                torch.testing.assert_close(result.snapshot, expected)
                torch.testing.assert_close(result.xor ^ previous, expected)
            torch.testing.assert_close(old, previous)
            unchanged = pool.submit(stage.prepare, previous.clone(), previous).result()
            assert unchanged.changed_counts == (0,) * len(counts)
            assert unchanged.xor is not None and not unchanged.xor.any()
            assert unchanged.copied_unchanged_bytes == sum(n for _, n in layout.entries)
        assert counters['stats']['unique_graphs'] == before
    print('3 layouts x 4 concurrent preparations + 3 no-change preparations passed; exact byte pairs, tails, >2^24 counts, ownership, and no timed recompilation')

if __name__ == '__main__':
    main()
