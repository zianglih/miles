"""Bounded CPU-only comparison of native PyTorch Inductor preparation variants."""
import gc
import json
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

OPTIONS = {"cpp_wrapper": True, "cpp.threads": 1, "cpp.dynamic_threads": False}


def count16(new, old):
    return new.view(-1, 4096).ne(old.view(-1, 4096)).sum(dim=1, dtype=torch.int16).to(torch.int64)


def count32(new, old):
    return new.view(-1, 4096).ne(old.view(-1, 4096)).sum(dim=1, dtype=torch.int32).to(torch.int64)


def count_words(new, old):
    difference = new.view(torch.int32).view(-1, 1024) ^ old.view(torch.int32).view(-1, 1024)
    counts = ((difference & 255) != 0).to(torch.int32)
    counts = counts + ((difference & 65280) != 0)
    counts = counts + ((difference & 16711680) != 0)
    counts = counts + ((difference & -16777216) != 0)
    return counts.sum(dim=1, dtype=torch.int32).to(torch.int64)


def fused_word_prepare(new, old):
    new_words, old_words = new.view(torch.int32).view(-1, 1024), old.view(torch.int32).view(-1, 1024)
    difference = new_words ^ old_words
    counts = ((difference & 255) != 0).to(torch.int32)
    counts = counts + ((difference & 65280) != 0)
    counts = counts + ((difference & 16711680) != 0)
    counts = counts + ((difference & -16777216) != 0)
    return counts.sum(dim=1, dtype=torch.int32).to(torch.int64), difference.view(torch.uint8).flatten(), new_words.clone().view(torch.uint8).flatten()


def byte_materialize(new, old):
    return new ^ old, new.clone()


def word_materialize(new, old):
    new_words, old_words = new.view(torch.int32), old.view(torch.int32)
    return (new_words ^ old_words).view(torch.uint8), new_words.clone().view(torch.uint8)


def main():
    output = Path(os.environ["PROBE_OUTPUT"])
    with output.open("x") as stream:
        def emit(record):
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            stream.write(line + "\n")
            stream.flush()
        emit({"event": "provenance", "host": platform.node(), "torch": torch.__version__, "torch_git": torch.version.git_version,
              "capability": torch.backends.cpu.get_cpu_capability(), "intra_threads": torch.get_num_threads(), "options": OPTIONS})
        functions = {f.__name__: torch.compile(f, dynamic=True, fullgraph=True, options=OPTIONS)
                     for f in (fused_word_prepare,)}
        old = torch.arange(128 << 20, dtype=torch.uint8)
        new = old.clone()
        new[::223] ^= 255
        with ThreadPoolExecutor(max_workers=1) as pool:
            for name, f in functions.items():
                start = time.perf_counter()
                result = pool.submit(f, new, old).result()
                if name.startswith("fused"):
                    assert int(result[0].sum()) == int(np.count_nonzero(new.numpy() != old.numpy()))
                    assert np.array_equal(result[1].numpy(), new.numpy() ^ old.numpy())
                    assert np.array_equal(result[2].numpy(), new.numpy())
                elif name.startswith("count"):
                    assert int(result.sum()) == int(np.count_nonzero(new.numpy() != old.numpy()))
                else:
                    assert np.array_equal(result[0].numpy(), new.numpy() ^ old.numpy())
                    assert np.array_equal(result[1].numpy(), new.numpy())
                    assert result[1].data_ptr() != new.data_ptr()
                del result
                emit({"event": "cold", "variant": name, "wall_s": time.perf_counter() - start})
            for size in (4 << 20, 128 << 20):
                rows = {name: [] for name in functions}
                for iteration in range(9):
                    for name in list(functions)[::1 if iteration % 2 == 0 else -1]:
                        f = functions[name]
                        cpu, wall = time.process_time(), time.perf_counter()
                        result = pool.submit(f, new[:size], old[:size]).result()
                        cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
                        del result
                        row = {"event": "sample", "variant": name, "bytes": size, "iteration": iteration,
                               "wall_s": wall, "cpu_s": cpu, "warmup": iteration < 2}
                        emit(row)
                        if iteration >= 2:
                            rows[name].append(row)
                emit({"event": "summary", "bytes": size, "medians": {name: {"wall_s": statistics.median(r['wall_s'] for r in values),
                      "cpu_s": statistics.median(r['cpu_s'] for r in values)} for name, values in rows.items()}})
        emit({"event": "complete"})

if __name__ == '__main__':
    main()
