"""Compare packed CPU delta preparation with the exact f17 CPU workers.

Representative buckets use sorted names, at most 128 MiB of padded storage
(except a single oversized tensor), and separate K=4 FP32 scalar-scale buckets.
This is not a capture of live iterator buckets. Checkpoint reads, reconstruction,
packing, incoming allocation, worker startup, and verification are untimed. The
timed stage includes submission/drain, counts, owned XOR/snapshot materialization,
one lease return per bucket, and native per-name zstd/checksum. The exact f17 AST
uses the same named bytes with its pinned=True path, including the exact scalar
batch worker for the same compact FP32 groups. Its lease returns are per tensor
or compact scalar group. Scalar batch copies are included even when unchanged.
No GPU packing/D2H, bounded staging-pool backpressure, snapshot commit, file
publication, or receiver behavior is measured. No global thread limit is changed.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import queue
import statistics
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zstandard

import benchmark_sender_replay as replay

ROOT = Path(__file__).resolve().parent
F17 = "f17ba4bce13bf7d357e7560182dc859c41a7cb37"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_preparation(path):
    name = "packed_delta_preparation_benchmark"
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def compile_counters():
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    result = {f"{group}.{key}": int(value) for group, values in counters.items()
              for key, value in values.items() if isinstance(value, (int, float))}
    result["inductor.generated_kernel_count"] = int(metrics.generated_kernel_count)
    return result


def counter_delta(before, after):
    return {key: after.get(key, 0) - before.get(key, 0)
            for key in sorted(set(before) | set(after)) if after.get(key, 0) != before.get(key, 0)}


def graph_activity(delta):
    return any(delta.get(key, 0) > 0 for key in (
        "frames.total", "stats.unique_graphs", "aot_autograd.total",
        "inductor.fxgraph_cache_hit", "inductor.fxgraph_cache_miss",
        "inductor.generated_kernel_count"))


def load_batch_worker(source_path, worker, baseline_info):
    """Compile the original batch AST in the exact per-tensor worker namespace."""
    source = source_path.read_text()
    model = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.ClassDef) and node.name == "UpdateWeightFromDiskDelta")
    method = next(node for node in model.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_diff_and_compress_batch")
    namespace = dict(worker.__globals__)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    baseline_info["batch_worker_source_sha256"] = hashlib.sha256(
        ast.get_source_segment(source, method).encode()).hexdigest()
    baseline_info["batch_worker_ast_sha256"] = hashlib.sha256(
        ast.dump(method, include_attributes=False).encode()).hexdigest()
    return namespace["_diff_and_compress_batch"]


def scalar_names(locations):
    scalars = set()
    for path in sorted({entry[0] for entry in locations.values()}):
        header, _, _ = replay.read_header(path)
        for name, metadata in header.items():
            if name in locations and name.endswith(".weight_scale_2") and metadata.get("dtype") == "F32" \
                    and metadata.get("shape") == [] and locations[name][2] == 4:
                scalars.add(name)
    return scalars


def make_layouts(module, locations, block_bytes, maximum):
    scalars = scalar_names(locations)
    layouts = []
    for kind, block, names in (
        ("weights", block_bytes, sorted(set(locations) - scalars)),
        ("fp32_scalar_weight_scale_2", 4, sorted(scalars)),
    ):
        entries, padded = [], 0
        for name in names:
            nbytes = locations[name][2]
            allocation = ((nbytes + block - 1) // block) * block
            if entries and padded + allocation > maximum:
                layouts.append((kind, module.PackedDeltaLayout(tuple(entries), block)))
                entries, padded = [], 0
            entries.append((name, nbytes))
            padded += allocation
            if allocation > maximum:
                layouts.append((kind, module.PackedDeltaLayout(tuple(entries), block)))
                entries, padded = [], 0
        if entries:
            layouts.append((kind, module.PackedDeltaLayout(tuple(entries), block)))
    return layouts


@dataclass
class Bucket:
    index: int
    kind: str
    layout: object
    old: torch.Tensor
    incoming: torch.Tensor | None = None
    preparer: object = None


def baseline_tasks(buckets):
    """Reuse stage scalar groups; ordinary weights keep f17's per-tensor path."""
    tasks = []
    for bucket in buckets:
        if bucket.kind == "fp32_scalar_weight_scale_2" and len(bucket.layout.entries) > 1:
            if bucket.layout.block_bytes != 4 or any(size != 4 for _, size in bucket.layout.entries):
                raise ValueError("The f17 scalar batch requires compact four-byte FP32 scalars")
            tasks.append((tuple(name for name, _ in bucket.layout.entries), bucket.incoming,
                          bucket.layout.padded_bytes))
        else:
            tasks.extend(((name,), view, view.numel()) for name, view in bucket.layout.views(bucket.incoming))
    return tasks


def named_arrays(buckets, field):
    return dict(sorted((name, view.numpy()) for bucket in buckets
                       for name, view in bucket.layout.views(getattr(bucket, field))))


def read_into_views(locations, arrays):
    # Only two complete input states are owned: old packed allocations and their
    # incoming clones. No canonical per-name arrays or reconstruction copies.
    for name, (path, offset, size) in locations.items():
        target = memoryview(arrays[name]).cast("B")
        with path.open("rb", buffering=0) as stream:
            stream.seek(offset)
            position = 0
            while position < size:
                count = stream.readinto(target[position:])
                if not count:
                    raise ValueError(f"Truncated canonical tensor {name!r}")
                position += count


def fixture(directory, algorithm, checksum):
    import safetensors.numpy

    directory.mkdir(parents=True, exist_ok=False)
    canonical, source = directory / "canonical", directory / "deltas"
    canonical.mkdir()
    source.mkdir()
    rng = np.random.default_rng(723)
    tensors = {
        "model.layers.0.weight": rng.integers(0, 256, 4099, dtype=np.uint8),
        "model.layers.1.weight": rng.integers(0, 256, 98317, dtype=np.uint8),
        "model.layers.2.weight": np.empty(0, dtype=np.uint8),
        "model.layers.3.weight": rng.integers(0, 256, 65536, dtype=np.uint8),
        "model.layers.0.experts.0.weight_scale_2": np.array(1.25, dtype=np.float32),
        "model.layers.0.experts.1.weight_scale_2": np.array(-2.5, dtype=np.float32),
        "model.layers.0.experts.2.weight_scale_2": np.array(0.0, dtype=np.float32),
        "model.router.bias": np.arange(257, dtype=np.float32),
    }
    safetensors.numpy.save_file(tensors, str(canonical / "model.safetensors"))
    previous = {name: value.reshape(-1).view(np.uint8).copy() for name, value in tensors.items()}
    for version in (1, 2):
        current = {name: value.copy() for name, value in previous.items()}
        current["model.layers.0.weight"][version::31] ^= np.uint8(129)
        if version == 2:
            current["model.layers.1.weight"][::17] ^= np.uint8(255)
            current["model.layers.0.experts.1.weight_scale_2"][0] ^= np.uint8(1)
        else:
            current["model.layers.0.experts.0.weight_scale_2"][0] ^= np.uint8(1)
        payloads, digests = {}, {}
        for name in sorted(previous):
            diff = current[name] ^ previous[name]
            if np.count_nonzero(diff):
                payloads[name] = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
                digests[name] = checksum(algorithm, current[name])
        target = source / f"weight_v{version:06d}"
        target.mkdir()
        shard = "model-00000-of-00001.safetensors"
        safetensors.numpy.save_file(payloads, str(target / shard), metadata=digests)
        index = {"metadata": {"version": version, "base_version": version - 1,
                             "delta_encoding": "xor", "compression_format": "zstd", "checksum_format": algorithm},
                 "weight_map": {name: shard for name in payloads}}
        (target / "model.safetensors.index.json").write_text(json.dumps(index, sort_keys=True))
        previous = current
    return canonical, source


def prepare_inputs(args, module, layouts, locations, case, checksum):
    buckets = [Bucket(index, kind, layout, layout.allocate())
               for index, (kind, layout) in enumerate(layouts)]
    old = named_arrays(buckets, "old")
    read_into_views(locations, old)
    lineage, algorithm = [], args.checksum
    if case == "synthetic-v1-v2":
        lineage.append(replay.apply_version(old, args.synthetic_source, 1, checksum))
        algorithm = lineage[0]["metadata"]["checksum_format"]
    for bucket in buckets:
        bucket.incoming = bucket.old.clone()
        bucket.preparer = module.CpuDeltaPreparer(bucket.layout, use_compile=not args.no_compile,
                                                kernel_threads=args.kernel_threads)
    incoming = named_arrays(buckets, "incoming")
    if case == "synthetic-v1-v2":
        lineage.append(replay.apply_version(incoming, args.synthetic_source, 2, checksum))
        if any(item["metadata"]["checksum_format"] != algorithm for item in lineage):
            raise ValueError("Synthetic checksum algorithm changes between versions")
    published = lineage[-1]["published_names"] if lineage else []
    changed, description = replay.describe_inputs(old, incoming, published)
    expected = {name: checksum(algorithm, value) for name, value in incoming.items()}
    old_checksums = {name: checksum(algorithm, value) for name, value in old.items()}
    return buckets, old, incoming, changed, expected, old_checksums, algorithm, lineage, description


def stage_task(bucket, free_queue, algorithm, checksum):
    prepared = None
    try:
        prepared = bucket.preparer.prepare(bucket.incoming, bucket.old)
        if prepared.xor is not None:
            incoming_storage = bucket.incoming.untyped_storage().data_ptr()
            if prepared.snapshot.untyped_storage().data_ptr() == incoming_storage \
                    or prepared.xor.untyped_storage().data_ptr() == incoming_storage:
                raise AssertionError("Prepared data still aliases the staging lease")
    finally:
        # Exactly once, before any NumPy/zstd/checksum handoff, including failures.
        free_queue.put(bucket.index)
    changed_views = {name: (snapshot, diff, count)
                     for name, snapshot, diff, count in prepared.changed_views()}
    old_views = dict(bucket.layout.views(bucket.old))
    results = []
    for (name, _), count in zip(bucket.layout.entries, prepared.changed_counts, strict=True):
        if not count:
            results.append((name, old_views[name].numpy(), None, None, 0))
            continue
        snapshot, diff, checked_count = changed_views[name]
        snapshot, diff = snapshot.numpy(), diff.numpy()
        compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
        results.append((name, snapshot, compressed, checksum(algorithm, snapshot), checked_count))
    return results, {"copied_unchanged_bytes": prepared.copied_unchanged_bytes,
                     "copied_padding_bytes": prepared.copied_padding_bytes,
                     "snapshot_copy_bytes": prepared.layout.padded_bytes if prepared.xor is not None else 0,
                     "materialized_buckets": int(prepared.xor is not None),
                     "lease_returns": 1}


def drain_returns(free_queue, expected):
    returned = []
    while True:
        try:
            returned.append(free_queue.get_nowait())
        except queue.Empty:
            break
    if len(returned) != expected:
        raise AssertionError(f"Expected {expected} lease returns, got {len(returned)}")
    return returned


def run_case(args, module, module_hash, module_path, layouts, locations, case, worker, batch_worker, checksum, emit,
             *, first_case, cache_initially_empty):
    setup_start = time.perf_counter()
    (buckets, old, incoming, changed, expected, old_checksums, algorithm,
     lineage, description) = prepare_inputs(args, module, layouts, locations, case, checksum)
    tasks = baseline_tasks(buckets)
    scalar_batch_names = {name for names, _, _ in tasks if len(names) > 1 for name in names}
    emit({"event": "workload", "case": case, "encoding": "xor", "checksum": algorithm,
          "setup_wall_s_untimed": time.perf_counter() - setup_start, "lineage": lineage,
          "old_state_checksum_manifest_sha256": hashlib.sha256(json.dumps(old_checksums, sort_keys=True).encode()).hexdigest(),
          "new_state_checksum_manifest_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
          "padded_input_bytes": 2 * sum(bucket.layout.padded_bytes for bucket in buckets),
          "baseline_task_count": len(tasks),
          "baseline_scalar_batch_count": sum(len(names) > 1 for names, _, _ in tasks),
          "baseline_scalar_batch_tensor_count": len(scalar_batch_names), **description})
    measured = {"exact_f17": [], "packed_stage": []}
    with ThreadPoolExecutor(max_workers=args.baseline_workers) as baseline_pool, \
            ThreadPoolExecutor(max_workers=args.stage_workers) as stage_pool:
        replay.warm_pool(baseline_pool, args.baseline_workers)
        replay.warm_pool(stage_pool, args.stage_workers)
        # Dispatcher initialization/compilation happens serially in the same
        # ordinary worker context used by stage tasks, not in actor inference mode.
        warmed = []
        before = compile_counters()
        cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
        if not args.no_compile:
            for bucket in buckets:
                stage_pool.submit(bucket.preparer.warmup, bucket.old, bucket.incoming).result()
                warmed.append(bucket.index)
        emit({"event": "compile_warmup", "case": case,
              "phase": "initial_compile_and_warmup" if first_case else "already_initialized_warmup",
              "initial_empty_cache_compile": first_case and cache_initially_empty and not args.no_compile,
              "wall_s": (time.perf_counter_ns() - wall_start) / 1e9,
              "process_cpu_s": (time.process_time_ns() - cpu_start) / 1e9,
              "process_cpu_excludes_compiler_children": True,
              "cache_may_be_warm": True, "new_compile_counters": counter_delta(before, compile_counters()),
              "block_sizes": sorted({buckets[index].layout.block_bytes for index in warmed}),
              "actual_layouts_warmed": warmed,
              "warmup_inputs": "each actual bucket.old and bucket.incoming, serially before fanout"})
        for iteration in range(args.warmups + args.repeats):
            order = ("exact_f17", "packed_stage") if iteration % 2 == 0 else ("packed_stage", "exact_f17")
            for variant in order:
                free_queue = queue.Queue()
                state = types.SimpleNamespace(_snapshot=old, _free_q=free_queue,
                                              delta_encoding="xor", checksum_algorithm=algorithm)
                state._diff_and_compress = types.MethodType(worker, state)
                gc.collect()
                before = compile_counters()
                cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
                if variant == "exact_f17":
                    futures = [baseline_pool.submit(batch_worker, state, names, view, size, True)
                               for names, view, size in tasks]
                    results = [result for future in futures for result in future.result()]
                else:
                    futures = [stage_pool.submit(stage_task, bucket, free_queue, algorithm, checksum) for bucket in buckets]
                    batches = [future.result() for future in futures]
                    results = [result for batch, _ in batches for result in batch]
                wall_s = (time.perf_counter_ns() - wall_start) / 1e9
                cpu_s = (time.process_time_ns() - cpu_start) / 1e9
                if variant == "exact_f17":
                    metadata = {"copied_unchanged_bytes": sum(old[name].nbytes for name in scalar_batch_names
                                                             if not changed[name]),
                                "copied_padding_bytes": 0,
                                "snapshot_copy_bytes": sum(old[name].nbytes for name, count in changed.items()
                                                           if count or name in scalar_batch_names),
                                "lease_returns": len(tasks)}
                else:
                    metadata = {key: sum(item[key] for _, item in batches) for key in batches[0][1]}
                delta = counter_delta(before, compile_counters())
                returned = drain_returns(free_queue, len(tasks) if variant == "exact_f17" else len(buckets))
                if variant == "packed_stage" and sorted(returned) != list(range(len(buckets))):
                    raise AssertionError("Stage duplicated or omitted a bucket lease return")
                if variant == "exact_f17" and sorted(map(id, returned)) != sorted(id(view) for _, view, _ in tasks):
                    raise AssertionError("Baseline duplicated or omitted a tensor/scalar-group lease return")
                if len({result[0] for result in results}) != len(old) or {result[0] for result in results} != set(old):
                    raise AssertionError("Results duplicated or omitted tensor names")
                payload = replay.verify_results(results, old, incoming, changed, expected, "xor", algorithm, checksum)
                for name in old:
                    if checksum(algorithm, old[name]) != old_checksums[name] \
                            or checksum(algorithm, incoming[name]) != expected[name]:
                        raise AssertionError(f"Input mutation: {name}")
                record = {"event": "sample", "case": case, "variant": variant, "iteration": iteration,
                          "warmup": iteration < args.warmups, "wall_s": wall_s, "process_cpu_s": cpu_s,
                          "new_compile_counters": delta, "new_graphs_or_kernels": graph_activity(delta),
                          "compressed_payload_bytes": payload, "exact_bytes_counts_checksums_verified": True,
                          "lease_return_once_before_compression": True if variant == "packed_stage" else None,
                          "lease_return_scope": "bucket" if variant == "packed_stage" else "tensor_or_scalar_group_exact_ast",
                          **metadata}
                emit(record)
                if iteration >= args.warmups:
                    if record["new_graphs_or_kernels"]:
                        raise RuntimeError(f"Compilation occurred in measured {variant} sample {iteration}: {delta}")
                    measured[variant].append(record)
                del results, futures, state, returned
                if variant == "packed_stage":
                    del batches
        summaries = {variant: {"median_wall_s": statistics.median(row["wall_s"] for row in rows),
                               "median_process_cpu_s": statistics.median(row["process_cpu_s"] for row in rows),
                               "raw_wall_s": [row["wall_s"] for row in rows],
                               "raw_process_cpu_s": [row["process_cpu_s"] for row in rows],
                               "steady_state_clean": not any(row["new_graphs_or_kernels"] for row in rows)}
                     for variant, rows in measured.items()}
        emit({"event": "summary", "case": case, "variants": summaries,
              "stage_wall_reduction_fraction": 1 - summaries["packed_stage"]["median_wall_s"] / summaries["exact_f17"]["median_wall_s"],
              "stage_cpu_reduction_fraction": 1 - summaries["packed_stage"]["median_process_cpu_s"] / summaries["exact_f17"]["median_process_cpu_s"]})
    if sha256(module_path) != module_hash:
        raise RuntimeError("Preparation module changed during benchmark; discard this run")
    return all(value["steady_state_clean"] for value in summaries.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles-path", type=Path, default=ROOT / "miles")
    parser.add_argument("--preparation-module", type=Path, default=ROOT / "miles-torch-compile/miles/utils/delta_preparation.py")
    parser.add_argument("--baseline-ref", default=F17)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--synthetic-source", type=Path)
    parser.add_argument("--fixture", action="store_true", help="Generate a deterministic tiny local checkpoint and v1/v2")
    parser.add_argument("--cases", nargs="+", choices=("native-unchanged", "synthetic-v1-v2"),
                        default=["native-unchanged", "synthetic-v1-v2"])
    parser.add_argument("--baseline-workers", type=int, default=32)
    parser.add_argument("--stage-workers", type=int, default=32)
    parser.add_argument("--kernel-threads", type=int, default=1)
    parser.add_argument("--block-bytes", type=int, choices=(256, 4096), default=4096)
    parser.add_argument("--bucket-mib", type=float, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checksum", default="xxh3-128")
    parser.add_argument("--exclude-suffix", nargs="*", default=[".input_scale"])
    parser.add_argument("--expected-tensors", type=int)
    parser.add_argument("--max-input-gib", type=float, default=64)
    parser.add_argument("--no-compile", action="store_true", help="Eager plumbing diagnostic; not a compiled performance claim")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.baseline_workers, args.stage_workers, args.kernel_threads, args.repeats) < 1 \
            or args.warmups < 0 or args.bucket_mib <= 0 or args.max_input_gib <= 0:
        parser.error("worker/thread/repeat/bucket/memory settings must be positive; warmups nonnegative")
    if args.fixture and (args.checkpoint or args.synthetic_source):
        parser.error("--fixture owns its inputs; omit --checkpoint and --synthetic-source")
    if not args.fixture and (args.checkpoint is None or
                            ("synthetic-v1-v2" in args.cases and args.synthetic_source is None)):
        parser.error("Supply --checkpoint and, for synthetic-v1-v2, --synthetic-source; or use --fixture")
    if args.output.exists():
        parser.error("Output exists; select a new evidence path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_dir = args.output.parent / (args.output.stem + "-sources")
    source_dir.mkdir(exist_ok=False)
    cache = args.cache_dir or args.output.parent / (args.output.stem + "-inductor-cache")
    cache.mkdir(parents=True, exist_ok=True)
    cache_was_empty = not any(cache.iterdir())
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache.resolve())
    worker, checksum, baseline_info = replay.load_worker(args.miles_path, args.baseline_ref, "exact-f17", source_dir)
    baseline_source = (source_dir / "exact-f17-delta.py").read_text()
    worker_ast = next(node for node in ast.walk(ast.parse(baseline_source))
                      if isinstance(node, ast.FunctionDef) and node.name == "_diff_and_compress")
    baseline_info["worker_ast_sha256"] = hashlib.sha256(ast.dump(worker_ast, include_attributes=False).encode()).hexdigest()
    batch_worker = load_batch_worker(source_dir / "exact-f17-delta.py", worker, baseline_info)
    module_hash = sha256(args.preparation_module)
    (source_dir / "delta_preparation.py").write_bytes(args.preparation_module.read_bytes())
    (source_dir / "benchmark_sender_replay.py").write_bytes(Path(replay.__file__).read_bytes())
    (source_dir / "benchmark_delta_preparation.py").write_bytes(Path(__file__).read_bytes())
    module = load_preparation(args.preparation_module)
    if args.fixture:
        args.checkpoint, args.synthetic_source = fixture(args.output.parent / (args.output.stem + "-fixture"),
                                                        args.checksum, checksum)
    canonical, files = replay.checkpoint_locations(args.checkpoint)
    locations = {name: value for name, value in canonical.items()
                 if not any(name.endswith(suffix) for suffix in args.exclude_suffix)}
    if not locations or (args.expected_tensors is not None and len(locations) != args.expected_tensors):
        raise ValueError("Selected tensor count is empty or differs from --expected-tensors")
    layouts = make_layouts(module, locations, args.block_bytes, int(args.bucket_mib * (1 << 20)))
    payload = sum(item[2] for item in locations.values())
    padded = sum(layout.padded_bytes for _, layout in layouts)
    if 2 * padded > args.max_input_gib * (1 << 30):
        raise ValueError("Two packed input states exceed --max-input-gib; raise it explicitly")
    layout_manifest = [{"index": index, "kind": kind, "block_bytes": layout.block_bytes,
                        "padded_bytes": layout.padded_bytes, "entries": layout.entries, "offsets": layout.offsets}
                       for index, (kind, layout) in enumerate(layouts)]
    manifest_path = source_dir / "layouts.json"
    manifest_path.write_text(json.dumps(layout_manifest, indent=2) + "\n")
    with args.output.open("x") as output:
        def emit(record):
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()

        emit({"event": "provenance", "scope": __doc__, "host": platform.node(), "platform": platform.platform(),
              "python": sys.version, "cpu_count": os.cpu_count(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "torch_intra_threads": torch.get_num_threads(), "torch_inter_threads": torch.get_num_interop_threads(),
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "zstandard", "xxhash")},
              "torch_git": torch.version.git_version,
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "baseline": baseline_info, "module_sha256": module_hash, "script_sha256": sha256(__file__),
              "replay_helper_sha256": sha256(replay.__file__), "cache_dir": str(cache), "cache_initially_empty": cache_was_empty,
              "canonical_files": files, "canonical_tensor_count": len(canonical), "selected_tensor_count": len(locations),
              "payload_bytes": payload, "padding_bytes": padded - payload, "owned_input_bytes": 2 * padded,
              "bucket_count": len(layouts), "layout_manifest": str(manifest_path), "layout_sha256": sha256(manifest_path),
              "bucket_distribution": [{"index": index, "kind": kind, "block_bytes": layout.block_bytes,
                                       "tensor_count": len(layout.entries), "payload_bytes": sum(n for _, n in layout.entries),
                                       "padded_bytes": layout.padded_bytes,
                                       "single_oversized": len(layout.entries) == 1 and layout.padded_bytes > args.bucket_mib * (1 << 20)}
                                      for index, (kind, layout) in enumerate(layouts)],
              "layout_scope": "representative stable sorted names; separate FP32 scalar scales; not live iterator capture",
              "primary_baseline_32_workers": args.baseline_workers == 32,
              "process_cpu_excludes_compiler_children": True})
        clean = True
        for index, case in enumerate(args.cases):
            clean = run_case(args, module, module_hash, args.preparation_module, layouts, locations,
                             case, worker, batch_worker, checksum, emit, first_case=index == 0,
                             cache_initially_empty=cache_was_empty) and clean
            gc.collect()
        emit({"event": "complete", "all_results_verified": True, "all_measured_samples_steady_state": clean})
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
