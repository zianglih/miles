"""Replay exact Miles CPU diff workers on canonical and reconstructed checkpoint bytes.

The primary case uses 32 threads and all sender-emitted checkpoint tensors. Inputs represent
completed D2H views; there is no CUDA work, pinned allocation, finite pinned-pool
backpressure, gather, serialization, filesystem publication, or receiver reload.
Input allocation/reconstruction, pool startup and verification are untimed. The
timed region includes task submission/drain and the exact worker's allocations,
diff, snapshot copy, compression, checksum and queue return.
"""

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import queue
import statistics
import struct
import subprocess
import sys
import threading
import time
import types
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import zstandard

ROOT = Path(__file__).resolve().parent
WORKER_PATH = "miles/backends/training_utils/weight_update/protocols/delta.py"
HELPER_PATH = "miles/utils/disk_delta.py"
DEFAULT_BASELINE = "81d43908021deaf916808675d7ac78e4fbccfa2f"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def source_at(repo, ref, path):
    if ref == "WORKTREE":
        return (repo / path).read_text()
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{path}"], text=True)


def load_worker(repo, ref, label, source_dir):
    """Compile the original AST nodes, without importing Miles or changing their code."""
    source = source_at(repo, ref, WORKER_PATH)
    helper_source = source_at(repo, ref, HELPER_PATH)
    model = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "UpdateWeightFromDiskDelta"
    )
    worker = next(
        node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_diff_and_compress"
    )
    helper_names = {"overwrite_encode", "checksum", "_new_hasher", "_Adler32"}
    helpers = [
        node
        for node in ast.parse(helper_source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in helper_names
    ]
    assert {node.name for node in helpers} == helper_names
    namespace = {"np": np, "zstandard": zstandard, "zlib": zlib}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(repo / HELPER_PATH), "exec"), namespace)
    exec(compile(ast.Module(body=[worker], type_ignores=[]), str(repo / WORKER_PATH), "exec"), namespace)
    (source_dir / f"{label}-delta.py").write_text(source)
    (source_dir / f"{label}-disk_delta.py").write_text(helper_source)
    info = {
        "ref": ref,
        "commit": git(repo, "rev-parse", "HEAD" if ref == "WORKTREE" else ref),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "helper_source_sha256": hashlib.sha256(helper_source.encode()).hexdigest(),
        "worker_source_sha256": hashlib.sha256(ast.get_source_segment(source, worker).encode()).hexdigest(),
        "working_tree": ref == "WORKTREE",
    }
    return namespace["_diff_and_compress"], namespace["checksum"], info


def read_header(path):
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        size = struct.unpack("<Q", raw_size)[0]
        if size > path.stat().st_size - 8:
            raise ValueError(f"Header exceeds file length: {path}")
        raw = stream.read(size)
    return json.loads(raw), 8 + size, hashlib.sha256(raw).hexdigest()


def checkpoint_locations(directory):
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    filenames = (
        sorted(set(index["weight_map"].values()))
        if index
        else sorted(path.name for path in directory.glob("*.safetensors"))
    )
    if not filenames:
        raise ValueError(f"No canonical safetensors found in {directory}")
    locations, files = {}, []
    for filename in filenames:
        path = directory / filename
        header, data_offset, header_hash = read_header(path)
        files.append({"name": filename, "file_bytes": path.stat().st_size, "header_sha256": header_hash})
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in locations:
                raise ValueError(f"Duplicate canonical tensor {name!r}")
            begin, end = metadata["data_offsets"]
            if not 0 <= begin <= end <= path.stat().st_size - data_offset:
                raise ValueError(f"Invalid byte range for {name!r}")
            locations[name] = (path, data_offset + begin, end - begin)
            if index and index["weight_map"].get(name) != filename:
                raise ValueError(f"Canonical index disagrees for {name!r}")
    if index and set(locations) != set(index["weight_map"]):
        raise ValueError("Canonical index contains missing tensors")
    return dict(sorted(locations.items())), files


def load_canonical(locations):
    arrays = {}
    for name, (path, offset, nbytes) in locations.items():
        with path.open("rb") as stream:
            stream.seek(offset)
            value = np.fromfile(stream, dtype=np.uint8, count=nbytes)
        if value.nbytes != nbytes:
            raise ValueError(f"Truncated canonical tensor {name!r}")
        arrays[name] = value
    return arrays


def version_index(source, version):
    directory = source / f"weight_v{version:06d}"
    path = directory / "model.safetensors.index.json"
    raw = path.read_bytes()
    index = json.loads(raw)
    metadata = index["metadata"]
    if int(metadata["version"]) != version or int(metadata["base_version"]) != version - 1:
        raise ValueError(f"Invalid version lineage in {path}")
    if metadata["compression_format"] != "zstd" or metadata["delta_encoding"] not in ("xor", "overwrite"):
        raise ValueError(f"Unsupported delta format in {path}")
    return directory, index, hashlib.sha256(raw).hexdigest()


def apply_version(arrays, source, version, checksum):
    directory, index, index_hash = version_index(source, version)
    metadata = index["metadata"]
    shards = {}
    for name, filename in index["weight_map"].items():
        if name not in arrays:
            raise ValueError(f"Published tensor {name!r} is absent from canonical checkpoint")
        shards.setdefault(filename, set()).add(name)
    decoder = zstandard.ZstdDecompressor()
    for filename, names in sorted(shards.items()):
        path = directory / filename
        header, data_offset, _ = read_header(path)
        if set(header) - {"__metadata__"} != names:
            raise ValueError(f"Published shard names disagree with index: {path}")
        for name in sorted(names):
            begin, end = header[name]["data_offsets"]
            with path.open("rb") as stream:
                stream.seek(data_offset + begin)
                compressed = stream.read(end - begin)
            decoded = np.frombuffer(decoder.decompress(compressed), dtype=np.uint8)
            apply_decoded(arrays[name], decoded, metadata["delta_encoding"])
            if checksum(metadata["checksum_format"], arrays[name]) != header["__metadata__"][name]:
                raise ValueError(f"Published v{version} checksum mismatch for {name!r}")
    return {
        "version": version,
        "index_sha256": index_hash,
        "metadata": metadata,
        "published_tensor_count": len(index["weight_map"]),
        "published_names": sorted(index["weight_map"]),
    }


def apply_decoded(target, decoded, encoding):
    if encoding == "xor":
        if target.size != decoded.size:
            raise ValueError("XOR byte count disagrees with canonical tensor")
        np.bitwise_xor(target, decoded, out=target)
        return
    if decoded.size < 4:
        raise ValueError("Truncated overwrite count")
    count = int(decoded[:4].view("<u4")[0])
    if decoded.size != 4 + 5 * count:
        raise ValueError("Invalid overwrite payload length")
    positions = decoded[4 : 4 + count * 4].view("<u4")
    if positions.size and positions.max() >= target.size:
        raise ValueError("Overwrite position exceeds canonical tensor")
    target[positions] = decoded[4 + count * 4 :]


def describe_inputs(old, incoming, published_names):
    sizes = np.array([array.nbytes for array in old.values()], dtype=np.int64)
    changed = {name: int(np.count_nonzero(array != incoming[name])) for name, array in old.items()}
    total = int(sizes.sum())
    changed_names = {name for name, count in changed.items() if count}
    if changed_names != set(published_names):
        raise ValueError("Published v2 names disagree with actual v1-to-v2 changed tensors")
    unchanged_bytes = sum(old[name].nbytes for name, count in changed.items() if not count)
    buckets = []
    lower = -1
    for upper in (16, 1024, 64 << 10, 1 << 20, 16 << 20, int(sizes.max())):
        if upper <= lower:
            continue
        members = [name for name, array in old.items() if lower < array.nbytes <= upper]
        buckets.append(
            {
                "bytes_gt": lower,
                "bytes_le": upper,
                "tensor_count": len(members),
                "tensor_bytes": sum(old[name].nbytes for name in members),
                "unchanged_tensor_bytes": sum(old[name].nbytes for name in members if not changed[name]),
            }
        )
        lower = upper
    description = {
        "tensor_count": len(old),
        "total_bytes": total,
        "changed_tensor_count": len(changed_names),
        "unchanged_tensor_count": len(old) - len(changed_names),
        "unchanged_tensor_bytes": unchanged_bytes,
        "unchanged_tensor_byte_fraction": unchanged_bytes / max(total, 1),
        "changed_bytes": sum(changed.values()),
        "changed_byte_density": sum(changed.values()) / max(total, 1),
        "tensor_size_min_bytes": int(sizes.min()),
        "tensor_size_max_bytes": int(sizes.max()),
        "tensor_size_quantiles_bytes": dict(zip(("p50", "p90", "p99"), np.quantile(sizes, (0.5, 0.9, 0.99)).tolist())),
        "size_buckets": buckets,
    }
    return changed, description


def verify_results(results, old, incoming, changed, expected_checksums, encoding, algorithm, checksum):
    wire_bytes = 0
    decoder = zstandard.ZstdDecompressor()
    for name, snapshot, compressed, digest, count in results:
        if count != changed[name] or not np.array_equal(snapshot, incoming[name]):
            raise AssertionError(f"Worker snapshot/count mismatch: {name}")
        if not count:
            if compressed is not None or digest is not None:
                raise AssertionError(f"Unexpected unchanged payload: {name}")
            continue
        if digest != expected_checksums[name] or digest != checksum(algorithm, snapshot):
            raise AssertionError(f"Worker checksum mismatch: {name}")
        replay = old[name].copy()
        decoded = np.frombuffer(decoder.decompress(compressed), dtype=np.uint8)
        apply_decoded(replay, decoded, encoding)
        if not np.array_equal(replay, incoming[name]):
            raise AssertionError(f"Worker decoded-byte mismatch: {name}")
        wire_bytes += compressed.nbytes
    if len(results) != len(old):
        raise AssertionError("Worker omitted tensors")
    return wire_bytes


def warm_pool(pool, workers):
    barrier = threading.Barrier(workers + 1)
    futures = [pool.submit(barrier.wait) for _ in range(workers)]
    barrier.wait()
    for future in futures:
        future.result()


def measure_case(args, label, locations, implementations, checksum, emit):
    old = load_canonical(locations)
    lineage = []
    algorithm, encoding = args.checksum, args.encoding
    if label == "synthetic-v1-v2":
        lineage.append(apply_version(old, args.synthetic_source, 1, checksum))
        algorithm = lineage[0]["metadata"]["checksum_format"]
        encoding = lineage[0]["metadata"]["delta_encoding"]
    incoming = {name: array.copy() for name, array in old.items()}
    if label == "synthetic-v1-v2":
        lineage.append(apply_version(incoming, args.synthetic_source, 2, checksum))
        if any(
            entry["metadata"]["checksum_format"] != algorithm or entry["metadata"]["delta_encoding"] != encoding
            for entry in lineage
        ):
            raise ValueError("Delta encoding/checksum changed between v1 and v2")
    names = lineage[-1]["published_names"] if lineage else []
    changed, description = describe_inputs(old, incoming, names)
    expected = {name: checksum(algorithm, array) for name, array in incoming.items()}
    old_checksums = {name: checksum(algorithm, array) for name, array in old.items()}
    for array in old.values():
        array.setflags(write=False)
    views = [(name, torch.from_numpy(array), array.nbytes) for name, array in incoming.items()]
    emit(
        {
            "event": "workload",
            "case": label,
            "encoding": encoding,
            "checksum": algorithm,
            "old_state_checksum_manifest_sha256": hashlib.sha256(
                json.dumps(old_checksums, sort_keys=True).encode()
            ).hexdigest(),
            "new_state_checksum_manifest_sha256": hashlib.sha256(
                json.dumps(expected, sort_keys=True).encode()
            ).hexdigest(),
            "lineage": lineage,
            "tensor_order": "canonical names sorted lexicographically",
            **description,
        }
    )
    for workers in args.workers:
        pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        if pool:
            warm_pool(pool, workers)
        measurements = {name: [] for name in implementations}
        try:
            for iteration in range(args.warmups + args.repeats):
                order = list(implementations) if iteration % 2 == 0 else list(reversed(implementations))
                for name in order:
                    method = implementations[name]
                    state = types.SimpleNamespace(
                        _snapshot=old, _free_q=queue.Queue(), delta_encoding=encoding, checksum_algorithm=algorithm
                    )
                    gc.collect()
                    cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
                    if pool:
                        futures = [pool.submit(method, state, key, buffer, size, True) for key, buffer, size in views]
                        results = [future.result() for future in futures]
                    else:
                        results = [method(state, key, buffer, size, True) for key, buffer, size in views]
                    wall_s = (time.perf_counter_ns() - wall_start) / 1e9
                    cpu_s = (time.process_time_ns() - cpu_start) / 1e9
                    if state._free_q.qsize() != len(views):
                        raise AssertionError("Worker did not return exactly one buffer per tensor")
                    payload_bytes = verify_results(
                        results, old, incoming, changed, expected, encoding, algorithm, checksum
                    )
                    for key, array in old.items():
                        if checksum(algorithm, array) != old_checksums[key]:
                            raise AssertionError(f"Worker modified the old snapshot: {key}")
                    record = {
                        "event": "sample",
                        "case": label,
                        "workers": workers,
                        "variant": name,
                        "iteration": iteration,
                        "warmup": iteration < args.warmups,
                        "wall_s": wall_s,
                        "process_cpu_s": cpu_s,
                        "compressed_payload_bytes": payload_bytes,
                        "exact_snapshots_decoded_bytes_and_checksums_verified": True,
                    }
                    emit(record)
                    if iteration >= args.warmups:
                        measurements[name].append(record)
                    del results, state
                    if pool:
                        del futures
            summary = {
                name: {
                    "median_wall_s": statistics.median(row["wall_s"] for row in rows),
                    "median_process_cpu_s": statistics.median(row["process_cpu_s"] for row in rows),
                    "raw_wall_s": [row["wall_s"] for row in rows],
                    "raw_process_cpu_s": [row["process_cpu_s"] for row in rows],
                }
                for name, rows in measurements.items()
            }
            emit(
                {
                    "event": "summary",
                    "case": label,
                    "workers": workers,
                    "variants": summary,
                    "candidate_wall_reduction_fraction": 1
                    - summary["candidate"]["median_wall_s"] / summary["baseline"]["median_wall_s"],
                    "candidate_cpu_reduction_fraction": 1
                    - summary["candidate"]["median_process_cpu_s"] / summary["baseline"]["median_process_cpu_s"],
                }
            )
        finally:
            if pool:
                pool.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles-path", type=Path, default=ROOT / "miles")
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-ref", default="WORKTREE")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic-source", type=Path, help="Published weight_v000001 and weight_v000002 parent")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("native-unchanged", "synthetic-v1-v2"),
        default=["native-unchanged", "synthetic-v1-v2"],
    )
    parser.add_argument("--workers", nargs="+", type=int, default=[32], help="1 selects direct sequential calls")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--expected-tensors", type=int)
    parser.add_argument(
        "--exclude-suffix",
        nargs="*",
        default=[".input_scale"],
        help="Conversion-only canonical entries absent from Miles NVFP4 export",
    )
    parser.add_argument(
        "--max-input-gib", type=float, default=64, help="Cap on old+incoming inputs, excluding worker allocations"
    )
    parser.add_argument("--encoding", choices=("xor", "overwrite"), default="xor", help="Native case only")
    parser.add_argument("--checksum", default="xxh3-128", help="Native case only; synthetic uses published metadata")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "synthetic-v1-v2" in args.cases and args.synthetic_source is None:
        parser.error("--synthetic-source is required for synthetic-v1-v2")
    if min(args.workers) < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("workers/repeats must be positive and warmups nonnegative")
    if args.output.exists():
        parser.error("--output already exists; select a new evidence file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_dir = args.output.parent / (args.output.stem + "-sources")
    source_dir.mkdir(exist_ok=True)
    baseline, checksum, baseline_info = load_worker(args.miles_path, args.baseline_ref, "baseline", source_dir)
    candidate, _, candidate_info = load_worker(args.miles_path, args.candidate_ref, "candidate", source_dir)
    if baseline_info["worker_source_sha256"] == candidate_info["worker_source_sha256"]:
        raise ValueError("Baseline and candidate workers are identical; check the selected source revisions")
    if baseline_info["helper_source_sha256"] != candidate_info["helper_source_sha256"]:
        raise ValueError("Checksum/overwrite helper source differs; audit before replaying these revisions")
    canonical_locations, files = checkpoint_locations(args.checkpoint)
    locations = {
        name: entry
        for name, entry in canonical_locations.items()
        if not any(name.endswith(suffix) for suffix in args.exclude_suffix)
    }
    if not locations:
        raise ValueError("No tensors remain in the selected sender scope")
    total = sum(size for _, _, size in locations.values())
    if args.expected_tensors is not None and len(locations) != args.expected_tensors:
        raise ValueError(f"Expected {args.expected_tensors} tensors, found {len(locations)}")
    if 2 * total > args.max_input_gib * (1 << 30):
        raise ValueError("Old+incoming arrays exceed --max-input-gib; explicitly increase for this checkpoint")
    with args.output.open("x") as output:

        def emit(record):
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()

        emit(
            {
                "event": "provenance",
                "scope": __doc__,
                "python": sys.version,
                "platform": platform.platform(),
                "host": platform.node(),
                "cpu_count": os.cpu_count(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "versions": {
                    name: importlib.metadata.version(name) for name in ("numpy", "torch", "zstandard", "xxhash")
                },
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "baseline": baseline_info,
                "candidate": candidate_info,
                "canonical_files": files,
                "canonical_tensor_count": len(canonical_locations),
                "selected_tensor_count": len(locations),
                "excluded_tensor_count": len(canonical_locations) - len(locations),
                "excluded_tensor_bytes": sum(size for _, _, size in canonical_locations.values()) - total,
                "input_bytes": 2 * total,
                "source_artifacts": str(source_dir),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
        for label in args.cases:
            measure_case(args, label, locations, {"baseline": baseline, "candidate": candidate}, checksum, emit)


if __name__ == "__main__":
    main()
