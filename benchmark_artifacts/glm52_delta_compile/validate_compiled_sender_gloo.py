"""Verify two-rank collective drain for packed layout and CPU-worker failures.

Run on a CPU-only process with the complete Miles image dependencies. This
uses real compiled preparation, real Gloo collectives, and a mocked finalizer;
it does not use CUDA, engine RPCs, or filesystem publication.
"""

import argparse
from argparse import Namespace
from contextlib import nullcontext
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock, patch

import safetensors.torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class CollectiveIterator:
    def __init__(self, failure):
        self.completed = []
        self.failure = failure

    def iter_hf_weights(self, weights, *, materialize, **kwargs):
        for index in range(4):
            token = torch.ones(1, dtype=torch.int64)
            dist.all_reduce(token)
            assert token.item() == 2
            self.completed.append(index)
            shape = (3, 2) if self.failure == "layout" and index == 1 else (2, 3)
            yield [(f"weight{index}", torch.ones(shape, dtype=torch.bfloat16))] if materialize else []


def worker(rank, miles_path, run_dir, failure):
    sys.path.insert(0, miles_path)
    from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
    from miles.backends.training_utils.weight_update.updater import WeightUpdater
    from miles.utils.disk_delta import make_tensor_reader

    directory = Path(run_dir)
    dist.init_process_group(
        "gloo", init_method=(directory / "gloo-store").as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=180),
    )
    try:
        protocol = UpdateWeightFromDiskDelta(Namespace(
            hf_checkpoint=run_dir,
            update_weight_disk_dir=str(directory / "deltas"),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="adler32",
            update_weight_delta_cpu_backend="torch-compile",
            custom_update_weight_post_write_path=None,
        ))
        protocol.is_sender = rank == 0
        protocol._baseline_captured = True
        if protocol.is_sender:
            read = make_tensor_reader(run_dir)
            for index in range(4):
                name = f"weight{index}"
                protocol._snapshot[name] = read(name)
                protocol._packed.capture([(name, torch.zeros((2, 3), dtype=torch.bfloat16))])
            protocol._packed.initialize()
        dist.barrier()
        updater = WeightUpdater.__new__(WeightUpdater)
        updater.protocol, updater.weight_version, updater.is_lora = protocol, 0, False
        updater.weights_getter = lambda: {}
        updater._hf_weight_iterator = CollectiveIterator(failure)
        protocol.finalize = MagicMock()
        injected = (
            patch("miles.backends.training_utils.weight_update.packed_delta.checksum",
                  side_effect=RuntimeError("injected packed checksum failure"))
            if failure == "worker" and rank == 0 else nullcontext()
        )
        with (
            patch("miles.backends.training_utils.weight_update.protocols.delta.get_gloo_group", return_value=dist.group.WORLD),
            patch("miles.backends.training_utils.weight_update.updater.get_gloo_group", return_value=dist.group.WORLD),
            injected,
        ):
            try:
                updater.update_weights()
            except RuntimeError as error:
                message = str(error)
                assert "rank 0" in message, message
                if failure == "worker":
                    assert "injected packed checksum failure" in message, message
                else:
                    assert "shape" in message or "layout" in message, message
            else:
                raise AssertionError("Malformed packed update unexpectedly succeeded")
        assert updater._hf_weight_iterator.completed == list(range(4))
        assert protocol._packed._pool is None and not protocol._packed._inflight
        protocol.finalize.assert_not_called()
        assert not list((directory / "deltas").rglob("*.safetensors"))
        assert not list((directory / "deltas").rglob("*.json"))
        (directory / f"result-{rank}.json").write_text(json.dumps({
            "rank": rank, "failure": failure, "completed_collectives": 4,
            "pool_closed": True, "published": False, "error": message,
        }))
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles-path", required=True, type=Path)
    options = parser.parse_args()
    for failure in ("layout", "worker"):
        with tempfile.TemporaryDirectory(prefix=f"compiled-gloo-{failure}-", dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            safetensors.torch.save_file({
                f"weight{index}": torch.zeros((2, 3), dtype=torch.bfloat16) for index in range(4)
            }, root / "model.safetensors")
            mp.spawn(worker, args=(str(options.miles_path.resolve()), directory, failure), nprocs=2, join=True)
            results = [json.loads((root / f"result-{rank}.json").read_text()) for rank in range(2)]
            assert results[0]["error"] == results[1]["error"]
            for result in results:
                print(json.dumps(result), flush=True)
    print("PASS: both failure modes drained all four collectives on both ranks without publication")


if __name__ == "__main__":
    main()
