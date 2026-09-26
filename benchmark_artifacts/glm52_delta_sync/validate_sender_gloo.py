"""Exercise a post-baseline validation failure through the real two-rank Gloo updater.

Run with the Miles checkout and CPU dependencies installed:
  python validate_sender_gloo.py --miles-path /path/to/miles

No GPU or rollout server is used. Temporary artifacts stay beside this script.
"""

import argparse
import json
import sys
import tempfile
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import safetensors.torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class CollectiveIterator:
    def __init__(self):
        self.completed = []

    def iter_hf_weights(self, weights, *, materialize, **kwargs):
        buckets = [
            [("weight", torch.ones((2, 3), dtype=torch.bfloat16))],
            [("weight", torch.ones((3, 2), dtype=torch.bfloat16))],
            [("missing", torch.zeros((2, 3), dtype=torch.bfloat16))],
            [("weight", torch.zeros((2, 3), dtype=torch.bfloat16))],
        ]
        for index, bucket in enumerate(buckets):
            # This collective must still complete after rank 0 sees the invalid
            # shape, otherwise rank 1 times out instead of receiving its error.
            token = torch.tensor([1], dtype=torch.int64)
            dist.all_reduce(token)
            assert token.item() == 2
            self.completed.append(index)
            yield bucket if materialize else []


def worker(rank, miles_path, run_dir):
    sys.path.insert(0, miles_path)
    from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
    from miles.backends.training_utils.weight_update.updater import WeightUpdater
    from miles.utils.disk_delta import make_tensor_reader

    directory = Path(run_dir)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "gloo-store").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        protocol = UpdateWeightFromDiskDelta(
            Namespace(
                hf_checkpoint=run_dir,
                update_weight_disk_dir=str(directory / "deltas"),
                update_weight_delta_encoding="xor",
                update_weight_delta_checksum="adler32",
                custom_update_weight_post_write_path=None,
            )
        )
        protocol.is_sender = rank == 0
        protocol._baseline_captured = True
        if protocol.is_sender:
            protocol._snapshot = {"weight": make_tensor_reader(run_dir)("weight")}
        updater = WeightUpdater.__new__(WeightUpdater)
        updater.protocol = protocol
        updater.weight_version = 0
        updater.is_lora = False
        updater.weights_getter = lambda: {}
        updater._hf_weight_iterator = CollectiveIterator()
        protocol.finalize = MagicMock()
        original_empty = torch.empty

        def pageable_empty(*args, **kwargs):
            if kwargs.get("pin_memory"):
                raise RuntimeError("CPU validation uses pageable buffers")
            return original_empty(*args, **kwargs)

        with (
            patch(
                "miles.backends.training_utils.weight_update.protocols.delta.get_gloo_group",
                return_value=dist.group.WORLD,
            ),
            patch(
                "miles.backends.training_utils.weight_update.updater.get_gloo_group",
                return_value=dist.group.WORLD,
            ),
            patch(
                "miles.backends.training_utils.weight_update.protocols.delta.torch.empty",
                side_effect=pageable_empty,
            ),
        ):
            try:
                updater.update_weights()
            except RuntimeError as error:
                message = str(error)
                assert message.startswith("Disk-delta update validation failed on rank 0:")
                assert "has shape (2, 3); trainer emitted (3, 2)" in message
            else:
                raise AssertionError("The malformed update unexpectedly succeeded")

        assert updater._hf_weight_iterator.completed == [0, 1, 2, 3]
        assert protocol._pool is None and not protocol._inflight
        protocol.finalize.assert_not_called()
        assert not list((directory / "deltas").rglob("*.safetensors"))
        assert not list((directory / "deltas").rglob("*.json"))
        (directory / f"result-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "sender": protocol.is_sender,
                    "completed_collectives": len(updater._hf_weight_iterator.completed),
                    "pool_closed": protocol._pool is None,
                    "published": False,
                    "error": message,
                }
            )
        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--miles-path", type=Path, default=Path(__file__).resolve().parent / "miles")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="sender-gloo-", dir=Path(__file__).resolve().parent) as run_dir:
        directory = Path(run_dir)
        safetensors.torch.save_file(
            {"weight": torch.zeros((2, 3), dtype=torch.bfloat16)},
            directory / "model.safetensors",
        )
        mp.spawn(worker, args=(str(args.miles_path.resolve()), run_dir), nprocs=2, join=True)
        for rank in range(2):
            print((directory / f"result-rank-{rank}.json").read_text())
    print("PASS: both ranks completed all gathers and rejected the update before publication")


if __name__ == "__main__":
    main()
