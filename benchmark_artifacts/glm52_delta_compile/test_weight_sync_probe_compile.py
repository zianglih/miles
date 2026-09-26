"""Validate compiler journal attribution and exception transparency."""

import json
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist

import weight_sync_probe_compile as probe


@pytest.mark.parametrize("fail", [False, True])
def test_compiler_probe_uses_trainer_rank_and_preserves_update_result(tmp_path, monkeypatch, fail):
    class Updater:
        def update_weights(self):
            if fail:
                raise RuntimeError("update failed")
            return "completed"

    module_name = "miles.backends.training_utils.weight_update.updater"
    module = ModuleType(module_name)
    module.WeightUpdater = Updater
    monkeypatch.setenv("WEIGHT_SYNC_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("WEIGHT_SYNC_MODE", "disk-delta")
    monkeypatch.setattr(probe.wall_probe, "_update_index", -1)

    def install_cpu(_):
        original = Updater.update_weights

        def update(self):
            probe.wall_probe._update_index += 1
            return original(self)

        Updater.update_weights = update

    counters = [
        {"inductor.generated_kernel_count": 0},  # Import/setup outside updates.
        {"inductor.generated_kernel_count": 1},
        {"inductor.generated_kernel_count": 3, "stats.unique_graphs": 1},
    ]
    with (
        patch.dict("sys.modules", {module_name: module}),
        patch.object(dist, "get_rank", return_value=2),
        patch.object(probe.cpu_probe, "install", side_effect=install_cpu),
        patch.object(probe, "compiler_counters", side_effect=counters),
    ):
        probe.install(SimpleNamespace(update_weight_delta_cpu_backend="torch-compile"))
        if fail:
            with pytest.raises(RuntimeError, match="update failed"):
                Updater().update_weights()
        else:
            assert Updater().update_weights() == "completed"

    record = json.loads((tmp_path / "trainer-rank2.jsonl").read_text())
    assert not (tmp_path / "driver.jsonl").exists()
    assert record["rank"] == 2 and record["update_index"] == 0
    assert record["success"] is not fail
    assert record["cpu_backend"] == "torch-compile"
    assert record["before"] == counters[1] and record["after"] == counters[2]
    assert record["delta"] == {"inductor.generated_kernel_count": 2, "stats.unique_graphs": 1}
