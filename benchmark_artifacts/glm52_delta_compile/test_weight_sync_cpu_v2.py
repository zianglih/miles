"""CPU-only validation: python -m unittest -v test_weight_sync_cpu_v2.py."""

import ast
import base64
import copy
import contextlib
import io
import os
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import process_cpu_clocks_cpu_v2 as clocks
import summarize_weight_sync_cpu_v2 as summary
import weight_sync_probe_cpu_v2 as probe
import launch_weight_sync_cpu_v2 as launcher
from build_benchmark_report import normalized_config

ROOT = Path(__file__).resolve().parent


def fixture():
    manifest = {"num_rollout": 7}
    wall = {"complete": True, "rows": []}
    drivers, trainers = [], []
    usage = dict(
        ru_utime=0.2, ru_stime=0.05, ru_minflt=3, ru_majflt=0, ru_nvcsw=2, ru_nivcsw=1
    )
    receiver = {
        "processes": [{"pid": rank, "role": "scheduler"} for rank in range(4)],
        "scheduler_cpu_ns": 1_000_000_000,
        "auxiliary_cpu_ns": 500_000_000,
        "total_cpu_ns": 1_500_000_000,
        "snapshot_cost_ns": 200_000,
    }
    for index in range(7):
        wall["rows"].append(
            {
                "update_index": index,
                "initial": index == 0,
                "driver_wall_s": float(index + 1),
            }
        )
        drivers.append(
            {
                "kind": "cpu_driver_update",
                "update_index": index,
                "wall_s": float(index + 1),
                "cpu_s": 0.01,
                "cpu_valid": True,
                "success": True,
            }
        )
        for rank in range(4):
            trainers.append(
                {
                    "kind": "cpu_trainer_update",
                    "rank": rank,
                    "update_index": index,
                    "cpu_s": 0.25,
                    "cpu_valid": True,
                    "success": True,
                    "resource_usage": usage,
                    "receiver": receiver if rank == 0 else None,
                    "observer_before_wall_ns": 100,
                    "observer_after_wall_ns": 100,
                }
            )
    return manifest, wall, drivers, trainers


class Counters(unittest.TestCase):
    def test_proc_stat_with_spaces(self):
        values = ["S"] + [str(index) for index in range(4, 53)]
        parsed = clocks.parse_proc_stat(
            "123 (a complicated ) name) " + " ".join(values)
        )
        self.assertEqual(parsed["user_ticks"], 14)
        self.assertEqual(parsed["system_ticks"], 15)
        self.assertEqual(parsed["start_ticks"], 22)

    def test_self_all_fields(self):
        before = clocks.self_cpu_snapshot(True)
        sum(index * index for index in range(10000))
        result = clocks.self_cpu_delta(before, clocks.self_cpu_snapshot(True))
        self.assertTrue(result["cpu_valid"])
        self.assertGreaterEqual(result["cpu_s"], 0)
        self.assertEqual(
            set(result["resource_usage"]),
            {"ru_utime", "ru_stime", "ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw"},
        )

    def test_self_failure_is_diagnostic(self):
        with patch.object(
            clocks.time, "process_time_ns", side_effect=OSError("unavailable")
        ):
            result = clocks.self_cpu_snapshot()
        self.assertIn("unavailable", result["error"])
        self.assertFalse(clocks.self_cpu_delta(result, result)["cpu_valid"])

    def test_receiver_totals_and_negative_rejection(self):
        receiver = clocks.ReceiverClocks.__new__(clocks.ReceiverClocks)
        receiver.processes = {
            1: {"pid": 1, "role": "scheduler", "engine": "a", "start_ticks": 3},
            2: {"pid": 2, "role": "auxiliary", "engine": "a", "start_ticks": 3},
        }
        before = {"cpu_ns": {"1": 10, "2": 20}, "duration_ns": 2}
        after = {"cpu_ns": {"1": 50, "2": 30}, "duration_ns": 2}
        self.assertEqual(receiver.difference(before, after)["total_cpu_ns"], 50)
        after["cpu_ns"]["1"] = 0
        with self.assertRaises(RuntimeError):
            receiver.difference(before, after)

    def test_refresh_detects_new_compile_worker(self):
        children = {100: [100, 101, 102], 200: [200, 201, 202]}
        receiver = clocks.ReceiverClocks.__new__(clocks.ReceiverClocks)
        receiver.roots = {"a": 100, "b": 200}
        receiver.processes = {}
        receiver.backend = "process-clock"
        receiver.expected_schedulers = 4

        def fake_clock(pid, backend):
            return types.SimpleNamespace(start_ticks=pid, resolution_s=1e-9)

        with (
            patch.object(
                clocks,
                "process_tree",
                side_effect=lambda root: [
                    (pid, {"start_ticks": pid, "ppid": root}) for pid in children[root]
                ],
            ),
            patch.object(
                clocks,
                "process_command",
                side_effect=lambda pid: "sglang::scheduler"
                if pid in (101, 102, 201, 202)
                else "auxiliary",
            ),
            patch.object(clocks, "ProcessClock", side_effect=fake_clock),
            patch.object(clocks, "runtime_metadata", return_value={}),
        ):
            self.assertTrue(receiver.refresh())
            self.assertFalse(receiver.refresh())
            children[100].append(103)
            self.assertTrue(receiver.refresh())
            self.assertEqual(receiver.processes[103]["role"], "auxiliary")
            self.assertEqual(len(receiver.processes), 7)
            children[100].remove(103)
            with patch.object(clocks, "identity", return_value={"start_ticks": 103}):
                with self.assertRaisesRegex(
                    RuntimeError, "disappeared from child traversal"
                ):
                    receiver.refresh()
            with patch.object(clocks, "identity", side_effect=FileNotFoundError):
                self.assertTrue(receiver.refresh())
                self.assertNotIn(103, receiver.processes)

    def test_proc_tree_includes_nonleader_thread_children_and_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def process(pid, parent, tasks):
                base = root / str(pid)
                base.mkdir()
                fields = ["S"] + ["0"] * 49
                fields[1], fields[17], fields[19] = (
                    str(parent),
                    str(len(tasks)),
                    str(pid * 10),
                )
                (base / "stat").write_text(
                    f"{pid} (fixture process) " + " ".join(fields)
                )
                for tid, children in tasks.items():
                    task = base / "task" / str(tid)
                    task.mkdir(parents=True)
                    (task / "children").write_text(
                        " ".join(str(child) for child in children)
                    )

            process(10, 1, {10: [20], 11: [30]})
            process(20, 10, {20: []})
            process(30, 10, {30: [40]})
            process(40, 30, {40: []})
            found = dict(clocks.process_tree(10, root))
            self.assertEqual(set(found), {10, 20, 30, 40})
            self.assertEqual(found[40]["start_ticks"], 400)
            # A partial/racing scan must be diagnostic, not silently incomplete.
            (root / "10/task/11/children").unlink()
            with self.assertRaises(FileNotFoundError):
                list(clocks.process_tree(10, root))

    def test_refresh_replacement_during_discovery_is_rejected(self):
        receiver = clocks.ReceiverClocks.__new__(clocks.ReceiverClocks)
        receiver.roots = {"a": 100}
        receiver.processes = {}
        receiver.backend = "process-clock"
        with (
            patch.object(
                clocks,
                "process_tree",
                return_value=[(100, {"start_ticks": 1, "ppid": 0})],
            ),
            patch.object(clocks, "process_command", return_value="server"),
            patch.object(
                clocks,
                "ProcessClock",
                return_value=types.SimpleNamespace(start_ticks=2),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "replaced during discovery"):
                receiver.refresh()

    def test_pid_replacement_rejected(self):
        instance = clocks.ProcessClock.__new__(clocks.ProcessClock)
        instance.pid, instance.start_ticks = 5, 10
        with patch.object(clocks, "identity", return_value={"start_ticks": 11}):
            with self.assertRaises(RuntimeError):
                instance.validate()


class Aggregation(unittest.TestCase):
    def test_five_steady_samples(self):
        result = summary.aggregate(*fixture())
        self.assertTrue(result["complete_cpu_steady"])
        self.assertEqual(
            result["steady"]["driver_wall_s"]["raw"], [3.0, 4.0, 5.0, 6.0, 7.0]
        )
        self.assertEqual(result["steady"]["trainer_cpu_sum_s"]["median"], 1.0)
        self.assertEqual(result["steady"]["receiver_total_cpu_s"]["median"], 1.5)
        self.assertEqual(result["steady"]["trainer_minor_faults"]["median"], 12)

    def test_invalid_steady_rejected(self):
        data = fixture()
        data[3][-1]["cpu_valid"] = False
        self.assertFalse(summary.aggregate(*data)["complete_cpu_steady"])

    def test_initial_process_change_does_not_poison_steady(self):
        data = fixture()
        data[3][0]["cpu_valid"] = False
        result = summary.aggregate(*data)
        self.assertTrue(result["complete_cpu_steady"])
        self.assertEqual(result["invalid_initial_or_first_post"], [0])

    def test_missing_receiver_and_duplicates_rejected(self):
        data = fixture()
        data[3][-4]["receiver"] = None
        self.assertFalse(summary.aggregate(*data)["complete_cpu_steady"])
        data = fixture()
        data[3].append(copy.deepcopy(data[3][-1]))
        self.assertFalse(summary.aggregate(*data)["complete_cpu_steady"])

    def test_wall_mismatch_rejected(self):
        data = fixture()
        data[2][-1]["wall_s"] = 999
        self.assertFalse(summary.aggregate(*data)["complete_cpu_steady"])


class HookFailureSafety(unittest.TestCase):
    def test_receiver_discovery_failure_keeps_update(self):
        class Updater:
            protocol = types.SimpleNamespace(
                rollout_engines=[
                    types.SimpleNamespace(server_url="http://a:1"),
                    types.SimpleNamespace(server_url="http://a:2"),
                ]
            )

            def update_weights(self):
                return "completed"

        fake_dist = types.ModuleType("torch.distributed")
        fake_dist.get_rank = lambda: 0
        fake_torch = types.ModuleType("torch")
        fake_torch.distributed = fake_dist
        fake_updater = types.ModuleType(
            "miles.backends.training_utils.weight_update.updater"
        )
        fake_updater.WeightUpdater = Updater

        def wall_install(args):
            inner = Updater.update_weights

            def outer(self):
                probe.wall_probe._update_index += 1
                return inner(self)

            Updater.update_weights = outer

        (ROOT / "validation").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "validation") as directory:
            with (
                patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch,
                        "torch.distributed": fake_dist,
                        "miles.backends.training_utils.weight_update.updater": fake_updater,
                    },
                ),
                patch.dict(
                    "os.environ",
                    {
                        "WEIGHT_SYNC_RUN_DIR": directory,
                        "WEIGHT_SYNC_CPU_CALIBRATION_ITERATIONS": "2",
                    },
                ),
                patch.object(probe.wall_probe, "install", wall_install),
                patch.object(
                    probe,
                    "ReceiverClocks",
                    side_effect=RuntimeError("PID mapping missing"),
                ),
            ):
                probe.install(types.SimpleNamespace())
                self.assertEqual(Updater().update_weights(), "completed")
            rows = [
                json.loads(line)
                for line in Path(directory, "cpu-trainer-rank0.jsonl")
                .read_text()
                .splitlines()
            ]
            event = next(row for row in rows if row["kind"] == "cpu_trainer_update")
            self.assertFalse(event["cpu_valid"])
            self.assertIn("PID mapping missing", event["error"])
            self.assertTrue(event["success"])


class LauncherRender(unittest.TestCase):
    def test_actual_recipe_pair_with_launch_utilities_stubbed(self):
        recipe_path = (
            ROOT
            / "miles/tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py"
        )
        tree = ast.parse(recipe_path.read_text(), filename=str(recipe_path))
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module == "tests.ci.ci_register"
                or isinstance(node, ast.Import)
                and any(
                    alias.name == "miles.utils.external_utils.command_utils"
                    for alias in node.names
                )
            )
        ]

        def must_not_launch(**kwargs):
            raise AssertionError("render-only attempted execution")

        utilities = types.SimpleNamespace(
            create_run_id=lambda: "fixture",
            execute_train=must_not_launch,
            encode_pseudo_file=lambda text: "base64:"
            + base64.b64encode(text.encode()).decode(),
            get_default_wandb_args=lambda *args, **kwargs: "",
        )
        recipe = types.ModuleType("actual_recipe_fixture")
        recipe.__dict__.update(
            U=utilities,
            register_cuda_ci=lambda **kwargs: None,
            __file__=str(recipe_path),
        )
        exec(compile(tree, str(recipe_path), "exec"), recipe.__dict__)
        (ROOT / "validation").mkdir(exist_ok=True)
        cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory(dir=ROOT / "validation") as directory:
                manifests = []
                for mode in ("broadcast", "disk-delta"):
                    arguments = [
                        "launch",
                        "--mode",
                        mode,
                        "--reward-mode",
                        "synthetic-balanced",
                        "--implementation-label",
                        "candidate",
                        "--run-name",
                        mode,
                        "--output-dir",
                        directory,
                        "--render-only",
                    ]
                    with (
                        patch.object(sys, "argv", arguments),
                        patch.object(
                            launcher.importlib, "import_module", return_value=recipe
                        ),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        launcher.main()
                    manifests.append(
                        json.loads(Path(directory, mode, "manifest.json").read_text())
                    )
                self.assertEqual(
                    normalized_config(manifests[0]), normalized_config(manifests[1])
                )
                for manifest in manifests:
                    self.assertEqual(manifest["num_rollout"], 7)
                    self.assertIn(
                        "weight_sync_probe_cpu_v2.install", manifest["train_argv"]
                    )
                    self.assertTrue(
                        manifest["launch_kwargs"]["train_script"].endswith(
                            "train_weight_sync_profiled_cpu_v2.py"
                        )
                    )
                    self.assertEqual(
                        manifest["launch_kwargs"]["extra_env_vars"][
                            "WEIGHT_SYNC_CPU_RECEIVER"
                        ],
                        "required",
                    )
                    self.assertEqual(manifest["source_manifest_version"], 3)
                    self.assertIn(
                        "process_cpu_clocks_cpu_v2.py", manifest["script_sha256"]
                    )
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
