"""CPU-only report admission tests; fixtures are not benchmark results."""

import copy
import tempfile
from pathlib import Path
import unittest

import build_cpu_benchmark_report as report
from test_weight_sync_cpu_v2 import fixture


def clock_fixture():
    manifest, wall, drivers, trainers = fixture()
    outer = []
    for event in drivers + trainers:
        index = event["update_index"]
        start = 10_000_000_000 * (index + 1)
        end = start + 1_000_000_000 * (index + 1)
        event.update(start_ns=start, end_ns=end, wall_s=(end - start) / 1e9)
        event.update(
            cpu_start_ns=index * 1_000_000_000,
            cpu_end_ns=index * 1_000_000_000 + round(event["cpu_s"] * 1e9),
        )
        if event["kind"] == "cpu_trainer_update":
            outer.append(
                dict(
                    update_index=index,
                    rank=event["rank"],
                    start_ns=start - 100,
                    end_ns=end + 100,
                )
            )
            if event["rank"] == 0:
                processes = [
                    dict(
                        pid=pid,
                        role="scheduler" if pid < 4 else "auxiliary",
                        engine="a" if pid % 2 else "b",
                        start_ticks=pid + 10,
                        cpu_ns=250_000_000,
                    )
                    for pid in range(6)
                ]
                event["receiver"] = {
                    "processes": processes,
                    "scheduler_cpu_ns": 1_000_000_000,
                    "auxiliary_cpu_ns": 500_000_000,
                    "total_cpu_ns": 1_500_000_000,
                    "snapshot_cost_ns": 20,
                    "before": dict(
                        start_ns=start - 20,
                        end_ns=start - 10,
                        duration_ns=10,
                        cpu_ns={str(pid): 1_000_000 for pid in range(6)},
                    ),
                    "after": dict(
                        start_ns=end + 10,
                        end_ns=end + 20,
                        duration_ns=10,
                        cpu_ns={str(pid): 251_000_000 for pid in range(6)},
                    ),
                }
    return manifest, wall, drivers, trainers, outer


def runtime_fixture():
    manifest, _, drivers, trainers, _ = clock_fixture()
    runtime = dict(
        pid=100,
        affinity=[0, 1],
        cpu_count=2,
        process_clock={"resolution": 1e-9},
        cpu_configuration={"/proc/100/cgroup": "0::/fixture"},
        thread_environment={"OMP_NUM_THREADS": "1"},
        metadata_error=None,
        pythonpath="/fixture/miles:/fixture/sglang/python",
        imported_sources={
            "miles": "/fixture/miles/miles/__init__.py",
            "sglang": "/fixture/sglang/python/sglang/__init__.py",
        },
    )
    manifest.update(
        repo="/fixture/miles",
        sglang_source={"repo": "/fixture/sglang"},
        launcher_runtime=copy.deepcopy(runtime),
    )
    drivers.append(dict(kind="cpu_driver_installed", runtime=copy.deepcopy(runtime)))
    trainers.extend(
        dict(kind="cpu_probe_installed", rank=rank, runtime=copy.deepcopy(runtime))
        for rank in range(4)
    )
    processes = copy.deepcopy(trainers[0]["receiver"]["processes"])
    for process in processes:
        process["runtime"] = copy.deepcopy(runtime)
    trainers.append(dict(kind="cpu_receiver_inventory", processes=processes))
    return manifest, drivers, trainers


class RawClocks(unittest.TestCase):
    def test_valid_seven_raw_updates(self):
        _, _, drivers, trainers, outer = clock_fixture()
        report.validate_cpu_clocks(drivers, trainers, outer)

    def test_bad_counter_sum_wall_and_resource_rejected(self):
        for mutation in (
            lambda d, t: d[-1].update(cpu_s=-1),
            lambda d, t: d[-1].update(wall_s=float("nan")),
            lambda d, t: t[-1]["resource_usage"].update(ru_minflt=-1),
            lambda d, t: t[-4]["receiver"].update(total_cpu_ns=1),
            lambda d, t: t[-4]["receiver"]["after"]["cpu_ns"].update({"0": 0}),
            lambda d, t: t[-4]["receiver"]["after"].update(duration_ns=-1),
        ):
            with self.subTest(mutation=mutation):
                _, _, drivers, trainers, outer = clock_fixture()
                mutation(drivers, trainers)
                with self.assertRaises(report.InvalidEvidence):
                    report.validate_cpu_clocks(drivers, trainers, outer)

    def test_four_steady_samples_rejected(self):
        _, _, drivers, trainers, outer = clock_fixture()
        with self.assertRaises(report.InvalidEvidence):
            report.validate_cpu_clocks(drivers[:-1], trainers[:-4], outer[:-4])


class RuntimeIdentity(unittest.TestCase):
    def test_valid_configured_receiver_source(self):
        result = report.validate_runtime_sources(*runtime_fixture())
        self.assertEqual(
            result["normalized_roles"]["receiver_roles"][0][0], "auxiliary"
        )
        self.assertIn("not independently", result["receiver_source_check"])

    def test_wrong_loaded_or_configured_source_rejected(self):
        data = runtime_fixture()
        data[1][-1]["runtime"]["imported_sources"]["miles"] = "/installed/miles.py"
        with self.assertRaises(report.InvalidEvidence):
            report.validate_runtime_sources(*data)
        data = runtime_fixture()
        data[2][-1]["processes"][0]["runtime"]["pythonpath"] = "/installed"
        with self.assertRaises(report.InvalidEvidence):
            report.validate_runtime_sources(*data)

    def test_pid_normalization_and_thread_difference(self):
        runtime = runtime_fixture()[1][-1]["runtime"]
        other = copy.deepcopy(runtime)
        other["pid"] = 999
        other["cpu_configuration"] = {"/proc/999/cgroup": "0::/fixture"}
        self.assertEqual(
            report.normalized_runtime(runtime), report.normalized_runtime(other)
        )
        other["thread_environment"]["OMP_NUM_THREADS"] = "2"
        self.assertNotEqual(
            report.normalized_runtime(runtime), report.normalized_runtime(other)
        )

    def test_missing_arms_does_not_write_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(report.InvalidEvidence):
                report.build(
                    [root / "missing"],
                    root / "campaign",
                    root / "plan",
                    root / "environment",
                )
            self.assertFalse(list(root.iterdir()))


if __name__ == "__main__":
    unittest.main()
