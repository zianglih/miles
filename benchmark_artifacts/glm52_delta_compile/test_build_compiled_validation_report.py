"""Admission fixtures only. Adapted records below are never benchmark evidence."""

import copy
import json
from pathlib import Path
import shlex
import struct
import tempfile
import unittest

import build_compiled_validation_report as report

ROOT = Path(__file__).resolve().parent


def plan_fixture():
    return {"schema_version": 1, "frozen": True, "sources": {"miles": "a" * 40, "sglang": "b" * 40},
            "script_sha256": {**report.base.FROZEN_MEASUREMENT_SHA256,
                              **{name: "c" * 64 for name in report.COMPILE_HELPERS}},
            "runs": [{"name": "fixture-compiled", "mode": "disk-delta", "reward_mode": "native",
                      "delta_cpu_backend": "torch-compile", "num_rollout": 7,
                      "compile_cache_dir": "/fixture/cache-compiled",
                      "compile_cache_state_before": {"exists": False, "empty": True}}]}


def compile_event(trainer, backend):
    index = trainer["update_index"]
    before_count = min(index, 2) if backend == "torch-compile" else 0
    after_count = min(index + 1, 2) if backend == "torch-compile" else 0
    before = {"inductor.generated_kernel_count": before_count, "stats.unique_graphs": before_count}
    after = {key: after_count for key in before}
    return {"kind": "compile_observation", "update_index": index, "rank": trainer["rank"],
            "pid": trainer["pid"], "success": True, "mode": "disk-delta", "cpu_backend": backend,
            "before": before, "after": after,
            "delta": {key: after[key] - value for key, value in before.items() if value != after[key]}}


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def compiler_fixture(root):
    write_jsonl(root / "driver.jsonl", [])
    for rank in range(4):
        events = []
        for index in range(7):
            trainer = {"kind": "trainer_update", "update_index": index, "rank": rank, "pid": 100 + rank}
            events.extend((trainer, compile_event(trainer, "torch-compile")))
        write_jsonl(root / f"trainer-rank{rank}.jsonl", events)
    return {"delta_cpu_backend": "torch-compile"}


class PlanAndCounters(unittest.TestCase):
    def test_frozen_helper_and_run_contract(self):
        self.assertEqual(set(report.validate_plan(plan_fixture())), {"fixture-compiled"})
        for mutation in (
            lambda p: p.update(frozen=False),
            lambda p: p.update(implementation_label="baseline"),
            lambda p: p["script_sha256"].pop("weight_sync_probe_compile.py"),
            lambda p: p["script_sha256"].update({"weight_sync_probe_cpu_v2.py": "0" * 64}),
            lambda p: p["runs"][0].update(compile_cache_dir="relative"),
            lambda p: p["runs"].append(copy.deepcopy(p["runs"][0])),
            lambda p: p["runs"].append({**p["runs"][0], "name": "fixture-numpy", "delta_cpu_backend": "numpy"}),
        ):
            plan = plan_fixture()
            mutation(plan)
            with self.subTest(mutation=mutation), self.assertRaises(report.InvalidEvidence):
                report.validate_plan(plan)

    def test_initial_and_first_post_compile_retained_steady_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = report.compiler_observations(root, compiler_fixture(root))
            self.assertEqual(len(records), 28)
            self.assertTrue(all(event["graph_or_kernel_activity"] for event in records[:8]))
            self.assertTrue(all(not event["graph_or_kernel_activity"] for event in records[8:]))

    def test_missing_misattributed_stale_and_steady_compile_rejected(self):
        for scenario in ("missing", "rank", "backend", "arithmetic", "steady", "driver"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = compiler_fixture(root)
                path = root / "trainer-rank0.jsonl"
                records = report.read_jsonl(path)
                if scenario == "missing":
                    records.pop()
                elif scenario == "rank":
                    records[1]["rank"] = None
                elif scenario == "backend":
                    records[1]["cpu_backend"] = "numpy"
                elif scenario == "arithmetic":
                    records[1]["delta"]["stats.unique_graphs"] = 999
                elif scenario == "steady":
                    records[5]["after"]["stats.unique_graphs"] += 1
                    records[5]["delta"]["stats.unique_graphs"] = 1
                else:
                    write_jsonl(root / "driver.jsonl", [records[1]])
                write_jsonl(path, records)
                with self.assertRaises(report.InvalidEvidence):
                    report.compiler_observations(root, manifest)


class Publication(unittest.TestCase):
    def test_shard_mapping_checksums_sizes_and_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"wire_bytes": None, "changed_bytes": None}]
            for version in range(1, 7):
                target = root / "delta-publication" / f"weight_v{version:06d}"
                target.mkdir(parents=True)
                header = json.dumps({"weight": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
                                     "__metadata__": {"weight": "a" * 32}}).encode()
                shard = target / "model.safetensors"
                shard.write_bytes(struct.pack("<Q", len(header)) + header + b"abc")
                (target / "model.safetensors.index.json").write_text(json.dumps({
                    "metadata": {"version": version, "base_version": version - 1, "delta_encoding": "xor",
                                 "compression_format": "zstd", "checksum_format": "xxh3-128"},
                    "weight_map": {"weight": shard.name}}))
                rows.append({"wire_bytes": shard.stat().st_size, "changed_bytes": 1})
            records = report.publications(root, rows)
            self.assertEqual(len(records), 6)
            rows[-1]["wire_bytes"] += 1
            with self.assertRaisesRegex(report.InvalidEvidence, "Publication bytes"):
                report.publications(root, rows)
            rows[-1]["wire_bytes"] -= 1
            final = root / "delta-publication/weight_v000006/model.safetensors"
            raw = final.read_bytes().replace(b'"a' + b"a" * 31 + b'"', b'"z' + b"a" * 31 + b'"')
            final.write_bytes(raw)
            with self.assertRaisesRegex(report.InvalidEvidence, "checksum"):
                report.publications(root, rows)


class FullAdapter(unittest.TestCase):
    @unittest.skipUnless((ROOT / "artifacts/cpu-v2-native-disk-delta-01/manifest.json").is_file(),
                         "Optional retained-artifact admission fixture")
    def test_original_admission_preserved_and_matching_pair(self):
        """Reuse immutable wall/CPU records only as a labeled temporary fixture."""
        source = ROOT / "artifacts/cpu-v2-native-disk-delta-01"
        original_globals = dict(report.base.load_arm.__globals__)
        original_sha = report.sha256(Path(report.base.__file__))
        with tempfile.TemporaryDirectory(prefix="compiled-report-fixture-") as directory:
            temporary = Path(directory)
            plan = plan_fixture()
            plan["runs"] = []
            for backend in ("numpy", "torch-compile"):
                name = "fixture-" + backend
                spec = {**plan_fixture()["runs"][0], "name": name, "delta_cpu_backend": backend,
                        "compile_cache_dir": "/fixture/cache-" + backend}
                plan["runs"].append(spec)
                target = temporary / name
                target.mkdir()
                for path in source.iterdir():
                    if path.name not in {"manifest.json", "cpu-summary.json", *(f"trainer-rank{r}.jsonl" for r in range(4))}:
                        (target / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
                for suffix in ("log", "exit"):
                    (temporary / f"{name}.{suffix}").symlink_to(source.parent / f"{source.name}.{suffix}")
                manifest = report.read_json(source / "manifest.json")
                plan["sources"] = {"miles": manifest["git_head"], "sglang": manifest["sglang_source"]["git_head"]}
                manifest.update({key: spec[key] for key in ("delta_cpu_backend", "compile_cache_dir", "compile_cache_state_before")})
                manifest.update(compile_observer_version=1, script_sha256=plan["script_sha256"])
                argv = manifest["train_argv"]
                argv[argv.index("--custom-megatron-init-path") + 1] = "weight_sync_probe_compile.install"
                argv.extend(("--update-weight-delta-cpu-backend", backend))
                manifest["launch_kwargs"]["train_args"] = shlex.join(argv)
                manifest["launch_kwargs"]["extra_env_vars"]["TORCHINDUCTOR_CACHE_DIR"] = spec["compile_cache_dir"]
                (target / "manifest.json").write_text(json.dumps(manifest))
                cpu = report.read_json(source / "cpu-summary.json")
                cpu["manifest"] = manifest
                (target / "cpu-summary.json").write_text(json.dumps(cpu))
                for rank in range(4):
                    records = report.read_jsonl(source / f"trainer-rank{rank}.jsonl")
                    records.extend(compile_event(event, backend) for event in list(records) if event["kind"] == "trainer_update")
                    write_jsonl(target / f"trainer-rank{rank}.jsonl", records)
            plan_path = temporary / "plan.json"
            plan_path.write_text(json.dumps(plan))
            arguments = ([temporary / run["name"] for run in plan["runs"]], plan_path, ROOT / "CAMPAIGN.json",
                         ROOT / "environment-image.json", ROOT / "artifacts/cpu-observer-fast-final-calibration.json")
            result = report.build(*arguments)
            self.assertTrue(result["validated_compiled_campaign"])
            self.assertEqual(set(result["backend_comparisons"]), {"native"})
            self.assertTrue(all(value["compiled_over_numpy"] == 1 for value in result["backend_comparisons"]["native"]["metrics"].values()))
            self.assertEqual(len(result["arms"]["fixture-torch-compile"]["compiler_observations"]), 28)
            self.assertIn("# Compiled disk-delta validation", report.markdown(result))
            with self.assertRaisesRegex(report.InvalidEvidence, "Every frozen planned run"):
                report.build(arguments[0][:-1], *arguments[1:])
            # Replacement metadata is selected by exact plan pins, and every
            # launch must bind to those bytes before its measurements begin.
            for field, original in zip(("campaign_metadata", "environment_metadata", "cpu_calibration_metadata"), arguments[2:]):
                name = field + "-fixture.json"
                (temporary / name).write_bytes(original.read_bytes())
                plan[field] = {"path": name, "sha256": report.sha256(temporary / name)}
            plan_path.write_text(json.dumps(plan))
            selected = report.resolve_metadata(plan_path, plan)
            record = {"plan_sha256": report.sha256(plan_path),
                      "metadata_selection": report.metadata_record(selected, relative_to=temporary)}
            for run in arguments[0]:
                (temporary / f"{run.name}-campaign-metadata.json").write_text(json.dumps(record))
            pinned_result = report.build(arguments[0], plan_path)
            self.assertEqual(pinned_result["campaign"], result["campaign"])
            self.assertIn("Historical observer calibration", pinned_result["external_calibration_scope"])
            marker = temporary / f"{arguments[0][0].name}-campaign-metadata.json"
            marker.write_text("{}")
            with self.assertRaisesRegex(report.InvalidEvidence, "launcher campaign metadata"):
                report.build(arguments[0], plan_path)
            marker.write_text(json.dumps(record))
            with self.assertRaisesRegex(report.InvalidEvidence, "override differs"):
                report.build(*arguments)
            # A source mismatch still goes through the unchanged strict loader.
            manifest_path = temporary / "fixture-torch-compile/manifest.json"
            manifest = report.read_json(manifest_path)
            manifest["git_head"] = "0" * 40
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(report.InvalidEvidence, "source mismatch"):
                report.build(arguments[0], plan_path)
        self.assertEqual(original_globals, report.base.load_arm.__globals__)
        self.assertEqual(original_sha, report.sha256(Path(report.base.__file__)))


if __name__ == "__main__":
    unittest.main()
