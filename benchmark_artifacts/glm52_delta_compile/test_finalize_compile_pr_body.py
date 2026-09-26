"""Small fixture checks for final body assembly, not GPU measurements."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import finalize_compile_pr_body as subject


class FinalBodyTests(unittest.TestCase):
    def test_hosted_ci_failure_remains_explicit(self):
        review = json.loads(subject.raw(subject.ROOT / "artifacts/compile-pr-ci-3720-6fb0e2a/final-review.json"))
        output = subject.raw(subject.ROOT / "artifacts/compile-pr-ci-3720-6fb0e2a/local-torch211-delta-tests.log")
        body = subject.hosted_ci_section(review, lambda name, label: f"[{label}](fixture://{name})", review["head"], output)
        self.assertIn("did **not** pass", body)
        self.assertIn(review["failed_job"]["url"], body)
        self.assertIn("not a confirmed signal or diagnosis", body)
        self.assertIn("No hosted GPU validation is claimed", body)
        self.assertNotIn("8,355,212", body)
        compatibility = json.loads(subject.raw(subject.ROOT / "COMPILE_CI_COMPATIBILITY.json"))
        diagnosed = subject.hosted_ci_section(review, lambda name, label: f"[{label}](fixture://{name})",
                                             review["head"], output, compatibility)
        self.assertIn("Historical hosted CI failure and compatibility fix", diagnosed)
        self.assertIn("PyTorch #178950", diagnosed)
        self.assertIn("no new hosted pass is claimed", diagnosed)
        self.assertNotIn("Investigation remains open", diagnosed)
        review["hosted_ci_passed"] = True
        with self.assertRaisesRegex(ValueError, "record/source changed"):
            subject.hosted_ci_section(review, lambda name, label: label, review["head"], output)

    def test_only_exact_fork_commit_url(self):
        url = "https://github.com/zianglih/miles/tree/" + "a" * 40 + "/benchmark_artifacts/glm52_delta_compile"
        self.assertEqual(subject.parse_url(url)[0], "a" * 40)
        for bad in (url.replace("a" * 40, "main"), url.replace("zianglih", "radixark"), url.replace("delta_compile", "delta_sync")):
            with self.subTest(url=bad), self.assertRaises(ValueError):
                subject.parse_url(bad)

    def test_published_bytes_and_cr_preservation(self):
        content = b"begin\rprogress\nend\n"
        name = "raw.log"
        manifest = {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / name).write_bytes(content)

            def git_show(argv):
                return json.dumps(manifest).encode() if argv[-1].endswith("SHA256.json") else content

            url = "https://github.com/zianglih/miles/tree/" + "b" * 40 + "/benchmark_artifacts/glm52_delta_compile"
            with patch.object(subject.subprocess, "check_output", side_effect=git_show):
                link = subject.publication_links(root, url)
                self.assertEqual(link(name, "Raw log"), f"[Raw log]({url}/raw.log)")
                self.assertEqual(subject.raw(root / name).encode(), content)
                (root / name).write_bytes(b"changed\n")
                with self.assertRaisesRegex(ValueError, "differs"):
                    link(name, "Raw log")

    def test_embedded_selected_preflight_logs_require_published_bytes(self):
        root = subject.ROOT
        plan = json.loads(subject.raw(root / "COMPILE_VALIDATION_PLAN.json"))
        attribution = subject.validate_attribution(root, plan)
        preflight = subject.selected_preflight(root, plan)
        checked = []
        def link(name, label):
            checked.append(name)
            return label
        text = subject.source_preflight_section(root, plan, attribution, preflight, link)
        for filename in ("focused-tests.log", "gloo-failures.log"):
            name = f"{preflight['directory']}/{filename}"
            self.assertIn(name, checked)
            self.assertIn(subject.raw(root / name).rstrip(), text)
        def altered(name, label):
            if name == f"{preflight['directory']}/focused-tests.log":
                raise ValueError("Local evidence differs from the published bytes")
            return label
        with self.assertRaisesRegex(ValueError, "differs from the published bytes"):
            subject.source_preflight_section(root, plan, attribution, preflight, altered)

    def test_all_fourteen_rows_and_steady_compiler_gate(self):
        metrics = ("driver_wall_s", "builtin_actor_s", "trainer_max_wall_s", "generation_pause_upper_bound_s",
                   "trainer_cpu_sum_s", "trainer_cpu_max_s", "receiver_total_cpu_s", "driver_cpu_s")
        report = {"plan": {"sources": {"miles": "fixture", "sglang": "fixture"}}, "arms": {},
                  "campaign": {"devbox": "fixture-v2", "host": "fixture-host", "image": "fixture-image",
                               "image_index_digest": "fixture-index", "image_amd64_digest": "fixture-amd64"},
                  "external_calibration_scope": "Historical calibration; per-run actual costs retained.",
                  "backend_comparisons": {"synthetic-balanced": {"numpy": "fixture-numpy", "torch_compile": "fixture-compiled",
                    "metrics": {name: {"numpy": {"median": 1}, "torch_compile": {"median": 2}, "compiled_over_numpy": 2}
                                for name in metrics}}}}
        for backend, name in (("numpy", "fixture-numpy"), ("torch-compile", "fixture-compiled")):
            report["arms"][name] = {
                "manifest": {"delta_cpu_backend": backend},
                "rows": [{"update_index": index, **{metric: 1 for metric in metrics},
                          "changed_bytes": 100 + index, "wire_bytes": 200 + index, "cpu_valid": True} for index in range(7)],
                "compiler_observations": [{"measured_steady": index >= 2, "graph_or_kernel_activity": {}}
                                          for index in range(7) for _ in range(4)],
                "training": {"gradient_norms": [1], "replay_checks": 7, "replay_nonzero_checks": 0},
                "cpu_observer": {"initial_or_first_post_cpu_invalid": []},
            }
        text = subject.gpu_section(report, lambda name, label: f"[{label}](fixture://{name})")
        rows = [line for line in text.splitlines() if line.startswith("| ") and "| True |" in line]
        self.assertEqual(len(rows), 14)
        self.assertIn("fixture-image", text)
        self.assertIn("they are not new-image reruns", text)
        self.assertIn("summary-only", text)
        self.assertIn("driver wall change is **+100.00%**", text)
        self.assertIn("not an isolated measurement of receiver apply work", text)
        self.assertIn("Changed/wire bytes and gradient trajectories differ", text)
        report["arms"]["fixture-compiled"]["compiler_observations"][-1]["graph_or_kernel_activity"] = {"stats.unique_graphs": 1}
        with self.assertRaisesRegex(ValueError, "compiler"):
            subject.gpu_section(report, lambda name, label: label)
        report["arms"]["fixture-compiled"]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "14 update"):
            subject.gpu_section(report, lambda name, label: label)


if __name__ == "__main__":
    unittest.main()
