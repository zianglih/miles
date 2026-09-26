"""Low-frequency process CPU counters; Linux receiver clocks need no server hooks.

CPU-only calibration (no GPU work):
  python process_cpu_clocks_cpu_v2.py --iterations 1000
  python process_cpu_clocks_cpu_v2.py --engine-url http://HOST:PORT --engine-url http://HOST:PORT
Use the latter after engines are ready; include both TP2 engine URLs. Calibration
records measurement cost, not an assumed/subtracted correction. Process clocks
count all threads. RUSAGE_CHILDREN and per-thread clocks are intentionally unused.
"""

import argparse
from collections import Counter
import ctypes
import json
import os
import resource
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlparse


def parse_proc_stat(text):
    fields = text[text.rindex(")") + 2 :].split()
    return {
        "ppid": int(fields[1]),
        "user_ticks": int(fields[11]),
        "system_ticks": int(fields[12]),
        "threads": int(fields[17]),
        "start_ticks": int(fields[19]),
    }


def identity(pid):
    return parse_proc_stat(Path(f"/proc/{pid}/stat").read_text())


def process_tree(root_pid, proc_root=Path("/proc")):
    """Visit only this tree; child processes can belong to any parent thread.

    Linux task children files avoid psutil's whole-node PPID scans. Stat and
    child-list failures propagate: incomplete membership must not look valid.
    Like any live /proc scan, this cannot observe children born and exited
    entirely between visits.
    """
    pending = [(root_pid, None)]
    seen = set()
    while pending:
        pid, parent_pid = pending.pop()
        if pid in seen:
            raise RuntimeError(f"Duplicate receiver child PID {pid}")
        seen.add(pid)
        base = proc_root / str(pid)
        stat = parse_proc_stat((base / "stat").read_text())
        if parent_pid is not None and stat["ppid"] != parent_pid:
            raise RuntimeError(f"Receiver child PID {pid} changed parent")
        if stat["threads"] == 1:
            tasks = [base / "task" / str(pid)]
        else:
            with os.scandir(base / "task") as entries:
                tasks = [Path(entry.path) for entry in entries if entry.name.isdigit()]
        children = set()
        for task in tasks:
            children.update(
                int(value) for value in (task / "children").read_text().split()
            )
        pending.extend((child, pid) for child in children)
        yield pid, stat


def process_command(pid):
    return (
        Path(f"/proc/{pid}/cmdline")
        .read_bytes()
        .replace(b"\0", b" ")
        .decode(errors="replace")
        .strip()
    )


def imported_sources():
    """Loaded modules only: never import a GPU dependency to observe its path."""
    names = (
        "miles",
        "miles.backends.training_utils.weight_update.updater",
        "sglang",
        "sglang.srt.managers.scheduler",
        "sglang.srt.managers.scheduler_components.weight_updater",
        "sglang.srt.model_executor.model_runner",
    )
    return {
        name: str(Path(module.__file__).resolve())
        for name in names
        if (module := sys.modules.get(name)) is not None
        and getattr(module, "__file__", None)
    }


def runtime_metadata(pid=None):
    pid = os.getpid() if pid is None else pid
    paths = [
        f"/proc/{pid}/cgroup",
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
    ]
    environment = os.environ
    error = None
    if pid != os.getpid():
        try:
            environment = dict(
                entry.split("=", 1)
                for entry in Path(f"/proc/{pid}/environ").read_text().split("\0")
                if "=" in entry
            )
        except Exception as exc:
            environment = {}
            error = f"{type(exc).__name__}: {exc}"
    result = {
        "pid": pid,
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(pid))
        if hasattr(os, "sched_getaffinity")
        else None,
        "process_clock": vars(time.get_clock_info("process_time")),
        "cpu_configuration": {
            p: Path(p).read_text().strip() for p in paths if Path(p).is_file()
        },
        "thread_environment": {
            k: v
            for k, v in environment.items()
            if k.startswith(("OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_"))
        },
        "metadata_error": error,
        "pythonpath": environment.get("PYTHONPATH"),
    }
    if pid == os.getpid():
        result["imported_sources"] = imported_sources()
    return result


def self_cpu_snapshot(include_resource=False):
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF) if include_resource else None
        return {"cpu_ns": time.process_time_ns(), "usage": usage, "error": None}
    except Exception as error:
        return {
            "cpu_ns": None,
            "usage": None,
            "error": f"{type(error).__name__}: {error}",
        }


def self_cpu_delta(before, after):
    error = before["error"] or after["error"]
    valid = error is None and after["cpu_ns"] >= before["cpu_ns"]
    usage = {}
    if before["usage"] is not None and after["usage"] is not None:
        usage = {
            name: getattr(after["usage"], name) - getattr(before["usage"], name)
            for name in (
                "ru_utime",
                "ru_stime",
                "ru_minflt",
                "ru_majflt",
                "ru_nvcsw",
                "ru_nivcsw",
            )
        }
    return {
        "cpu_start_ns": before["cpu_ns"],
        "cpu_end_ns": after["cpu_ns"],
        "cpu_s": (after["cpu_ns"] - before["cpu_ns"]) / 1e9 if valid else None,
        "cpu_valid": valid,
        "resource_usage": usage,
        "error": error if error else None if valid else "CPU clock moved backwards",
    }


def calibrate(read, iterations=200):
    values = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        read()
        read()
        values.append(time.perf_counter_ns() - start)
    return {
        "iterations": iterations,
        "two_reads_median_ns": statistics.median(values),
        "two_reads_min_ns": min(values),
        "two_reads_max_ns": max(values),
    }


class ProcessClock:
    def __init__(self, pid, backend="process-clock"):
        self.pid = pid
        self.backend = backend
        self.start_ticks = identity(pid)["start_ticks"]
        if backend == "process-clock":
            libc = ctypes.CDLL(None)
            function = libc.clock_getcpuclockid
            function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            function.restype = ctypes.c_int
            clock_id = ctypes.c_int()
            error = function(pid, ctypes.byref(clock_id))
            if error:
                raise OSError(error, os.strerror(error))
            self.clock_id = clock_id.value
            self.resolution_s = time.clock_getres(self.clock_id)
        elif backend == "proc-stat":
            self.ticks_per_second = os.sysconf("SC_CLK_TCK")
            self.resolution_s = 1 / self.ticks_per_second
        else:
            raise ValueError(f"Unknown CPU clock backend: {backend}")

    def validate(self):
        if identity(self.pid)["start_ticks"] != self.start_ticks:
            raise RuntimeError(f"PID {self.pid} was replaced")

    def read(self):
        if self.backend == "process-clock":
            return time.clock_gettime_ns(self.clock_id)
        values = identity(self.pid)
        return round(
            (values["user_ticks"] + values["system_ticks"])
            * 1e9
            / self.ticks_per_second
        )


class ReceiverClocks:
    def __init__(self, engine_urls, backend="process-clock", expected_schedulers=4):
        import psutil

        endpoints = sorted(set(engine_urls))
        if len(endpoints) != 2:
            raise RuntimeError(f"Expected two receiver engines, found {endpoints}")
        listeners = psutil.net_connections(kind="tcp")
        self.roots = {}
        for engine in endpoints:
            port = urlparse(engine).port
            roots = {
                connection.pid
                for connection in listeners
                if connection.status == psutil.CONN_LISTEN
                and connection.laddr.port == port
                and connection.pid
            }
            if len(roots) != 1:
                raise RuntimeError(
                    f"Cannot map {engine} to exactly one listening process: {roots}"
                )
            self.roots[engine] = roots.pop()
        self.processes = {}
        self.backend = backend
        self.engine_urls = endpoints
        self.expected_schedulers = expected_schedulers
        self.refresh()

    def membership(self):
        return sorted(
            (pid, record["start_ticks"]) for pid, record in self.processes.items()
        )

    def refresh(self):
        """Discover children outside timed updates; reuse stable cached clock handles."""
        previous = self.membership()
        current = {}
        for engine, root_pid in self.roots.items():
            for pid, stat in process_tree(root_pid):
                if pid in current:
                    raise RuntimeError(f"Receiver trees overlap at PID {pid}")
                old = self.processes.get(pid)
                start_ticks = stat["start_ticks"]
                if old is not None and old["start_ticks"] == start_ticks:
                    current[pid] = old
                else:
                    command = process_command(pid)
                    clock = ProcessClock(pid, self.backend)
                    if clock.start_ticks != start_ticks:
                        raise RuntimeError(
                            f"Receiver PID {pid} was replaced during discovery"
                        )
                    current[pid] = {
                        "pid": pid,
                        "engine": engine,
                        "role": "scheduler"
                        if command.startswith("sglang::scheduler")
                        else "auxiliary",
                        "command": command,
                        "ppid": stat["ppid"],
                        "start_ticks": start_ticks,
                        "resolution_s": clock.resolution_s,
                        "runtime": runtime_metadata(pid),
                        "clock": clock,
                    }
        # Linux children files can omit a surviving sibling while another child
        # exits. Never silently drop a previously inventoried process that still
        # has the same identity; invalidate this live scan instead.
        for pid in self.processes.keys() - current.keys():
            try:
                remaining = identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if remaining["start_ticks"] == self.processes[pid]["start_ticks"]:
                raise RuntimeError(
                    f"Live receiver PID {pid} disappeared from child traversal"
                )
        expected = self.expected_schedulers
        if (
            sum(record["role"] == "scheduler" for record in current.values())
            != expected
        ):
            raise RuntimeError(
                f"Receiver scheduler membership is not exactly {expected}"
            )
        self.processes = current
        return previous != self.membership()

    def validate(self):
        for process in self.processes.values():
            process["clock"].validate()

    def inventory(self):
        return [
            {k: v for k, v in record.items() if k != "clock"}
            for record in self.processes.values()
        ]

    def snapshot(self):
        start = time.perf_counter_ns()
        counters = {
            str(pid): record["clock"].read() for pid, record in self.processes.items()
        }
        end = time.perf_counter_ns()
        return {
            "start_ns": start,
            "end_ns": end,
            "duration_ns": end - start,
            "cpu_ns": counters,
        }

    def difference(self, before, after):
        rows = []
        for record in self.inventory():
            pid = str(record["pid"])
            elapsed = after["cpu_ns"][pid] - before["cpu_ns"][pid]
            if elapsed < 0:
                raise RuntimeError(f"CPU clock moved backwards for PID {pid}")
            rows.append(
                {k: record[k] for k in ("pid", "engine", "role", "start_ticks")}
                | {"cpu_ns": elapsed}
            )
        totals = Counter()
        for row in rows:
            totals[row["role"]] += row["cpu_ns"]
        return {
            "processes": rows,
            "scheduler_cpu_ns": totals["scheduler"],
            "auxiliary_cpu_ns": totals["auxiliary"],
            "total_cpu_ns": sum(totals.values()),
            "snapshot_cost_ns": before["duration_ns"] + after["duration_ns"],
            "before": before,
            "after": after,
        }


class Journal:
    """One buffered append after an update; no fsync, no hot-path per-tensor writes."""

    def __init__(self, path):
        self.file = Path(path).open("a", buffering=65536)
        self.pending = []
        self.previous_flush_ns = None

    def append(self, event):
        self.pending.append(event)

    def flush(self):
        if not self.pending:
            return
        start = time.perf_counter_ns()
        payload = [
            {**event, "previous_flush_ns": self.previous_flush_ns}
            for event in self.pending
        ]
        self.file.write(
            "".join(
                json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
                for event in payload
            )
        )
        self.file.flush()
        self.pending.clear()
        self.previous_flush_ns = time.perf_counter_ns() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", action="append", default=[])
    parser.add_argument(
        "--backend", choices=("process-clock", "proc-stat"), default="process-clock"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--membership-iterations", type=int, default=20)
    options = parser.parse_args()
    if options.iterations < 1 or options.membership_iterations < 1:
        parser.error("iterations must be positive")
    result = {
        "runtime": runtime_metadata(),
        "self": calibrate(time.process_time_ns, options.iterations),
    }
    if options.engine_url:
        try:
            receiver = ReceiverClocks(options.engine_url, options.backend)
            result["receiver"] = {
                "inventory": receiver.inventory(),
                "calibration": calibrate(receiver.snapshot, options.iterations),
                "membership_refresh_pair": calibrate(
                    receiver.refresh, options.membership_iterations
                ),
            }
        except Exception as error:
            result["receiver_error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(result, indent=2))
    return int("receiver_error" in result)


if __name__ == "__main__":
    raise SystemExit(main())
