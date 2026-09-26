#!/usr/bin/env python3
"""Calibrate idle receiver observation and cross-check its descendant inventory."""

import argparse
import json
from pathlib import Path

import psutil

from process_cpu_clocks_cpu_v2 import ReceiverClocks, calibrate, self_cpu_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receiver = ReceiverClocks(args.engine_url)
    receiver.refresh()
    fast = set(receiver.processes)
    reference = set()
    for pid in receiver.roots.values():
        reference.add(pid)
        reference.update(child.pid for child in psutil.Process(pid).children(recursive=True))
    receiver.refresh()
    assert fast == reference == set(receiver.processes), (fast - reference, reference - fast)
    result = {
        "independent_psutil_inventory_matches": True,
        "process_count": len(fast),
        "scheduler_count": sum(r["role"] == "scheduler" for r in receiver.processes.values()),
        "self": calibrate(self_cpu_snapshot, 1000),
        "receiver_snapshot_pair": calibrate(receiver.snapshot, 1000),
        "membership_refresh_pair": calibrate(receiver.refresh, 20),
        "inventory": receiver.inventory(),
        "scope": "Idle engines before campaign; non-atomic live membership and transient-child limits still apply.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "inventory"}, indent=2))


if __name__ == "__main__":
    main()
