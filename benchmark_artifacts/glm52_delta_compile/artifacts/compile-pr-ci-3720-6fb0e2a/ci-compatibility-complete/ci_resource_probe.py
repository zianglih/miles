"""Project-only native-crash and resource diagnostics; no production monkeypatches."""
import faulthandler
import json
import os
import sys
from pathlib import Path
import threading
import time

_fatal_stream = None


def snapshot(where):
    # Do not import Torch before the original suite does; preserve import order.
    torch = sys.modules.get('torch')
    status = Path('/proc/self/status').read_text()
    fields = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
              for line in status.splitlines() if line.split(':', 1)[0] in
              ('VmPeak', 'VmSize', 'VmHWM', 'VmRSS', 'Threads')}
    cgroup = {}
    for filename in ('memory.current', 'memory.peak', 'memory.max', 'memory.events', 'pids.current', 'pids.max'):
        path = Path('/sys/fs/cgroup') / filename
        if path.exists():
            cgroup[filename] = path.read_text().strip()
    print('CI_REPRO_RESOURCE ' + json.dumps({
        'where': where, 'time_ns': time.time_ns(), 'pid': os.getpid(), 'status': fields,
        'python_threads': [(t.name, t.ident) for t in threading.enumerate()],
        'torch_intra_threads': torch.get_num_threads() if torch else None, 'torch_inter_threads': torch.get_num_interop_threads() if torch else None,
        'grad_enabled': torch.is_grad_enabled() if torch else None, 'inference_enabled': torch.is_inference_mode_enabled() if torch else None,
        'affinity': sorted(os.sched_getaffinity(0)), 'cgroup': cgroup,
    }), flush=True)


def pytest_sessionstart(session):
    global _fatal_stream
    _fatal_stream = open(os.environ['CI_REPRO_FATAL_PATH'], 'w', buffering=1)
    faulthandler.enable(file=_fatal_stream, all_threads=True)
    snapshot('session_start')


def pytest_runtest_setup(item):
    if 'test_delta_preparation.py' in item.nodeid:
        snapshot('setup:' + item.nodeid)


def pytest_runtest_teardown(item):
    if 'test_delta_preparation.py' in item.nodeid:
        snapshot('teardown:' + item.nodeid)


def pytest_sessionfinish(session, exitstatus):
    snapshot('session_finish:' + str(exitstatus))
