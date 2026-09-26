"""Run frozen focused tests, then the exact failed hosted shard, with native stderr visible."""
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'miles'
OUTPUT = ROOT / 'compatibility-full-16-logical-gpus-results'
OUTPUT.mkdir(exist_ok=False)
PYTHON = Path((ROOT / 'python-path.txt').read_text().strip())
log = OUTPUT / 'phases.jsonl'


def emit(event):
    row = json.dumps(event, sort_keys=True)
    with log.open('a') as f:
        f.write(row + '\n')
    print(row, flush=True)


def run_phase(name, arguments, limit):
    cache = ROOT / (name + '-compatibility-full-16-logical-gpus-fresh-inductor-cache')
    assert not cache.exists(), cache
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS', 'PYTEST_ADDOPTS', 'RAY_ADDRESS', 'LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update({
        'CUDA_VISIBLE_DEVICES': '99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84', 'PYTHONFAULTHANDLER': '1',
        'GITHUB_WORKSPACE': str(SOURCE),
        'TORCHINDUCTOR_CACHE_DIR': str(cache),
        'SGLANG_SOURCE_ROOT': str(ROOT / 'sglang/python'),
        'MEGATRON_SOURCE_ROOT': str(ROOT / 'Megatron-LM'),
        'PYTHONPATH': ':'.join(map(str, [ROOT, SOURCE, ROOT / 'sglang/python', ROOT / 'Megatron-LM'])),
        'CI_REPRO_FATAL_PATH': str(OUTPUT / (name + '-fatal.log')),
        'PATH': str(ROOT / 'venv/bin') + ':' + str(ROOT / 'bin') + ':' + env['PATH'],
    })
    command = [str(PYTHON), '-X', 'faulthandler', '-m', 'pytest', *arguments, '-s', '-p', 'ci_resource_probe']
    started = time.monotonic()
    emit({'event': 'start', 'phase': name, 'command': command, 'cwd': str(SOURCE), 'timeout_s': limit,
          'cache': str(cache), 'capturing': False, 'thread_env': {k: env.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS')}})
    with (OUTPUT / (name + '.log')).open('wb') as stream:
        process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait(timeout=limit)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                result = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                result = process.wait(timeout=10)
    emit({'event': 'finish', 'phase': name, 'returncode': result,
          'signal': signal.Signals(-result).name if result < 0 else None,
          'wall_s': time.monotonic() - started, 'timed_out': timed_out})
    return result == 0 and not timed_out


emit({'event': 'scope', 'source_merge': '5093563d01e81e6a71a1425470f32e359338f37f',
      'sglang': '106ef6d1262a8c1bf61cabcb9053102ff80f71f9', 'megatron': 'f148a32b4385b758b66a77c9c3ad1641f1295d4b',
      'hardware': platform.platform(), 'scope': 'C2 Linux diagnostic; same source/dependencies and installed-version constraints; hosted hardware/memory limits are not reproduced'})
argv = json.loads((ROOT / 'hosted-shard-1-pytest-argv.json').read_text())
assert argv[0] == 'pytest' and argv[-2:] == ['-v', '-x']
raise SystemExit(0 if run_phase('full-shard-1', argv[1:], 1800) else 1)
