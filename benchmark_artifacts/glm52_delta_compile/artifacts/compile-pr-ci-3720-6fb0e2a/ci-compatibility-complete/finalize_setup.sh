#!/usr/bin/env bash
set -euo pipefail
CI_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export UV_CACHE_DIR="$CI_ROOT/uv-cache" UV_PYTHON_INSTALL_DIR="$CI_ROOT/managed-python" UV_MANAGED_PYTHON=1 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=2
ulimit -c 0
exec > >(tee "$CI_ROOT/setup-logs/setup-finalize.log") 2>&1
CI_PYTHON="$CI_ROOT/venv/bin/python"
export GITHUB_WORKSPACE="$CI_ROOT/miles"
export SGLANG_SOURCE_ROOT="$CI_ROOT/sglang/python"
export MEGATRON_SOURCE_ROOT="$CI_ROOT/Megatron-LM"
export PYTHONPATH="$CI_ROOT/miles:$SGLANG_SOURCE_ROOT:$MEGATRON_SOURCE_ROOT"
export CUDA_VISIBLE_DEVICES=''
cd "$CI_ROOT/miles"
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CI_PYTHON" tests/ci/verify_source_resolution.py

# Hosted suite usesHelm4.2.3; retain the binary in this task's bin directory.
timeout --kill-after=10 120 curl -fL --retry 2 https://get.helm.sh/helm-v4.2.3-linux-amd64.tar.gz -o "$CI_ROOT/helm-v4.2.3-linux-amd64.tar.gz"
tar -xzf "$CI_ROOT/helm-v4.2.3-linux-amd64.tar.gz" -C "$CI_ROOT" linux-amd64/helm
cp "$CI_ROOT/linux-amd64/helm" "$CI_ROOT/bin/helm"
"$CI_ROOT/bin/helm" version

env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CI_PYTHON" - <<'PY' > "$CI_ROOT/setup-logs/runtime.json"
import json, os, platform, sys
import numpy, torch
print(json.dumps({'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                  'torch_git': torch.version.git_version, 'numpy': numpy.__version__,
                  'cuda_available': torch.cuda.is_available(), 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
                  'intra_threads': torch.get_num_threads(), 'inter_threads': torch.get_num_interop_threads(),
                  'affinity': sorted(os.sched_getaffinity(0)), 'omp_num_threads': os.environ.get('OMP_NUM_THREADS')}, indent=2))
PY
c++ --version > "$CI_ROOT/setup-logs/compiler.txt"
lscpu > "$CI_ROOT/setup-logs/lscpu.txt"
cat "$CI_ROOT/setup-logs/runtime.json"
git status --porcelain
echo 'Setup complete. Run: venv/bin/python run_reproduction.py from the ci-torch211-repro directory.'
