#!/usr/bin/env bash
set -euo pipefail
cd /hai-workspace/glm52-delta
git -C miles-torch-compile fetch ../artifacts/torch-compile-final-validation.bundle refs/heads/glm52-delta-torch-compile-cpu
git -C miles-torch-compile worktree add --detach ../miles-torch-compile-final 6fb0e2a9d81fc343863ba6015512fe93992892e4
cd miles-torch-compile-final
test "$(git rev-parse HEAD)" = 6fb0e2a9d81fc343863ba6015512fe93992892e4
test -z "$(git status --porcelain)"
git rev-parse HEAD > ../artifacts/torch-compile-final-production-tested-head.txt
sha256sum miles/utils/delta_preparation.py miles/backends/training_utils/weight_update/packed_delta.py miles/backends/training_utils/weight_update/protocols/delta.py miles/utils/arguments.py tests/fast/utils/test_delta_preparation.py tests/fast/backends/training_utils/weight_update/test_delta_compiled_cpu.py > ../artifacts/torch-compile-final-source-sha256.txt
set +e
CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/hai-workspace/glm52-delta/artifacts/torch-compile-final-production-unit-cache PYTHONPATH=. timeout --kill-after=5 240 /opt/sglang/bin/python -m pytest tests/fast/utils/test_delta_preparation.py tests/fast/backends/training_utils/weight_update/test_delta_compiled_cpu.py tests/fast/backends/training_utils/weight_update/test_delta.py tests/fast/backends/training_utils/weight_update/test_disk_delta_weight_version.py tests/fast/backends/training_utils/weight_update/test_disk_delta_engine_calls.py -q > ../artifacts/torch-compile-final-production-tests.log 2>&1
result=$?
echo "$result" > ../artifacts/torch-compile-final-production-tests.exit
tail -8 ../artifacts/torch-compile-final-production-tests.log
exit "$result"
