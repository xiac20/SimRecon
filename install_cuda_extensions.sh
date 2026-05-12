#!/usr/bin/env bash
# 在 conda env create 完成且已 conda activate、PyTorch 已裝好後執行。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# conda / 系統有時帶入 -Werror，PyTorch 2.x 標頭會觸發大量 deprecated 警告而導致 ninja 失敗
if [[ "${CFLAGS:-}" == *-Werror* ]] || [[ "${CXXFLAGS:-}" == *-Werror* ]]; then
  echo "[install_cuda_extensions] 偵測到 CFLAGS/CXXFLAGS 含 -Werror，已暫時 unset（避免擴充編譯失敗）"
  unset CFLAGS CXXFLAGS
fi

PY=python3
command -v "${PY}" >/dev/null 2>&1 || PY=python

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST="$("${PY}" - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{major}.{minor}")
else:
    print("12.1")
PY
)"
  echo "[install_cuda_extensions] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi

pip install --no-build-isolation -e submodules/diff-surfel-rasterization
pip install --no-build-isolation submodules/simple-knn
