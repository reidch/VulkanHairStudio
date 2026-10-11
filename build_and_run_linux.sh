#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${V47_PREFLIGHT:-0}" == "1" ]] && command -v python3 >/dev/null; then
  python3 verify_project.py
  python3 tools/validate_groom_assets.py
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
exec ./build/vulkan_hair_v47 "$@"
