#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
VENV=${H3_VENV:-"$ROOT/.venv"}
LIGHTX=${H3_LIGHTX2V_SOURCE:-"$ROOT/.deps/LightX2V"}
COMMIT=8e86555809efd20138d3a4e1f937ee306cdf0ee0
"$PYTHON" -m venv "$VENV"
P="$VENV/bin/python"
"$P" -m pip install --upgrade pip setuptools wheel ninja packaging
"$P" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
"$P" -m pip install -r "$ROOT/requirements-runtime.txt"
if [ ! -d "$LIGHTX/.git" ]; then
    mkdir -p "$(dirname "$LIGHTX")"
    git init "$LIGHTX"
    git -C "$LIGHTX" remote add origin https://github.com/ModelTC/LightX2V.git
    git -C "$LIGHTX" fetch --depth 1 origin "$COMMIT"
    git -C "$LIGHTX" checkout --detach FETCH_HEAD
fi
if [ "$(git -C "$LIGHTX" rev-parse HEAD)" != "$COMMIT" ]; then
    echo "Use the pinned LightX2V commit $COMMIT at $LIGHTX" >&2
    exit 1
fi
if git -C "$LIGHTX" apply --check "$ROOT/runtime/patches/lightx2v-h3.patch"; then
    git -C "$LIGHTX" apply "$ROOT/runtime/patches/lightx2v-h3.patch"
else
    git -C "$LIGHTX" apply --reverse --check "$ROOT/runtime/patches/lightx2v-h3.patch"
fi
"$P" -m pip install --no-deps -e "$LIGHTX"
if [ "${H3_SKIP_SAGE:-0}" != 1 ]; then
    SAGE="$ROOT/.deps/SageAttention"
    if [ ! -d "$SAGE" ]; then
        git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git "$SAGE"
    fi
    "$P" -m pip install --no-build-isolation "$SAGE"
fi
echo "Ready. Activate with: source $VENV/bin/activate"
echo "Run: python scripts/run.py --help"
