#!/usr/bin/env bash
# Creates .venv and installs PyTorch + LIBERO-PRO.
# Tested on Ubuntu 22.04, Python 3.10, CUDA 12.4, RTX 4060 Ti (8 GB).
set -euo pipefail
cd "$(dirname "$0")"

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

mkdir -p third_party
if [ ! -d third_party/LIBERO-PRO ]; then
    git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git third_party/LIBERO-PRO
fi
pip install --no-deps -e third_party/LIBERO-PRO
echo "done. run: source env.sh"
