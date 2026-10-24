#!/usr/bin/env bash
# One-shot run: main experiments on four datasets + mixed ablations
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/eval_main.py --dataset all
python scripts/eval_ablation.py --ablation all
