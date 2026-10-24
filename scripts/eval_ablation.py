#!/usr/bin/env python3
"""Ablation evaluation: Abl-01–04 on mixed only (department-specific datasets ship no ablation artifacts)."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from cd2r.eval.engine import run_eval

ABLATIONS = ["abl01", "abl02", "abl03", "abl04"]

def main():
    parser = argparse.ArgumentParser(description="CD2R ablation evaluation (mixed only)")
    parser.add_argument("--ablation", choices=ABLATIONS + ["all"], default="all")
    parser.add_argument(
        "--dataset",
        default="mixed",
        choices=list(config.ABLATION_DATASETS),
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    variants = ABLATIONS if args.ablation == "all" else [args.ablation]
    for var in variants:
        print(f"\n===== CD2R ablation | {args.dataset}/{var} =====")
        run_eval(args.dataset, variant=var, seed=args.seed)

if __name__ == "__main__":
    main()
