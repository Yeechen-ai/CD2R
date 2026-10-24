#!/usr/bin/env python3
"""Main evaluation: load trained weights on mixed / resp / oral / inf and report Recall/MRR/SMR@5/10 for T1/T2."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from cd2r.eval.engine import run_eval

def main():
    parser = argparse.ArgumentParser(description="CD2R main evaluation (T1/T2)")
    parser.add_argument(
        "--dataset",
        choices=list(config.DATASET_CHOICES) + ["all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    datasets = list(config.DATASET_CHOICES) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n===== CD2R main | dataset={ds} =====")
        run_eval(ds, variant="full", seed=args.seed)

if __name__ == "__main__":
    main()
