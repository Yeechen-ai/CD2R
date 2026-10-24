"""Global config: dataset paths, eval hyperparameters, and final CD2R inference protocol (eval-only)."""
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_FILES = {
    "mixed": None,
    "resp": None,
    "oral": None,
    "inf": None,
}
DATASET_CHOICES = tuple(DATASET_FILES.keys())
DATASET_NAME = "mixed"
ABLATION_DATASETS = ("mixed",)

LEGACY_TO_RELEASE = {
    "raw": "mixed",
    "breathe": "resp",
    "oral": "oral",
    "inject": "inf",
}

ARTIFACTS_ROOT = os.path.join(PROJECT_ROOT, "artifacts")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

EMB_DIM = 64
N_LAYERS = 2
SEED = 42
EVAL_K = [5, 10]

CAUSAL_ALPHA = 0.5
LAMBDA_ALIGN = 0.15

USE_GPU = os.environ.get("CD2R_USE_GPU", "auto")
FORCE_CPU = os.environ.get("CD2R_FORCE_CPU", "0") == "1"

def resolve_device() -> str:
    if FORCE_CPU:
        return "cpu"
    if USE_GPU == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if USE_GPU == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"

DEVICE = resolve_device()

def artifact_dir(dataset: str, variant: str = "full") -> str:
    return os.path.join(ARTIFACTS_ROOT, dataset, variant)

def results_dir(dataset: str, variant: str = "full") -> str:
    return os.path.join(RESULTS_ROOT, dataset, variant)

def assert_ablation_allowed(dataset: str, variant: str) -> None:
    if variant.startswith("abl") and dataset not in ABLATION_DATASETS:
        raise RuntimeError(
            f"Ablation '{variant}' is only supported for datasets {ABLATION_DATASETS}, got {dataset}."
        )
