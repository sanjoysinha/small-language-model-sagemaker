"""
Evaluation gate for CI/CD.

Compares a candidate model (challenger) against the current production model (champion).
The candidate must:
  1. Meet absolute metric thresholds (MIN_F1, MIN_ACCURACY, etc.)
  2. Match or beat the champion model's F1 score

If both conditions pass, the challenger is promoted to champion.

Uses MLflow 3.x aliases: "champion" = production, "challenger" = staging.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple

import mlflow
import mlflow.pytorch
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data_loader import load_and_preprocess, create_dataloaders
from train import evaluate as evaluate_model_on_loader
from mlops.config_mlops import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_MODEL_NAME,
    MIN_F1,
    MIN_ACCURACY,
    MIN_PRECISION,
    MIN_RECALL,
)


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_by_alias(alias: str) -> Optional[Tuple]:
    """
    Load a model from MLflow registry by alias.

    Aliases in MLflow 3.x replace the old stage system:
      "challenger" = staging candidate
      "champion"   = production model

    Returns (model, tokenizer, version) or None if alias not found.
    """
    client = mlflow.MlflowClient()

    try:
        mv = client.get_model_version_by_alias(
            name=MLFLOW_REGISTRY_MODEL_NAME, alias=alias
        )
    except mlflow.exceptions.MlflowException:
        logger.warning("No model found with alias '{}' in registry", alias)
        return None

    logger.info(
        "Loading '{}' model: version={}, run_id={}",
        alias, mv.version, mv.run_id,
    )

    # Load the PyTorch model from MLflow
    model_uri = f"models:/{MLFLOW_REGISTRY_MODEL_NAME}/{mv.version}"
    model = mlflow.pytorch.load_model(model_uri)

    # Load tokenizer from the model_files artifact
    tokenizer_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="model_files"
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)

    return model, tokenizer, mv.version


def load_model_from_local(model_dir: str) -> Tuple:
    """Load model and tokenizer from a local directory."""
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    return model, tokenizer


def evaluate_on_test_set(model, tokenizer) -> Dict[str, float]:
    """Evaluate model on the test split and return metrics."""
    device = get_device()
    model.to(device)

    texts, labels = load_and_preprocess()
    _, _, test_loader = create_dataloaders(texts, labels, tokenizer)

    metrics = evaluate_model_on_loader(model, test_loader, device, verbose=True)
    return metrics


def check_thresholds(metrics: Dict[str, float]) -> Tuple[bool, str]:
    """
    Check if metrics meet minimum thresholds.

    Returns (passed, report_string).
    """
    checks = [
        ("f1", metrics["f1"], MIN_F1),
        ("accuracy", metrics["accuracy"], MIN_ACCURACY),
        ("precision", metrics["precision"], MIN_PRECISION),
        ("recall", metrics["recall"], MIN_RECALL),
    ]

    lines = []
    all_passed = True

    for name, value, threshold in checks:
        passed = value >= threshold
        status = "PASS" if passed else "FAIL"
        lines.append(f"  {name:>10}: {value:.4f} (min: {threshold:.2f}) [{status}]")
        if not passed:
            all_passed = False

    report = "\n".join(lines)
    return all_passed, report


def compare_and_gate() -> bool:
    """
    Run the full evaluation gate.

    1. Load challenger model from MLflow registry
    2. Load champion model (if exists)
    3. Evaluate both on test set
    4. Check thresholds + compare
    5. Promote challenger → champion if pass

    Returns True if gate passed, False otherwise.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    # ── Load challenger (staging) model ──
    challenger = load_model_by_alias("challenger")
    if challenger is None:
        logger.error("No 'challenger' model found. Run training first.")
        return False

    challenger_model, challenger_tokenizer, challenger_version = challenger

    # ── Evaluate challenger ──
    logger.info("=" * 60)
    logger.info("Evaluating CHALLENGER model (version {})", challenger_version)
    logger.info("=" * 60)
    challenger_metrics = evaluate_on_test_set(challenger_model, challenger_tokenizer)
    logger.info("Challenger metrics: {}", challenger_metrics)

    # ── Check absolute thresholds ──
    thresholds_passed, threshold_report = check_thresholds(challenger_metrics)
    logger.info("Threshold check:\n{}", threshold_report)

    if not thresholds_passed:
        logger.error("GATE FAILED: Challenger model does not meet minimum thresholds")
        return False

    # ── Load champion (production) model if exists ──
    champion = load_model_by_alias("champion")

    if champion is not None:
        champ_model, champ_tokenizer, champ_version = champion
        logger.info("=" * 60)
        logger.info("Evaluating CHAMPION model (version {})", champ_version)
        logger.info("=" * 60)
        champ_metrics = evaluate_on_test_set(champ_model, champ_tokenizer)
        logger.info("Champion metrics: {}", champ_metrics)

        # ── Compare F1 ──
        logger.info(
            "Comparison: Challenger F1={:.4f} vs Champion F1={:.4f}",
            challenger_metrics["f1"], champ_metrics["f1"],
        )

        if challenger_metrics["f1"] < champ_metrics["f1"]:
            logger.error(
                "GATE FAILED: Challenger F1 ({:.4f}) < Champion F1 ({:.4f})",
                challenger_metrics["f1"], champ_metrics["f1"],
            )
            return False
    else:
        logger.info("No champion model exists — first deployment, skipping comparison")

    # ── Promote challenger → champion ──
    logger.info("GATE PASSED — Promoting version {} to champion", challenger_version)

    client.set_registered_model_alias(
        name=MLFLOW_REGISTRY_MODEL_NAME,
        alias="champion",
        version=challenger_version,
    )
    logger.info("Version {} is now 'champion' (production)", challenger_version)

    # Remove challenger alias (it's now the champion)
    try:
        client.delete_registered_model_alias(
            name=MLFLOW_REGISTRY_MODEL_NAME,
            alias="challenger",
        )
    except mlflow.exceptions.MlflowException:
        pass

    return True


def evaluate_local(model_dir: str) -> Dict[str, float]:
    """Evaluate a local model directory (for ad-hoc testing)."""
    model, tokenizer = load_model_from_local(model_dir)
    metrics = evaluate_on_test_set(model, tokenizer)

    passed, report = check_thresholds(metrics)
    logger.info("Threshold check:\n{}", report)
    logger.info("Overall: {}", "PASS" if passed else "FAIL")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation gate")
    parser.add_argument(
        "--local", type=str, default=None,
        help="Evaluate a local model directory instead of MLflow registry",
    )
    args = parser.parse_args()

    if args.local:
        evaluate_local(args.local)
    else:
        passed = compare_and_gate()
        sys.exit(0 if passed else 1)
