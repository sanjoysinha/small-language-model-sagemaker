"""
Production monitoring.

Checks SageMaker endpoint health, API endpoint health (via API Gateway),
runs smoke tests, and detects prediction drift.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional

import boto3
import requests
import mlflow
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.config_mlops import (
    MLFLOW_TRACKING_URI,
    AWS_REGION,
    SAGEMAKER_ENDPOINT_NAME,
    API_BASE_URL,
    MONITOR_INTERVAL_SECONDS,
)

# Known test samples for smoke testing
SMOKE_PHISHING = (
    "URGENT: Your account has been compromised! Click here immediately to verify "
    "your identity and restore access. Failure to act within 24 hours will result "
    "in permanent account suspension. http://totally-legit-bank.com/verify"
)
SMOKE_SAFE = (
    "Hi team, please find attached the Q3 financial report. "
    "Let me know if you have any questions. Best regards, John"
)


def check_sagemaker_endpoint_health() -> Dict:
    """Check SageMaker endpoint health via AWS API."""
    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    try:
        response = sm.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
        status = response["EndpointStatus"]
        return {
            "endpoint_name": SAGEMAKER_ENDPOINT_NAME,
            "status": status,
            "healthy": status == "InService",
            "creation_time": str(response.get("CreationTime", "")),
            "last_modified": str(response.get("LastModifiedTime", "")),
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e), "healthy": False}


def check_endpoint_health(base_url: Optional[str] = None) -> Dict:
    """Check the /health endpoint of the deployed API (via API Gateway)."""
    url = (base_url or API_BASE_URL).rstrip("/") + "/health"

    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        return {
            "status_code": resp.status_code,
            "is_ready": data.get("is_ready", False),
            "api_status": data.get("status", "unknown"),
            "healthy": resp.status_code == 200 and data.get("is_ready", False),
        }
    except requests.ConnectionError:
        return {"status_code": None, "healthy": False, "message": "Connection refused"}
    except Exception as e:
        return {"status_code": None, "healthy": False, "message": str(e)}


def run_smoke_test(base_url: Optional[str] = None) -> Dict:
    """Send known phishing + safe emails and verify correct classification."""
    url = (base_url or API_BASE_URL).rstrip("/") + "/predict"
    results = {}

    # Test phishing detection
    try:
        resp = requests.post(url, json={"email_text": SMOKE_PHISHING}, timeout=30)
        data = resp.json()
        results["phishing_test"] = {
            "label": data.get("label"),
            "confidence": data.get("confidence"),
            "correct": data.get("label") == "phishing",
            "latency_ms": data.get("latency_ms"),
        }
    except Exception as e:
        results["phishing_test"] = {"correct": False, "error": str(e)}

    # Test safe detection
    try:
        resp = requests.post(url, json={"email_text": SMOKE_SAFE}, timeout=30)
        data = resp.json()
        results["safe_test"] = {
            "label": data.get("label"),
            "confidence": data.get("confidence"),
            "correct": data.get("label") == "safe",
            "latency_ms": data.get("latency_ms"),
        }
    except Exception as e:
        results["safe_test"] = {"correct": False, "error": str(e)}

    results["all_passed"] = (
        results.get("phishing_test", {}).get("correct", False)
        and results.get("safe_test", {}).get("correct", False)
    )

    return results


def check_prediction_drift(
    base_url: Optional[str] = None,
    n_samples: int = 50,
) -> Dict:
    """
    Send test set samples and check if phishing ratio drifts from expected.

    Expected ratio is ~50% for the balanced test set.
    Drift alert if ratio deviates by more than 15 percentage points.
    """
    url = (base_url or API_BASE_URL).rstrip("/") + "/predict"

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_loader import load_and_preprocess

    texts, labels = load_and_preprocess()
    sample_texts = texts[-n_samples:]
    sample_labels = labels[-n_samples:]

    phishing_count = 0
    total = 0
    latencies = []

    for text in sample_texts:
        try:
            resp = requests.post(url, json={"email_text": text}, timeout=30)
            data = resp.json()
            if data.get("label") == "phishing":
                phishing_count += 1
            total += 1
            latencies.append(data.get("latency_ms", 0))
        except Exception:
            continue

    if total == 0:
        return {"status": "ERROR", "message": "No successful predictions"}

    predicted_ratio = phishing_count / total
    expected_ratio = sum(sample_labels) / len(sample_labels)
    drift = abs(predicted_ratio - expected_ratio)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "total_samples": total,
        "predicted_phishing_ratio": round(predicted_ratio, 4),
        "expected_phishing_ratio": round(expected_ratio, 4),
        "drift": round(drift, 4),
        "drift_alert": drift > 0.15,
        "avg_latency_ms": round(avg_latency, 2),
    }


def log_metrics_to_mlflow(metrics: Dict):
    """Log monitoring metrics to a dedicated MLflow experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phishing-detector-monitoring")

    with mlflow.start_run():
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                flat[f"monitor_{key}"] = value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        flat[f"monitor_{key}_{k}"] = v
        if flat:
            mlflow.log_metrics(flat)


def run_all_checks(
    base_url: Optional[str] = None,
    skip_drift: bool = False,
) -> Dict:
    """Run all monitoring checks and return combined results."""
    results = {}

    # SageMaker endpoint health
    logger.info("Checking SageMaker endpoint health...")
    results["sagemaker"] = check_sagemaker_endpoint_health()
    status = "HEALTHY" if results["sagemaker"].get("healthy") else "UNHEALTHY"
    logger.info("SageMaker endpoint: {} (status={})", status, results["sagemaker"].get("status"))

    # API Gateway / HTTP endpoint health
    logger.info("Checking API endpoint health...")
    results["endpoint"] = check_endpoint_health(base_url)
    status = "HEALTHY" if results["endpoint"].get("healthy") else "UNHEALTHY"
    logger.info("API endpoint: {}", status)

    # Smoke tests
    if results["endpoint"].get("healthy"):
        logger.info("Running smoke tests...")
        results["smoke"] = run_smoke_test(base_url)
        status = "PASSED" if results["smoke"]["all_passed"] else "FAILED"
        logger.info("Smoke tests: {}", status)

        for test_name in ["phishing_test", "safe_test"]:
            t = results["smoke"].get(test_name, {})
            logger.info(
                "  {}: label={}, confidence={:.4f}, correct={}",
                test_name,
                t.get("label", "?"),
                t.get("confidence", 0),
                t.get("correct", False),
            )

        # Drift detection
        if not skip_drift:
            logger.info("Checking prediction drift (this may take a moment)...")
            results["drift"] = check_prediction_drift(base_url)
            drift_status = "ALERT" if results["drift"].get("drift_alert") else "OK"
            logger.info(
                "Drift: {} (predicted={:.1%}, expected={:.1%}, drift={:.1%})",
                drift_status,
                results["drift"]["predicted_phishing_ratio"],
                results["drift"]["expected_phishing_ratio"],
                results["drift"]["drift"],
            )

    # Overall health
    results["overall_healthy"] = (
        results.get("sagemaker", {}).get("healthy", False)
        and results.get("endpoint", {}).get("healthy", False)
        and results.get("smoke", {}).get("all_passed", False)
        and not results.get("drift", {}).get("drift_alert", False)
    )

    return results


def monitor_loop(
    base_url: Optional[str] = None,
    interval: Optional[int] = None,
    once: bool = False,
):
    """Run monitoring checks in a loop."""
    interval = interval or MONITOR_INTERVAL_SECONDS

    while True:
        logger.info("=" * 60)
        logger.info("Running monitoring checks...")
        logger.info("=" * 60)

        results = run_all_checks(base_url)

        overall = "HEALTHY" if results["overall_healthy"] else "UNHEALTHY"
        logger.info("=" * 60)
        logger.info("Overall status: {}", overall)
        logger.info("=" * 60)

        if once:
            return results

        logger.info("Next check in {} seconds...", interval)
        time.sleep(interval)


if __name__ == "__main__":
    monitor_loop(once=True)
