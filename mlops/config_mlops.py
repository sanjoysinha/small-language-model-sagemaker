import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

# ──────────────────────────────────────────────
# MLflow
# ──────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "phishing-detector")
MLFLOW_REGISTRY_MODEL_NAME = os.getenv("MLFLOW_REGISTRY_MODEL_NAME", "phishing-distilbert")

# ──────────────────────────────────────────────
# S3 artifact store
# ──────────────────────────────────────────────
S3_ARTIFACT_BUCKET = os.getenv("S3_ARTIFACT_BUCKET", "phishing-mlops-artifacts")
S3_ARTIFACT_PREFIX = os.getenv("S3_ARTIFACT_PREFIX", "mlflow-artifacts")

# ──────────────────────────────────────────────
# AWS - General
# ──────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ──────────────────────────────────────────────
# AWS - ECR (Docker image registry)
# ──────────────────────────────────────────────
ECR_REGISTRY = os.getenv("ECR_REGISTRY", "")  # e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com
ECR_REPO = os.getenv("ECR_REPO", "phishing-detector")

# ──────────────────────────────────────────────
# AWS - SageMaker
# ──────────────────────────────────────────────
SAGEMAKER_EXECUTION_ROLE_ARN = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN", "")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "phishing-detector")
SAGEMAKER_MODEL_NAME = os.getenv("SAGEMAKER_MODEL_NAME", "phishing-distilbert")
SAGEMAKER_INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")

# ──────────────────────────────────────────────
# Evaluation gate thresholds
# ──────────────────────────────────────────────
MIN_F1 = float(os.getenv("MIN_F1", "0.95"))
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.95"))
MIN_PRECISION = float(os.getenv("MIN_PRECISION", "0.93"))
MIN_RECALL = float(os.getenv("MIN_RECALL", "0.93"))

# ──────────────────────────────────────────────
# Monitoring
# ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
MONITOR_INTERVAL_SECONDS = int(os.getenv("MONITOR_INTERVAL_SECONDS", "300"))
