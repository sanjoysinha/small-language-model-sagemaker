"""
AWS SageMaker deployment automation.

Downloads the Production model from MLflow registry, packages it as model.tar.gz,
uploads to S3, builds a Docker image, pushes to ECR, and deploys a SageMaker
Real-Time Endpoint.
"""

import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Optional

import boto3
import mlflow
import mlflow.pytorch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from mlops.config_mlops import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_MODEL_NAME,
    AWS_REGION,
    ECR_REGISTRY,
    ECR_REPO,
    S3_ARTIFACT_BUCKET,
    SAGEMAKER_EXECUTION_ROLE_ARN,
    SAGEMAKER_ENDPOINT_NAME,
    SAGEMAKER_MODEL_NAME,
    SAGEMAKER_INSTANCE_TYPE,
)


def download_production_model() -> str:
    """
    Download the Production model from MLflow registry to local disk.

    Returns the local model directory path.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    try:
        mv = client.get_model_version_by_alias(
            name=MLFLOW_REGISTRY_MODEL_NAME, alias="champion"
        )
    except mlflow.exceptions.MlflowException:
        raise RuntimeError("No 'champion' (production) model found in MLflow registry")

    logger.info("Downloading champion model: version={}, run_id={}", mv.version, mv.run_id)

    local_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="model_files"
    )

    model_dir = config.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    for item in model_dir.iterdir():
        if item.is_file():
            item.unlink()
    src = Path(local_path)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, model_dir / item.name)

    logger.info("Production model saved to {}", model_dir)
    return str(model_dir)


def package_model_for_sagemaker(model_dir: str) -> str:
    """
    Package model files into model.tar.gz for SageMaker.

    SageMaker extracts this archive to /opt/ml/model inside the container.
    Returns the path to the created archive.
    """
    model_dir = Path(model_dir)
    tar_path = model_dir.parent / "model.tar.gz"

    logger.info("Packaging model artifacts from {} -> {}", model_dir, tar_path)
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in model_dir.iterdir():
            if f.is_file():
                tar.add(f, arcname=f.name)

    logger.info("Model packaged: {} ({:.1f} MB)", tar_path, tar_path.stat().st_size / 1e6)
    return str(tar_path)


def upload_model_to_s3(local_tar_path: str, tag: str) -> str:
    """
    Upload model.tar.gz to S3.

    Returns the S3 URI (s3://bucket/key).
    """
    s3_key = f"sagemaker/models/{tag}/model.tar.gz"
    s3 = boto3.client("s3", region_name=AWS_REGION)

    logger.info("Uploading model to s3://{}/{}", S3_ARTIFACT_BUCKET, s3_key)
    s3.upload_file(local_tar_path, S3_ARTIFACT_BUCKET, s3_key)

    s3_uri = f"s3://{S3_ARTIFACT_BUCKET}/{s3_key}"
    logger.info("Model uploaded: {}", s3_uri)
    return s3_uri


def build_and_push_image(tag: str) -> str:
    """
    Build Docker image for linux/amd64 and push to ECR.

    Returns the full image URI.
    """
    if not ECR_REGISTRY:
        raise RuntimeError("ECR_REGISTRY env var not set")

    image_uri = f"{ECR_REGISTRY}/{ECR_REPO}:{tag}"
    project_root = Path(__file__).parent.parent

    logger.info("Logging in to ECR...")
    login_cmd = (
        f"aws ecr get-login-password --region {AWS_REGION} | "
        f"docker login --username AWS --password-stdin {ECR_REGISTRY}"
    )
    _run_cmd(login_cmd, shell=True)

    logger.info("Building Docker image: {}", image_uri)
    _run_cmd(
        f"docker buildx build --platform linux/amd64 -t {image_uri} {project_root}",
        shell=True,
    )

    logger.info("Pushing image to ECR...")
    _run_cmd(f"docker push {image_uri}", shell=True)

    logger.info("Image pushed: {}", image_uri)
    return image_uri


def create_sagemaker_model(
    model_name: str,
    image_uri: str,
    model_data_url: str,
) -> str:
    """
    Create a SageMaker Model resource.

    Deletes any existing model with the same name first.
    Returns the model ARN.
    """
    if not SAGEMAKER_EXECUTION_ROLE_ARN:
        raise RuntimeError("SAGEMAKER_EXECUTION_ROLE_ARN env var not set")

    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    # Delete existing model with same name (idempotent)
    try:
        sm.delete_model(ModelName=model_name)
        logger.info("Deleted existing SageMaker model: {}", model_name)
    except sm.exceptions.ClientError:
        pass

    logger.info("Creating SageMaker model: {}", model_name)
    response = sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
        },
        ExecutionRoleArn=SAGEMAKER_EXECUTION_ROLE_ARN,
    )

    model_arn = response["ModelArn"]
    logger.info("SageMaker model created: {}", model_arn)
    return model_arn


def deploy_sagemaker_endpoint(
    model_name: str,
    endpoint_name: str,
    instance_type: str,
) -> str:
    """
    Create or update a SageMaker Real-Time Endpoint.

    Creates a new EndpointConfig, then creates or updates the endpoint.
    Waits for the endpoint to reach InService status.
    Returns the endpoint name.
    """
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"

    logger.info("Creating EndpointConfig: {}", endpoint_config_name)
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
            }
        ],
    )

    # Check if endpoint already exists → update, else create
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        logger.info("Updating existing endpoint: {}", endpoint_name)
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
    except sm.exceptions.ClientError:
        logger.info("Creating new endpoint: {}", endpoint_name)
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )

    logger.info("Waiting for endpoint to reach InService status...")
    waiter = sm.get_waiter("endpoint_in_service")
    try:
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 40},
        )
        logger.info("Endpoint is InService: {}", endpoint_name)
    except Exception as e:
        logger.error("Endpoint did not reach InService: {}", e)
        raise

    return endpoint_name


def rollback(previous_endpoint_config: Optional[str]):
    """Roll back SageMaker endpoint to a previous EndpointConfig."""
    if not previous_endpoint_config:
        logger.warning("No previous EndpointConfig available for rollback")
        return

    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    logger.warning("Rolling back to EndpointConfig: {}", previous_endpoint_config)
    sm.update_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        EndpointConfigName=previous_endpoint_config,
    )

    logger.info("Waiting for rollback to stabilize...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},
    )
    logger.info("Rollback complete")


def get_current_endpoint_config() -> Optional[str]:
    """Get the current EndpointConfig name for the SageMaker endpoint."""
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        response = sm.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
        return response.get("EndpointConfigName")
    except Exception:
        return None


def deploy(tag: str) -> str:
    """
    Full SageMaker deployment pipeline:
    1. Download Production model from MLflow
    2. Package model as model.tar.gz
    3. Upload model artifact to S3
    4. Build and push Docker image to ECR
    5. Create SageMaker Model
    6. Create/update SageMaker Endpoint

    Returns the endpoint name.
    """
    previous_endpoint_config = get_current_endpoint_config()
    if previous_endpoint_config:
        logger.info("Current EndpointConfig (for rollback): {}", previous_endpoint_config)

    # Step 1: Download production model
    model_dir = download_production_model()

    # Step 2: Package model
    tar_path = package_model_for_sagemaker(model_dir)

    # Step 3: Upload to S3
    model_data_url = upload_model_to_s3(tar_path, tag)

    # Step 4: Build and push Docker image
    image_uri = build_and_push_image(tag)

    # Step 5: Create SageMaker Model
    sagemaker_model_name = f"{SAGEMAKER_MODEL_NAME}-{tag}"
    create_sagemaker_model(sagemaker_model_name, image_uri, model_data_url)

    # Step 6: Deploy endpoint
    try:
        deploy_sagemaker_endpoint(sagemaker_model_name, SAGEMAKER_ENDPOINT_NAME, SAGEMAKER_INSTANCE_TYPE)
        logger.info("Deployment successful! Endpoint: {}", SAGEMAKER_ENDPOINT_NAME)
    except Exception as e:
        logger.error("Deployment failed: {}", e)
        if previous_endpoint_config:
            logger.info("Attempting rollback...")
            rollback(previous_endpoint_config)
        raise

    return SAGEMAKER_ENDPOINT_NAME


def _run_cmd(cmd: str, shell: bool = False):
    """Run a shell command and raise on failure."""
    logger.debug("Running: {}", cmd)
    result = subprocess.run(
        cmd if shell else cmd.split(),
        shell=shell,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Command failed:\n{}", result.stderr)
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy to AWS SageMaker")
    parser.add_argument("--tag", required=True, help="Docker image tag / deployment version")
    args = parser.parse_args()

    deploy(args.tag)
