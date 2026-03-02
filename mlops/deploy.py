"""
AWS deployment automation.

Downloads the Production model from MLflow registry, builds a Docker image,
pushes to ECR, and updates the ECS service with a rolling deployment.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import boto3
import mlflow
import mlflow.pytorch
from loguru import logger
from transformers import DistilBertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from mlops.config_mlops import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_MODEL_NAME,
    AWS_REGION,
    ECR_REGISTRY,
    ECR_REPO,
    ECS_CLUSTER,
    ECS_SERVICE,
    ECS_TASK_FAMILY,
    CONTAINER_NAME,
    ECS_TASK_CPU,
    ECS_TASK_MEMORY,
    ECS_EXECUTION_ROLE_ARN,
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

    # Download tokenizer + model files
    local_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="model_files"
    )

    # Copy to the expected model directory for Docker build
    model_dir = config.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from downloaded artifacts to model dir
    import shutil
    # Clear existing model files
    for item in model_dir.iterdir():
        if item.is_file():
            item.unlink()
    # Copy new files
    src = Path(local_path)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, model_dir / item.name)

    logger.info("Production model saved to {}", model_dir)
    return str(model_dir)


def build_and_push_image(tag: str) -> str:
    """
    Build Docker image for linux/amd64 and push to ECR.

    Returns the full image URI.
    """
    if not ECR_REGISTRY:
        raise RuntimeError("ECR_REGISTRY env var not set")

    image_uri = f"{ECR_REGISTRY}/{ECR_REPO}:{tag}"
    project_root = Path(__file__).parent.parent

    # ECR login
    logger.info("Logging in to ECR...")
    login_cmd = (
        f"aws ecr get-login-password --region {AWS_REGION} | "
        f"docker login --username AWS --password-stdin {ECR_REGISTRY}"
    )
    _run_cmd(login_cmd, shell=True)

    # Build for amd64 (Fargate)
    logger.info("Building Docker image: {}", image_uri)
    _run_cmd(
        f"docker buildx build --platform linux/amd64 -t {image_uri} {project_root}",
        shell=True,
    )

    # Push
    logger.info("Pushing image to ECR...")
    _run_cmd(f"docker push {image_uri}", shell=True)

    logger.info("Image pushed: {}", image_uri)
    return image_uri


def update_ecs_service(image_uri: str) -> str:
    """
    Register a new ECS task definition and update the service.

    Returns the new task definition ARN.
    """
    ecs = boto3.client("ecs", region_name=AWS_REGION)

    # Register new task definition
    logger.info("Registering new task definition...")
    response = ecs.register_task_definition(
        family=ECS_TASK_FAMILY,
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu=ECS_TASK_CPU,
        memory=ECS_TASK_MEMORY,
        executionRoleArn=ECS_EXECUTION_ROLE_ARN,
        containerDefinitions=[
            {
                "name": CONTAINER_NAME,
                "image": image_uri,
                "portMappings": [
                    {"containerPort": 8000, "protocol": "tcp"}
                ],
                "essential": True,
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": f"/ecs/{ECS_TASK_FAMILY}",
                        "awslogs-region": AWS_REGION,
                        "awslogs-stream-prefix": "ecs",
                    },
                },
            }
        ],
    )

    task_def_arn = response["taskDefinition"]["taskDefinitionArn"]
    logger.info("New task definition: {}", task_def_arn)

    # Update service
    logger.info("Updating ECS service...")
    ecs.update_service(
        cluster=ECS_CLUSTER,
        service=ECS_SERVICE,
        taskDefinition=task_def_arn,
        forceNewDeployment=True,
    )

    # Wait for deployment to stabilize
    logger.info("Waiting for deployment to stabilize...")
    waiter = ecs.get_waiter("services_stable")
    try:
        waiter.wait(
            cluster=ECS_CLUSTER,
            services=[ECS_SERVICE],
            WaiterConfig={"Delay": 15, "MaxAttempts": 40},
        )
        logger.info("Deployment stabilized successfully")
    except Exception as e:
        logger.error("Deployment did not stabilize: {}", e)
        raise

    return task_def_arn


def rollback(previous_task_def_arn: str):
    """Roll back ECS service to a previous task definition."""
    ecs = boto3.client("ecs", region_name=AWS_REGION)

    logger.warning("Rolling back to: {}", previous_task_def_arn)
    ecs.update_service(
        cluster=ECS_CLUSTER,
        service=ECS_SERVICE,
        taskDefinition=previous_task_def_arn,
        forceNewDeployment=True,
    )

    logger.info("Waiting for rollback to stabilize...")
    waiter = ecs.get_waiter("services_stable")
    waiter.wait(
        cluster=ECS_CLUSTER,
        services=[ECS_SERVICE],
        WaiterConfig={"Delay": 15, "MaxAttempts": 40},
    )
    logger.info("Rollback complete")


def get_current_task_def() -> Optional[str]:
    """Get the current task definition ARN for the ECS service."""
    ecs = boto3.client("ecs", region_name=AWS_REGION)
    try:
        response = ecs.describe_services(
            cluster=ECS_CLUSTER, services=[ECS_SERVICE]
        )
        if response["services"]:
            return response["services"][0]["taskDefinition"]
    except Exception:
        pass
    return None


def deploy(tag: str):
    """
    Full deployment pipeline:
    1. Download Production model from MLflow
    2. Build and push Docker image
    3. Update ECS service
    """
    # Save current task def for rollback
    previous_task_def = get_current_task_def()
    if previous_task_def:
        logger.info("Current task definition (for rollback): {}", previous_task_def)

    # Step 1: Download production model
    download_production_model()

    # Step 2: Build and push
    image_uri = build_and_push_image(tag)

    # Step 3: Update ECS
    try:
        new_task_def = update_ecs_service(image_uri)
        logger.info("Deployment successful! Task definition: {}", new_task_def)
    except Exception as e:
        logger.error("Deployment failed: {}", e)
        if previous_task_def:
            logger.info("Attempting rollback...")
            rollback(previous_task_def)
        raise

    return image_uri


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

    parser = argparse.ArgumentParser(description="Deploy to AWS ECS")
    parser.add_argument("--tag", required=True, help="Docker image tag")
    args = parser.parse_args()

    deploy(args.tag)
