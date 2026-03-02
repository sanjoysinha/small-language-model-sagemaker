"""
MLOps pipeline CLI orchestrator.

Usage:
    python -m mlops.pipeline train           # MLflow-tracked training + register to Staging
    python -m mlops.pipeline evaluate        # Evaluation gate: Staging vs Production
    python -m mlops.pipeline deploy --tag v1 # Build/push/deploy Production model
    python -m mlops.pipeline promote --version 3  # Manually promote a version to Production
    python -m mlops.pipeline monitor [--once]     # Health + smoke tests
    python -m mlops.pipeline status               # Show registry state + ECS status
"""

import argparse
import sys
from pathlib import Path

import mlflow
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.config_mlops import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_MODEL_NAME,
    API_BASE_URL,
)


def cmd_train(args):
    """Run MLflow-tracked training."""
    from mlops.train_tracked import train_with_mlflow

    logger.info("Starting MLflow-tracked training...")
    run_id = train_with_mlflow()
    logger.info("Training complete. Run ID: {}", run_id)


def cmd_evaluate(args):
    """Run evaluation gate."""
    from mlops.evaluate_gate import compare_and_gate, evaluate_local

    if args.model_dir:
        logger.info("Evaluating local model: {}", args.model_dir)
        evaluate_local(args.model_dir)
    else:
        logger.info("Running evaluation gate (Staging vs Production)...")
        passed = compare_and_gate()
        if passed:
            logger.info("Evaluation gate PASSED")
        else:
            logger.error("Evaluation gate FAILED")
            sys.exit(1)


def cmd_deploy(args):
    """Deploy to AWS ECS."""
    from mlops.deploy import deploy

    logger.info("Starting deployment with tag '{}'...", args.tag)
    image_uri = deploy(args.tag)
    logger.info("Deployment complete: {}", image_uri)


def cmd_promote(args):
    """Manually promote a model version to champion (production)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    version = args.version
    logger.info("Promoting version {} to champion...", version)

    # Set champion alias to the specified version
    client.set_registered_model_alias(
        name=MLFLOW_REGISTRY_MODEL_NAME,
        alias="champion",
        version=version,
    )
    logger.info("Version {} is now 'champion' (production)", version)

    # Remove challenger alias if it pointed to this version
    try:
        client.delete_registered_model_alias(
            name=MLFLOW_REGISTRY_MODEL_NAME,
            alias="challenger",
        )
    except mlflow.exceptions.MlflowException:
        pass


def cmd_monitor(args):
    """Run monitoring checks."""
    from mlops.monitor import monitor_loop

    base_url = args.url or API_BASE_URL
    logger.info("Monitoring endpoint: {}", base_url)
    results = monitor_loop(base_url=base_url, once=args.once)

    if args.once and results:
        if not results.get("overall_healthy"):
            sys.exit(1)


def cmd_status(args):
    """Show MLflow registry state and ECS service status."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    # MLflow registry status
    logger.info("=" * 60)
    logger.info("MLflow Model Registry: {}", MLFLOW_REGISTRY_MODEL_NAME)
    logger.info("=" * 60)

    try:
        versions = client.search_model_versions(
            f"name='{MLFLOW_REGISTRY_MODEL_NAME}'"
        )
        if not versions:
            logger.info("No model versions registered yet")
        else:
            # Check which versions have aliases
            alias_map = {}
            for alias in ["champion", "challenger"]:
                try:
                    mv = client.get_model_version_by_alias(
                        name=MLFLOW_REGISTRY_MODEL_NAME, alias=alias
                    )
                    alias_map[mv.version] = alias
                except mlflow.exceptions.MlflowException:
                    pass

            for v in sorted(versions, key=lambda x: int(x.version)):
                alias_label = alias_map.get(v.version, "-")
                logger.info(
                    "  Version {} | Alias: {:>12} | Run: {} | Created: {}",
                    v.version,
                    alias_label,
                    v.run_id[:8],
                    v.creation_timestamp,
                )
    except mlflow.exceptions.MlflowException:
        logger.info("Model '{}' not found in registry", MLFLOW_REGISTRY_MODEL_NAME)

    # ECS status
    logger.info("")
    logger.info("=" * 60)
    logger.info("ECS Service Status")
    logger.info("=" * 60)

    try:
        from mlops.monitor import check_ecs_health
        ecs_status = check_ecs_health()
        if ecs_status.get("status") == "ERROR" and "ClusterNotFoundException" in ecs_status.get("message", ""):
            logger.info("  No ECS cluster deployed (AWS resources not provisioned)")
        elif ecs_status.get("healthy"):
            logger.info("  Status: HEALTHY (running={}/{})",
                         ecs_status.get("running_count", "?"),
                         ecs_status.get("desired_count", "?"))
            logger.info("  Task definition: {}", ecs_status.get("task_definition", "?"))
        else:
            for key, value in ecs_status.items():
                logger.info("  {}: {}", key, value)
    except Exception as e:
        logger.info("  ECS check unavailable: {}", e)


def main():
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline for Phishing Email Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mlops.pipeline train              # Train with MLflow tracking
  python -m mlops.pipeline evaluate           # Run evaluation gate
  python -m mlops.pipeline deploy --tag v1    # Deploy to ECS
  python -m mlops.pipeline promote --version 3  # Promote model version
  python -m mlops.pipeline monitor --once     # Run one health check
  python -m mlops.pipeline status             # Show registry + ECS status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # train
    subparsers.add_parser("train", help="MLflow-tracked training")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation gate")
    eval_parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Evaluate a local model directory instead of MLflow registry",
    )

    # deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to AWS ECS")
    deploy_parser.add_argument("--tag", required=True, help="Docker image tag")

    # promote
    promote_parser = subparsers.add_parser("promote", help="Promote model version")
    promote_parser.add_argument("--version", required=True, help="Model version to promote")

    # monitor
    monitor_parser = subparsers.add_parser("monitor", help="Run monitoring checks")
    monitor_parser.add_argument("--once", action="store_true", help="Run once and exit")
    monitor_parser.add_argument("--url", type=str, default=None, help="API base URL")

    # status
    subparsers.add_parser("status", help="Show registry + ECS status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "deploy": cmd_deploy,
        "promote": cmd_promote,
        "monitor": cmd_monitor,
        "status": cmd_status,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
