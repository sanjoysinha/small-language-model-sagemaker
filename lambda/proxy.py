"""
Lambda proxy function for API Gateway → SageMaker endpoint.

Deployed as part of the API Gateway stack (sagemaker/api_gateway_template.yaml).
Maps incoming HTTP requests to SageMaker invoke_endpoint calls.

Environment variables:
  SAGEMAKER_ENDPOINT_NAME - Name of the SageMaker Real-Time Endpoint
"""

import json
import os

import boto3

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]

sm_runtime = boto3.client("sagemaker-runtime")
sm = boto3.client("sagemaker")


def lambda_handler(event, context):
    path = event.get("rawPath", event.get("path", "/"))
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    body = event.get("body", "{}")

    # Health check routes
    if method == "GET" and path in ("/health", "/ping"):
        return _health_response()

    # Inference routes
    if method == "POST" and path in ("/predict", "/invocations"):
        return _invoke_endpoint(body, batch=False)

    if method == "POST" and path == "/predict/batch":
        return _invoke_endpoint(body, batch=True)

    return {
        "statusCode": 404,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": f"Route not found: {method} {path}"}),
    }


def _health_response():
    """Check SageMaker endpoint status and return appropriate HTTP response."""
    try:
        response = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response["EndpointStatus"]
        is_healthy = status == "InService"
        return {
            "statusCode": 200 if is_healthy else 503,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "status": "healthy" if is_healthy else status,
                "is_ready": is_healthy,
                "endpoint_name": ENDPOINT_NAME,
            }),
        }
    except Exception as e:
        return {
            "statusCode": 503,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "error", "message": str(e)}),
        }


def _invoke_endpoint(body: str, batch: bool):
    """Forward request body to SageMaker endpoint and return response."""
    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=body if isinstance(body, bytes) else body.encode("utf-8"),
        )
        result = response["Body"].read().decode("utf-8")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": result,
        }
    except sm_runtime.exceptions.ModelError as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Model error: {str(e)}"}),
        }
    except Exception as e:
        return {
            "statusCode": 502,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Invocation failed: {str(e)}"}),
        }
