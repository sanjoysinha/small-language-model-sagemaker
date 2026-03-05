import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

import config
from inference import PhishingDetector


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────

class EmailRequest(BaseModel):
    email_text: str = Field(
        ...,
        min_length=1,
        max_length=500_000,
        description="Raw email body text to classify",
    )


class EmailResponse(BaseModel):
    label: str
    confidence: float
    phishing_probability: float
    latency_ms: float


class BatchEmailRequest(BaseModel):
    emails: List[EmailRequest] = Field(
        ...,
        min_length=1,
        max_length=config.MAX_BATCH_SIZE,
    )


class BatchEmailResponse(BaseModel):
    predictions: List[EmailResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    is_ready: bool
    onnx_path: str


class PingResponse(BaseModel):
    status: str


# ──────────────────────────────────────────────
# Application Lifecycle
# ──────────────────────────────────────────────

detector: Optional[PhishingDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global detector
    logger.info("Starting up: loading phishing detection model...")
    detector = PhishingDetector()
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down")
    detector = None


app = FastAPI(
    title="Phishing Email Detection API",
    description="On-premise DistilBERT-based phishing email classifier",
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/ping", response_model=PingResponse)
async def ping():
    """SageMaker health check endpoint. Returns 200 when model is ready."""
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return PingResponse(status="healthy")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are operational."""
    return HealthResponse(
        status="healthy" if detector else "not_ready",
        is_ready=detector is not None,
        onnx_path=str(config.ONNX_DIR),
    )


@app.post("/predict", response_model=EmailResponse)
async def predict_email(request: EmailRequest):
    """Classify a single email as phishing or safe."""
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = detector.predict(request.email_text)
        return EmailResponse(**result)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchEmailResponse)
async def predict_batch(request: BatchEmailRequest):
    """Classify a batch of emails (max 64)."""
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start = time.perf_counter()
        texts = [item.email_text for item in request.emails]
        results = detector.predict_batch(texts)
        total_ms = (time.perf_counter() - start) * 1000

        return BatchEmailResponse(
            predictions=[EmailResponse(**r) for r in results],
            total_latency_ms=round(total_ms, 2),
        )
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction error: {str(e)}"
        )


@app.post("/invocations", response_model=EmailResponse)
async def invocations(request: EmailRequest):
    """SageMaker inference endpoint. Accepts same payload as /predict."""
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = detector.predict(request.email_text)
        return EmailResponse(**result)
    except Exception as e:
        logger.exception("SageMaker invocation failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
