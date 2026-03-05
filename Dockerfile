FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py data_loader.py inference.py app.py ./

# SageMaker injects model artifacts to /opt/ml/model at runtime.
# For local testing, mount your model directory to /opt/ml/model:
#   docker run -v ./models/distilbert-phishing:/opt/ml/model -p 8080:8080 <image>

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
