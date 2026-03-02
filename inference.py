import time

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from loguru import logger

import config
from data_loader import clean_text


class PhishingDetector:
    """
    PyTorch-powered phishing email detector.

    Loads the model once at init and provides predict/predict_batch methods.
    Uses dynamic padding for optimal latency.
    """

    def __init__(self, model_dir=None):
        model_dir = model_dir or config.MODEL_DIR
        logger.info("Loading model from {}", model_dir)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
        self.model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.labels = {0: "safe", 1: "phishing"}

        self._warmup()
        logger.info("PhishingDetector initialized and warmed up")

    def _warmup(self):
        """Run a dummy prediction to warm up the model."""
        dummy = self.tokenizer(
            "warmup text",
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )
        with torch.no_grad():
            self.model(**dummy)

    def predict(self, email_text: str) -> dict:
        """
        Classify a single email.

        Returns:
            {
                "label": "phishing" | "safe",
                "confidence": float (0-1),
                "phishing_probability": float (0-1),
                "latency_ms": float,
            }
        """
        start = time.perf_counter()

        cleaned = clean_text(email_text)
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQ_LENGTH,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.numpy()[0]

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        phishing_prob = float(probs[1])
        label_id = 1 if phishing_prob >= config.CONFIDENCE_THRESHOLD else 0

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "label": self.labels[label_id],
            "confidence": float(max(probs)),
            "phishing_probability": phishing_prob,
            "latency_ms": round(elapsed_ms, 2),
        }

    def predict_batch(self, email_texts: list) -> list:
        """Classify a batch of emails."""
        start = time.perf_counter()

        cleaned = [clean_text(t) for t in email_texts]
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQ_LENGTH,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.numpy()

        # Batch softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_email_ms = elapsed_ms / len(email_texts)

        results = []
        for i, prob in enumerate(probs):
            phishing_prob = float(prob[1])
            label_id = 1 if phishing_prob >= config.CONFIDENCE_THRESHOLD else 0
            results.append({
                "label": self.labels[label_id],
                "confidence": float(max(prob)),
                "phishing_probability": phishing_prob,
                "latency_ms": round(per_email_ms, 2),
            })
        return results
