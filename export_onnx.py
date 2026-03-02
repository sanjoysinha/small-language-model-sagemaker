from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import DistilBertTokenizerFast
from loguru import logger

import config


def export_to_onnx():
    """Export fine-tuned DistilBERT to ONNX format for fast inference."""
    logger.info("Loading fine-tuned model from {}", config.MODEL_DIR)

    # Load the fine-tuned PyTorch model and export to ONNX
    model = ORTModelForSequenceClassification.from_pretrained(
        config.MODEL_DIR,
        export=True,
    )

    # Save ONNX model + tokenizer together
    config.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.ONNX_DIR)

    tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_DIR)
    tokenizer.save_pretrained(config.ONNX_DIR)

    logger.info("ONNX model exported to {}", config.ONNX_DIR)

    # Verify the exported model works
    verify_onnx_model(tokenizer)


def verify_onnx_model(tokenizer):
    """Quick sanity check on the exported model."""
    logger.info("Verifying ONNX model...")

    model = ORTModelForSequenceClassification.from_pretrained(config.ONNX_DIR)

    test_texts = [
        "Dear user, your account has been compromised. Click here to verify.",
        "Hi team, please find attached the Q3 budget report.",
    ]

    for text in test_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )
        outputs = model(**inputs)
        logits = outputs.logits.detach().numpy()[0]
        pred = "Phishing" if logits[1] > logits[0] else "Safe"
        logger.info("Text: '{}...' -> {} (logits: {})", text[:50], pred, logits)

    logger.info("ONNX verification complete!")


if __name__ == "__main__":
    export_to_onnx()
