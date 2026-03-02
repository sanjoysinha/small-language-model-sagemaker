import torch
from torch.optim import AdamW
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from loguru import logger
from tqdm import tqdm

import config
from data_loader import load_and_preprocess, create_dataloaders


def evaluate(model, dataloader, device, verbose=False):
    """Run model on dataloader, return metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.tolist())

            if device.type == "mps":
                torch.mps.empty_cache()

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "recall": recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "f1": f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
    }

    if verbose:
        logger.info(
            "\n{}",
            classification_report(
                all_labels, all_preds,
                target_names=["Safe Email", "Phishing Email"],
                zero_division=0,
            ),
        )
        logger.info("Confusion Matrix:\n{}", confusion_matrix(all_labels, all_preds))

    return metrics


def train():
    # ── Device setup ──
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: {}", device)

    # ── Load tokenizer and model ──
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=2,
        id2label={0: "Safe Email", 1: "Phishing Email"},
        label2id=config.LABEL_MAP,
    )
    model.to(device)

    # ── Prepare data ──
    texts, labels = load_and_preprocess()
    train_loader, val_loader, test_loader = create_dataloaders(
        texts, labels, tokenizer
    )

    # ── Optimizer and scheduler ──
    accum_steps = config.GRADIENT_ACCUMULATION_STEPS
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    # Scheduler counts optimizer steps, not batch steps
    total_optimizer_steps = (len(train_loader) // accum_steps) * config.NUM_EPOCHS
    warmup_steps = int(total_optimizer_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    logger.info(
        "Training config: batch_size={}, accum_steps={}, effective_batch={}",
        config.BATCH_SIZE, accum_steps, config.BATCH_SIZE * accum_steps,
    )

    # ── Training loop ──
    best_val_f1 = 0.0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}",
            unit="batch",
        )

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch,
            )

            # Scale loss by accumulation steps
            loss = outputs.loss / accum_steps
            loss.backward()

            total_loss += outputs.loss.item()  # log unscaled loss

            # Optimizer step every accum_steps batches
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Free MPS memory cache periodically
            if device.type == "mps" and (step + 1) % accum_steps == 0:
                torch.mps.empty_cache()

            progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # ── Validation ──
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch {}/{} | Loss: {:.4f} | Val Acc: {:.4f} | "
            "Val P: {:.4f} | Val R: {:.4f} | Val F1: {:.4f}",
            epoch + 1,
            config.NUM_EPOCHS,
            avg_loss,
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
        )

        # ── Save best model ──
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(config.MODEL_DIR)
            tokenizer.save_pretrained(config.MODEL_DIR)
            logger.info("Saved best model (F1={:.4f}) to {}", best_val_f1, config.MODEL_DIR)

    # ── Final test evaluation ──
    logger.info("=" * 50)
    logger.info("Test Set Evaluation (best model)")
    logger.info("=" * 50)

    best_model = DistilBertForSequenceClassification.from_pretrained(config.MODEL_DIR)
    best_model.to(device)
    test_metrics = evaluate(best_model, test_loader, device, verbose=True)

    logger.info("Test Metrics: {}", test_metrics)
    logger.info("Training complete!")


if __name__ == "__main__":
    train()
