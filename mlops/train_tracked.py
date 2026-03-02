"""
MLflow-instrumented training.

Wraps the existing train.py logic with experiment tracking:
  - Logs all hyperparameters
  - Logs per-epoch metrics (train_loss, val_accuracy, val_precision, val_recall, val_f1)
  - Logs final test metrics
  - Saves model artifact to MLflow
  - Registers model in MLflow Model Registry as "Staging"
"""

import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch.optim import AdamW
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from loguru import logger
from tqdm import tqdm

# Ensure project root is on path so we can import config, data_loader, train
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data_loader import load_and_preprocess, create_dataloaders
from train import evaluate
from mlops.config_mlops import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTRY_MODEL_NAME,
)


def train_with_mlflow():
    """Run full training with MLflow experiment tracking."""

    # ── MLflow setup ──
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: {}", run_id)

        # ── Log hyperparameters ──
        mlflow.log_params({
            "model_name": config.MODEL_NAME,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "batch_size": config.BATCH_SIZE,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": config.LEARNING_RATE,
            "num_epochs": config.NUM_EPOCHS,
            "weight_decay": config.WEIGHT_DECAY,
            "warmup_ratio": config.WARMUP_RATIO,
            "seed": config.SEED,
            "train_split": config.TRAIN_SPLIT,
            "val_split": config.VAL_SPLIT,
            "test_split": config.TEST_SPLIT,
        })

        # ── Device setup ──
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info("Using device: {}", device)
        mlflow.log_param("device", str(device))

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
        mlflow.log_param("train_samples", len(train_loader.dataset))
        mlflow.log_param("val_samples", len(val_loader.dataset))
        mlflow.log_param("test_samples", len(test_loader.dataset))

        # ── Optimizer and scheduler ──
        accum_steps = config.GRADIENT_ACCUMULATION_STEPS
        optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        total_optimizer_steps = (len(train_loader) // accum_steps) * config.NUM_EPOCHS
        warmup_steps = int(total_optimizer_steps * config.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
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

                loss = outputs.loss / accum_steps
                loss.backward()
                total_loss += outputs.loss.item()

                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if device.type == "mps" and (step + 1) % accum_steps == 0:
                    torch.mps.empty_cache()

                progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)

            # ── Validation ──
            val_metrics = evaluate(model, val_loader, device)

            # ── Log epoch metrics to MLflow ──
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }, step=epoch + 1)

            logger.info(
                "Epoch {}/{} | Loss: {:.4f} | Val Acc: {:.4f} | "
                "Val P: {:.4f} | Val R: {:.4f} | Val F1: {:.4f}",
                epoch + 1, config.NUM_EPOCHS, avg_loss,
                val_metrics["accuracy"], val_metrics["precision"],
                val_metrics["recall"], val_metrics["f1"],
            )

            # ── Save best model locally ──
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

        # Log final test metrics
        mlflow.log_metrics({
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        })

        logger.info("Test Metrics: {}", test_metrics)

        # ── Log model artifact to MLflow ──
        logger.info("Logging model artifact to MLflow...")
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            artifact_path="model",
            registered_model_name=MLFLOW_REGISTRY_MODEL_NAME,
        )

        # ── Also log tokenizer files as artifacts ──
        mlflow.log_artifacts(str(config.MODEL_DIR), artifact_path="model_files")

        # ── Set alias to "challenger" (staging equivalent in MLflow 3.x) ──
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(
            f"name='{MLFLOW_REGISTRY_MODEL_NAME}'"
        )
        if versions:
            latest_version = max(versions, key=lambda v: int(v.version))
            client.set_registered_model_alias(
                name=MLFLOW_REGISTRY_MODEL_NAME,
                alias="challenger",
                version=latest_version.version,
            )
            logger.info(
                "Model version {} aliased as 'challenger'",
                latest_version.version,
            )

        logger.info("Training complete! MLflow run ID: {}", run_id)
        return run_id


if __name__ == "__main__":
    train_with_mlflow()
