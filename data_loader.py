import re

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from loguru import logger

import config


class PhishingDataset(Dataset):
    """Wraps tokenized emails for PyTorch DataLoader."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip HTML artifacts."""
    text = re.sub(r"<[^>]+>", " ", text)   # strip HTML tags
    text = re.sub(r"\s+", " ", text)        # collapse whitespace
    return text.strip()


def load_and_preprocess():
    """Load dataset from HuggingFace, clean, and return texts + labels."""
    logger.info("Loading dataset: {}", config.DATASET_NAME)
    ds = load_dataset(config.DATASET_NAME, split="train")

    texts = []
    labels = []
    skipped = 0

    for row in ds:
        text = row[config.TEXT_COLUMN]
        label_str = row[config.LABEL_COLUMN]

        if not text or not text.strip() or label_str not in config.LABEL_MAP:
            skipped += 1
            continue

        texts.append(clean_text(text))
        labels.append(config.LABEL_MAP[label_str])

    # Log class distribution
    n_safe = labels.count(0)
    n_phishing = labels.count(1)
    logger.info(
        "Loaded {} samples, skipped {} invalid rows", len(texts), skipped
    )
    logger.info(
        "Class distribution: Safe={} ({:.1f}%), Phishing={} ({:.1f}%)",
        n_safe, 100 * n_safe / len(labels),
        n_phishing, 100 * n_phishing / len(labels),
    )
    return texts, labels


def tokenize_texts(texts: list, tokenizer: DistilBertTokenizerFast):
    """
    Tokenize with head+tail truncation in a single pass.

    For emails longer than MAX_SEQ_LENGTH tokens, keep the first
    HEAD_TOKENS and last TAIL_TOKENS. This preserves the greeting/subject
    at the top and the call-to-action/links at the bottom.
    """
    HEAD_TOKENS = 64
    # minus 2 for [CLS] and [SEP]
    TAIL_TOKENS = config.MAX_SEQ_LENGTH - HEAD_TOKENS - 2

    all_input_ids = []
    all_attention_mask = []
    head_tail_count = 0

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= config.MAX_SEQ_LENGTH - 2:
            # Fits — use as-is with padding
            combined = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
        else:
            # Head+tail strategy
            head_tail_count += 1
            combined = (
                [tokenizer.cls_token_id]
                + tokens[:HEAD_TOKENS]
                + tokens[-TAIL_TOKENS:]
                + [tokenizer.sep_token_id]
            )

        pad_len = config.MAX_SEQ_LENGTH - len(combined)
        input_ids = combined + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(combined) + [0] * pad_len

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

    logger.info(
        "Head+tail truncation applied to {} / {} emails ({:.1f}%)",
        head_tail_count, len(texts), 100 * head_tail_count / len(texts),
    )

    return {
        "input_ids": torch.tensor(all_input_ids),
        "attention_mask": torch.tensor(all_attention_mask),
    }


def create_dataloaders(texts, labels, tokenizer, batch_size=None, seed=None):
    """Split into train/val/test and return DataLoaders."""
    batch_size = batch_size or config.BATCH_SIZE
    seed = seed or config.SEED

    encodings = tokenize_texts(texts, tokenizer)
    dataset = PhishingDataset(encodings, labels)

    total = len(dataset)
    train_size = int(total * config.TRAIN_SPLIT)
    val_size = int(total * config.VAL_SPLIT)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        "Splits: train={}, val={}, test={}", train_size, val_size, test_size
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick verification that data pipeline works
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)
    texts, labels = load_and_preprocess()
    train_loader, val_loader, test_loader = create_dataloaders(
        texts, labels, tokenizer
    )
    # Check first batch
    batch = next(iter(train_loader))
    logger.info("First batch keys: {}", list(batch.keys()))
    logger.info("input_ids shape: {}", batch["input_ids"].shape)
    logger.info("labels shape: {}", batch["labels"].shape)
