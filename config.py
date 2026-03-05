from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent

# SageMaker injects model artifacts to /opt/ml/model at runtime.
# Fall back to local path for development.
_SM_MODEL_DIR = Path("/opt/ml/model")
MODEL_DIR = _SM_MODEL_DIR if _SM_MODEL_DIR.exists() else PROJECT_ROOT / "models" / "distilbert-phishing"

ONNX_DIR = PROJECT_ROOT / "models" / "onnx"
LOG_DIR = PROJECT_ROOT / "logs"

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
DATASET_NAME = "zefang-liu/phishing-email-dataset"
TEXT_COLUMN = "Email Text"
LABEL_COLUMN = "Email Type"
LABEL_MAP = {"Safe Email": 0, "Phishing Email": 1}

# ──────────────────────────────────────────────
# Tokenizer / Model
# ──────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 256  # 256 captures phishing signals; 4x faster than 512

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch size = 16 * 2 = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
MAX_BATCH_SIZE = 64
