"""
Model configuration for detector weights and paths.
"""
import os
from pathlib import Path

# Base directory for model weights
MODELS_DIR = Path("models")
WEIGHTS_DIR = MODELS_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Model weight file paths
CNN_MODEL_PATH = os.environ.get(
    "CNN_MODEL_PATH",
    str(WEIGHTS_DIR / "xception_deepfake.pth")
)

TEMPORAL_MODEL_PATH = os.environ.get(
    "TEMPORAL_MODEL_PATH",
    str(WEIGHTS_DIR / "temporal_3dcnn.pth")
)

LIPSYNC_MODEL_PATH = os.environ.get(
    "LIPSYNC_MODEL_PATH",
    str(WEIGHTS_DIR / "syncnet.pth")
)

# Model settings
USE_PRETRAINED_IF_AVAILABLE = True  # Try to use pretrained if weights not found
FAIL_IF_NO_WEIGHTS = False  # If True, raise error if weights missing (set True for production)

# Model validation
VALIDATE_MODELS_ON_STARTUP = True

# Logging configuration
DEBUG_MODE = os.environ.get("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Performance optimization settings
ENABLE_MIXED_PRECISION = os.environ.get("ENABLE_MIXED_PRECISION", "True").lower() == "true"
ENABLE_TORCHSCRIPT = os.environ.get("ENABLE_TORCHSCRIPT", "False").lower() == "true"
ENABLE_QUANTIZATION = os.environ.get("ENABLE_QUANTIZATION", "False").lower() == "true"
BATCH_SIZE_CNN = int(os.environ.get("BATCH_SIZE_CNN", "8"))
BATCH_SIZE_TEMPORAL = int(os.environ.get("BATCH_SIZE_TEMPORAL", "4"))

