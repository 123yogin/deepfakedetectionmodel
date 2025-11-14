"""
Script to validate all models and check system health.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.model_validator import print_validation_report
from backend.utils.model_health import print_model_health


if __name__ == "__main__":
    print("=" * 60)
    print("Deepfake Detection System - Model Validation")
    print("=" * 60)
    
    # Validate weights
    print_validation_report()
    
    # Check model health
    print_model_health()
    
    print("\n[INFO] Validation complete!")
    print("[INFO] If weights are missing, run: python scripts/download_weights.py")

