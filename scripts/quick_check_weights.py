"""
Quick script to check if model weights have been added.
Run this after downloading weights to verify they're loaded correctly.
"""
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config.model_config import CNN_MODEL_PATH, TEMPORAL_MODEL_PATH, LIPSYNC_MODEL_PATH
from backend.utils.model_validator import validate_all_models


def quick_check():
    """Quick check of weight files."""
    print("\n" + "="*70)
    print("QUICK WEIGHT CHECK")
    print("="*70)
    
    weights = {
        "CNN": CNN_MODEL_PATH,
        "Temporal": TEMPORAL_MODEL_PATH,
        "LipSync": LIPSYNC_MODEL_PATH
    }
    
    print("\n[FILE EXISTENCE CHECK]")
    print("-" * 70)
    all_exist = True
    for name, path in weights.items():
        exists = os.path.exists(path)
        status = "[FOUND]" if exists else "[MISSING]"
        print(f"  {name:12} {status:10} {path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n[OK] All weight files exist!")
    else:
        print("\n[WARNING] Some weight files are missing.")
        print("  Download weights and place them in models/weights/")
    
    print("\n[VALIDATION CHECK]")
    print("-" * 70)
    validation_results = validate_all_models()
    
    all_valid = True
    for model_name, result in validation_results.items():
        status = "[VALID]" if result["valid"] else "[INVALID/MISSING]"
        print(f"  {model_name.upper():12} {status:20}")
        if not result["valid"]:
            all_valid = False
            print(f"    {result['message']}")
    
    if all_valid:
        print("\n[SUCCESS] All weights are valid and ready to use!")
        print("\nNext steps:")
        print("  1. Restart your application")
        print("  2. Run: python scripts/diagnose_model_predictions.py")
        print("  3. Verify predictions are no longer random")
    else:
        print("\n[ACTION NEEDED] Some weights need to be downloaded.")
        print("\nSee WEIGHTS_SETUP.md for download instructions.")
    
    print("\n" + "="*70)
    return all_valid


if __name__ == "__main__":
    quick_check()

