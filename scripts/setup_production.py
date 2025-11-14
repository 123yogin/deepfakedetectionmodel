"""
Production setup script - validates and prepares the system.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.model_validator import print_validation_report, validate_all_models
from backend.utils.model_health import print_model_health, check_model_health


def setup_production():
    """Run production setup checks."""
    print("=" * 60)
    print("Production Setup - Deepfake Detection System")
    print("=" * 60)
    
    # Step 1: Validate weights
    print("\n[STEP 1] Validating model weights...")
    validation_results = validate_all_models()
    
    # Step 2: Check model health
    print("\n[STEP 2] Checking model health...")
    health_status = check_model_health()
    
    # Step 3: Summary
    print("\n[STEP 3] Production Readiness Summary")
    print("=" * 60)
    
    models_with_weights = sum(1 for r in validation_results.values() if r["valid"])
    models_healthy = sum(1 for h in health_status.values() if h.get("status") == "healthy")
    
    print(f"\nModel Weights: {models_with_weights}/3 loaded")
    print(f"Model Health: {models_healthy}/4 healthy")
    
    # Recommendations
    print("\n[RECOMMENDATIONS]:")
    
    if models_with_weights == 0:
        print("  [CRITICAL] No model weights found!")
        print("  -> Run: python scripts/download_weights.py")
        print("  -> Or manually add weights to models/weights/")
    elif models_with_weights < 3:
        print(f"  [WARNING] Only {models_with_weights}/3 models have weights")
        print("  -> Add remaining weights for full accuracy")
    else:
        print("  [OK] All models have weights - system ready!")
    
    if models_healthy < 4:
        print(f"  [WARNING] {4 - models_healthy} models have health issues")
        print("  -> Check error messages above")
    else:
        print("  [OK] All models are healthy!")
    
    # Final status
    print("\n" + "=" * 60)
    if models_with_weights >= 2 and models_healthy == 4:
        print("[STATUS] [OK] PRODUCTION READY")
        print("System is ready for production use!")
    elif models_with_weights >= 1:
        print("[STATUS] [WARNING] PARTIALLY READY")
        print("System works but accuracy can be improved with more weights.")
    else:
        print("[STATUS] [ERROR] NEEDS SETUP")
        print("Add model weights for production use.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    setup_production()

