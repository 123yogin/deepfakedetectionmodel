"""
Diagnostic script to check if models are giving random predictions or actual predictions.
Tests model behavior with and without trained weights.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xception_detector import XceptionDeepfakeDetector
from models.temporal_detector import TemporalDetector
from models.lipsync_detector import LipSyncDetector
from backend.config.model_config import CNN_MODEL_PATH, TEMPORAL_MODEL_PATH, LIPSYNC_MODEL_PATH
from backend.utils.model_validator import validate_all_models


def create_test_image(seed: int = 42, size: tuple = (224, 224)) -> Image.Image:
    """Create a deterministic test image."""
    np.random.seed(seed)
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def create_test_frames(num_frames: int = 16, seed: int = 42) -> list:
    """Create a sequence of test frames."""
    frames = []
    for i in range(num_frames):
        # Slight variation between frames
        np.random.seed(seed + i)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frames.append(Image.fromarray(img_array))
    return frames


def test_prediction_consistency(detector, test_input, num_runs: int = 10):
    """
    Test if predictions are consistent (deterministic) or random.
    
    Returns:
        Tuple of (is_consistent, predictions_list, std_dev)
    """
    predictions = []
    
    for _ in range(num_runs):
        if isinstance(test_input, str):  # File path
            pred = detector.predict(test_input)
        elif isinstance(test_input, list):  # List of frames
            pred = detector.predict_clip(test_input)
        else:
            pred = 0.5
        predictions.append(pred)
    
    predictions = np.array(predictions)
    std_dev = np.std(predictions)
    mean_pred = np.mean(predictions)
    
    # If std_dev is very small (< 0.001), predictions are consistent
    is_consistent = std_dev < 0.001
    
    return is_consistent, predictions.tolist(), std_dev, mean_pred


def test_prediction_variance(detector, num_tests: int = 20):
    """
    Test prediction variance across different inputs.
    High variance suggests model is responding to input features.
    Low variance suggests random or constant predictions.
    """
    predictions = []
    
    for i in range(num_tests):
        # Create different test images
        test_img = create_test_image(seed=i)
        
        # Save temporarily
        temp_path = Path("temp_test_img.jpg")
        test_img.save(temp_path)
        
        try:
            if hasattr(detector, 'predict'):
                pred = detector.predict(str(temp_path))
            else:
                pred = 0.5
            predictions.append(pred)
        except:
            predictions.append(0.5)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    predictions = np.array(predictions)
    variance = np.var(predictions)
    mean_pred = np.mean(predictions)
    std_dev = np.std(predictions)
    
    return {
        "variance": variance,
        "mean": mean_pred,
        "std_dev": std_dev,
        "min": float(np.min(predictions)),
        "max": float(np.max(predictions)),
        "range": float(np.max(predictions) - np.min(predictions))
    }


def analyze_model_status(model_name: str, detector, weight_path: str):
    """Analyze a single model's status."""
    print(f"\n{'='*70}")
    print(f"Analyzing {model_name.upper()} Model")
    print(f"{'='*70}")
    
    # Check if weights exist
    weights_exist = os.path.exists(weight_path) if weight_path else False
    model_loaded = getattr(detector, 'model_loaded', False)
    
    print(f"\n[WEIGHTS STATUS]")
    print(f"  Weight file path: {weight_path}")
    print(f"  Weight file exists: {'YES [OK]' if weights_exist else 'NO [MISSING]'}")
    print(f"  Model loaded flag: {'YES [OK]' if model_loaded else 'NO [NOT LOADED]'}")
    
    if weights_exist:
        # Validate weights
        from backend.utils.model_validator import ModelValidator
        is_valid, msg = ModelValidator.validate_weight_file(weight_path)
        print(f"  Weight file valid: {'YES [OK]' if is_valid else 'NO [INVALID]'}")
        print(f"  Validation message: {msg}")
    
    # Test prediction consistency
    print(f"\n[PREDICTION CONSISTENCY TEST]")
    print(f"  Testing if same input gives same output (deterministic)...")
    
    if model_name.lower() == "cnn":
        # Create test image
        test_img = create_test_image()
        temp_path = Path("temp_test_consistency.jpg")
        test_img.save(temp_path)
        
        is_consistent, preds, std_dev, mean_pred = test_prediction_consistency(
            detector, str(temp_path), num_runs=10
        )
        temp_path.unlink()
        
    elif model_name.lower() == "temporal":
        test_frames = create_test_frames(num_frames=16)
        is_consistent, preds, std_dev, mean_pred = test_prediction_consistency(
            detector, test_frames, num_runs=10
        )
    else:  # lipsync
        # Skip consistency test for lipsync (requires audio)
        is_consistent, preds, std_dev, mean_pred = True, [0.5], 0.0, 0.5
    
    print(f"  Deterministic: {'YES [OK]' if is_consistent else 'NO [RANDOM!]'}")
    print(f"  Mean prediction: {mean_pred:.4f}")
    print(f"  Std deviation: {std_dev:.6f}")
    if not is_consistent:
        print(f"  [WARNING] Predictions vary for same input - model may be non-deterministic!")
    
    # Test prediction variance across different inputs
    print(f"\n[PREDICTION VARIANCE TEST]")
    print(f"  Testing variance across different inputs...")
    
    if model_name.lower() == "lipsync":
        print(f"  Skipped (requires audio files)")
        variance_stats = {"variance": 0.0, "mean": 0.5, "std_dev": 0.0, 
                         "min": 0.5, "max": 0.5, "range": 0.0}
    else:
        variance_stats = test_prediction_variance(detector, num_tests=20)
    
    print(f"  Mean prediction: {variance_stats['mean']:.4f}")
    print(f"  Std deviation: {variance_stats['std_dev']:.4f}")
    print(f"  Prediction range: [{variance_stats['min']:.4f}, {variance_stats['max']:.4f}]")
    print(f"  Range span: {variance_stats['range']:.4f}")
    
    # Analyze if predictions are random
    print(f"\n[ANALYSIS]")
    
    if not model_loaded:
        print(f"  [WARNING] MODEL IS USING UNTRAINED WEIGHTS (Random Initialization)")
        print(f"     - Predictions are based on random weights + heuristics")
        print(f"     - Not reliable for actual deepfake detection")
        
        # Check if predictions are in the compressed range (0.3-0.7 or 0.35-0.65)
        if 0.3 <= variance_stats['mean'] <= 0.7:
            print(f"  [WARNING] Predictions are in compressed range (0.3-0.7)")
            print(f"     This indicates untrained model with fallback heuristics")
        
        if variance_stats['range'] < 0.2:
            print(f"  [WARNING] Low prediction variance ({variance_stats['range']:.4f})")
            print(f"     Predictions are clustered - likely random/untrained")
        elif variance_stats['range'] > 0.3:
            print(f"  [OK] Good prediction variance ({variance_stats['range']:.4f})")
            print(f"     Model is responding to input features (even if untrained)")
    else:
        print(f"  [OK] MODEL HAS TRAINED WEIGHTS")
        print(f"     - Predictions should be based on learned features")
        
        if variance_stats['range'] < 0.1:
            print(f"  [WARNING] Very low variance - model may not be learning properly")
        elif variance_stats['range'] > 0.5:
            print(f"  [OK] Good variance - model is responding to inputs")
    
    # Recommendations
    print(f"\n[RECOMMENDATIONS]")
    if not model_loaded:
        print(f"  1. Download trained weights:")
        print(f"     Run: python scripts/download_weights.py")
        print(f"  2. Or manually add weights to: {weight_path}")
        print(f"  3. Restart the application after adding weights")
    else:
        print(f"  [OK] Model has trained weights - predictions should be accurate")
        if variance_stats['range'] < 0.2:
            print(f"  [WARNING] Consider retraining or checking model architecture")
    
    return {
        "model_loaded": model_loaded,
        "weights_exist": weights_exist,
        "is_consistent": is_consistent,
        "variance_stats": variance_stats,
        "mean_prediction": variance_stats['mean']
    }


def main():
    """Main diagnostic function."""
    print("\n" + "="*70)
    print("MODEL PREDICTION DIAGNOSTIC TOOL")
    print("="*70)
    print("\nThis tool checks if your models are giving random or actual predictions.")
    print("It tests prediction consistency, variance, and weight loading status.\n")
    
    # Validate all models first
    print("\n[STEP 1] Validating Model Weights")
    print("-" * 70)
    validation_results = validate_all_models()
    
    for model_name, result in validation_results.items():
        status = "[VALID]" if result["valid"] else "[MISSING/INVALID]"
        print(f"  {model_name.upper()}: {status}")
        print(f"    {result['message']}")
    
    # Initialize detectors
    print("\n[STEP 2] Initializing Models")
    print("-" * 70)
    
    try:
        cnn_detector = XceptionDeepfakeDetector()
        print("  [OK] CNN (Xception) detector initialized")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize CNN detector: {e}")
        cnn_detector = None
    
    try:
        temporal_detector = TemporalDetector()
        print("  [OK] Temporal detector initialized")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize Temporal detector: {e}")
        temporal_detector = None
    
    try:
        lipsync_detector = LipSyncDetector()
        print("  [OK] LipSync detector initialized")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize LipSync detector: {e}")
        lipsync_detector = None
    
    # Analyze each model
    results = {}
    
    if cnn_detector:
        results['cnn'] = analyze_model_status(
            "CNN", cnn_detector, CNN_MODEL_PATH
        )
    
    if temporal_detector:
        results['temporal'] = analyze_model_status(
            "Temporal", temporal_detector, TEMPORAL_MODEL_PATH
        )
    
    if lipsync_detector:
        results['lipsync'] = analyze_model_status(
            "LipSync", lipsync_detector, LIPSYNC_MODEL_PATH
        )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    models_with_weights = sum(1 for r in results.values() if r.get('model_loaded', False))
    total_models = len(results)
    
    print(f"\nModels with trained weights: {models_with_weights}/{total_models}")
    
    if models_with_weights == 0:
        print("\n[WARNING] ALL MODELS ARE USING UNTRAINED WEIGHTS!")
        print("   Your predictions are based on random initialization + heuristics.")
        print("   They are NOT reliable for actual deepfake detection.")
        print("\n   ACTION REQUIRED:")
        print("   1. Download trained weights: python scripts/download_weights.py")
        print("   2. Or train your own models using the training scripts")
        print("   3. Restart the application after adding weights")
    elif models_with_weights < total_models:
        print(f"\n[WARNING] {total_models - models_with_weights} model(s) are missing trained weights.")
        print("   Consider adding weights for better accuracy.")
    else:
        print("\n[OK] All models have trained weights!")
        print("   Your predictions should be based on learned features.")
    
    # Check if predictions are random
    print("\n[PREDICTION QUALITY ASSESSMENT]")
    for model_name, result in results.items():
        variance = result['variance_stats']['range']
        model_loaded = result.get('model_loaded', False)
        
        if not model_loaded:
            if variance < 0.15:
                quality = "LOW (Random/Untrained)"
            elif variance < 0.3:
                quality = "MEDIUM (Heuristics + Untrained Model)"
            else:
                quality = "MEDIUM-HIGH (Good Feature Extraction)"
        else:
            if variance < 0.2:
                quality = "MEDIUM (May need retraining)"
            else:
                quality = "HIGH (Good Learned Features)"
        
        print(f"  {model_name.upper()}: {quality}")
    
    print("\n" + "="*70)
    print("Diagnostic complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

