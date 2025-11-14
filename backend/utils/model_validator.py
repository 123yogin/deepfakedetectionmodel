"""
Model validation utilities for checking model health and compatibility.
"""
import torch
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
import io


class ModelValidator:
    """Validates model weights and architectures."""
    
    @staticmethod
    def validate_weight_file(weight_path: str) -> Tuple[bool, str]:
        """
        Validate a weight file.
        
        Args:
            weight_path: Path to weight file
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not os.path.exists(weight_path):
            return False, f"Weight file not found: {weight_path}"
        
        try:
            # Try to load the file
            state_dict = torch.load(weight_path, map_location='cpu')
            
            # Check if it's a valid state dict
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # Check if it has keys (model parameters)
                if len(state_dict) == 0:
                    return False, "Weight file is empty"
                
                # Check file size (should be reasonable)
                file_size = os.path.getsize(weight_path) / (1024 * 1024)  # MB
                if file_size < 0.1:  # Less than 100KB is suspicious
                    return False, f"Weight file too small ({file_size:.2f} MB) - may be corrupted"
                
                return True, f"Weight file valid ({file_size:.2f} MB, {len(state_dict)} parameters)"
            else:
                return False, "Weight file format not recognized"
                
        except Exception as e:
            return False, f"Error loading weight file: {str(e)}"
    
    @staticmethod
    def validate_model_architecture(model, weight_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate model architecture compatibility with weights.
        
        Args:
            model: PyTorch model
            weight_path: Optional path to weights
            
        Returns:
            Tuple of (is_compatible, message)
        """
        if weight_path and os.path.exists(weight_path):
            try:
                state_dict = torch.load(weight_path, map_location='cpu')
                
                # Handle different formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # Try to load with strict=False (allows partial loading)
                try:
                    model.load_state_dict(state_dict, strict=False)
                    return True, "Model architecture compatible with weights"
                except Exception as e:
                    return False, f"Architecture mismatch: {str(e)}"
            except Exception as e:
                return False, f"Error validating compatibility: {str(e)}"
        else:
            return True, "No weights provided - architecture is valid"
    
    @staticmethod
    def test_model_inference(model, input_shape: Tuple) -> Tuple[bool, str]:
        """
        Test model inference with dummy input.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (e.g., (1, 3, 224, 224))
            
        Returns:
            Tuple of (success, message)
        """
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(input_shape)
                output = model(dummy_input)
                
                if output is not None:
                    return True, f"Inference test passed (output shape: {output.shape})"
                else:
                    return False, "Model returned None output"
        except Exception as e:
            return False, f"Inference test failed: {str(e)}"


def validate_all_models() -> Dict[str, Dict]:
    """
    Validate all models in the system.
    
    Returns:
        Dictionary with validation results for each model
    """
    from backend.config.model_config import CNN_MODEL_PATH, TEMPORAL_MODEL_PATH, LIPSYNC_MODEL_PATH
    
    results = {}
    
    # Validate CNN model
    cnn_valid, cnn_msg = ModelValidator.validate_weight_file(CNN_MODEL_PATH)
    results["cnn"] = {
        "valid": cnn_valid,
        "message": cnn_msg,
        "path": CNN_MODEL_PATH
    }
    
    # Validate Temporal model
    temporal_valid, temporal_msg = ModelValidator.validate_weight_file(TEMPORAL_MODEL_PATH)
    results["temporal"] = {
        "valid": temporal_valid,
        "message": temporal_msg,
        "path": TEMPORAL_MODEL_PATH
    }
    
    # Validate LipSync model
    lipsync_valid, lipsync_msg = ModelValidator.validate_weight_file(LIPSYNC_MODEL_PATH)
    results["lipsync"] = {
        "valid": lipsync_valid,
        "message": lipsync_msg,
        "path": LIPSYNC_MODEL_PATH
    }
    
    return results


def print_validation_report():
    """Print a validation report for all models."""
    print("\n" + "=" * 60)
    print("Model Validation Report")
    print("=" * 60)
    
    results = validate_all_models()
    
    for model_name, result in results.items():
        status = "[OK]" if result["valid"] else "[MISSING/INVALID]"
        print(f"\n{status} {model_name.upper()} Model")
        print(f"  Path: {result['path']}")
        print(f"  Status: {result['message']}")
    
    print("\n" + "=" * 60)
    
    # Summary
    all_valid = all(r["valid"] for r in results.values())
    if all_valid:
        print("\n[SUCCESS] All models have valid weights!")
    else:
        print("\n[WARNING] Some models are missing weights.")
        print("Run: python scripts/download_weights.py")
        print("Or manually add weights to models/weights/")
    
    print("=" * 60 + "\n")
    
    return results

