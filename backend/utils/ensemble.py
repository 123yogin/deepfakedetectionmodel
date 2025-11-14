"""
Ensemble combiner for fusing multiple detector scores.
"""
import numpy as np
from typing import Dict, Any, Optional


class EnsembleCombiner:
    """
    Combines scores from multiple detectors using weighted fusion.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize ensemble combiner.
        
        Args:
            threshold: Decision threshold for final label
        """
        self.threshold = threshold
        
        # Default weights for each detector
        # CNN (max_score): 50%
        # Temporal (max): 25%
        # LipSync (inverted): 15%
        # Frequency: 10%
        self.weights = {
            "cnn": 0.50,
            "temporal": 0.25,
            "lipsync": 0.15,
            "frequency": 0.10
        }
        
        # Calibration parameters (can be set via calibrate method)
        self.calibration_params = None
    
    def set_weights(self, cnn: float, temporal: float, lipsync: float, frequency: float):
        """
        Set custom weights for detectors.
        
        Args:
            cnn: Weight for CNN detector
            temporal: Weight for temporal detector
            lipsync: Weight for lip-sync detector
            frequency: Weight for frequency detector
        """
        total = cnn + temporal + lipsync + frequency
        if abs(total - 1.0) > 1e-6:
            # Normalize weights
            self.weights["cnn"] = cnn / total
            self.weights["temporal"] = temporal / total
            self.weights["lipsync"] = lipsync / total
            self.weights["frequency"] = frequency / total
        else:
            self.weights["cnn"] = cnn
            self.weights["temporal"] = temporal
            self.weights["lipsync"] = lipsync
            self.weights["frequency"] = frequency
    
    def combine(self, aggregation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine scores from all detectors.
        
        Args:
            aggregation: Aggregated scores dictionary
            
        Returns:
            Dictionary with final_score, final_label, and raw_score
        """
        # Extract scores
        cnn_score = aggregation.get("max_score", 0.5)
        temporal_score = aggregation.get("temporal_max", 0.5)
        lip_sync_score = aggregation.get("lip_sync_score", 0.5)
        frequency_score = aggregation.get("frequency_score", 0.5)
        
        # Handle lip-sync: lower sync score = higher manipulation probability
        # So we invert it: (1 - lip_sync_score)
        if lip_sync_score is None:
            lip_sync_manipulation = 0.5
        else:
            lip_sync_manipulation = 1.0 - lip_sync_score
        
        # Weighted combination
        raw_score = (
            self.weights["cnn"] * cnn_score +
            self.weights["temporal"] * temporal_score +
            self.weights["lipsync"] * lip_sync_manipulation +
            self.weights["frequency"] * frequency_score
        )
        
        # Apply calibration if available
        if self.calibration_params:
            raw_score = self._apply_calibration(raw_score)
        
        # Apply sigmoid normalization for final score
        # Less aggressive transformation to avoid false positives on real videos
        # Using: sigmoid((x - 0.5) * 3) instead of 4 for better calibration
        final_score = 1.0 / (1.0 + np.exp(-3.0 * (raw_score - 0.5)))
        
        # Determine label based on threshold
        # Use a slightly higher threshold (0.55) to reduce false positives
        if final_score >= 0.55:  # Changed from self.threshold (0.5) to 0.55
            final_label = "LIKELY_MANIPULATED"
        else:
            final_label = "LIKELY_AUTHENTIC"
        
        return {
            "final_score": float(final_score),
            "final_label": final_label,
            "raw_score": float(raw_score)
        }
    
    def _apply_calibration(self, raw_score: float) -> float:
        """
        Apply Platt scaling calibration.
        
        Args:
            raw_score: Raw combined score
            
        Returns:
            Calibrated score
        """
        if not self.calibration_params:
            return raw_score
        
        slope = self.calibration_params.get("slope", 1.0)
        intercept = self.calibration_params.get("intercept", 0.0)
        
        # Platt scaling: 1 / (1 + exp(-(slope * x + intercept)))
        calibrated = 1.0 / (1.0 + np.exp(-(slope * raw_score + intercept)))
        return float(calibrated)
    
    def calibrate(self, validation_X: np.ndarray, validation_y: np.ndarray, method: str = 'platt') -> Dict[str, Any]:
        """
        Calibrate ensemble using validation data.
        
        Args:
            validation_X: Array of raw scores
            validation_y: Array of labels (0=real, 1=fake)
            method: Calibration method ('platt' for Platt scaling)
            
        Returns:
            Calibration parameters
        """
        if method == 'platt':
            # Simple Platt scaling using logistic regression
            from sklearn.linear_model import LogisticRegression
            
            # Reshape if needed
            if validation_X.ndim == 1:
                validation_X = validation_X.reshape(-1, 1)
            
            # Fit logistic regression
            lr = LogisticRegression()
            lr.fit(validation_X, validation_y)
            
            # Extract parameters
            slope = lr.coef_[0][0]
            intercept = lr.intercept_[0]
            
            self.calibration_params = {
                "slope": float(slope),
                "intercept": float(intercept),
                "method": method
            }
            
            return self.calibration_params
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def save_config(self, filepath: str):
        """Save ensemble configuration to file."""
        import json
        
        config = {
            "weights": self.weights,
            "threshold": self.threshold,
            "calibration_params": self.calibration_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filepath: str):
        """Load ensemble configuration from file."""
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.weights = config.get("weights", self.weights)
        self.threshold = config.get("threshold", self.threshold)
        self.calibration_params = config.get("calibration_params", None)

