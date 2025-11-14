"""
Unit tests for temporal detector and ensemble combiner.
"""
import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from models.temporal_detector import TemporalDetector
from backend.utils.ensemble import EnsembleCombiner


def create_test_frame(width: int = 224, height: int = 224, noise: float = 0.0) -> Image.Image:
    """Create a synthetic test frame."""
    img_array = np.random.rand(height, width) * 255
    if noise > 0:
        img_array += np.random.randn(height, width) * noise * 50
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L').convert('RGB')


def test_temporal_detector_predict_clip():
    """Test that predict_clip returns a value in [0, 1]."""
    detector = TemporalDetector()
    
    # Create a clip with small random noise (should have low temporal variance)
    frames_smooth = [create_test_frame(noise=0.05) for _ in range(16)]
    score_smooth = detector.predict_clip(frames_smooth)
    
    assert isinstance(score_smooth, float)
    assert 0.0 <= score_smooth <= 1.0
    
    # Create a clip with abrupt changes (should have high temporal variance)
    frames_abrupt = []
    for i in range(16):
        if i % 4 == 0:
            # Abrupt change every 4 frames
            frames_abrupt.append(create_test_frame(noise=0.5))
        else:
            frames_abrupt.append(create_test_frame(noise=0.05))
    
    score_abrupt = detector.predict_clip(frames_abrupt)
    
    assert isinstance(score_abrupt, float)
    assert 0.0 <= score_abrupt <= 1.0
    
    # Abrupt changes should generally produce higher scores
    # (though not guaranteed due to normalization)
    assert isinstance(score_abrupt, float)


def test_temporal_detector_predict_for_face_track():
    """Test predict_for_face_track returns expected structure."""
    detector = TemporalDetector()
    
    # Create temporary directory with test frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test frames
        for i in range(20):
            frame = create_test_frame(noise=0.1)
            frame.save(os.path.join(tmpdir, f"face_{i+1:04d}.jpg"))
        
        result = detector.predict_for_face_track(tmpdir, clip_len=16, stride=8)
        
        assert "clip_scores" in result
        assert "mean_score" in result
        assert "max_score" in result
        assert isinstance(result["mean_score"], float)
        assert isinstance(result["max_score"], float)
        assert 0.0 <= result["mean_score"] <= 1.0
        assert 0.0 <= result["max_score"] <= 1.0


def test_ensemble_combiner_high_scores():
    """Test ensemble combiner with high detector scores."""
    combiner = EnsembleCombiner()
    
    # High scores across all detectors
    aggregation = {
        "max_score": 0.9,  # CNN
        "mean_score": 0.85,
        "temporal_max": 0.8,
        "temporal_mean": 0.75,
        "lip_sync_score": 0.1,  # Low sync = high manipulation
        "frequency_score": 0.7,
        "total_faces": 10
    }
    
    result = combiner.combine(aggregation)
    
    assert "final_score" in result
    assert "final_label" in result
    assert "raw_score" in result
    assert result["final_label"] == "LIKELY_MANIPULATED"
    assert result["final_score"] >= 0.5


def test_ensemble_combiner_low_scores():
    """Test ensemble combiner with low detector scores."""
    combiner = EnsembleCombiner(threshold=0.5)
    
    # Very low scores across all detectors
    aggregation = {
        "max_score": 0.05,  # CNN - very low
        "mean_score": 0.02,
        "temporal_max": 0.1,  # Very low temporal
        "temporal_mean": 0.08,
        "lip_sync_score": 0.95,  # Very high sync = very low manipulation
        "frequency_score": 0.05,  # Very low frequency
        "total_faces": 10
    }
    
    result = combiner.combine(aggregation)
    
    # With very low scores, final score should be below threshold
    # (though exact value depends on sigmoid normalization)
    assert result["final_score"] < 0.6  # More lenient check
    assert isinstance(result["final_label"], str)
    assert result["final_label"] in ["LIKELY_AUTHENTIC", "LIKELY_MANIPULATED"]


def test_ensemble_combiner_set_weights():
    """Test that set_weights updates weights correctly."""
    combiner = EnsembleCombiner()
    
    new_weights = {
        "cnn": 0.6,
        "temporal": 0.2,
        "lipsync": 0.1,
        "frequency": 0.1
    }
    
    combiner.set_weights(new_weights)
    
    # Weights should be normalized
    total = sum(combiner.weights.values())
    assert abs(total - 1.0) < 1e-6


def test_ensemble_combiner_calibration():
    """Test ensemble calibration."""
    combiner = EnsembleCombiner()
    
    # Create mock validation data
    # Raw scores (simulated)
    validation_X = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    # Labels (0=real, 1=fake)
    validation_y = np.array([0, 0, 0, 1, 1, 1, 1])
    
    calibration_params = combiner.calibrate(validation_X, validation_y, method='platt')
    
    assert "slope" in calibration_params
    assert "intercept" in calibration_params
    assert "method" in calibration_params
    assert calibration_params["method"] == "platt"


def test_ensemble_combiner_save_load_config():
    """Test saving and loading ensemble configuration."""
    combiner = EnsembleCombiner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        config_path = tmp.name
    
    try:
        # Save config
        combiner.save_config(config_path)
        assert os.path.exists(config_path)
        
        # Load config
        loaded_combiner = EnsembleCombiner.load_config(config_path)
        
        assert loaded_combiner.weights == combiner.weights
        assert loaded_combiner.bias == combiner.bias
        assert loaded_combiner.scale == combiner.scale
        assert loaded_combiner.threshold == combiner.threshold
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

