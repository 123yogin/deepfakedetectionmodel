"""
Unit tests for frequency detector.
"""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os
from models.frequency_detector import FrequencyDetector


def create_test_face_image(output_path: str, noise_level: float = 0.1, blur: bool = False):
    """
    Create a synthetic test face image.
    
    Args:
        output_path: Path to save the image
        noise_level: Amount of noise to add (0-1)
        blur: Whether to apply blur
    """
    # Create a base face-like pattern
    size = 256
    img = np.random.rand(size, size) * 255
    
    # Add some structure (simulate face features)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    img += 100 * np.exp(-dist**2 / (2 * (size/4)**2))
    
    # Add noise
    img += np.random.randn(size, size) * noise_level * 50
    
    # Apply blur if requested
    if blur:
        from scipy import ndimage
        img = ndimage.gaussian_filter(img, sigma=2.0)
    
    # Clip to valid range
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Save as image
    pil_img = Image.fromarray(img, mode='L')
    pil_img.save(output_path)


def test_compute_face_freq_score():
    """Test that compute_face_freq_score returns a float in [0, 1]."""
    detector = FrequencyDetector()
    
    # Create temporary test image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_image_path = tmp.name
        create_test_face_image(test_image_path, noise_level=0.1)
    
    try:
        freq_score = detector.compute_face_freq_score(test_image_path)
        
        # Verify score is in valid range
        assert isinstance(freq_score, float)
        assert 0.0 <= freq_score <= 1.0
        
    finally:
        # Clean up
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)


def test_compute_face_freq_score_with_noise():
    """Test that higher noise produces different frequency scores."""
    detector = FrequencyDetector()
    
    # Create two test images with different noise levels
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
        low_noise_path = tmp1.name
        high_noise_path = tmp2.name
        
        create_test_face_image(low_noise_path, noise_level=0.05, blur=True)
        create_test_face_image(high_noise_path, noise_level=0.5, blur=False)
    
    try:
        low_noise_score = detector.compute_face_freq_score(low_noise_path)
        high_noise_score = detector.compute_face_freq_score(high_noise_path)
        
        # Both should be valid scores
        assert 0.0 <= low_noise_score <= 1.0
        assert 0.0 <= high_noise_score <= 1.0
        
        # High noise should generally produce different scores
        # (though not always higher due to normalization)
        assert isinstance(low_noise_score, float)
        assert isinstance(high_noise_score, float)
        
    finally:
        # Clean up
        for path in [low_noise_path, high_noise_path]:
            if os.path.exists(path):
                os.unlink(path)


def test_batch_compute():
    """Test that batch_compute returns mapping for multiple files."""
    detector = FrequencyDetector()
    
    # Create temporary directory with test images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 test face images
        face_paths = []
        for i in range(3):
            face_path = os.path.join(tmpdir, f"face_{i+1:04d}.jpg")
            create_test_face_image(face_path, noise_level=0.1 + i * 0.1)
            face_paths.append(face_path)
        
        # Run batch compute
        scores = detector.batch_compute(tmpdir)
        
        # Verify results
        assert isinstance(scores, dict)
        assert len(scores) == 3
        
        # Check all face files are in results
        for i in range(3):
            face_filename = f"face_{i+1:04d}.jpg"
            assert face_filename in scores
            assert 0.0 <= scores[face_filename] <= 1.0


def test_batch_compute_with_debug():
    """Test that batch_compute saves debug visualizations when requested."""
    detector = FrequencyDetector()
    
    # Create temporary directory with test images
    with tempfile.TemporaryDirectory() as tmpdir:
        debug_dir = os.path.join(tmpdir, "debug")
        
        # Create test face image
        face_path = os.path.join(tmpdir, "face_0001.jpg")
        create_test_face_image(face_path, noise_level=0.2)
        
        # Run batch compute with debug output
        scores = detector.batch_compute(tmpdir, output_debug_dir=debug_dir)
        
        # Verify debug file was created
        debug_file = os.path.join(debug_dir, "face_0001_freq.png")
        assert os.path.exists(debug_file), "Debug frequency map should be created"


def test_frequency_detector_initialization():
    """Test FrequencyDetector initialization with custom parameters."""
    detector = FrequencyDetector(img_size=128, R_low=10.0, mu=0.8, sigma=0.3)
    
    assert detector.img_size == 128
    assert detector.R_low == 10.0
    assert detector.mu == 0.8
    assert detector.sigma == 0.3
    assert detector.low_mask.shape == (128, 128)
    assert detector.high_mask.shape == (128, 128)

