"""
Unit tests for aggregation and verdict logic.
"""
import pytest
from pathlib import Path
from backend.utils.aggregation import aggregate_scores, decide_verdict, generate_result
from backend.config.detection_config import FACE_THRESHOLD, HIGH_THRESHOLD


def test_aggregate_scores_high_max():
    """Test aggregation with high max score (should trigger LIKELY_MANIPULATED)."""
    detections = [
        {"face_file": "face_0001.jpg", "frame": 1, "fake_score": 0.95},
        {"face_file": "face_0002.jpg", "frame": 2, "fake_score": 0.02},
        {"face_file": "face_0003.jpg", "frame": 3, "fake_score": 0.1}
    ]
    
    aggregation = aggregate_scores(detections)
    
    assert aggregation["total_faces"] == 3
    assert aggregation["max_score"] == 0.95
    assert aggregation["mean_score"] > 0.0
    assert aggregation["count_above_0.5"] == 1  # Only the 0.95 score
    
    verdict = decide_verdict(aggregation)
    assert verdict["label"] == "LIKELY_MANIPULATED"
    assert "high_max_score" in verdict["reason"]
    assert verdict["confidence"] > 0.5


def test_aggregate_scores_low_scores():
    """Test aggregation with all low scores (should trigger LIKELY_AUTHENTIC)."""
    detections = [
        {"face_file": "face_0001.jpg", "frame": 1, "fake_score": 0.05},
        {"face_file": "face_0002.jpg", "frame": 2, "fake_score": 0.08},
        {"face_file": "face_0003.jpg", "frame": 3, "fake_score": 0.10},
        {"face_file": "face_0004.jpg", "frame": 4, "fake_score": 0.12}
    ]
    
    aggregation = aggregate_scores(detections)
    
    assert aggregation["total_faces"] == 4
    assert aggregation["max_score"] < 0.25
    assert aggregation["mean_score"] < 0.15
    assert aggregation["count_above_0.5"] == 0
    
    verdict = decide_verdict(aggregation)
    assert verdict["label"] == "LIKELY_AUTHENTIC"
    assert "very_low_scores" in verdict["reason"]


def test_aggregate_scores_insufficient_faces():
    """Test aggregation with insufficient faces (should trigger INCONCLUSIVE)."""
    detections = [
        {"face_file": "face_0001.jpg", "frame": 1, "fake_score": 0.5}
    ]
    
    aggregation = aggregate_scores(detections)
    
    assert aggregation["total_faces"] == 1
    
    verdict = decide_verdict(aggregation)
    assert verdict["label"] == "INCONCLUSIVE"
    assert "insufficient_faces" in verdict["reason"]


def test_aggregate_scores_multiple_suspicious():
    """Test aggregation with multiple suspicious faces."""
    detections = [
        {"face_file": "face_0001.jpg", "frame": 1, "fake_score": 0.6},
        {"face_file": "face_0002.jpg", "frame": 2, "fake_score": 0.55},
        {"face_file": "face_0003.jpg", "frame": 3, "fake_score": 0.75},
        {"face_file": "face_0004.jpg", "frame": 4, "fake_score": 0.12}
    ]
    
    aggregation = aggregate_scores(detections)
    
    assert aggregation["count_above_0.5"] >= 2
    assert aggregation["p90_score"] >= 0.7
    
    verdict = decide_verdict(aggregation)
    # Should be LIKELY_MANIPULATED due to multiple suspicious faces
    assert verdict["label"] == "LIKELY_MANIPULATED"
    assert "multiple_suspicious_faces" in verdict["reason"]


def test_generate_result():
    """Test complete result generation."""
    job_id = "test-agg-001"
    video_path = "storage/uploads/test.mp4"
    frames = 10
    
    # Mix of suspicious and benign
    detections = [
        {"face_file": "face_0001.jpg", "frame": 1, "fake_score": 0.95},
        {"face_file": "face_0002.jpg", "frame": 2, "fake_score": 0.02},
        {"face_file": "face_0003.jpg", "frame": 3, "fake_score": 0.1},
        {"face_file": "face_0004.jpg", "frame": 4, "fake_score": 0.6},
        {"face_file": "face_0005.jpg", "frame": 5, "fake_score": 0.55},
        {"face_file": "face_0006.jpg", "frame": 6, "fake_score": 0.12}
    ]
    
    result = generate_result(job_id, video_path, frames, detections)
    
    # Verify structure
    assert result["job_id"] == job_id
    assert result["video_path"] == video_path
    assert result["frames"] == frames
    assert "aggregation" in result
    assert "verdict" in result
    assert "detections" in result
    assert "report_meta" in result
    
    # Verify verdict
    assert result["verdict"]["label"] == "LIKELY_MANIPULATED"
    assert "high_max_score" in result["verdict"]["reason"]
    assert result["verdict"]["confidence"] > 0.0
    
    # Verify file was created
    result_file = Path("results") / f"{job_id}.json"
    assert result_file.exists()
    
    # Clean up test file
    if result_file.exists():
        result_file.unlink()


def test_empty_detections():
    """Test aggregation with empty detections."""
    detections = []
    
    aggregation = aggregate_scores(detections)
    
    assert aggregation["total_faces"] == 0
    assert aggregation["max_score"] == 0.0
    assert aggregation["mean_score"] == 0.0
    
    verdict = decide_verdict(aggregation)
    assert verdict["label"] == "INCONCLUSIVE"
    assert "insufficient_faces" in verdict["reason"]

