"""
Aggregation and verdict logic for deepfake detection results.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.config.detection_config import (
    FACE_THRESHOLD,
    HIGH_THRESHOLD,
    MIN_FACES_FOR_DECISION,
    SUSPICIOUS_COUNT_NEEDED,
    P90_THRESHOLD
)


def aggregate_scores(
    detections: List[Dict[str, Any]],
    lip_sync_score: Optional[float] = None,
    temporal_mean: float = 0.5,
    temporal_max: float = 0.5
) -> Dict[str, Any]:
    """
    Aggregate detection scores from all detectors.
    
    Args:
        detections: List of detection dictionaries with 'fake_score' and optionally 'freq_score'
        lip_sync_score: Lip-sync score (lower = more suspicious)
        temporal_mean: Mean temporal consistency score
        temporal_max: Max temporal consistency score
        
    Returns:
        Dictionary with aggregated statistics
    """
    total_faces = len(detections)
    
    if total_faces == 0:
        return {
            "total_faces": 0,
            "max_score": 0.0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "p90_score": 0.0,
            "count_above_0.5": 0,
            "frequency_score": 0.5,
            "lip_sync_score": lip_sync_score,
            "temporal_mean": temporal_mean,
            "temporal_max": temporal_max
        }
    
    # Extract CNN scores (fake_score)
    cnn_scores = [d.get("fake_score", 0.5) for d in detections]
    
    # Extract frequency scores
    freq_scores = [d.get("freq_score", 0.5) for d in detections]
    
    # Calculate CNN statistics
    max_score = max(cnn_scores) if cnn_scores else 0.0
    mean_score = sum(cnn_scores) / len(cnn_scores) if cnn_scores else 0.0
    sorted_cnn = sorted(cnn_scores, reverse=True)
    median_score = sorted_cnn[len(sorted_cnn) // 2] if sorted_cnn else 0.0
    p90_index = int(len(sorted_cnn) * 0.9)
    p90_score = sorted_cnn[p90_index] if p90_index < len(sorted_cnn) else sorted_cnn[-1] if sorted_cnn else 0.0
    count_above_threshold = sum(1 for score in cnn_scores if score > FACE_THRESHOLD)
    
    # Calculate frequency statistics (mean of all frequency scores)
    frequency_score = sum(freq_scores) / len(freq_scores) if freq_scores else 0.5
    
    return {
        "total_faces": total_faces,
        "max_score": max_score,
        "mean_score": mean_score,
        "median_score": median_score,
        "p90_score": p90_score,
        "count_above_0.5": count_above_threshold,
        "frequency_score": frequency_score,
        "lip_sync_score": lip_sync_score,
        "temporal_mean": temporal_mean,
        "temporal_max": temporal_max
    }


def decide_verdict(aggregation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide verdict based on aggregated scores.
    
    Args:
        aggregation: Aggregated scores dictionary
        
    Returns:
        Dictionary with label, confidence, and reasons
    """
    total_faces = aggregation.get("total_faces", 0)
    max_score = aggregation.get("max_score", 0.0)
    mean_score = aggregation.get("mean_score", 0.0)
    p90_score = aggregation.get("p90_score", 0.0)
    count_above_threshold = aggregation.get("count_above_0.5", 0)
    
    reasons = []
    confidence = 0.0
    
    # Check for insufficient data
    if total_faces < MIN_FACES_FOR_DECISION:
        return {
            "label": "INCONCLUSIVE",
            "confidence": 0.0,
            "reason": ["insufficient_faces"],
            "final_score": 0.5,
            "final_label": "INCONCLUSIVE",
            "raw_score": 0.5
        }
    
    # Check for very low scores (likely authentic) - RELAXED for real videos
    if max_score < 0.4 and mean_score < 0.35:
        confidence = 0.7
        return {
            "label": "LIKELY_AUTHENTIC",
            "confidence": confidence,
            "reason": ["very_low_scores"],
            "final_score": 1.0 - max_score,  # Invert for authentic
            "final_label": "LIKELY_AUTHENTIC",
            "raw_score": 1.0 - max_score
        }
    
    # Strong Signal Rule: Multiple strong indicators of manipulation
    lip_sync_score = aggregation.get("lip_sync_score")
    frequency_score = aggregation.get("frequency_score", 0.5)
    
    strong_signal = False
    if lip_sync_score is not None and lip_sync_score < 0.2 and max_score > 0.6:
        reasons.append("strong_signal_lip_sync")
        confidence = 0.85
        strong_signal = True
    elif frequency_score > 0.8 and max_score > 0.6:
        reasons.append("strong_signal_frequency")
        confidence = 0.85
        strong_signal = True
    
    # If strong signal detected, return early with LIKELY_MANIPULATED
    if strong_signal:
        return {
            "label": "LIKELY_MANIPULATED",
            "confidence": confidence,
            "reason": reasons,
            "final_score": max_score,
            "final_label": "LIKELY_MANIPULATED",
            "raw_score": max_score
        }
    
    # Check for extremely high score
    if max_score >= HIGH_THRESHOLD:
        reasons.append("high_max_score")
        confidence = 0.9
    
    # Check for multiple suspicious faces
    if count_above_threshold >= SUSPICIOUS_COUNT_NEEDED:
        reasons.append("multiple_suspicious_faces")
        confidence = max(confidence, 0.75)
    
    # Check for high p90 score
    if p90_score >= P90_THRESHOLD:
        reasons.append("high_p90_score")
        confidence = max(confidence, 0.7)
    
    # Check lip sync mismatch (lip_sync_score already retrieved in Strong Signal rule)
    if lip_sync_score is not None and lip_sync_score < 0.3:
        reasons.append("lip_sync_mismatch")
        confidence = max(confidence, 0.65)
    
    # Determine label - RELAXED thresholds for authentic detection
    if confidence >= 0.6 or max_score >= HIGH_THRESHOLD:
        label = "LIKELY_MANIPULATED"
        if not reasons:
            reasons.append("strong_manipulation_signal")
    elif max_score < 0.45 and mean_score < 0.4:
        label = "LIKELY_AUTHENTIC"
        confidence = 0.65
        reasons = ["low_scores_across_faces"]
    else:
        label = "INCONCLUSIVE"
        confidence = 0.4
        reasons = ["mixed_signals"]
    
    return {
        "label": label,
        "confidence": confidence,
        "reason": reasons,
        "final_score": max_score,  # Will be overridden by ensemble
        "final_label": label,  # Will be overridden by ensemble
        "raw_score": max_score  # Will be overridden by ensemble
    }


def generate_result(
    job_id: str,
    video_path: str,
    frames: int,
    detections: List[Dict[str, Any]],
    lip_sync_score: Optional[float] = None,
    temporal_mean: float = 0.5,
    temporal_max: float = 0.5,
    abnormality_report: Optional[Dict[str, Any]] = None,
    technique_report: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate complete result structure.
    
    Args:
        job_id: Job identifier
        video_path: Path to processed video
        frames: Number of frames extracted
        detections: List of detection results
        lip_sync_score: Lip-sync score
        temporal_mean: Mean temporal score
        temporal_max: Max temporal score
        abnormality_report: Optional abnormality analysis report
        technique_report: Optional technique identification report
        
    Returns:
        Complete result dictionary
    """
    # Aggregate scores
    aggregation = aggregate_scores(
        detections=detections,
        lip_sync_score=lip_sync_score,
        temporal_mean=temporal_mean,
        temporal_max=temporal_max
    )
    
    # Decide verdict
    verdict = decide_verdict(aggregation)
    
    # Create result structure
    result = {
        "job_id": job_id,
        "video_path": video_path,
        "frames": frames,
        "faces": len(detections),
        "detections": detections,
        "aggregation": aggregation,
        "verdict": verdict,
        "report_meta": {
            "timestamp": datetime.now().isoformat(),
            "model_versions": {
                "xception": "v0.1",
                "temporal": "v0.1",
                "lipsync": "v0.1",
                "frequency": "v0.1"
            }
        }
    }
    
    # Add abnormality report if provided
    if abnormality_report:
        result["abnormalities"] = abnormality_report
    
    # Add technique identification if provided
    if technique_report:
        result["detected_techniques"] = technique_report
    
    return result

