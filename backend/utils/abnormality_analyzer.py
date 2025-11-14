"""
Abnormality analyzer for detecting and describing specific deepfake artifacts.
"""
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import cv2


class AbnormalityAnalyzer:
    """
    Analyzes faces and frames to detect specific abnormalities and artifacts.
    """
    
    def __init__(self):
        """Initialize the abnormality analyzer."""
        pass
    
    def analyze_face_artifacts(self, face_path: str) -> Dict[str, Any]:
        """
        Analyze a single face image for specific artifacts.
        
        Args:
            face_path: Path to face image
            
        Returns:
            Dictionary with detected artifacts and their descriptions
        """
        artifacts = []
        artifact_details = []
        
        try:
            # Load image
            img = Image.open(face_path).convert('RGB')
            img_array = np.array(img)
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # 1. Check for blur artifacts (common in GAN-generated faces)
            blur_score = self._detect_blur_artifacts(gray)
            if blur_score > 0.6:
                artifacts.append("blur_artifacts")
                artifact_details.append({
                    "type": "blur_artifacts",
                    "severity": "high" if blur_score > 0.8 else "medium",
                    "description": f"Detected unnatural blur patterns (score: {blur_score:.2f}). "
                                 f"Common in GAN-based face swaps where blending creates soft edges.",
                    "location": "face_boundary",
                    "confidence": float(blur_score)
                })
            
            # 2. Check for color inconsistencies
            color_inconsistency = self._detect_color_inconsistencies(hsv)
            if color_inconsistency > 0.5:
                artifacts.append("color_inconsistency")
                artifact_details.append({
                    "type": "color_inconsistency",
                    "severity": "high" if color_inconsistency > 0.7 else "medium",
                    "description": f"Detected color tone mismatches (score: {color_inconsistency:.2f}). "
                                 f"Face region shows inconsistent skin tone, suggesting face swap artifacts.",
                    "location": "skin_region",
                    "confidence": float(color_inconsistency)
                })
            
            # 3. Check for edge artifacts
            edge_artifacts = self._detect_edge_artifacts(gray)
            if edge_artifacts > 0.5:
                artifacts.append("edge_artifacts")
                artifact_details.append({
                    "type": "edge_artifacts",
                    "severity": "high" if edge_artifacts > 0.7 else "medium",
                    "description": f"Detected unnatural edge patterns (score: {edge_artifacts:.2f}). "
                                 f"Sharp transitions or halos around face boundaries indicate manipulation.",
                    "location": "face_boundary",
                    "confidence": float(edge_artifacts)
                })
            
            # 4. Check for texture anomalies
            texture_anomaly = self._detect_texture_anomalies(gray)
            if texture_anomaly > 0.5:
                artifacts.append("texture_anomaly")
                artifact_details.append({
                    "type": "texture_anomaly",
                    "severity": "high" if texture_anomaly > 0.7 else "medium",
                    "description": f"Detected unnatural skin texture patterns (score: {texture_anomaly:.2f}). "
                                 f"Overly smooth or inconsistent texture suggests GAN generation artifacts.",
                    "location": "skin_region",
                    "confidence": float(texture_anomaly)
                })
            
            # 5. Check for eye region artifacts
            eye_artifacts = self._detect_eye_artifacts(img_array)
            if eye_artifacts > 0.5:
                artifacts.append("eye_artifacts")
                artifact_details.append({
                    "type": "eye_artifacts",
                    "severity": "high" if eye_artifacts > 0.7 else "medium",
                    "description": f"Detected anomalies in eye region (score: {eye_artifacts:.2f}). "
                                 f"Unnatural eye shape, reflections, or alignment issues detected.",
                    "location": "eye_region",
                    "confidence": float(eye_artifacts)
                })
            
        except Exception as e:
            print(f"Error analyzing face artifacts for {face_path}: {e}")
        
        return {
            "artifacts": artifacts,
            "details": artifact_details
        }
    
    def _detect_blur_artifacts(self, gray: np.ndarray) -> float:
        """Detect blur artifacts using Laplacian variance."""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Low variance indicates blur
        # Normalize to 0-1 range (lower variance = higher blur score)
        blur_score = 1.0 - min(laplacian_var / 100.0, 1.0)
        return float(blur_score)
    
    def _detect_color_inconsistencies(self, hsv: np.ndarray) -> float:
        """Detect color inconsistencies in HSV space."""
        # Calculate standard deviation of hue and saturation
        hue_std = np.std(hsv[:, :, 0])
        sat_std = np.std(hsv[:, :, 1])
        
        # High variation in hue/saturation suggests inconsistencies
        inconsistency = (hue_std / 180.0 + sat_std / 255.0) / 2.0
        return float(min(inconsistency, 1.0))
    
    def _detect_edge_artifacts(self, gray: np.ndarray) -> float:
        """Detect edge artifacts using Canny edge detection."""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Very high or very low edge density can indicate artifacts
        if edge_density > 0.3 or edge_density < 0.05:
            return float(min(edge_density * 2.0, 1.0))
        return 0.3
    
    def _detect_texture_anomalies(self, gray: np.ndarray) -> float:
        """Detect texture anomalies using local binary patterns."""
        # Calculate local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Very low or very high variance indicates texture anomalies
        mean_var = np.mean(local_var)
        if mean_var < 50 or mean_var > 500:
            return float(min(abs(mean_var - 200) / 300.0, 1.0))
        return 0.2
    
    def _detect_eye_artifacts(self, img: np.ndarray) -> float:
        """Detect artifacts in eye region (simplified detection)."""
        h, w = img.shape[:2]
        # Focus on upper third of face (eye region)
        eye_region = img[0:h//3, :]
        
        # Check for symmetry issues
        left_half = eye_region[:, :w//2]
        right_half = eye_region[:, w//2:]
        
        # Flip right half and compare
        right_flipped = cv2.flip(right_half, 1)
        
        # Resize if needed for comparison
        if left_half.shape != right_flipped.shape:
            right_flipped = cv2.resize(right_flipped, (left_half.shape[1], left_half.shape[0]))
        
        # Calculate difference
        diff = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
        anomaly_score = min(diff / 30.0, 1.0)  # Normalize
        
        return float(anomaly_score)
    
    def analyze_temporal_inconsistencies(
        self, 
        detections: List[Dict[str, Any]], 
        temporal_mean: float,
        temporal_max: float
    ) -> Dict[str, Any]:
        """
        Analyze temporal inconsistencies across frames.
        
        Args:
            detections: List of detection results with frame numbers
            temporal_mean: Mean temporal consistency score
            temporal_max: Max temporal inconsistency score
            
        Returns:
            Dictionary with temporal abnormality details
        """
        temporal_issues = []
        
        # Check for high temporal inconsistency
        if temporal_max > 0.6:
            temporal_issues.append({
                "type": "temporal_inconsistency",
                "severity": "high" if temporal_max > 0.8 else "medium",
                "description": f"Detected significant temporal inconsistencies (score: {temporal_max:.2f}). "
                             f"Frames show unnatural transitions or flickering, indicating frame-by-frame manipulation.",
                "affected_frames": "multiple",
                "confidence": float(temporal_max)
            })
        
        # Analyze score variations across frames
        if len(detections) > 1:
            scores = [d.get("fake_score", 0.5) for d in detections]
            score_variance = np.var(scores)
            
            if score_variance > 0.05:  # High variance indicates inconsistent manipulation
                temporal_issues.append({
                    "type": "inconsistent_manipulation",
                    "severity": "medium",
                    "description": f"Detected inconsistent manipulation quality across frames (variance: {score_variance:.3f}). "
                                 f"Some frames show stronger manipulation artifacts than others.",
                    "affected_frames": "multiple",
                    "confidence": float(min(score_variance * 10, 1.0))
                })
        
        return {
            "temporal_issues": temporal_issues,
            "temporal_mean": temporal_mean,
            "temporal_max": temporal_max
        }
    
    def analyze_audio_visual_mismatch(
        self, 
        lip_sync_score: Optional[float]
    ) -> Dict[str, Any]:
        """
        Analyze audio-visual synchronization issues.
        
        Args:
            lip_sync_score: Lip-sync score (lower = worse sync)
            
        Returns:
            Dictionary with audio-visual mismatch details
        """
        audio_visual_issues = []
        
        if lip_sync_score is not None:
            if lip_sync_score < 0.3:
                audio_visual_issues.append({
                    "type": "severe_lip_sync_mismatch",
                    "severity": "high",
                    "description": f"Severe lip-sync mismatch detected (score: {lip_sync_score:.2f}). "
                                 f"Lip movements do not match audio track, indicating face replacement or audio manipulation.",
                    "confidence": float(1.0 - lip_sync_score)
                })
            elif lip_sync_score < 0.5:
                audio_visual_issues.append({
                    "type": "moderate_lip_sync_mismatch",
                    "severity": "medium",
                    "description": f"Moderate lip-sync mismatch detected (score: {lip_sync_score:.2f}). "
                                 f"Some inconsistencies between lip movements and audio.",
                    "confidence": float(0.7 - lip_sync_score)
                })
        
        return {
            "audio_visual_issues": audio_visual_issues,
            "lip_sync_score": lip_sync_score
        }
    
    def generate_abnormality_report(
        self,
        faces_dir: str,
        detections: List[Dict[str, Any]],
        temporal_mean: float,
        temporal_max: float,
        lip_sync_score: Optional[float]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive abnormality report.
        
        Args:
            faces_dir: Directory containing face images
            detections: List of detection results
            temporal_mean: Mean temporal score
            temporal_max: Max temporal score
            lip_sync_score: Lip-sync score
            
        Returns:
            Complete abnormality report
        """
        all_artifacts = []
        all_artifact_details = []
        
        # Analyze each face for artifacts
        face_files = sorted(Path(faces_dir).glob("face_*.jpg"))
        for face_path in face_files:
            face_analysis = self.analyze_face_artifacts(str(face_path))
            if face_analysis["artifacts"]:
                all_artifacts.extend(face_analysis["artifacts"])
                # Add frame info to details
                face_filename = face_path.name
                for detail in face_analysis["details"]:
                    # Find corresponding detection
                    detection = next((d for d in detections if d.get("face_file") == face_filename), None)
                    if detection:
                        detail["frame"] = detection.get("frame", "unknown")
                    all_artifact_details.append(detail)
        
        # Analyze temporal inconsistencies
        temporal_analysis = self.analyze_temporal_inconsistencies(
            detections, temporal_mean, temporal_max
        )
        
        # Analyze audio-visual mismatch
        audio_visual_analysis = self.analyze_audio_visual_mismatch(lip_sync_score)
        
        # Count unique artifact types
        artifact_counts = {}
        for artifact in all_artifacts:
            artifact_counts[artifact] = artifact_counts.get(artifact, 0) + 1
        
        # Generate summary
        total_artifacts = len(all_artifact_details)
        high_severity_count = sum(1 for d in all_artifact_details if d.get("severity") == "high")
        medium_severity_count = sum(1 for d in all_artifact_details if d.get("severity") == "medium")
        
        return {
            "summary": {
                "total_abnormalities": total_artifacts,
                "high_severity_count": high_severity_count,
                "medium_severity_count": medium_severity_count,
                "artifact_types_detected": list(artifact_counts.keys()),
                "artifact_counts": artifact_counts
            },
            "spatial_artifacts": all_artifact_details,
            "temporal_abnormalities": temporal_analysis["temporal_issues"],
            "audio_visual_abnormalities": audio_visual_analysis["audio_visual_issues"],
            "detailed_findings": {
                "spatial": all_artifact_details,
                "temporal": temporal_analysis["temporal_issues"],
                "audio_visual": audio_visual_analysis["audio_visual_issues"]
            }
        }

