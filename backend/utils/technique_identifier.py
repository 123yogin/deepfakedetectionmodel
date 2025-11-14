"""
Technique identifier for detecting mathematical/algorithmic methods used in deepfake creation.
"""
from typing import Dict, Any, List, Optional
import numpy as np


class TechniqueIdentifier:
    """
    Identifies the mathematical techniques and algorithms likely used to create deepfakes.
    """
    
    # Technique signatures based on artifact patterns
    TECHNIQUE_SIGNATURES = {
        "gan_based": {
            "name": "GAN-based Face Swap",
            "description": "Generative Adversarial Network (GAN) based face swapping technique",
            "indicators": ["blur_artifacts", "texture_anomaly", "high_frequency_artifacts"],
            "mathematical_basis": "Uses generator-discriminator architecture. Generator learns mapping G: X→Y, "
                               "discriminator D distinguishes real from fake. Training objective: "
                               "min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]. "
                               "Common implementations: StyleGAN, StarGAN, FaceSwap-GAN.",
            "typical_artifacts": [
                "Blur artifacts at face boundaries due to generator's learned blending",
                "Texture inconsistencies from adversarial training",
                "High-frequency artifacts in frequency domain analysis"
            ]
        },
        "autoencoder_based": {
            "name": "Autoencoder-based Face Swap",
            "description": "Autoencoder architecture for face reconstruction and swapping",
            "indicators": ["edge_artifacts", "color_inconsistency", "moderate_temporal_inconsistency"],
            "mathematical_basis": "Uses encoder-decoder architecture: E: X→Z, D: Z→X. "
                               "Encoder compresses face to latent representation z, decoder reconstructs. "
                               "Face swap: z_target = E(x_target), x_swapped = D(z_source). "
                               "Loss function: L = ||x - D(E(x))||² + λ·R(z). "
                               "Common implementations: DeepFaceLab, FaceSwap.",
            "typical_artifacts": [
                "Edge artifacts from imperfect reconstruction",
                "Color mismatches due to encoder-decoder limitations",
                "Moderate temporal flickering"
            ]
        },
        "first_order_motion": {
            "name": "First-Order Motion Model",
            "description": "First-order motion model for face reenactment",
            "indicators": ["temporal_inconsistency", "moderate_lip_sync_mismatch", "eye_artifacts"],
            "mathematical_basis": "Uses keypoint detection and motion transfer. "
                               "Keypoints: K = {k₁, k₂, ..., kₙ}. Motion: M = K_driver - K_source. "
                               "Warping: W(x, M) applies motion to source. "
                               "Generator: G(x_source, M) = W(x_source, M) + residual. "
                               "Common implementations: FOMM, Face2Face.",
            "typical_artifacts": [
                "Temporal inconsistencies from keypoint tracking errors",
                "Eye region artifacts from motion transfer",
                "Moderate lip-sync issues"
            ]
        },
        "neural_texture": {
            "name": "Neural Texture Synthesis",
            "description": "Neural texture synthesis for face manipulation",
            "indicators": ["texture_anomaly", "blur_artifacts", "high_frequency_artifacts"],
            "mathematical_basis": "Uses neural texture fields: T(x,y) = f_θ(x,y). "
                               "Texture mapping: I = T(UV_map). "
                               "Optimization: min_θ ||I_target - T(UV_target)||². "
                               "Common implementations: NeuralTextures, PIFu.",
            "typical_artifacts": [
                "Texture anomalies from neural synthesis",
                "Blur from texture interpolation",
                "High-frequency artifacts in synthesized regions"
            ]
        },
        "face2face": {
            "name": "Face2Face",
            "description": "Real-time face reenactment using expression transfer",
            "indicators": ["severe_lip_sync_mismatch", "temporal_inconsistency", "eye_artifacts"],
            "mathematical_basis": "Uses 3D face model and expression transfer. "
                               "3D model: M = (V, F) where V are vertices, F are faces. "
                               "Expression: E = {e₁, e₂, ..., eₙ}. "
                               "Transfer: M_target = M_source + ΔE. "
                               "Rendering: I = R(M_target, P, L). "
                               "Common implementations: Face2Face, FaceForensics.",
            "typical_artifacts": [
                "Severe lip-sync mismatches from expression transfer",
                "Temporal flickering from frame-by-frame processing",
                "Eye region artifacts from 3D model fitting"
            ]
        },
        "deepfacelab": {
            "name": "DeepFaceLab",
            "description": "Autoencoder-based face swap with advanced training",
            "indicators": ["color_inconsistency", "edge_artifacts", "texture_anomaly", "moderate_temporal_inconsistency"],
            "mathematical_basis": "Uses SAE (Stacked Autoencoder) architecture. "
                               "Multiple encoders: E₁, E₂, ..., Eₙ. "
                               "Progressive training with increasing resolution. "
                               "Loss: L = L_recon + λ₁·L_perceptual + λ₂·L_adversarial. "
                               "Common implementations: DeepFaceLab, FaceSwap.",
            "typical_artifacts": [
                "Color inconsistencies from encoder limitations",
                "Edge artifacts at boundaries",
                "Texture anomalies in reconstructed regions"
            ]
        }
    }
    
    def __init__(self):
        """Initialize the technique identifier."""
        pass
    
    def identify_technique(
        self,
        abnormality_report: Dict[str, Any],
        aggregation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify the most likely deepfake creation technique.
        
        Args:
            abnormality_report: Report from AbnormalityAnalyzer
            aggregation: Aggregated scores from detection
            
        Returns:
            Dictionary with identified techniques and confidence scores
        """
        # Extract artifact types
        spatial_artifacts = abnormality_report.get("spatial_artifacts", [])
        temporal_issues = abnormality_report.get("temporal_abnormalities", [])
        audio_visual_issues = abnormality_report.get("audio_visual_abnormalities", [])
        
        # Collect all artifact types
        artifact_types = set()
        for artifact in spatial_artifacts:
            artifact_types.add(artifact.get("type", ""))
        for issue in temporal_issues:
            artifact_types.add(issue.get("type", ""))
        for issue in audio_visual_issues:
            artifact_types.add(issue.get("type", ""))
        
        # Add frequency-based indicators
        frequency_score = aggregation.get("frequency_score", 0.5)
        if frequency_score > 0.7:
            artifact_types.add("high_frequency_artifacts")
        
        # Score each technique based on matching indicators
        technique_scores = {}
        for technique_id, signature in self.TECHNIQUE_SIGNATURES.items():
            score = 0.0
            matching_indicators = []
            
            # Check indicator matches
            for indicator in signature["indicators"]:
                if indicator in artifact_types:
                    score += 1.0
                    matching_indicators.append(indicator)
            
            # Normalize score
            if len(signature["indicators"]) > 0:
                score = score / len(signature["indicators"])
            
            # Boost score based on aggregation metrics
            if technique_id == "gan_based" and frequency_score > 0.7:
                score += 0.2
            if technique_id in ["autoencoder_based", "deepfacelab"] and aggregation.get("max_score", 0) > 0.6:
                score += 0.15
            if technique_id == "face2face" and aggregation.get("lip_sync_score", 1.0) < 0.3:
                score += 0.2
            
            technique_scores[technique_id] = {
                "score": min(score, 1.0),
                "matching_indicators": matching_indicators
            }
        
        # Sort techniques by score
        sorted_techniques = sorted(
            technique_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Get top techniques
        identified_techniques = []
        for technique_id, score_info in sorted_techniques:
            if score_info["score"] > 0.2:  # Only include if score > 20%
                signature = self.TECHNIQUE_SIGNATURES[technique_id]
                identified_techniques.append({
                    "technique_id": technique_id,
                    "name": signature["name"],
                    "description": signature["description"],
                    "confidence": float(score_info["score"]),
                    "matching_indicators": score_info["matching_indicators"],
                    "mathematical_basis": signature["mathematical_basis"],
                    "typical_artifacts": signature["typical_artifacts"]
                })
        
        # Determine primary technique
        primary_technique = None
        if identified_techniques:
            primary = identified_techniques[0]
            if primary["confidence"] > 0.4:
                primary_technique = {
                    "technique_id": primary["technique_id"],
                    "name": primary["name"],
                    "confidence": primary["confidence"],
                    "description": primary["description"]
                }
        
        return {
            "primary_technique": primary_technique,
            "all_techniques": identified_techniques,
            "analysis": {
                "total_techniques_identified": len(identified_techniques),
                "highest_confidence": identified_techniques[0]["confidence"] if identified_techniques else 0.0,
                "artifact_evidence": list(artifact_types)
            }
        }
    
    def generate_technique_report(
        self,
        abnormality_report: Dict[str, Any],
        aggregation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive technique identification report.
        
        Args:
            abnormality_report: Report from AbnormalityAnalyzer
            aggregation: Aggregated scores from detection
            
        Returns:
            Complete technique identification report
        """
        identification = self.identify_technique(abnormality_report, aggregation)
        
        # Generate detailed explanation
        explanation = self._generate_explanation(identification, abnormality_report)
        
        return {
            **identification,
            "explanation": explanation,
            "methodology": {
                "approach": "Pattern-based technique identification using artifact signatures",
                "indicators_analyzed": len(abnormality_report.get("summary", {}).get("artifact_types_detected", [])),
                "techniques_considered": len(self.TECHNIQUE_SIGNATURES)
            }
        }
    
    def _generate_explanation(
        self,
        identification: Dict[str, Any],
        abnormality_report: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation of technique identification."""
        primary = identification.get("primary_technique")
        
        if not primary:
            return "Insufficient evidence to confidently identify a specific deepfake creation technique. " \
                   "The video may be authentic, or the manipulation method is not clearly identifiable from the detected artifacts."
        
        explanation = f"Based on the analysis of detected artifacts, the video most likely uses **{primary['name']}** " \
                     f"(confidence: {primary['confidence']:.1%}). "
        
        # Add details about why this technique was identified
        all_techniques = identification.get("all_techniques", [])
        if all_techniques:
            primary_tech = all_techniques[0]
            if primary_tech.get("matching_indicators"):
                explanation += f"\n\n**Key Evidence:**\n"
                for indicator in primary_tech["matching_indicators"]:
                    explanation += f"- {indicator.replace('_', ' ').title()}\n"
        
        # Add mathematical basis
        if all_techniques:
            explanation += f"\n**Mathematical Basis:**\n{all_techniques[0].get('mathematical_basis', 'N/A')}"
        
        return explanation

