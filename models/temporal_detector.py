"""
Temporal detector for deepfake artifacts.
Uses 3D-CNN to detect frame-to-frame inconsistencies.
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.special import expit  # Sigmoid function
from backend.config.model_config import TEMPORAL_MODEL_PATH, FAIL_IF_NO_WEIGHTS
from models.architectures import Simple3DCNN


class TemporalDetector:
    """
    Temporal detector that analyzes sequences of frames for temporal artifacts.
    Detects flicker, inconsistent blending, and frame-to-frame anomalies.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize temporal detector.
        
        Args:
            model_path: Optional path to pretrained 3D-CNN weights.
                       If None, uses TEMPORAL_MODEL_PATH from config.
            device: Device to run on ('cpu' or 'cuda')
                       
        Raises:
            FileNotFoundError: If weights not found and FAIL_IF_NO_WEIGHTS=True
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        
        # GPU memory optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Use config path if model_path not provided
        if model_path is None:
            model_path = TEMPORAL_MODEL_PATH
        
        # Initialize model (with or without weights)
        try:
            # Always create model architecture
            # If weights exist, they'll be loaded; otherwise use untrained model
            self.model, weights_loaded = self._load_3d_model(model_path if (model_path and os.path.exists(model_path)) else None)
            
            # Set model_loaded based on actual successful weight loading
            self.model_loaded = weights_loaded
            
            if weights_loaded:
                print(f"[OK] Loaded trained temporal 3D-CNN model from {model_path}")
            else:
                if model_path and os.path.exists(model_path):
                    print(f"[WARNING] Temporal model weights found but failed to load from {model_path}")
                else:
                    print(f"[WARNING] Temporal model weights not found at {model_path}")
                print(f"[WARNING] Using untrained 3D-CNN architecture (predictions will be unreliable)")
                print(f"[INFO] To get accurate temporal predictions:")
                print(f"  1. Train or download 3D-CNN weights (I3D, 3D-ResNet)")
                print(f"  2. Place weights at: {model_path}")
                print(f"  3. Restart the application")
        except Exception as e:
            print(f"[WARNING] Error initializing temporal model: {e}")
            if FAIL_IF_NO_WEIGHTS:
                raise FileNotFoundError(
                    f"Failed to initialize temporal model. "
                    f"Please ensure trained 3D-CNN weights are available at {model_path}."
                )
            # Create model without weights as fallback
            self.model = Simple3DCNN(num_classes=2, input_channels=3).to(self.device)
            self.model.eval()
            self.model_loaded = False
            print(f"[WARNING] Continuing with untrained temporal model")
        
        # Image preprocessing for temporal model (optimized)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),  # Faster interpolation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Performance flags
        self.use_mixed_precision = torch.cuda.is_available() and self.device.type == 'cuda'
        self.batch_size = 4  # Default batch size for clip processing
    
    def _load_3d_model(self, model_path: Optional[str] = None) -> tuple:
        """
        Load 3D-CNN model from weights file.
        
        Args:
            model_path: Optional path to model weights. If None, creates untrained model.
            
        Returns:
            Tuple of (Loaded 3D-CNN model, weights_loaded_successfully)
        """
        # Create model architecture
        model = Simple3DCNN(num_classes=2, input_channels=3)
        weights_loaded = False
        
        # Try to load weights if path provided and file exists
        if model_path and os.path.exists(model_path):
            try:
                # Validate weight file first
                from backend.utils.model_validator import ModelValidator
                is_valid, msg = ModelValidator.validate_weight_file(model_path)
                
                if not is_valid:
                    print(f"[WARNING] Weight file validation failed: {msg}")
                    print(f"[INFO] Using untrained 3D-CNN architecture")
                else:
                    state_dict = torch.load(model_path, map_location=self.device)
                    # Handle different state dict formats
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    
                    # Load weights (strict=False allows partial loading)
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        weights_loaded = True
                        print(f"[OK] Loaded 3D-CNN weights from {model_path}")
                        print(f"     {msg}")
                    except Exception as load_error:
                        print(f"[WARNING] Error loading weights into model: {load_error}")
                        print(f"[INFO] Using untrained 3D-CNN architecture")
            except Exception as e:
                print(f"[WARNING] Could not load weights: {e}")
                print(f"[INFO] Using untrained 3D-CNN architecture (will need training)")
        else:
            # No weights provided or file doesn't exist
            print(f"[INFO] Creating untrained 3D-CNN architecture")
            print(f"[INFO] To add weights: Run python scripts/download_weights.py")
        
        model = model.to(self.device)
        model.eval()
        return model, weights_loaded
    
    def predict_clip(self, frames: List[Image.Image]) -> float:
        """
        Predict temporal score for a clip of frames.
        
        Args:
            frames: List of PIL Images (temporal sequence)
            
        Returns:
            Temporal score between 0 and 1 (higher = more likely fake)
            
        Note:
            If 3D-CNN model is loaded, uses learned features.
            Otherwise, uses fallback variance method (not accurate).
        """
        if len(frames) < 2:
            return 0.5  # Neutral score for insufficient frames
        
        # Use 3D-CNN if model is loaded
        if self.model_loaded and self.model is not None:
            return self._predict_with_3dcnn(frames)
        else:
            # Fallback: variance-based method (not accurate, but prevents errors)
            return self._predict_with_variance_fallback(frames)
    
    def _predict_with_3dcnn(self, frames: List[Image.Image]) -> float:
        """
        Predict using 3D-CNN model (trained or untrained).
        
        Args:
            frames: List of PIL Images
            
        Returns:
            Temporal score from 3D-CNN
        """
        if self.model is None:
            return 0.5  # Neutral if model not initialized
        
        # Convert frames to tensor sequence
        frame_tensors = []
        for frame in frames:
            tensor = self.transform(frame.convert('RGB'))
            frame_tensors.append(tensor)
        
        if len(frame_tensors) == 0:
            return 0.5
        
        # Stack into (C, T, H, W) format for 3D-CNN
        # Each tensor is (3, 224, 224), stack along new dimension
        # Result: (3, num_frames, 224, 224)
        clip_tensor = torch.stack(frame_tensors, dim=1)  # (3, T, 224, 224)
        clip_tensor = clip_tensor.unsqueeze(0)  # Add batch dim: (1, 3, T, 224, 224)
        clip_tensor = clip_tensor.to(self.device)
        
        # Apply mixed precision if enabled
        if self.use_mixed_precision and self.device.type == 'cuda':
            clip_tensor = clip_tensor.half()
        
        # Run inference
        with torch.no_grad():
            try:
                outputs = self.model(clip_tensor)
                # Apply softmax to get probabilities
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    score = probabilities[0][1].item()  # Fake class probability
                else:
                    # Single output, use sigmoid
                    score = torch.sigmoid(outputs).item()
                
                # If model not trained, normalize the output to be less extreme
                if not self.model_loaded:
                    # Compress untrained predictions to reasonable range (0.35-0.65)
                    score = 0.35 + (score * 0.3)
                
                return float(np.clip(score, 0.0, 1.0))
            except Exception as e:
                print(f"Error in 3D-CNN prediction: {e}")
                return 0.5  # Neutral on error
    
    def _predict_with_variance_fallback(self, frames: List[Image.Image]) -> float:
        """
        Fallback method using temporal variance analysis.
        
        Args:
            frames: List of PIL Images
            
        Returns:
            Temporal score based on variance (higher variance = more likely fake)
        """
        # Use untrained model if available, otherwise use variance heuristic
        if self.model is not None:
            try:
                # Try to use untrained model for basic feature extraction
                return self._predict_with_3dcnn(frames)
            except:
                pass
        
        # Fallback to variance-based analysis
        frame_arrays = []
        for frame in frames:
            gray = frame.convert('L')
            resized = gray.resize((224, 224), Image.LANCZOS)
            frame_arrays.append(np.array(resized, dtype=np.float32))
        
        frames_stack = np.stack(frame_arrays, axis=0)
        diffs = [np.abs(frames_stack[i+1] - frames_stack[i]) 
                 for i in range(len(frames_stack) - 1)]
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # Calculate temporal inconsistency score
        # Higher variance and std = more temporal artifacts (likely fake)
        # Normalize to reasonable range
        raw_score = (mean_diff / 50.0) + (std_diff / 30.0)
        temporal_score = expit((raw_score - 0.5) / 0.3)
        
        # Compress to reasonable range for untrained predictions (0.35-0.65)
        temporal_score = 0.35 + (temporal_score * 0.3)
        
        return float(np.clip(temporal_score, 0.0, 1.0))
    
    def predict_for_face_track(self, frames_dir: str, clip_len: int = 16, stride: int = 8) -> Dict:
        """
        Predict temporal scores for all clips in a face track.
        
        Args:
            frames_dir: Directory containing face frame images
            clip_len: Number of frames per clip
            stride: Stride between clips (overlap = clip_len - stride)
            
        Returns:
            Dictionary with clip_scores, mean_score, max_score
        """
        # Get all frame files sorted
        frame_files = sorted(Path(frames_dir).glob("face_*.jpg"))
        
        if len(frame_files) < clip_len:
            # Not enough frames for a clip
            return {
                "clip_scores": [],
                "mean_score": 0.5,
                "max_score": 0.5
            }
        
        clip_scores = []
        
        # Extract clips with stride
        for start_idx in range(0, len(frame_files) - clip_len + 1, stride):
            # Load clip frames
            clip_frames = []
            for i in range(start_idx, min(start_idx + clip_len, len(frame_files))):
                try:
                    frame = Image.open(frame_files[i]).convert('RGB')
                    clip_frames.append(frame)
                except Exception as e:
                    print(f"Error loading frame {frame_files[i]}: {e}")
                    continue
            
            if len(clip_frames) >= 2:  # Need at least 2 frames
                score = self.predict_clip(clip_frames)
                clip_scores.append(score)
        
        if not clip_scores:
            return {
                "clip_scores": [],
                "mean_score": 0.5,
                "max_score": 0.5
            }
        
        return {
            "clip_scores": clip_scores,
            "mean_score": float(np.mean(clip_scores)),
            "max_score": float(np.max(clip_scores))
        }
    
    def predict_clips_batch(self, clips: List[List[Image.Image]], batch_size: int = None) -> List[float]:
        """
        Predict temporal scores for multiple clips in batches.
        More efficient than calling predict_clip() multiple times.
        
        Args:
            clips: List of clips, where each clip is a list of PIL Images
            batch_size: Batch size for processing (default: self.batch_size)
            
        Returns:
            List of temporal scores between 0 and 1
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if not clips:
            return []
        
        results = []
        
        # Process clips in batches
        for i in range(0, len(clips), batch_size):
            batch_clips = clips[i:i+batch_size]
            batch_tensors = []
            valid_indices = []
            
            for clip_idx, clip_frames in enumerate(batch_clips):
                if len(clip_frames) < 2:
                    continue
                
                # Convert frames to tensor sequence
                frame_tensors = []
                for frame in clip_frames:
                    tensor = self.transform(frame.convert('RGB'))
                    frame_tensors.append(tensor)
                
                if len(frame_tensors) == 0:
                    continue
                
                # Stack into (C, T, H, W) format
                clip_tensor = torch.stack(frame_tensors, dim=1)  # (3, T, 224, 224)
                clip_tensor = clip_tensor.unsqueeze(0)  # (1, 3, T, 224, 224)
                batch_tensors.append(clip_tensor)
                valid_indices.append(i + clip_idx)
            
            if not batch_tensors:
                # Add neutral scores for invalid clips
                results.extend([0.5] * len(batch_clips))
                continue
            
            # Pad clips to same temporal length for batching
            max_temporal_len = max(t.shape[2] for t in batch_tensors)
            padded_tensors = []
            
            for tensor in batch_tensors:
                current_len = tensor.shape[2]
                if current_len < max_temporal_len:
                    # Pad with last frame
                    padding = tensor[:, :, -1:, :, :].repeat(1, 1, max_temporal_len - current_len, 1, 1)
                    tensor = torch.cat([tensor, padding], dim=2)
                padded_tensors.append(tensor)
            
            # Stack into batch: (B, 3, T, 224, 224)
            batch_tensor = torch.cat(padded_tensors, dim=0).to(self.device)
            
            # Apply mixed precision if enabled
            if self.use_mixed_precision and self.device.type == 'cuda':
                batch_tensor = batch_tensor.half()
            
            # Run inference
            with torch.no_grad():
                try:
                    outputs = self.model(batch_tensor)
                    
                    # Process outputs
                    batch_scores = []
                    for idx, output in enumerate(outputs):
                        if output.dim() > 0 and output.size(0) > 1:
                            probabilities = torch.nn.functional.softmax(output.unsqueeze(0), dim=1)
                            score = probabilities[0][1].item()
                        else:
                            score = torch.sigmoid(output).item()
                        
                        # If model not trained, normalize
                        if not self.model_loaded:
                            score = 0.35 + (score * 0.3)
                        
                        batch_scores.append(float(np.clip(score, 0.0, 1.0)))
                    
                    # Map scores back to original clip indices
                    clip_scores_dict = {valid_indices[j]: batch_scores[j] for j in range(len(batch_scores))}
                    
                    # Fill in results for this batch
                    for clip_idx in range(i, i + len(batch_clips)):
                        if clip_idx in clip_scores_dict:
                            results.append(clip_scores_dict[clip_idx])
                        else:
                            results.append(0.5)  # Neutral for invalid clips
                            
                except Exception as e:
                    print(f"Error in batch temporal prediction: {e}")
                    results.extend([0.5] * len(batch_clips))
        
        return results
    
    def enable_mixed_precision(self):
        """Enable FP16 mixed precision for faster inference on GPU."""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            try:
                self.model = self.model.half()
                self.use_mixed_precision = True
                print(f"[OK] Enabled FP16 mixed precision for Temporal detector")
            except Exception as e:
                print(f"[WARNING] Failed to enable mixed precision: {e}")
                self.use_mixed_precision = False
    
    def compile_with_torchscript(self):
        """Compile model with TorchScript for faster inference."""
        try:
            # Create example input (1, 3, 16, 224, 224) - 16 frames
            example_input = torch.randn(1, 3, 16, 224, 224).to(self.device)
            if self.use_mixed_precision and self.device.type == 'cuda':
                example_input = example_input.half()
            
            self.model = torch.jit.trace(self.model, example_input)
            print(f"[OK] Temporal model compiled with TorchScript")
        except Exception as e:
            print(f"[WARNING] TorchScript compilation failed: {e}")


