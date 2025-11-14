"""
Frequency-domain detector for deepfake artifacts.
Uses FFT/DCT analysis to detect high-frequency anomalies.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from scipy.special import expit  # Sigmoid function


class FrequencyDetector:
    """
    Frequency-domain detector that analyzes FFT patterns in face images.
    Detects anomalies in high-frequency energy distribution.
    """
    
    def __init__(self, img_size: int = 256, R_low: float = None, mu: float = 3.0, sigma: float = 1.5):
        """
        Initialize frequency detector.
        
        Args:
            img_size: Size to resize images to (default: 256)
            R_low: Radius for low-frequency region (default: img_size * 0.08)
            mu: Mean parameter for sigmoid normalization (default: 3.0, calibrated to prevent all 1.0)
            sigma: Standard deviation parameter for sigmoid normalization (default: 1.5, calibrated to prevent all 1.0)
        """
        self.img_size = img_size
        self.R_low = R_low if R_low is not None else img_size * 0.08
        self.mu = mu
        self.sigma = sigma
        self._debug_count = 0  # For debug logging
        
        # Create low-frequency mask (circular region)
        center = img_size // 2
        y, x = np.ogrid[:img_size, :img_size]
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        self.low_mask = dist_from_center <= self.R_low
        self.high_mask = ~self.low_mask
    
    def compute_face_freq_score(self, face_path: str) -> float:
        """
        Compute frequency anomaly score for a face image.
        
        Args:
            face_path: Path to face image file
            
        Returns:
            Frequency score between 0 and 1 (higher = more likely fake)
        """
        try:
            # 1) Load face image
            img = Image.open(face_path).convert('RGB')
            
            # 2) Convert to grayscale
            img_gray = img.convert('L')
            
            # 3) Resize to fixed size
            img_resized = img_gray.resize((self.img_size, self.img_size), Image.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32)
            
            # 4) Compute 2D FFT magnitude and center it
            fft_result = np.fft.fft2(img_array)
            fft_shifted = np.fft.fftshift(fft_result)
            fft_magnitude = np.abs(fft_shifted)
            
            # Use log magnitude for stability
            log_magnitude = np.log1p(fft_magnitude)
            
            # 5) Compute high-frequency vs low-frequency energy ratio
            energy_low = np.sum(log_magnitude[self.low_mask])
            energy_high = np.sum(log_magnitude[self.high_mask])
            
            # Avoid division by zero
            raw_ratio = energy_high / (energy_low + 1e-8)
            
            # Debug logging for first few faces
            if self._debug_count < 3:
                print(f"[DEBUG] Frequency - Face: {Path(face_path).name}, Raw ratio: {raw_ratio:.3f}, Energy low: {energy_low:.1f}, Energy high: {energy_high:.1f}")
                self._debug_count += 1
            
            # 6) Normalize using sigmoid
            # sigmoid((raw - mu) / sigma) maps to [0, 1]
            freq_score = expit((raw_ratio - self.mu) / self.sigma)
            
            # Apply clamping for extreme values to prevent all 1.0
            if freq_score > 0.95:
                if raw_ratio > 10.0:
                    # Very high ratio - compress but keep high
                    freq_score = 0.75 + ((freq_score - 0.75) * 0.2)  # Map 0.75-1.0 to 0.75-0.8
                else:
                    # High but not extreme - cap at 0.85
                    freq_score = min(freq_score, 0.85)
            
            return float(np.clip(freq_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error computing frequency score for {face_path}: {e}")
            return 0.5  # Neutral score on error
    
    def save_frequency_map(self, face_path: str, output_path: str) -> None:
        """
        Save a debug visualization of the frequency map.
        
        Args:
            face_path: Path to input face image
            output_path: Path to save frequency map visualization
        """
        try:
            # Load and process image
            img = Image.open(face_path).convert('RGB')
            img_gray = img.convert('L')
            img_resized = img_gray.resize((self.img_size, self.img_size), Image.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32)
            
            # Compute FFT
            fft_result = np.fft.fft2(img_array)
            fft_shifted = np.fft.fftshift(fft_result)
            fft_magnitude = np.abs(fft_shifted)
            log_magnitude = np.log1p(fft_magnitude)
            
            # Normalize for visualization (0-255)
            log_mag_normalized = ((log_magnitude - log_magnitude.min()) / 
                                 (log_magnitude.max() - log_magnitude.min() + 1e-8) * 255)
            log_mag_normalized = log_mag_normalized.astype(np.uint8)
            
            # Apply colormap for better visualization
            freq_map = cv2.applyColorMap(log_mag_normalized, cv2.COLORMAP_JET)
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            cv2.imwrite(output_path, freq_map)
            
        except Exception as e:
            print(f"Error saving frequency map for {face_path}: {e}")
    
    def batch_compute(self, faces_dir: str, output_debug_dir: str = None) -> dict:
        """
        Compute frequency scores for all face images in a directory.
        
        Args:
            faces_dir: Directory containing face images
            output_debug_dir: Optional directory to save frequency map visualizations
            
        Returns:
            Dictionary mapping face filename to frequency score
        """
        face_files = sorted(Path(faces_dir).glob("face_*.jpg"))
        scores = {}
        
        for face_path in face_files:
            face_filename = face_path.name
            freq_score = self.compute_face_freq_score(str(face_path))
            scores[face_filename] = freq_score
            
            # Save debug visualization if output directory provided
            if output_debug_dir:
                debug_filename = face_filename.replace('.jpg', '_freq.png')
                debug_path = os.path.join(output_debug_dir, debug_filename)
                self.save_frequency_map(str(face_path), debug_path)
        
        return scores

