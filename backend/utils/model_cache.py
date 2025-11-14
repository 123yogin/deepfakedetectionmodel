"""
Model caching utility to avoid reloading models on each request.
"""
from typing import Optional
from models.xception_detector import XceptionDeepfakeDetector
from models.temporal_detector import TemporalDetector
from models.lipsync_detector import LipSyncDetector
from models.frequency_detector import FrequencyDetector


class ModelCache:
    """Singleton cache for model instances."""
    
    _instance = None
    _cnn_detector: Optional[XceptionDeepfakeDetector] = None
    _temporal_detector: Optional[TemporalDetector] = None
    _lipsync_detector: Optional[LipSyncDetector] = None
    _frequency_detector: Optional[FrequencyDetector] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, enable_optimizations: bool = True):
        """
        Initialize all models (call on startup).
        
        Args:
            enable_optimizations: If True, enables mixed precision, TorchScript, etc.
        """
        if self._initialized:
            return
        
        print("[INFO] Initializing model cache...")
        
        try:
            self._cnn_detector = XceptionDeepfakeDetector()
            if enable_optimizations:
                self._apply_optimizations(self._cnn_detector, "CNN")
            print("[OK] CNN detector cached")
        except Exception as e:
            print(f"[WARNING] Failed to initialize CNN detector: {e}")
        
        try:
            # Use GPU if available for temporal detector
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._temporal_detector = TemporalDetector(device=device)
            if enable_optimizations:
                self._apply_optimizations(self._temporal_detector, "Temporal")
            print("[OK] Temporal detector cached")
        except Exception as e:
            print(f"[WARNING] Failed to initialize Temporal detector: {e}")
        
        try:
            self._lipsync_detector = LipSyncDetector()
            if enable_optimizations:
                self._apply_optimizations(self._lipsync_detector, "LipSync")
            print("[OK] LipSync detector cached")
        except Exception as e:
            print(f"[WARNING] Failed to initialize LipSync detector: {e}")
        
        try:
            self._frequency_detector = FrequencyDetector()
            print("[OK] Frequency detector cached")
        except Exception as e:
            print(f"[WARNING] Failed to initialize Frequency detector: {e}")
        
        self._initialized = True
        print("[OK] Model cache initialized")
    
    def _apply_optimizations(self, detector, name: str):
        """Apply performance optimizations to a detector."""
        import torch
        
        if detector is None:
            return
        
        try:
            # Enable mixed precision on GPU
            if hasattr(detector, 'enable_mixed_precision'):
                detector.enable_mixed_precision()
        except Exception as e:
            print(f"[WARNING] Failed to enable mixed precision for {name}: {e}")
        
        # Note: TorchScript and quantization can be enabled via config if needed
        # They're disabled by default as they may not work with all models
    
    def get_cnn_detector(self) -> XceptionDeepfakeDetector:
        """Get cached CNN detector or create new one."""
        if self._cnn_detector is None:
            self._cnn_detector = XceptionDeepfakeDetector()
        return self._cnn_detector
    
    def get_temporal_detector(self) -> TemporalDetector:
        """Get cached Temporal detector or create new one."""
        if self._temporal_detector is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._temporal_detector = TemporalDetector(device=device)
        return self._temporal_detector
    
    def get_lipsync_detector(self) -> LipSyncDetector:
        """Get cached LipSync detector or create new one."""
        if self._lipsync_detector is None:
            self._lipsync_detector = LipSyncDetector()
        return self._lipsync_detector
    
    def get_frequency_detector(self) -> FrequencyDetector:
        """Get cached Frequency detector or create new one."""
        if self._frequency_detector is None:
            self._frequency_detector = FrequencyDetector()
        return self._frequency_detector
    
    def clear_cache(self):
        """Clear all cached models (for testing/reloading)."""
        self._cnn_detector = None
        self._temporal_detector = None
        self._lipsync_detector = None
        self._frequency_detector = None
        self._initialized = False


# Global cache instance
model_cache = ModelCache()

