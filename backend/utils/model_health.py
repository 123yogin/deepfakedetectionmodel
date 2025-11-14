"""
Model health monitoring and performance tracking.
"""
import time
from typing import Dict, List, Optional
from collections import defaultdict
import torch


class ModelHealthMonitor:
    """Monitor model health and performance."""
    
    def __init__(self):
        self.metrics = {
            "predictions": 0,
            "total_time": 0.0,
            "errors": 0,
            "model_loads": 0,
            "cache_hits": 0
        }
        self.prediction_times = []
        self.error_log = []
    
    def record_prediction(self, duration: float, success: bool = True):
        """Record a prediction attempt."""
        self.metrics["predictions"] += 1
        self.metrics["total_time"] += duration
        self.prediction_times.append(duration)
        
        if not success:
            self.metrics["errors"] += 1
    
    def record_model_load(self):
        """Record a model load event."""
        self.metrics["model_loads"] += 1
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        avg_time = (
            self.metrics["total_time"] / self.metrics["predictions"]
            if self.metrics["predictions"] > 0
            else 0.0
        )
        
        p95_time = (
            sorted(self.prediction_times)[int(len(self.prediction_times) * 0.95)]
            if self.prediction_times
            else 0.0
        )
        
        return {
            "total_predictions": self.metrics["predictions"],
            "total_errors": self.metrics["errors"],
            "error_rate": (
                self.metrics["errors"] / self.metrics["predictions"]
                if self.metrics["predictions"] > 0
                else 0.0
            ),
            "avg_prediction_time": avg_time,
            "p95_prediction_time": p95_time,
            "total_time": self.metrics["total_time"],
            "model_loads": self.metrics["model_loads"]
        }
    
    def print_health_report(self):
        """Print a health report."""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("Model Health Report")
        print("=" * 60)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Error Rate: {stats['error_rate'] * 100:.2f}%")
        print(f"Average Prediction Time: {stats['avg_prediction_time'] * 1000:.2f} ms")
        print(f"P95 Prediction Time: {stats['p95_prediction_time'] * 1000:.2f} ms")
        print(f"Model Loads: {stats['model_loads']}")
        print("=" * 60 + "\n")


# Global health monitor instance
health_monitor = ModelHealthMonitor()


def check_model_health() -> Dict[str, bool]:
    """
    Check health of all models.
    
    Returns:
        Dictionary with health status for each model
    """
    from models.xception_detector import XceptionDeepfakeDetector
    from models.temporal_detector import TemporalDetector
    from models.lipsync_detector import LipSyncDetector
    from models.frequency_detector import FrequencyDetector
    
    health_status = {}
    
    # Check CNN detector
    try:
        cnn = XceptionDeepfakeDetector()
        health_status["cnn"] = {
            "loaded": cnn.model_loaded,
            "device": str(cnn.device),
            "status": "healthy" if cnn.model_loaded else "no_weights"
        }
    except Exception as e:
        health_status["cnn"] = {
            "loaded": False,
            "error": str(e),
            "status": "error"
        }
    
    # Check Temporal detector
    try:
        temporal = TemporalDetector()
        health_status["temporal"] = {
            "loaded": temporal.model_loaded,
            "device": str(temporal.device),
            "status": "healthy" if temporal.model_loaded else "no_weights"
        }
    except Exception as e:
        health_status["temporal"] = {
            "loaded": False,
            "error": str(e),
            "status": "error"
        }
    
    # Check LipSync detector
    try:
        lipsync = LipSyncDetector()
        health_status["lipsync"] = {
            "loaded": lipsync.model_loaded,
            "device": str(lipsync.device),
            "status": "healthy" if lipsync.model_loaded else "no_weights"
        }
    except Exception as e:
        health_status["lipsync"] = {
            "loaded": False,
            "error": str(e),
            "status": "error"
        }
    
    # Check Frequency detector (always works)
    try:
        freq = FrequencyDetector()
        health_status["frequency"] = {
            "loaded": True,
            "status": "healthy"
        }
    except Exception as e:
        health_status["frequency"] = {
            "loaded": False,
            "error": str(e),
            "status": "error"
        }
    
    return health_status


def print_model_health():
    """Print model health status."""
    print("\n" + "=" * 60)
    print("Model Health Check")
    print("=" * 60)
    
    health = check_model_health()
    
    for model_name, status in health.items():
        if status["status"] == "healthy":
            print(f"[OK] {model_name.upper()}: Healthy")
            if "device" in status:
                print(f"     Device: {status['device']}")
        elif status["status"] == "no_weights":
            print(f"[WARNING] {model_name.upper()}: No weights loaded")
        else:
            print(f"[ERROR] {model_name.upper()}: {status.get('error', 'Unknown error')}")
    
    print("=" * 60 + "\n")
    
    return health

