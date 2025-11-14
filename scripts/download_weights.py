"""
Script to download pretrained model weights for deepfake detection.
Supports downloading from multiple sources.
"""
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Model weights URLs (public sources)
WEIGHT_URLS = {
    "cnn": {
        "url": None,  # Will be set to actual URL if available
        "filename": "xception_deepfake.pth",
        "description": "Xception-based deepfake detection model"
    },
    "temporal": {
        "url": None,
        "filename": "temporal_3dcnn.pth",
        "description": "3D-CNN temporal deepfake detection model"
    },
    "lipsync": {
        "url": None,
        "filename": "syncnet.pth",
        "description": "SyncNet lip-sync detection model"
    }
}

# Weights directory
WEIGHTS_DIR = Path(__file__).parent.parent / "models" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filepath: Path, description: str = "") -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        filepath: Path to save file
        description: Description for progress bar
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {description}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"[OK] Downloaded to {filepath}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial download
        return False


def check_weights_exist() -> dict:
    """
    Check which weights already exist.
    
    Returns:
        Dictionary with status of each weight file
    """
    status = {}
    for key, info in WEIGHT_URLS.items():
        filepath = WEIGHTS_DIR / info["filename"]
        status[key] = {
            "exists": filepath.exists(),
            "path": str(filepath),
            "size": filepath.stat().st_size if filepath.exists() else 0
        }
    return status


def print_weights_status():
    """Print current status of model weights."""
    print("\n" + "=" * 60)
    print("Model Weights Status")
    print("=" * 60)
    
    status = check_weights_exist()
    
    for key, info in WEIGHT_URLS.items():
        weight_status = status[key]
        if weight_status["exists"]:
            size_mb = weight_status["size"] / (1024 * 1024)
            print(f"[OK] {info['description']}")
            print(f"     File: {weight_status['path']}")
            print(f"     Size: {size_mb:.2f} MB")
        else:
            print(f"[MISSING] {info['description']}")
            print(f"          File: {info['filename']}")
        print()
    
    print("=" * 60)


def download_weights_interactive():
    """
    Interactive function to download weights.
    Shows available options and downloads selected weights.
    """
    print("\n" + "=" * 60)
    print("Model Weights Downloader")
    print("=" * 60)
    print("\nThis script helps you download pretrained model weights.")
    print("Note: Actual weight URLs need to be configured.")
    print("\nOptions:")
    print("1. Check current weights status")
    print("2. Download CNN weights (Xception)")
    print("3. Download Temporal weights (3D-CNN)")
    print("4. Download LipSync weights (SyncNet)")
    print("5. Download all available weights")
    print("6. Exit")
    
    while True:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            print_weights_status()
        
        elif choice == "2":
            info = WEIGHT_URLS["cnn"]
            if info["url"]:
                filepath = WEIGHTS_DIR / info["filename"]
                if download_file(info["url"], filepath, info["description"]):
                    print(f"[SUCCESS] CNN weights downloaded!")
            else:
                print("[INFO] CNN weights URL not configured.")
                print("Please download manually from:")
                print("  - FaceForensics++ repository")
                print("  - Celeb-DF models")
                print(f"  - Place in: {filepath}")
        
        elif choice == "3":
            info = WEIGHT_URLS["temporal"]
            if info["url"]:
                filepath = WEIGHTS_DIR / info["filename"]
                if download_file(info["url"], filepath, info["description"]):
                    print(f"[SUCCESS] Temporal weights downloaded!")
            else:
                print("[INFO] Temporal weights URL not configured.")
                print("Please download manually from:")
                print("  - 3D-CNN/I3D repositories")
                print(f"  - Place in: {filepath}")
        
        elif choice == "4":
            info = WEIGHT_URLS["lipsync"]
            if info["url"]:
                filepath = WEIGHTS_DIR / info["filename"]
                if download_file(info["url"], filepath, info["description"]):
                    print(f"[SUCCESS] LipSync weights downloaded!")
            else:
                print("[INFO] LipSync weights URL not configured.")
                print("Please download manually from:")
                print("  - SyncNet repository")
                print(f"  - Place in: {filepath}")
        
        elif choice == "5":
            print("\nDownloading all available weights...")
            for key, info in WEIGHT_URLS.items():
                if info["url"]:
                    filepath = WEIGHTS_DIR / info["filename"]
                    download_file(info["url"], filepath, info["description"])
            print_weights_status()
        
        elif choice == "6":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1-6.")


def create_weights_guide():
    """Create a guide for manually adding weights."""
    guide_path = WEIGHTS_DIR / "README_WEIGHTS.md"
    
    guide_content = """# Model Weights Guide

## How to Add Pretrained Weights

### Option 1: Manual Download

1. **CNN Weights (Xception)**
   - Download from: FaceForensics++ or Celeb-DF repositories
   - File: `xception_deepfake.pth`
   - Place in: `models/weights/xception_deepfake.pth`

2. **Temporal Weights (3D-CNN)**
   - Download from: 3D-CNN/I3D repositories
   - File: `temporal_3dcnn.pth`
   - Place in: `models/weights/temporal_3dcnn.pth`

3. **LipSync Weights (SyncNet)**
   - Download from: SyncNet repository
   - File: `syncnet.pth`
   - Place in: `models/weights/syncnet.pth`

### Option 2: Use Download Script

Run:
```bash
python scripts/download_weights.py
```

### Option 3: Train Your Own

Use the training scripts in `training/` directory to train your own models.

## Weight File Formats

- PyTorch: `.pth`, `.pt` files
- Should contain model state_dict
- Compatible with model architectures in `models/architectures.py`

## Verification

After adding weights, restart the server. You should see:
```
[OK] Loaded trained Xception model from models/weights/xception_deepfake.pth
```

If you see warnings, check:
- File exists and is readable
- Model architecture matches weights
- File is not corrupted
"""
    
    guide_path.write_text(guide_content)
    print(f"[OK] Created weights guide at {guide_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Deepfake Detection - Model Weights Manager")
    print("=" * 60)
    
    # Create weights guide
    create_weights_guide()
    
    # Show current status
    print_weights_status()
    
    # Interactive download
    download_weights_interactive()

