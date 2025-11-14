"""
Automated weight download script with direct links and instructions.
Helps download pretrained weights from various sources.
"""
import sys
import os
import requests
from pathlib import Path
from tqdm import tqdm
import webbrowser

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config.model_config import WEIGHTS_DIR, CNN_MODEL_PATH, TEMPORAL_MODEL_PATH, LIPSYNC_MODEL_PATH

# Known repositories and download sources
REPOSITORIES = {
    "cnn": {
        "name": "CNN/Xception Deepfake Detection",
        "sources": [
            {
                "name": "DeepfakeBench (Recommended)",
                "url": "https://github.com/SCLBD/DeepfakeBench",
                "description": "Multiple pretrained models including Xception",
                "direct_download": False,
                "instructions": "Visit repository, download weights from releases or weights folder"
            },
            {
                "name": "FaceForensics++",
                "url": "https://github.com/ondyari/FaceForensics",
                "description": "Original FaceForensics++ repository",
                "direct_download": False,
                "instructions": "Check repository for pretrained model downloads"
            },
            {
                "name": "Deepfake Detection Project v4",
                "url": "https://github.com/ameencaslam/deepfake-detection-project-v4",
                "description": "Pre-trained models on Google Drive",
                "direct_download": False,
                "instructions": "Models hosted on Google Drive - follow repository instructions"
            }
        ],
        "target_file": CNN_MODEL_PATH
    },
    "temporal": {
        "name": "Temporal/3D-CNN",
        "sources": [
            {
                "name": "I3D (Inflated 3D ConvNet)",
                "url": "https://github.com/deepmind/kinetics-i3d",
                "description": "Pretrained I3D models on Kinetics dataset",
                "direct_download": False,
                "instructions": "Download I3D weights and fine-tune for deepfake detection"
            },
            {
                "name": "DeepfakeBench",
                "url": "https://github.com/SCLBD/DeepfakeBench",
                "description": "May include temporal models",
                "direct_download": False,
                "instructions": "Check repository for temporal detection models"
            }
        ],
        "target_file": TEMPORAL_MODEL_PATH
    },
    "lipsync": {
        "name": "LipSync/SyncNet",
        "sources": [
            {
                "name": "SyncNet Python",
                "url": "https://github.com/joonson/syncnet_python",
                "description": "Original SyncNet implementation",
                "direct_download": False,
                "instructions": "Download pretrained SyncNet weights from repository"
            },
            {
                "name": "Wav2Lip",
                "url": "https://github.com/Rudrabha/Wav2Lip",
                "description": "Wav2Lip uses SyncNet - weights may be compatible",
                "direct_download": False,
                "instructions": "Check for SyncNet weights in Wav2Lip repository"
            }
        ],
        "target_file": LIPSYNC_MODEL_PATH
    }
}


def download_from_url(url: str, filepath: Path, description: str = "") -> bool:
    """Download file from URL with progress bar."""
    try:
        print(f"\nDownloading {description}...")
        print(f"URL: {url}")
        print(f"Save to: {filepath}")
        
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
            filepath.unlink()
        return False


def show_download_options(model_key: str):
    """Show download options for a specific model."""
    model_info = REPOSITORIES[model_key]
    
    print(f"\n{'='*70}")
    print(f"Download Options for: {model_info['name']}")
    print(f"{'='*70}")
    print(f"\nTarget file: {model_info['target_file']}")
    print(f"\nAvailable Sources:")
    
    for idx, source in enumerate(model_info['sources'], 1):
        print(f"\n{idx}. {source['name']}")
        print(f"   Description: {source['description']}")
        print(f"   URL: {source['url']}")
        print(f"   Instructions: {source['instructions']}")
    
    print(f"\n{'='*70}")
    
    # Ask if user wants to open browser
    choice = input("\nOpen browser to download? (y/n): ").strip().lower()
    if choice == 'y':
        print("\nOpening repositories in browser...")
        for source in model_info['sources']:
            webbrowser.open(source['url'])
            input(f"Opened {source['name']}. Press Enter to open next...")
    
    # Check if file was downloaded
    if os.path.exists(model_info['target_file']):
        print(f"\n[SUCCESS] File found at {model_info['target_file']}")
        return True
    else:
        print(f"\n[INFO] Please download the weights manually and place at:")
        print(f"  {model_info['target_file']}")
        return False


def interactive_download():
    """Interactive download interface."""
    print("\n" + "="*70)
    print("AUTOMATED WEIGHT DOWNLOAD HELPER")
    print("="*70)
    print("\nThis script helps you download pretrained model weights.")
    print("Most weights require manual download from GitHub repositories.")
    
    while True:
        print("\n" + "="*70)
        print("Select Model to Download:")
        print("="*70)
        print("1. CNN/Xception Deepfake Detection (HIGHEST PRIORITY)")
        print("2. Temporal/3D-CNN Model")
        print("3. LipSync/SyncNet Model")
        print("4. Show all download sources")
        print("5. Check current weights status")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            show_download_options("cnn")
        elif choice == "2":
            show_download_options("temporal")
        elif choice == "3":
            show_download_options("lipsync")
        elif choice == "4":
            print("\n" + "="*70)
            print("ALL DOWNLOAD SOURCES")
            print("="*70)
            for key, model_info in REPOSITORIES.items():
                print(f"\n{model_info['name']}:")
                for source in model_info['sources']:
                    print(f"  - {source['name']}: {source['url']}")
        elif choice == "5":
            from scripts.quick_check_weights import quick_check
            quick_check()
        elif choice == "6":
            print("\nExiting...")
            print("\nRemember:")
            print("  1. Download weights from the repositories")
            print("  2. Place them in models/weights/ directory")
            print("  3. Run: python scripts/quick_check_weights.py")
            print("  4. Run: python scripts/diagnose_model_predictions.py")
            break
        else:
            print("Invalid choice. Please enter 1-6.")


def create_download_instructions():
    """Create a markdown file with detailed download instructions."""
    instructions = """# Detailed Weight Download Instructions

## Priority 1: CNN/Xception Weights (MOST IMPORTANT)

### Option A: DeepfakeBench (Easiest)
1. Visit: https://github.com/SCLBD/DeepfakeBench
2. Look for "Releases" or "weights" folder
3. Download Xception-based deepfake detection weights
4. Save as: `models/weights/xception_deepfake.pth`

### Option B: FaceForensics++
1. Visit: https://github.com/ondyari/FaceForensics
2. Check README for pretrained model links
3. Download Xception weights
4. Save as: `models/weights/xception_deepfake.pth`

### Option C: Deepfake Detection Project v4
1. Visit: https://github.com/ameencaslam/deepfake-detection-project-v4
2. Follow instructions for Google Drive download
3. Extract Xception weights
4. Save as: `models/weights/xception_deepfake.pth`

## Priority 2: Temporal/3D-CNN Weights

1. Visit: https://github.com/deepmind/kinetics-i3d
2. Download I3D pretrained weights
3. (Optional) Fine-tune on deepfake dataset
4. Save as: `models/weights/temporal_3dcnn.pth`

## Priority 3: LipSync/SyncNet Weights

1. Visit: https://github.com/joonson/syncnet_python
2. Download pretrained SyncNet weights
3. Save as: `models/weights/syncnet.pth`

## After Downloading

1. Verify files exist:
   ```bash
   python scripts/quick_check_weights.py
   ```

2. Run diagnostic:
   ```bash
   python scripts/diagnose_model_predictions.py
   ```

3. Restart your application

## File Size Expectations

- CNN weights: Usually 50-200 MB
- Temporal weights: Usually 100-500 MB
- LipSync weights: Usually 10-50 MB

If files are much smaller, they may be corrupted or incomplete.
"""
    
    instructions_path = Path(__file__).parent.parent / "DOWNLOAD_INSTRUCTIONS.md"
    instructions_path.write_text(instructions)
    print(f"\n[OK] Created detailed instructions at: {instructions_path}")


if __name__ == "__main__":
    create_download_instructions()
    interactive_download()

