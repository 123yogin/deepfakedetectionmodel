"""
Script to add weight files to the models/weights directory.
"""
import shutil
import os
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config.model_config import TEMPORAL_MODEL_PATH, LIPSYNC_MODEL_PATH, CNN_MODEL_PATH, WEIGHTS_DIR

def validate_and_copy(source_path: str, target_path: Path, model_name: str):
    """Validate and copy a weight file."""
    source = Path(source_path)
    target = Path(target_path)
    
    print(f"\n{'='*70}")
    print(f"Adding {model_name} Weights")
    print(f"{'='*70}")
    
    # Check if source exists
    if not source.exists():
        print(f"[ERROR] Source file not found: {source_path}")
        return False
    
    # Get file info
    file_size_mb = source.stat().st_size / (1024 * 1024)
    print(f"\nSource file: {source_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Validate the file (try to load it)
    print(f"\n[VALIDATING] Checking weight file...")
    try:
        state_dict = torch.load(str(source), map_location='cpu')
        
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            num_params = len(state_dict)
            print(f"[OK] Valid PyTorch weight file")
            print(f"     Contains {num_params} parameter groups")
            if num_params > 0:
                print(f"     Sample keys: {list(state_dict.keys())[:3]}")
        else:
            print(f"[WARNING] File format may not be standard state_dict")
            print(f"         File will still be copied, but may need special handling")
            
    except Exception as e:
        print(f"[WARNING] Could not fully validate file: {e}")
        print(f"         File will still be copied, but may not be compatible")
        print(f"         Note: Some model formats (.model) may need special loading")
    
    # Ensure target directory exists
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy file
    print(f"\n[COPYING] Copying to: {target}")
    try:
        shutil.copy2(source, target)
        print(f"[SUCCESS] File copied successfully!")
        print(f"         Target: {target}")
        
        # Verify copy
        if target.exists():
            target_size = target.stat().st_size / (1024 * 1024)
            print(f"         Verified: {target_size:.2f} MB")
            return True
        else:
            print(f"[ERROR] Copy verification failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to copy file: {e}")
        return False

def add_all_weights():
    """Add all weight files."""
    results = {}
    
    # SyncNet weights
    syncnet_source = r"C:\Users\yogin\Downloads\new downloads\syncnet_v2.model"
    results['syncnet'] = validate_and_copy(
        syncnet_source,
        LIPSYNC_MODEL_PATH,
        "LipSync/SyncNet"
    )
    
    # SFD Face detection weights (might be for CNN or face detection)
    sfd_source = r"C:\Users\yogin\Downloads\new downloads\sfd_face.pth"
    # Try as CNN first, but note it might be for face detection
    results['sfd'] = validate_and_copy(
        sfd_source,
        CNN_MODEL_PATH,
        "CNN/SFD Face Detection"
    )
    
    return results

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADDING MODEL WEIGHTS")
    print("="*70)
    
    results = add_all_weights()
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    success_count = sum(1 for r in results.values() if r)
    total_count = len(results)
    
    for name, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {name.upper():15} {status}")
    
    if success_count == total_count:
        print(f"\n[SUCCESS] All {total_count} weight files added successfully!")
    elif success_count > 0:
        print(f"\n[PARTIAL] {success_count}/{total_count} weight files added successfully.")
    else:
        print(f"\n[ERROR] Failed to add weight files.")
    
    print(f"\n{'='*70}")
    print("Next Steps:")
    print(f"{'='*70}")
    print("1. Run: python scripts/quick_check_weights.py")
    print("2. Run: python scripts/diagnose_model_predictions.py")
    print("3. Restart your application")
    print(f"{'='*70}\n")
