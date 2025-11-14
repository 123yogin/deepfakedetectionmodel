# Detailed Weight Download Instructions

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
