# Model Weights Setup Guide

## Current Status

Your models are currently using **untrained weights (random initialization)**. This means predictions are not reliable for actual deepfake detection.

## Quick Solution: Download Pretrained Weights

### Option 1: FaceForensics++ (Recommended for CNN)

1. **Visit the repository:**
   - Go to: https://github.com/ondyari/FaceForensics
   - Or search for "FaceForensics++ pretrained models"

2. **Download Xception weights:**
   - Look for pretrained model downloads
   - Download Xception-based deepfake detection weights
   - Save as: `models/weights/xception_deepfake.pth`

3. **Verify:**
   ```bash
   python scripts/diagnose_model_predictions.py
   ```

### Option 2: Use ImageNet Pretrained + Fine-tune

Your code already loads ImageNet pretrained weights for the base architecture. However, the final classification layers are still random.

**To improve:**
- Fine-tune on a deepfake dataset
- Or download deepfake-specific weights (Option 1)

### Option 3: Manual Download from Multiple Sources

#### CNN/Xception Weights:
- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics
- **Kaggle DeepFake Challenge**: https://www.kaggle.com/c/deepfake-detection-challenge

#### Temporal/3D-CNN Weights:
- **I3D (Inflated 3D ConvNet)**: https://github.com/deepmind/kinetics-i3d
- **FaceForensics++ temporal models**

#### LipSync/SyncNet Weights:
- **SyncNet**: https://github.com/joonson/syncnet_python
- **Wav2Lip**: https://github.com/Rudrabha/Wav2Lip

## File Locations

After downloading, place weights in:
- CNN: `models/weights/xception_deepfake.pth`
- Temporal: `models/weights/temporal_3dcnn.pth`
- LipSync: `models/weights/syncnet.pth`

## Verification

After adding weights, run:
```bash
python scripts/diagnose_model_predictions.py
```

You should see:
- `[OK] MODEL HAS TRAINED WEIGHTS` for each model
- Higher prediction variance (indicating learned features)
- Predictions outside the 0.3-0.7 compressed range

## Priority Order

1. **Start with CNN weights** (most important for basic detection)
2. Add temporal weights (for better video analysis)
3. Add lip-sync weights (for audio-visual consistency)

## Training Your Own Models

If you have a deepfake dataset, you can train your own models:

1. Prepare dataset (real and fake videos/images)
2. Use training scripts (when available)
3. Train models on your specific data
4. Save weights to `models/weights/` directory

## Troubleshooting

**Problem**: Weights not loading
- Check file path matches exactly
- Verify file is not corrupted
- Check file format (should be PyTorch .pth or .pt)

**Problem**: Predictions still seem random
- Run diagnostic: `python scripts/diagnose_model_predictions.py`
- Check if `model_loaded` flag is True
- Verify weight file size (should be > 100MB for CNN)

**Problem**: Architecture mismatch
- Ensure weights match your model architecture
- Check model configuration in `backend/config/model_config.py`

## Next Steps

1. Download at least CNN weights for immediate improvement
2. Run diagnostic to verify: `python scripts/diagnose_model_predictions.py`
3. Restart your application
4. Test with real videos to see improved accuracy

