# Deepfake Detection System

A comprehensive deepfake detection system with multiple detectors and ensemble fusion.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Or manually:**
```bash
python start_server.py
```

The server will start on `http://localhost:8000`

### 3. Open the Frontend

Simply open `frontend/index.html` in your web browser, or:

```bash
# Using Python HTTP server
cd frontend
python -m http.server 8080
# Then open http://localhost:8080
```

## 📋 System Architecture

```
Video Upload
    ↓
Frame Extraction (FFmpeg)
    ↓
Face Detection (MTCNN)
    ↓
┌─────────────────────────────────────┐
│  Multiple Detectors (Parallel)      │
├─────────────────────────────────────┤
│ • CNN (Xception/ResNet)             │
│ • Temporal (3D-CNN placeholder)     │
│ • Lip-Sync (SyncNet placeholder)    │
│ • Frequency (FFT-based)              │
└─────────────────────────────────────┘
    ↓
Ensemble Fusion
    ↓
Final Verdict
```

## 🎯 Features

### Detectors

1. **CNN Detector** - Spatial deepfake detection using Xception/ResNet architecture
2. **Temporal Detector** - Frame-to-frame consistency analysis
3. **Lip-Sync Detector** - Audio-visual synchronization check
4. **Frequency Detector** - FFT-based artifact detection

### Pipeline

- ✅ Video upload and processing
- ✅ Automatic frame extraction (1 fps)
- ✅ Face detection and cropping
- ✅ Multi-detector analysis
- ✅ Ensemble fusion with weighted averaging
- ✅ Comprehensive result reports

### API Endpoints

- `POST /upload-video` - Upload and analyze video
- `GET /result/{job_id}` - Get specific result
- `GET /latest-result` - Get most recent result
- `GET /docs` - API documentation (Swagger UI)

## 📁 Project Structure

```
repo/
├── backend/              # FastAPI backend
│   ├── main.py          # Application entry point
│   ├── routers/         # API routes
│   ├── utils/           # Utility functions
│   └── config/          # Configuration
├── models/              # ML models
│   ├── xception_detector.py
│   ├── lipsync_detector.py
│   ├── frequency_detector.py
│   └── temporal_detector.py
├── frontend/            # Web interface
│   └── index.html      # Main UI
├── storage/             # Uploaded files
├── results/             # Detection results
└── requirements.txt     # Python dependencies
```

## 🔧 Configuration

### Detector Weights (Ensemble)

Default weights in `backend/utils/ensemble.py`:
- CNN: 0.50
- Temporal: 0.25
- LipSync: 0.15
- Frequency: 0.10

### Thresholds

Configurable in `backend/config/detection_config.py`:
- `FACE_THRESHOLD = 0.5`
- `HIGH_THRESHOLD = 0.9`
- `MIN_FACES_FOR_DECISION = 2`

## 📊 Result Format

```json
{
  "job_id": "uuid",
  "video_path": "storage/uploads/uuid.mp4",
  "frames": 20,
  "faces": 18,
  "aggregation": {
    "max_score": 0.85,
    "mean_score": 0.58,
    "temporal_mean": 0.7,
    "temporal_max": 0.8,
    "lip_sync_score": 0.2,
    "frequency_score": 0.58
  },
  "verdict": {
    "final_score": 0.6906,
    "final_label": "LIKELY_MANIPULATED",
    "confidence": 0.737,
    "reason": ["high_max_score", "lip_sync_mismatch"]
  }
}
```

## 🧪 Testing

Run unit tests:
```bash
pytest backend/tests/ -v
```

## 📝 Notes

- **Placeholder Models**: Some detectors use placeholder implementations (temporal, lipsync). Replace with pretrained weights for production.
- **GPU Support**: Temporal and CNN detectors can use GPU if available (automatic detection).
- **FFmpeg Required**: Must be installed and in PATH for video processing.

## 🐛 Troubleshooting

### CORS Errors
- Backend is configured to allow CORS from any origin
- If issues persist, check `backend/main.py` CORS settings

### Import Errors
- Install all dependencies: `pip install -r requirements.txt`
- Ensure Python 3.8+ is being used

### FFmpeg Not Found
- Install FFmpeg: https://ffmpeg.org/download.html
- Add to system PATH

### Frontend Not Connecting
- Verify backend is running on `http://localhost:8000`
- Check browser console for errors
- Ensure CORS is enabled in backend

## 📚 API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔒 Production Considerations

- [ ] Add authentication/authorization
- [ ] Replace placeholder models with trained weights
- [ ] Implement rate limiting
- [ ] Add database for result persistence
- [ ] Set up proper logging
- [ ] Configure CORS for specific domains
- [ ] Add input validation and sanitization
- [ ] Implement file size limits
- [ ] Set up monitoring and alerts

## 📄 License

_To be determined._

## 🤝 Contributing

This is a step-by-step implementation following a structured development plan.

---

**Status**: ✅ Full application ready - Backend + Frontend integrated
