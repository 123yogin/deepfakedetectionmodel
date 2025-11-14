# Frontend Setup Guide

## Quick Start

The frontend is now ready to use! Here's how to set it up:

### Option 1: Direct File Open (Simplest)

1. **Start the backend:**
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Open the frontend:**
   - Navigate to the `frontend` folder
   - Double-click `index.html` to open in your browser
   - Or right-click → "Open with" → Your browser

### Option 2: Using a Local Server (Recommended)

1. **Start the backend:**
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Start a simple HTTP server for the frontend:**
   ```bash
   # From the project root
   cd frontend
   python -m http.server 8080
   ```

3. **Open in browser:**
   - Go to: `http://localhost:8080`

### Option 3: Using Node.js (if you have it)

```bash
# Install http-server globally
npm install -g http-server

# Run from frontend directory
cd frontend
http-server -p 8080
```

## Features

✅ **Video Upload**
- Drag & drop support
- Click to browse
- File size display

✅ **Real-time Progress**
- Upload progress bar
- Processing status

✅ **Results Display**
- Frame and face counts
- All detector scores (CNN, Temporal, LipSync, Frequency)
- Final verdict with confidence
- Color-coded verdict (Red=Manipulated, Green=Authentic, Yellow=Inconclusive)

## Troubleshooting

### CORS Errors
The backend is configured to allow CORS from any origin. If you still see CORS errors:
- Make sure the backend is running on `http://localhost:8000`
- Check browser console for specific error messages

### Connection Errors
- Verify backend is running: `curl http://localhost:8000/`
- Check API_URL in `index.html` matches your backend URL
- Ensure no firewall is blocking the connection

### File Upload Issues
- Check file size (large files may take time)
- Verify video format is supported (MP4, AVI, MOV, etc.)
- Check browser console for error details

## API Endpoints Used

- `POST /upload-video` - Upload and analyze video
- `GET /result/{job_id}` - Get specific result (not used in current UI, but available)

## Next Steps

For production, consider:
- Adding authentication
- Implementing result history
- Adding video preview
- Showing detection visualizations
- Exporting reports as PDF

