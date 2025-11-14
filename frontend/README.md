# Frontend - Deepfake Detection System

Simple HTML/JavaScript frontend for uploading videos and viewing detection results.

## Features

- 📹 Video upload (drag & drop or click to select)
- 📊 Real-time progress tracking
- 🎯 Display of all detector scores
- ⚖️ Final verdict with confidence scores
- 🎨 Modern, responsive UI

## Usage

1. **Start the backend server:**
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Open the frontend:**
   - Simply open `frontend/index.html` in your web browser
   - Or serve it using a simple HTTP server:
     ```bash
     # Python 3
     python -m http.server 8080 --directory frontend
     
     # Then open: http://localhost:8080
     ```

3. **Upload a video:**
   - Click the upload area or drag & drop a video file
   - Click "Upload & Analyze"
   - Wait for processing to complete
   - View the results

## API Configuration

The frontend connects to the backend API at `http://localhost:8000` by default.

To change the API URL, edit the `API_URL` constant in `index.html`:
```javascript
const API_URL = 'http://localhost:8000';
```

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Modern browsers with ES6 support

## Notes

- The frontend is a single-page application (SPA)
- No build process required - just open the HTML file
- For production, consider using a framework like React/Vue for better maintainability

