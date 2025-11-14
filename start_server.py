"""
Startup script for the Deepfake Detection System.
Starts the FastAPI backend server and automatically opens the frontend.
"""
import uvicorn
import sys
import os
import webbrowser
import time
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

# Frontend server port
FRONTEND_PORT = 8080

def start_frontend_server():
    """Start a simple HTTP server for the frontend."""
    try:
        frontend_dir = Path(__file__).parent / "frontend"
        original_dir = os.getcwd()
        
        # Change to frontend directory
        os.chdir(str(frontend_dir))
        
        # Create server
        httpd = socketserver.TCPServer(("", FRONTEND_PORT), SimpleHTTPRequestHandler)
        print(f"  [OK] Frontend server started on http://localhost:{FRONTEND_PORT}")
        
        # Serve forever
        httpd.serve_forever()
    except OSError as e:
        os.chdir(original_dir)
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            print(f"  [WARNING] Port {FRONTEND_PORT} already in use. Frontend may already be running.")
        else:
            print(f"  [ERROR] Could not start frontend server: {e}")
    except Exception as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        print(f"  [ERROR] Frontend server error: {e}")

def open_frontend():
    """Open the frontend in the default browser."""
    # Wait a moment for servers to start
    time.sleep(2)
    
    frontend_url = f"http://localhost:{FRONTEND_PORT}"
    print(f"  [OK] Opening frontend in browser: {frontend_url}")
    webbrowser.open(frontend_url)

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    
    print("=" * 60)
    print("Deepfake Detection System - Starting Full Application")
    print("=" * 60)
    print(f"Backend Server: http://{host}:{port}")
    print(f"Frontend Server: http://localhost:{FRONTEND_PORT}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print("=" * 60)
    print("\nStarting servers...\n")
    
    # Start frontend server in a separate thread
    frontend_thread = threading.Thread(target=start_frontend_server, daemon=True)
    frontend_thread.start()
    
    # Open frontend in browser after a short delay
    browser_thread = threading.Thread(target=open_frontend, daemon=True)
    browser_thread.start()
    
    print("Press Ctrl+C to stop all servers\n")
    print("=" * 60)
    print("Servers running! Frontend will open automatically...")
    print("=" * 60)
    print()
    
    try:
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        print("Server stopped.")
        sys.exit(0)

