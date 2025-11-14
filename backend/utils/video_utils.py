"""
Video processing utilities using FFmpeg.
"""
import subprocess
import os
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: int = 1) -> int:
    """
    Extracts frames from the video using FFmpeg.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        fps: Number of frames per second to extract (default: 1)
        
    Returns:
        Total number of extracted frames
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command to extract frames
    # -i: input file
    # -vf fps=1: extract 1 frame per second
    # -q:v 2: high quality JPEG
    # frame_%04d.jpg: output filename pattern (frame_0001.jpg, frame_0002.jpg, etc.)
    # Normalize paths for Windows compatibility
    output_dir = os.path.normpath(output_dir)
    video_path = os.path.normpath(video_path)
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    # Convert to forward slashes for FFmpeg (works on both Windows and Unix)
    output_pattern = output_pattern.replace("\\", "/")
    video_path_ffmpeg = video_path.replace("\\", "/")
    
    command = [
        "ffmpeg",
        "-i", video_path_ffmpeg,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        output_pattern,
        "-y"  # Overwrite output files if they exist
    ]
    
    try:
        print(f"[DEBUG] FFmpeg command: {' '.join(command)}")
        print(f"[DEBUG] Video path: {video_path}")
        print(f"[DEBUG] Output directory: {output_dir}")
        
        # Run FFmpeg command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        print(f"[DEBUG] FFmpeg stdout: {result.stdout[:200] if result.stdout else 'None'}")
        if result.stderr:
            print(f"[DEBUG] FFmpeg stderr: {result.stderr[:200]}")
        
        # Count extracted frames by listing files in output directory
        frame_files = list(Path(output_dir).glob("frame_*.jpg"))
        frame_count = len(frame_files)
        
        if frame_count == 0:
            print(f"[WARNING] No frames extracted. Checking output directory: {output_dir}")
            if Path(output_dir).exists():
                all_files = list(Path(output_dir).glob("*"))
                print(f"[WARNING] Files in directory: {[f.name for f in all_files]}")
        
        return frame_count
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg command timed out after 5 minutes. Video might be too long or corrupted.")
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error (code {e.returncode}): {e.stderr or e.stdout or 'Unknown error'}"
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg) from e
    except FileNotFoundError:
        error_msg = "FFmpeg not found. Please install FFmpeg to use frame extraction."
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error in frame extraction: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg) from e

