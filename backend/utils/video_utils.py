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
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        output_pattern,
        "-y"  # Overwrite output files if they exist
    ]
    
    try:
        # Run FFmpeg command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Count extracted frames by listing files in output directory
        frame_files = list(Path(output_dir).glob("frame_*.jpg"))
        frame_count = len(frame_files)
        
        return frame_count
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg to use frame extraction.")

