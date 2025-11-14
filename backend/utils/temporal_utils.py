"""
Temporal utilities for frame sampling and face tracking.
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image


def group_faces_into_tracks(faces_dir: str, max_distance: float = 20.0) -> List[List[str]]:
    """
    Group face crops into tracks based on sequential proximity.
    Simple tracker: faces in consecutive frames with similar positions form a track.
    
    Args:
        faces_dir: Directory containing face images
        max_distance: Maximum pixel distance for faces to be in same track
        
    Returns:
        List of tracks, where each track is a list of face filenames
    """
    face_files = sorted(Path(faces_dir).glob("face_*.jpg"))
    
    if not face_files:
        return []
    
    tracks = []
    current_track = [face_files[0].name]
    
    for i in range(1, len(face_files)):
        prev_file = face_files[i-1]
        curr_file = face_files[i]
        
        # Extract frame numbers from filenames (face_0001.jpg -> 1)
        try:
            prev_frame = int(prev_file.stem.split('_')[1])
            curr_frame = int(curr_file.stem.split('_')[1])
        except (ValueError, IndexError):
            # If parsing fails, assume sequential
            prev_frame = i - 1
            curr_frame = i
        
        # Check if frames are consecutive or close
        frame_diff = curr_frame - prev_frame
        
        if frame_diff <= 2:  # Consecutive or very close frames
            current_track.append(curr_file.name)
        else:
            # Start new track
            if current_track:
                tracks.append(current_track)
            current_track = [curr_file.name]
    
    # Add final track
    if current_track:
        tracks.append(current_track)
    
    return tracks


def sample_clips_from_track(track_frames: List[str], clip_len: int = 16, stride: int = 8) -> List[List[str]]:
    """
    Sample clips from a face track.
    
    Args:
        track_frames: List of frame filenames in the track
        clip_len: Number of frames per clip
        stride: Stride between clips
        
    Returns:
        List of clips, where each clip is a list of frame filenames
    """
    if len(track_frames) < clip_len:
        return [track_frames] if track_frames else []
    
    clips = []
    for start_idx in range(0, len(track_frames) - clip_len + 1, stride):
        clip = track_frames[start_idx:start_idx + clip_len]
        clips.append(clip)
    
    return clips

