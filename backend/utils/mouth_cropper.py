"""
Mouth region cropping utilities.
Extracts mouth regions from face crops using facial landmarks.
"""
import os
from pathlib import Path
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import torch


def crop_mouth_from_face(face_img: Image.Image, mtcnn: MTCNN = None) -> Image.Image:
    """
    Crop mouth region from a face image using facial landmarks.
    
    Args:
        face_img: PIL Image of a face crop
        mtcnn: Optional MTCNN instance (will create if not provided)
        
    Returns:
        PIL Image of cropped mouth region, or original face if detection fails
    """
    if mtcnn is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=device
        )
    
    try:
        # Detect facial landmarks
        # MTCNN returns bounding boxes, but we need landmarks
        # For MVP, we'll use a simple heuristic based on face dimensions
        width, height = face_img.size
        
        # Mouth region is typically in the lower 1/3 of the face
        # and centered horizontally
        mouth_top = int(height * 0.5)  # Start from middle of face
        mouth_bottom = int(height * 0.85)  # End at 85% of face height
        mouth_left = int(width * 0.25)  # Start from 25% of width
        mouth_right = int(width * 0.75)  # End at 75% of width
        
        # Crop mouth region
        mouth_crop = face_img.crop((mouth_left, mouth_top, mouth_right, mouth_bottom))
        
        return mouth_crop
        
    except Exception as e:
        print(f"Error cropping mouth: {e}")
        # Return a default crop if detection fails
        width, height = face_img.size
        return face_img.crop((int(width * 0.25), int(height * 0.5), 
                             int(width * 0.75), int(height * 0.85)))


def extract_mouth_frames(faces_dir: str, output_dir: str) -> int:
    """
    Extract mouth regions from all face crops in a directory.
    
    Args:
        faces_dir: Directory containing face crop images
        output_dir: Directory to save mouth crops
        
    Returns:
        Number of mouth frames extracted
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize MTCNN (will be reused for all faces)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device=device
    )
    
    # Get all face files
    face_files = sorted(Path(faces_dir).glob("face_*.jpg"))
    
    mouth_count = 0
    
    for face_path in face_files:
        try:
            # Load face image
            face_img = Image.open(face_path).convert('RGB')
            
            # Crop mouth region
            mouth_img = crop_mouth_from_face(face_img, mtcnn)
            
            # Save mouth crop (normalize path for Windows)
            mouth_count += 1
            mouth_filename = f"mouth_{mouth_count:04d}.jpg"
            mouth_path = os.path.normpath(os.path.join(output_dir, mouth_filename))
            mouth_img.save(mouth_path, 'JPEG', quality=95)
            
        except Exception as e:
            print(f"Error processing face {face_path}: {e}")
            continue
    
    return mouth_count

