"""
Face detection and cropping utilities using MTCNN.
"""
import os
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
import torch


def extract_faces_from_frames(frames_dir: str, output_dir: str) -> int:
    """
    Detects faces in all frames inside frames_dir,
    crops them, and saves them into output_dir.
    
    Args:
        frames_dir: Directory containing frame images
        output_dir: Directory to save cropped faces
        
    Returns:
        Number of cropped faces extracted
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize MTCNN face detector
    # Keep it on CPU for simplicity (can move to GPU later)
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
    
    # Get all frame files sorted
    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    
    face_count = 0
    
    for frame_path in frame_files:
        try:
            # Load image
            img = Image.open(frame_path).convert('RGB')
            
            # Detect faces
            # MTCNN returns bounding boxes and probabilities
            boxes, probs = mtcnn.detect(img)
            
            # Skip if no faces detected
            if boxes is None or len(boxes) == 0:
                continue
            
            # Process each detected face
            # For MVP, we'll take the first (largest) face if multiple detected
            for idx, (box, prob) in enumerate(zip(boxes, probs)):
                # Only process faces with high confidence
                if prob < 0.9:
                    continue
                
                # Crop face from image
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure coordinates are within image bounds
                width, height = img.size
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Crop the face
                face_img = img.crop((x1, y1, x2, y2))
                
                # Save cropped face (normalize path for Windows)
                face_count += 1
                face_filename = f"face_{face_count:04d}.jpg"
                face_path = os.path.normpath(os.path.join(output_dir, face_filename))
                face_img.save(face_path, 'JPEG', quality=95)
                
                # For MVP, only extract the first face per frame
                break
                
        except Exception as e:
            # Skip frames that cause errors
            print(f"Error processing frame {frame_path}: {e}")
            continue
    
    return face_count

