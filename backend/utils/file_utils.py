"""
File utility functions for handling file operations.
"""
import os
import uuid
from pathlib import Path
from typing import Optional


def ensure_directory_exists(directory_path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def save_uploaded_file(file_content: bytes, original_filename: str, upload_dir: str = "storage/uploads") -> str:
    """
    Save an uploaded file with a unique UUID-based filename.
    
    Args:
        file_content: The file content as bytes
        original_filename: Original filename to extract extension
        upload_dir: Directory to save the file in
        
    Returns:
        The full path where the file was saved
    """
    # Ensure upload directory exists
    ensure_directory_exists(upload_dir)
    
    # Extract file extension from original filename
    file_extension = Path(original_filename).suffix
    
    # Generate unique filename using UUID
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Full path to save the file
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Write file
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return file_path

