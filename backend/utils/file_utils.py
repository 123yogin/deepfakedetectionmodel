"""
File utility functions for handling file operations.
"""
import os
import uuid
import re
from pathlib import Path
from typing import Optional


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove invalid characters for Windows.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for Windows
    """
    # Remove or replace invalid characters for Windows
    # Invalid: < > : " / \ | ? *
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "file"
    
    return sanitized


def ensure_directory_exists(directory_path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    try:
        # Try relative path first
        dir_path = Path(directory_path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if e.errno == 22:  # Invalid argument
            # Try with absolute path
            try:
                abs_path = os.path.abspath(directory_path)
                Path(abs_path).mkdir(parents=True, exist_ok=True)
            except OSError as e2:
                # Last resort: try creating parent directories one by one
                parts = Path(directory_path).parts
                current = Path(parts[0] if parts else ".")
                for part in parts[1:]:
                    current = current / part
                    try:
                        if not current.exists():
                            current.mkdir(exist_ok=True)
                    except OSError:
                        # If we can't create, maybe it already exists or we don't have permission
                        if not current.exists():
                            raise
        else:
            raise


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
    # Sanitize original filename
    safe_filename = sanitize_filename(original_filename or "video.mp4")
    
    # Extract file extension from sanitized filename
    file_extension = Path(safe_filename).suffix or ".mp4"
    
    # Generate very short unique filename (8 chars only)
    unique_filename = f"{uuid.uuid4().hex[:8]}{file_extension}"
    
    # Use simple relative path approach (tested and working)
    upload_path = Path(upload_dir)
    
    # Ensure directory exists
    try:
        upload_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[WARNING] Could not create directory {upload_dir}: {e}")
        # Try absolute path
        abs_upload_dir = os.path.abspath(upload_dir)
        Path(abs_upload_dir).mkdir(parents=True, exist_ok=True)
        upload_path = Path(abs_upload_dir)
    
    # Create file path
    file_path_obj = upload_path / unique_filename
    file_path_str = str(file_path_obj)
    
    # Write file directly (simplest approach)
    try:
        with open(file_path_str, "wb") as f:
            f.write(file_content)
        
        # Verify file was written
        if not file_path_obj.exists():
            raise OSError(22, "File was not created after write")
        
        # Return absolute path
        result_path = os.path.abspath(file_path_str)
        print(f"[INFO] File saved successfully: {result_path} (length: {len(result_path)})")
        return result_path
        
    except OSError as e:
        if e.errno == 22:
            # Provide detailed diagnostics
            abs_path = os.path.abspath(file_path_str)
            error_info = {
                "relative_path": file_path_str,
                "relative_length": len(file_path_str),
                "absolute_path": abs_path,
                "absolute_length": len(abs_path),
                "directory_exists": upload_path.exists(),
                "directory_writable": os.access(str(upload_path), os.W_OK) if upload_path.exists() else False,
                "error": str(e)
            }
            print(f"[ERROR] File save failed: {error_info}")
            
            raise OSError(
                22,
                f"Failed to save file. Path details:\n"
                f"  Relative: {file_path_str} ({len(file_path_str)} chars)\n"
                f"  Absolute: {abs_path} ({len(abs_path)} chars)\n"
                f"  Directory exists: {upload_path.exists()}\n"
                f"  Directory writable: {error_info['directory_writable']}\n"
                f"  Error: {e.strerror if hasattr(e, 'strerror') else str(e)}\n"
                f"  If path is too long, move project to shorter location (e.g., C:\\hack)"
            ) from e
        raise


def _save_with_strategy(upload_dir: str, unique_filename: str, file_content: bytes, use_relative: bool = True) -> str:
    """Internal function to try saving with a specific strategy."""
    if use_relative:
        # Use relative path (shortest) - don't resolve
        upload_path = Path(upload_dir)
    else:
        # Use absolute path but don't resolve yet
        upload_path = Path(os.path.abspath(upload_dir))
    
    # Ensure directory exists
    try:
        upload_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if e.errno == 22:
            raise
        # Try creating with absolute path
        abs_path = os.path.abspath(upload_dir)
        Path(abs_path).mkdir(parents=True, exist_ok=True)
        upload_path = Path(abs_path)
    
    # Create file path
    file_path_obj = upload_path / unique_filename
    
    # Check if path would be too long (use string representation, not resolve)
    full_path_str = str(file_path_obj)
    if len(full_path_str) > 250:  # Leave some margin
        # Try to get actual length without resolve
        try:
            actual_path = os.path.abspath(str(file_path_obj))
            if len(actual_path) > 250:
                raise OSError(22, "Path too long")
        except:
            raise OSError(22, "Path too long")
    
    # Write file - use direct file writing with explicit error handling
    file_path_str = str(file_path_obj)
    
    # Debug: Print path info
    print(f"[DEBUG] Attempting to save file:")
    print(f"  Path: {file_path_str}")
    print(f"  Path length: {len(file_path_str)}")
    print(f"  Absolute: {os.path.abspath(file_path_str)}")
    print(f"  Absolute length: {len(os.path.abspath(file_path_str))}")
    print(f"  Directory exists: {upload_path.exists()}")
    
    try:
        # Ensure directory exists one more time
        if not upload_path.exists():
            upload_path.mkdir(parents=True, exist_ok=True)
    
        # Try writing with explicit mode
        with open(file_path_str, "wb") as f:
            f.write(file_content)
        
        # Verify file was written
        if not Path(file_path_str).exists():
            raise OSError(22, "File was not created after write")
        
        # Return absolute path for consistency
        result_path = os.path.abspath(file_path_str)
        print(f"[DEBUG] File saved successfully to: {result_path}")
        return result_path
        
    except OSError as e:
        error_details = {
            "errno": e.errno,
            "strerror": e.strerror,
            "filename": file_path_str,
            "path_length": len(file_path_str),
            "abs_path_length": len(os.path.abspath(file_path_str)) if file_path_str else 0
        }
        print(f"[ERROR] File write failed: {error_details}")
        
        if e.errno == 22:
            # Provide detailed error message
            raise OSError(
                22,
                f"Invalid file path (errno 22). Details:\n"
                f"  Path: {file_path_str}\n"
                f"  Path length: {len(file_path_str)} chars\n"
                f"  Absolute path length: {len(os.path.abspath(file_path_str))} chars\n"
                f"  Directory exists: {upload_path.exists()}\n"
                f"  Directory writable: {os.access(str(upload_path), os.W_OK) if upload_path.exists() else 'N/A'}\n"
                f"  Error: {e.strerror}\n"
                f"Possible solutions:\n"
                f"  1. Check if path contains invalid characters\n"
                f"  2. Ensure directory is writable\n"
                f"  3. Try running as administrator\n"
                f"  4. Move project to shorter path (C:\\hack)"
            ) from e
        raise

