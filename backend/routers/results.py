"""
Results router for retrieving detection results.
"""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(tags=["results"])


@router.get("/result/{job_id}")
async def get_result(job_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific result by job ID.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Complete result JSON
    """
    result_file = Path("results") / f"{job_id}.json"
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"Result not found for job_id: {job_id}")
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result file: {str(e)}")


@router.get("/latest-result")
async def get_latest_result() -> Dict[str, Any]:
    """
    Retrieve the most recent result file.
    
    Returns:
        Most recent result JSON
    """
    results_dir = Path("results")
    
    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="No results directory found")
    
    # Find all JSON files and get the most recent one
    result_files = list(results_dir.glob("*.json"))
    
    if not result_files:
        raise HTTPException(status_code=404, detail="No results found")
    
    # Sort by modification time (most recent first)
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_file = result_files[0]
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result file: {str(e)}")

