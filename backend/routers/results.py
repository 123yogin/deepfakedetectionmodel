"""
Results router for retrieving detection results from SQLite database.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from backend.utils.database import db

router = APIRouter(tags=["results"])


@router.get("/result/{job_id}")
async def get_result(job_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific result by job ID from database.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Complete result dictionary
    """
    result = db.get_result(job_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Result not found for job_id: {job_id}")
    
        return result


@router.get("/latest-result")
async def get_latest_result() -> Dict[str, Any]:
    """
    Retrieve the most recent result from database.
    
    Returns:
        Most recent result dictionary
    """
    result = db.get_latest_result()
    
    if not result:
        raise HTTPException(status_code=404, detail="No results found")
    
    return result


@router.get("/jobs")
async def list_jobs(limit: int = 100) -> List[Dict[str, Any]]:
    """
    List all jobs with basic information.
    
    Args:
        limit: Maximum number of jobs to return (default: 100)
        
    Returns:
        List of job dictionaries
    """
    return db.list_jobs(limit=limit)
