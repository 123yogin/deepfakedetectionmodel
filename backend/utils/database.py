"""
SQLite database module for storing detection results.
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager


class Database:
    """SQLite database handler for detection results."""
    
    def __init__(self, db_path: str = "detection_results.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Jobs table - main job information
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    frames INTEGER NOT NULL,
                    faces INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Detections table - individual face detections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    face_file TEXT NOT NULL,
                    frame INTEGER NOT NULL,
                    fake_score REAL NOT NULL,
                    freq_score REAL NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
            """)
            
            # Aggregations table - aggregated scores
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS aggregations (
                    job_id TEXT PRIMARY KEY,
                    total_faces INTEGER NOT NULL,
                    max_score REAL NOT NULL,
                    mean_score REAL NOT NULL,
                    median_score REAL NOT NULL,
                    p90_score REAL NOT NULL,
                    count_above_0_5 INTEGER NOT NULL,
                    frequency_score REAL,
                    lip_sync_score REAL,
                    temporal_mean REAL,
                    temporal_max REAL,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
            """)
            
            # Verdicts table - final verdict information
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verdicts (
                    job_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    final_score REAL,
                    final_label TEXT,
                    raw_score REAL,
                    reasons TEXT,  -- JSON array of reasons
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
            """)
            
            # Abnormalities table - abnormality reports (stored as JSON)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS abnormalities (
                    job_id TEXT PRIMARY KEY,
                    report_data TEXT,  -- JSON string
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
            """)
            
            # Techniques table - technique identification reports (stored as JSON)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS techniques (
                    job_id TEXT PRIMARY KEY,
                    report_data TEXT,  -- JSON string
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
            """)
            
            # Report metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS report_meta (
                    job_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_versions TEXT,  -- JSON string
                    notes TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_job_id ON detections(job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def save_result(self, result: Dict[str, Any]) -> bool:
        """
        Save a complete result to the database.
        
        Args:
            result: Complete result dictionary from generate_result()
            
        Returns:
            True if successful
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                job_id = result["job_id"]
                
                # Save job
                cursor.execute("""
                    INSERT OR REPLACE INTO jobs (job_id, video_path, frames, faces, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    job_id,
                    result["video_path"],
                    result["frames"],
                    result["faces"],
                    datetime.now().isoformat()
                ))
                
                # Save detections (delete old ones first)
                cursor.execute("DELETE FROM detections WHERE job_id = ?", (job_id,))
                for detection in result.get("detections", []):
                    cursor.execute("""
                        INSERT INTO detections (job_id, face_file, frame, fake_score, freq_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        job_id,
                        detection["face_file"],
                        detection["frame"],
                        detection["fake_score"],
                        detection.get("freq_score", 0.5)
                    ))
                
                # Save aggregation
                agg = result.get("aggregation", {})
                cursor.execute("""
                    INSERT OR REPLACE INTO aggregations (
                        job_id, total_faces, max_score, mean_score, median_score,
                        p90_score, count_above_0_5, frequency_score,
                        lip_sync_score, temporal_mean, temporal_max
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_id,
                    agg.get("total_faces", 0),
                    agg.get("max_score", 0.0),
                    agg.get("mean_score", 0.0),
                    agg.get("median_score", 0.0),
                    agg.get("p90_score", 0.0),
                    agg.get("count_above_0.5", 0),
                    agg.get("frequency_score"),
                    agg.get("lip_sync_score"),
                    agg.get("temporal_mean"),
                    agg.get("temporal_max")
                ))
                
                # Save verdict
                verdict = result.get("verdict", {})
                reasons_json = json.dumps(verdict.get("reason", []))
                cursor.execute("""
                    INSERT OR REPLACE INTO verdicts (
                        job_id, label, confidence, final_score, final_label, raw_score, reasons
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_id,
                    verdict.get("label", "UNKNOWN"),
                    verdict.get("confidence", 0.0),
                    verdict.get("final_score"),
                    verdict.get("final_label"),
                    verdict.get("raw_score"),
                    reasons_json
                ))
                
                # Save abnormalities if present
                if "abnormalities" in result:
                    cursor.execute("""
                        INSERT OR REPLACE INTO abnormalities (job_id, report_data)
                        VALUES (?, ?)
                    """, (job_id, json.dumps(result["abnormalities"])))
                else:
                    cursor.execute("DELETE FROM abnormalities WHERE job_id = ?", (job_id,))
                
                # Save techniques if present
                if "detected_techniques" in result:
                    cursor.execute("""
                        INSERT OR REPLACE INTO techniques (job_id, report_data)
                        VALUES (?, ?)
                    """, (job_id, json.dumps(result["detected_techniques"])))
                else:
                    cursor.execute("DELETE FROM techniques WHERE job_id = ?", (job_id,))
                
                # Save report metadata
                meta = result.get("report_meta", {})
                model_versions_json = json.dumps(meta.get("model_versions", {}))
                cursor.execute("""
                    INSERT OR REPLACE INTO report_meta (job_id, timestamp, model_versions, notes)
                    VALUES (?, ?, ?, ?)
                """, (
                    job_id,
                    meta.get("timestamp", datetime.now().isoformat()),
                    model_versions_json,
                    meta.get("notes")
                ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"[ERROR] Failed to save result to database: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete result by job_id.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Complete result dictionary or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get job
                cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                job_row = cursor.fetchone()
                if not job_row:
                    return None
                
                # Get detections
                cursor.execute("""
                    SELECT face_file, frame, fake_score, freq_score
                    FROM detections WHERE job_id = ?
                    ORDER BY frame
                """, (job_id,))
                detections = [
                    {
                        "face_file": row["face_file"],
                        "frame": row["frame"],
                        "fake_score": row["fake_score"],
                        "freq_score": row["freq_score"]
                    }
                    for row in cursor.fetchall()
                ]
                
                # Get aggregation
                cursor.execute("SELECT * FROM aggregations WHERE job_id = ?", (job_id,))
                agg_row = cursor.fetchone()
                aggregation = {}
                if agg_row:
                    aggregation = {
                        "total_faces": agg_row["total_faces"],
                        "max_score": agg_row["max_score"],
                        "mean_score": agg_row["mean_score"],
                        "median_score": agg_row["median_score"],
                        "p90_score": agg_row["p90_score"],
                        "count_above_0.5": agg_row["count_above_0_5"],
                        "frequency_score": agg_row["frequency_score"],
                        "lip_sync_score": agg_row["lip_sync_score"],
                        "temporal_mean": agg_row["temporal_mean"],
                        "temporal_max": agg_row["temporal_max"]
                    }
                
                # Get verdict
                cursor.execute("SELECT * FROM verdicts WHERE job_id = ?", (job_id,))
                verdict_row = cursor.fetchone()
                verdict = {}
                if verdict_row:
                    reasons = json.loads(verdict_row["reasons"]) if verdict_row["reasons"] else []
                    verdict = {
                        "label": verdict_row["label"],
                        "confidence": verdict_row["confidence"],
                        "final_score": verdict_row["final_score"],
                        "final_label": verdict_row["final_label"],
                        "raw_score": verdict_row["raw_score"],
                        "reason": reasons
                    }
                
                # Get abnormalities
                cursor.execute("SELECT report_data FROM abnormalities WHERE job_id = ?", (job_id,))
                abn_row = cursor.fetchone()
                abnormalities = None
                if abn_row and abn_row["report_data"]:
                    abnormalities = json.loads(abn_row["report_data"])
                
                # Get techniques
                cursor.execute("SELECT report_data FROM techniques WHERE job_id = ?", (job_id,))
                tech_row = cursor.fetchone()
                techniques = None
                if tech_row and tech_row["report_data"]:
                    techniques = json.loads(tech_row["report_data"])
                
                # Get report metadata
                cursor.execute("SELECT * FROM report_meta WHERE job_id = ?", (job_id,))
                meta_row = cursor.fetchone()
                report_meta = {}
                if meta_row:
                    model_versions = json.loads(meta_row["model_versions"]) if meta_row["model_versions"] else {}
                    report_meta = {
                        "timestamp": meta_row["timestamp"],
                        "model_versions": model_versions,
                        "notes": meta_row["notes"]
                    }
                
                # Build complete result
                result = {
                    "job_id": job_row["job_id"],
                    "video_path": job_row["video_path"],
                    "frames": job_row["frames"],
                    "faces": job_row["faces"],
                    "detections": detections,
                    "aggregation": aggregation,
                    "verdict": verdict,
                    "report_meta": report_meta
                }
                
                if abnormalities:
                    result["abnormalities"] = abnormalities
                if techniques:
                    result["detected_techniques"] = techniques
                
                return result
        except Exception as e:
            print(f"[ERROR] Failed to get result from database: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent result.
        
        Returns:
            Most recent result dictionary or None if no results exist
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT job_id FROM jobs
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    return self.get_result(row["job_id"])
                return None
        except Exception as e:
            print(f"[ERROR] Failed to get latest result: {e}")
            return None
    
    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all jobs with basic information.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job dictionaries with basic info
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT j.job_id, j.video_path, j.frames, j.faces, j.created_at,
                           v.label, v.confidence, v.final_score
                    FROM jobs j
                    LEFT JOIN verdicts v ON j.job_id = v.job_id
                    ORDER BY j.created_at DESC
                    LIMIT ?
                """, (limit,))
                
                return [
                    {
                        "job_id": row["job_id"],
                        "video_path": row["video_path"],
                        "frames": row["frames"],
                        "faces": row["faces"],
                        "created_at": row["created_at"],
                        "verdict_label": row["label"],
                        "verdict_confidence": row["confidence"],
                        "final_score": row["final_score"]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            print(f"[ERROR] Failed to list jobs: {e}")
            return []


# Global database instance
db = Database()

