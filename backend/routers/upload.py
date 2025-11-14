"""
Upload router for handling video file uploads.
"""
import os
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any, List
import backend.utils.file_utils as file_utils
import backend.utils.video_utils as video_utils
import backend.utils.face_utils as face_utils
import backend.utils.mouth_cropper as mouth_cropper
import backend.utils.temporal_utils as temporal_utils
import backend.utils.aggregation as aggregation
import backend.utils.ensemble as ensemble
from backend.utils.model_cache import model_cache
from backend.utils.abnormality_analyzer import AbnormalityAnalyzer
from backend.utils.technique_identifier import TechniqueIdentifier


router = APIRouter(tags=["upload"])


@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a video file, extract frames, detect faces, run deepfake detection, and lip-sync analysis.
    
    Accepts multipart/form-data with a file field.
    Saves the file to storage/uploads/ with a unique UUID filename.
    Extracts frames to storage/frames/<uuid>/
    Extracts faces to storage/faces/<uuid>/
    Extracts mouth regions to storage/mouth/<uuid>/
    Extracts audio to storage/audio/<uuid>.wav
    Runs deepfake detection on each face.
    Runs lip-sync analysis.
    
    Returns:
        Dictionary with message, video path, frame count, face count, detections, and verdict
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Save file with unique name
        file_path = file_utils.save_uploaded_file(
            file_content=file_content,
            original_filename=file.filename or "video.mp4"
        )
        
        # Extract UUID from filename (filename format: <uuid>.<ext>)
        video_filename = Path(file_path).stem  # Gets filename without extension
        frames_output_dir = os.path.join("storage", "frames", video_filename)
        
        # Extract frames (1 frame per second)
        frame_count = video_utils.extract_frames(
            video_path=file_path,
            output_dir=frames_output_dir,
            fps=1
        )
        
        # Extract faces from frames
        faces_output_dir = os.path.join("storage", "faces", video_filename)
        face_count = face_utils.extract_faces_from_frames(
            frames_dir=frames_output_dir,
            output_dir=faces_output_dir
        )
        
        # Run deepfake detection on each face (use cached model)
        detector = model_cache.get_cnn_detector()
        detections: List[Dict[str, Any]] = []
        
        # Compute frequency scores for all faces (use cached model)
        freq_detector = model_cache.get_frequency_detector()
        frequency_debug_dir = os.path.join("results", video_filename, "frequency_maps")
        freq_scores_dict = {}
        
        if face_count > 0:
            # Batch compute frequency scores
            freq_scores_dict = freq_detector.batch_compute(
                faces_dir=faces_output_dir,
                output_debug_dir=frequency_debug_dir
            )
            
            # Get all face files sorted
            face_files = sorted(Path(faces_output_dir).glob("face_*.jpg"))
            face_paths = [str(f) for f in face_files]
            
            # Use batch prediction if available (much faster)
            if hasattr(detector, 'predict_batch'):
                print(f"[INFO] Using batch processing for {len(face_paths)} faces")
                fake_scores = detector.predict_batch(face_paths)
            else:
                print(f"[INFO] Using individual predictions for {len(face_paths)} faces")
                fake_scores = [detector.predict(path) for path in face_paths]
            
            # Create detections list
            for idx, (face_path, fake_score) in enumerate(zip(face_files, fake_scores)):
                face_filename = face_path.name
                
                # Get frequency score for this face
                freq_score = freq_scores_dict.get(face_filename, 0.5)
                
                # Extract frame number from face filename (face_0001.jpg -> frame 1)
                # For now, use the index as frame number (can be improved later)
                frame_num = idx + 1
                
                detections.append({
                    "face_file": face_filename,
                    "frame": frame_num,
                    "fake_score": round(fake_score, 4),  # Round to 4 decimal places
                    "freq_score": round(freq_score, 4)   # Add frequency score
                })
            
            # Sort detections by highest fake_score (most suspicious first)
            detections.sort(key=lambda x: x["fake_score"], reverse=True)
        
        # Extract mouth regions from faces for lip-sync analysis
        mouth_output_dir = os.path.join("storage", "mouth", video_filename)
        mouth_count = 0
        lip_sync_score = None
        
        if face_count > 0:
            mouth_count = mouth_cropper.extract_mouth_frames(
                faces_dir=faces_output_dir,
                output_dir=mouth_output_dir
            )
            
            # Extract audio from video
            audio_output_path = os.path.join("storage", "audio", f"{video_filename}.wav")
            try:
                print(f"[INFO] Initializing LipSync detector...")
                lipsync_detector = model_cache.get_lipsync_detector()
                print(f"[INFO] LipSync detector initialized, model: {lipsync_detector.model is not None}")
                
                print(f"[INFO] Extracting audio from video...")
                lipsync_detector.extract_audio(video_path=file_path, out_wav=audio_output_path)
                print(f"[INFO] Audio extracted to: {audio_output_path}")
                
                # Compute lip-sync score
                print(f"[INFO] Computing lip-sync score...")
                print(f"[INFO] Mouth frames dir: {mouth_output_dir}")
                print(f"[INFO] Audio path: {audio_output_path}")
                lip_sync_score = lipsync_detector.compute_sync_score(
                    mouth_frames_dir=mouth_output_dir,
                    audio_path=audio_output_path
                )
                print(f"[INFO] Lip-sync score computed: {lip_sync_score}")
            except Exception as e:
                import traceback
                print(f"[ERROR] Error in lip-sync analysis: {e}")
                print(f"[ERROR] Traceback:")
                traceback.print_exc()
                # Continue without lip-sync score if extraction fails
                lip_sync_score = None
        
        # Run temporal detection on face tracks
        temporal_mean = 0.5
        temporal_max = 0.5
        
        if face_count > 0:
            try:
                temporal_detector = model_cache.get_temporal_detector()
                # Group faces into tracks
                tracks = temporal_utils.group_faces_into_tracks(faces_output_dir)
                
                track_scores = []
                for track in tracks:
                    # Create temporary directory for track frames
                    # For simplicity, use the main faces directory
                    # In production, would create per-track subdirectories
                    track_result = temporal_detector.predict_for_face_track(
                        frames_dir=faces_output_dir,
                        clip_len=16,
                        stride=8
                    )
                    if track_result["clip_scores"]:
                        track_scores.extend(track_result["clip_scores"])
                
                if track_scores:
                    temporal_mean = sum(track_scores) / len(track_scores)
                    temporal_max = max(track_scores)
            except Exception as e:
                print(f"Error in temporal detection: {e}")
                # Use default values on error
        
        # Generate aggregation first
        job_id = video_filename  # Use UUID as job_id
        aggregation_result = aggregation.aggregate_scores(
            detections=detections,
            lip_sync_score=lip_sync_score,
            temporal_mean=temporal_mean,
            temporal_max=temporal_max
        )
        
        # Decide verdict using aggregation (for reasons/confidence) - DO THIS FIRST
        verdict_result = aggregation.decide_verdict(aggregation_result)
        
        # Apply ensemble combiner for final verdict
        ensemble_combiner = ensemble.EnsembleCombiner()
        ensemble_result = ensemble_combiner.combine(aggregation_result)
        
        # Override verdict label and final_score with ensemble results
        # BUT: Respect LIKELY_AUTHENTIC verdict when decide_verdict says so and ensemble score is low
        if verdict_result.get("label") == "LIKELY_AUTHENTIC":
            # If decide_verdict says authentic, trust it unless ensemble score is very high
            if ensemble_result["final_score"] < 0.6:  # Allow margin for ensemble
                verdict_result["final_score"] = ensemble_result["final_score"]
                verdict_result["final_label"] = "LIKELY_AUTHENTIC"  # Keep authentic verdict
                verdict_result["raw_score"] = ensemble_result["raw_score"]
            else:
                # If ensemble score is high despite decide_verdict, use ensemble but lower confidence
                verdict_result["final_score"] = ensemble_result["final_score"]
                verdict_result["final_label"] = ensemble_result["final_label"]
                verdict_result["raw_score"] = ensemble_result["raw_score"]
                verdict_result["confidence"] = min(verdict_result.get("confidence", 0.5), 0.6)  # Lower confidence
        else:
            # For other verdicts, use ensemble result
            verdict_result["final_score"] = ensemble_result["final_score"]
            verdict_result["final_label"] = ensemble_result["final_label"]
            verdict_result["raw_score"] = ensemble_result["raw_score"]
        
        # Generate abnormality report
        abnormality_report = None
        technique_report = None
        
        if face_count > 0:
            try:
                print(f"[INFO] Generating abnormality report...")
                abnormality_analyzer = AbnormalityAnalyzer()
                abnormality_report = abnormality_analyzer.generate_abnormality_report(
                    faces_dir=faces_output_dir,
                    detections=detections,
                    temporal_mean=temporal_mean,
                    temporal_max=temporal_max,
                    lip_sync_score=lip_sync_score
                )
                print(f"[INFO] Abnormality report generated: {len(abnormality_report.get('spatial_artifacts', []))} spatial artifacts detected")
                
                # Generate technique identification
                print(f"[INFO] Identifying deepfake creation technique...")
                technique_identifier = TechniqueIdentifier()
                technique_report = technique_identifier.generate_technique_report(
                    abnormality_report=abnormality_report,
                    aggregation=aggregation_result
                )
                if technique_report.get("primary_technique"):
                    print(f"[INFO] Primary technique identified: {technique_report['primary_technique']['name']} "
                          f"(confidence: {technique_report['primary_technique']['confidence']:.1%})")
                else:
                    print(f"[INFO] No specific technique identified with sufficient confidence")
            except Exception as e:
                import traceback
                print(f"[WARNING] Error generating abnormality/technique reports: {e}")
                print(f"[WARNING] Traceback:")
                traceback.print_exc()
                # Continue without reports if analysis fails
        
        # Generate complete result with ensemble verdict
        result = aggregation.generate_result(
            job_id=job_id,
            video_path=file_path,
            frames=frame_count,
            detections=detections,
            lip_sync_score=lip_sync_score,
            temporal_mean=temporal_mean,
            temporal_max=temporal_max,
            abnormality_report=abnormality_report,
            technique_report=technique_report
        )
        
        # Update with ensemble verdict (override the decide_verdict result)
        result["verdict"] = verdict_result
        
        # Save updated result with ensemble verdict to JSON
        result_file = Path("results") / f"{job_id}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Return result with all information
        response = {
            "message": "processed",
            "job_id": job_id,
            "video": file_path,
            "frames": frame_count,
            "faces": face_count,
            "detections": detections,
            "aggregation": result["aggregation"],
            "verdict": result["verdict"]
        }
        
        # Include abnormalities and techniques in response if available
        if abnormality_report:
            response["abnormalities"] = abnormality_report
        if technique_report:
            response["detected_techniques"] = technique_report
        
        return response
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"[ERROR] Upload video error: {error_msg}")
        print(f"[ERROR] Traceback:\n{error_traceback}")
        
        # Return a proper error response
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {error_msg}. Check server logs for details."
        )

