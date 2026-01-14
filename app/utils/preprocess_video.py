import cv2
import os
import base64
from pathlib import Path
from typing import Optional, List
from app.utils.logger import logger


def extract_frames_from_video(
    video_path: str,
    frame_interval: int = 30,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extract frames from a video file and return them as base64 encoded strings.
    
    Args:
        video_path (str): Path to the input video file
        frame_interval (int): Extract every Nth frame (default: 30, i.e., 1 frame per second at 30fps)
        max_frames (Optional[int]): Maximum number of frames to extract (default: None, extract all)
    
    Returns:
        List[str]: List of base64 encoded frame images (JPEG format)
    
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video cannot be opened or read
    """
    # Validate input video path
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
    
    base64_frames = []
    frame_count = 0
    extracted_count = 0
    
    try:
        while True:
            # Read the next frame
            success, frame = video_capture.read()
            
            if not success:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Encode frame to JPEG format
                success, buffer = cv2.imencode('.jpg', frame)
                
                if success:
                    # Convert to base64
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    base64_frames.append(frame_base64)
                    extracted_count += 1
                    
                    logger.info(f"Extracted frame {extracted_count} at frame position {frame_count}")
                    
                    # Check if we've reached the maximum number of frames
                    if max_frames and extracted_count >= max_frames:
                        logger.info(f"Reached maximum frame limit: {max_frames}")
                        break
                else:
                    logger.warning(f"Failed to encode frame {frame_count}")
            
            frame_count += 1
    
    finally:
        # Release the video capture object
        video_capture.release()
    
    logger.info(f"Frame extraction complete. Extracted {extracted_count} frames from {frame_count} total frames")
    
    return base64_frames


def extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float]
) -> List[str]:
    """
    Extract frames from a video at specific timestamps and return as base64.
    
    Args:
        video_path (str): Path to the input video file
        timestamps (List[float]): List of timestamps (in seconds) to extract frames
    
    Returns:
        List[str]: List of base64 encoded frame images (JPEG format)
    
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video cannot be opened or read
    """
    # Validate input video path
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    logger.info(f"Processing video: {video_path} (FPS: {fps:.2f})")
    
    base64_frames = []
    
    try:
        for idx, timestamp in enumerate(sorted(timestamps)):
            # Calculate frame number from timestamp
            frame_number = int(timestamp * fps)
            
            # Set the video to the specific frame
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read the frame
            success, frame = video_capture.read()
            
            if success:
                # Encode frame to JPEG format
                success_encode, buffer = cv2.imencode('.jpg', frame)
                
                if success_encode:
                    # Convert to base64
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    base64_frames.append(frame_base64)
                    logger.info(f"Extracted frame at timestamp {timestamp}s")
                else:
                    logger.warning(f"Failed to encode frame at timestamp {timestamp}s")
            else:
                logger.warning(f"Failed to extract frame at timestamp {timestamp}s")
    
    finally:
        # Release the video capture object
        video_capture.release()
    
    logger.info(f"Timestamp extraction complete. Extracted {len(base64_frames)} frames")
    
    return base64_frames


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Dictionary containing video information (fps, frame_count, duration, width, height)
    
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video cannot be opened
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        info = {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}"
        }
        
        return info
    
    finally:
        video_capture.release()
