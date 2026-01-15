"""
Spare Part Recommendation API endpoints for spare part recommendation using Claude AI.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import base64
import json
import os
import re
import tempfile
import time
from typing import List
from pathlib import Path
from app.services.claude_client import claude_client
from app.services.voyage_client import voyage_client
from app.services.supabase_client import supabase_vector_client
from app.utils.logger import logger
from app.config import settings
from app.utils.preprocess_video import extract_frames_from_video


def clean_json_response(response_text: str) -> str:
    """
    Clean JSON response by removing markdown code fences if present.
    
    Args:
        response_text: Raw response text from Claude
        
    Returns:
        Cleaned JSON string
    """
    # Remove markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = response_text.strip()
    
    # Check if response starts with code fence
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
        # Remove closing fence (```)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    
    return cleaned.strip()


async def generate_embedding_with_rate_limit(
    voyage_client_instance,
    search_text: str,
    delay_seconds: float = 20.0,
    max_retries: int = 3
) -> list:
    """
    Generate embedding with rate limiting and retry logic.
    
    Voyage AI free tier has limits of:
    - 3 RPM (requests per minute) = 1 request every 20 seconds
    - 10K TPM (tokens per minute)
    
    Args:
        voyage_client_instance: Voyage client instance
        search_text: Text to generate embedding for
        delay_seconds: Delay between requests (default: 20 seconds for 3 RPM)
        max_retries: Maximum number of retries on rate limit errors
        
    Returns:
        Embedding vector
    """
    for attempt in range(max_retries):
        try:
            # Generate the embedding
            embedding = voyage_client_instance.generate_query_embedding(search_text)
            
            # Add delay after successful request to respect rate limits
            if delay_seconds > 0:
                logger.info(f"Rate limit buffer: waiting {delay_seconds} seconds before next request")
                await asyncio.sleep(delay_seconds)
            
            return embedding
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if "rate limit" in error_msg or "rpm" in error_msg or "tpm" in error_msg:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait longer on each retry
                    wait_time = delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait_time} seconds before retry..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Rate limit error persists after {max_retries} attempts: {e}")
                    raise HTTPException(
                        status_code=429,
                        detail=f"Voyage AI rate limit exceeded. Please try again later or add payment method. Error: {str(e)}"
                    )
            else:
                # Not a rate limit error, raise immediately
                logger.error(f"Error generating embedding: {e}")
                raise
    
    # Should not reach here, but just in case
    raise HTTPException(
        status_code=500,
        detail="Failed to generate embedding after multiple retries"
    )
# Create router for police report endpoints
router = APIRouter(prefix="/spare/part/recommendation", tags=["Spare Part Recommendation"])

@router.post("/spare-part-recommendation")
async def analyze_spare_part_recommendation(
    files: List[UploadFile] = File(..., description="Image or video file(s) to analyze")
):
    """
    Analyze image file(s) and return recommendation spare parts for repair.
    
    This endpoint accepts image files, analyzes them using Claude AI, and returns
    a JSON response with spare part recommendations for vehicle damage repair.
    
    Args:
        files: List of image files to analyze (supports PNG, JPEG, JPG)
        
    Returns:
        JSON response with analysis results including damaged parts and recommendations
    """
    try:
        # Validate file types
        allowed_extensions = ('.png', '.jpg', '.jpeg', '.mp4', '.mov', '.avi', '.mkv', '.webm')
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="One or more files have no filename"
                )
            
            if not file.filename.lower().endswith(allowed_extensions):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a supported image format. Only PNG, JPEG, and JPG files are supported."
                )
        
        # Check if files are provided
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No files provided. Please upload at least one image file."
            )
        
        # Convert files to base64 (supports both images and videos)
        base64_data_list = []
        
        if any(file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')) for file in files):
            logger.info("Processing video file(s) for spare part recommendation")
            
            # Get the first video file
            video_file = files[0]
            temp_video_path = None
            
            try:
                # Read the uploaded video content
                video_content = await video_file.read()
                
                if not video_content:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Video file {video_file.filename} is empty"
                    )
                
                # Create a temporary file to save the video
                # Use the original file extension
                file_extension = Path(video_file.filename).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_video_path = temp_file.name
                    temp_file.write(video_content)
                
                logger.info(f"Saved uploaded video to temporary file: {temp_video_path}")
                
                # Extract frames from the video as base64
                base64_frames = extract_frames_from_video(
                    video_path=temp_video_path,
                    frame_interval=30,  # Extract 1 frame per second (assuming 30 fps)
                    max_frames=10  # Limit to 10 frames to avoid too many images for Claude
                )
                
                logger.info(f"Successfully extracted {len(base64_frames)} frames from video")
                
                # Add all extracted frames to base64_data_list
                base64_data_list.extend(base64_frames)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing video file: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing video: {str(e)}"
                )
            finally:
                # Clean up: Remove the temporary video file
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                        logger.info(f"Cleaned up temporary video file: {temp_video_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary video file {temp_video_path}: {e}")
        else:
            logger.info("Processing image file(s) for spare part recommendation")
            # Convert image files to base64
            for file in files:
                try:
                    # Read file content
                    file_content = await file.read()
                    
                    # Validate that file is not empty
                    if not file_content:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File {file.filename} is empty"
                        )
            
                    # Convert to base64
                    base64_data = base64.b64encode(file_content).decode('utf-8')
                    base64_data_list.append(base64_data)
                    
                    logger.info(f"Successfully encoded {file.filename} to base64")
                    
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {e}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing file {file.filename}: {str(e)}"
                    )
        
        # Process with Claude client
        logger.info("Sending image files to Claude for spare part recommendation")
        result = claude_client.analyze_spare_part_recommendation(base64_data_list)
        
        if result["success"]:
            logger.info("Claude spare part recommendation analysis completed successfully")
            
            # Check if response exists and is not empty
            response_text = result.get("response")
            if not response_text:
                logger.error("Claude returned empty response")
                raise HTTPException(
                    status_code=500,
                    detail="Claude returned an empty response. Please try again."
                )
            
            # Try to parse the JSON response
            try:
                # First, try to parse as-is
                try:
                    parsed_response = json.loads(response_text)
                except json.JSONDecodeError:
                    # If that fails, clean up markdown code fences and try again
                    logger.info("Initial JSON parse failed, attempting to clean markdown code fences")
                    cleaned_response = clean_json_response(response_text)
                    parsed_response = json.loads(cleaned_response)
                    logger.info("Successfully parsed JSON after cleaning markdown code fences")

                # Extract damaged parts from Claude response
                damaged_parts = parsed_response.get("damaged_parts", [])
                vehicle_info = parsed_response.get("vehicle_identification", {})
                
                if not damaged_parts:
                    logger.warning("No damaged parts found in Claude response")
                    return JSONResponse(
                        status_code=200,
                        content={
                            "success": True,
                            "analysis": parsed_response,
                            "spare_part_recommendations": [],
                            "usage": result.get("usage"),
                            "message": "Analysis completed, but no damaged parts identified"
                        }
                    )
                
                # Check if clients are available
                if voyage_client is None:
                    logger.error("Voyage AI client not initialized")
                    raise HTTPException(
                        status_code=500,
                        detail="Voyage AI client not initialized. Please configure VOYAGE_API_KEY."
                    )
                
                if supabase_vector_client is None:
                    logger.error("Supabase client not initialized")
                    raise HTTPException(
                        status_code=500,
                        detail="Supabase client not initialized. Please configure SUPABASE_URL and SUPABASE_KEY."
                    )
                
                logger.info(f"Processing {len(damaged_parts)} damaged parts for spare part recommendations")
                logger.info(f"Note: Rate limiting applied - 20 second delay between Voyage AI requests (3 RPM limit)")
                
                # Process each damaged part to find matching spare parts
                spare_part_recommendations = []
                
                for idx, damaged_part in enumerate(damaged_parts, 1):
                    try:
                        # Create text representation for embedding
                        part_name = damaged_part.get("part_name", "")
                        part_category = damaged_part.get("part_category", "")
                        damage_type = damaged_part.get("damage_type", "")
                        damage_description = damaged_part.get("damage_description", "")
                        vehicle_type = vehicle_info.get("vehicle_type", "")
                        brand = vehicle_info.get("brand", "")
                        
                        # Combine relevant information for embedding search
                        search_text = f"{part_name} {part_category} {damage_type} {vehicle_type} {brand}".strip()
                        
                        logger.info(f"Processing damaged part {idx}/{len(damaged_parts)}: {part_name}")
                        
                        # Generate query embedding with rate limiting
                        # 20 seconds delay for 3 RPM limit (except for last item)
                        delay = 20.0 if idx < len(damaged_parts) else 0.0
                        query_embedding = await generate_embedding_with_rate_limit(
                            voyage_client,
                            search_text,
                            delay_seconds=delay
                        )
                        
                        # Query spare parts database using search_motorcycle_parts RPC function
                        # Map vehicle info to filters: brand -> brand_filter, make/model -> make_filter
                        # Note: Filters are exact match - if brand doesn't match exactly, no results will be returned
                        # For now, we'll try without filters first, then with filters if needed
                        brand_filter = brand if brand else None
                        make_filter = vehicle_info.get("model") or vehicle_info.get("make") or None
                        
                        # Note: region_filter and max_price can be added later if needed
                        matched_parts = []
                        rpc_error = None
                        
                        try:
                            # First attempt: Try with filters if available
                            if brand_filter or make_filter:
                                logger.debug(f"Querying spare parts with filters: brand={brand_filter}, make={make_filter}")
                                search_result = supabase_vector_client.client.rpc(
                                    'search_motorcycle_parts',
                                    {
                                        'query_embedding': query_embedding,
                                        'match_limit': 5,  # Top 5 matches per part
                                        'brand_filter': brand_filter,
                                        'make_filter': make_filter,
                                        'region_filter': None,
                                        'max_price': None
                                    }
                                ).execute()
                                
                                matched_parts = search_result.data if search_result.data else []
                                
                                # If no results with filters, try without filters (broader search)
                                if not matched_parts:
                                    logger.warning(f"No results with filters (brand={brand_filter}, make={make_filter}). Trying without filters...")
                                    search_result = supabase_vector_client.client.rpc(
                                        'search_motorcycle_parts',
                                        {
                                            'query_embedding': query_embedding,
                                            'match_limit': 10,  # Get more results when no filters
                                            'brand_filter': None,
                                            'make_filter': None,
                                            'region_filter': None,
                                            'max_price': None
                                        }
                                    ).execute()
                                    
                                    matched_parts = search_result.data if search_result.data else []
                                    if matched_parts:
                                        logger.info(f"Found {len(matched_parts)} results without filters. Filters may be too restrictive (exact match required).")
                            else:
                                # No filters, search broadly
                                logger.debug(f"Querying spare parts without filters")
                                search_result = supabase_vector_client.client.rpc(
                                    'search_motorcycle_parts',
                                    {
                                        'query_embedding': query_embedding,
                                        'match_limit': 10,  # Top 10 matches per part when no filters
                                        'brand_filter': None,
                                        'make_filter': None,
                                        'region_filter': None,
                                        'max_price': None
                                    }
                                ).execute()
                                
                                matched_parts = search_result.data if search_result.data else []
                            
                            logger.info(f"Successfully queried spare parts database, found {len(matched_parts)} matching parts for {part_name}")
                            
                            # Log warning if no results found
                            if not matched_parts:
                                logger.warning(
                                    f"No matching spare parts found for '{part_name}'. "
                                    f"Possible reasons: 1) Embeddings not generated for this part in database, "
                                    f"2) Embedding column is NULL, 3) Semantic similarity too low. "
                                    f"Please verify embeddings exist in motorcycle_spareparts table."
                                )
                            
                        except Exception as e:
                            rpc_error = e
                            error_str = str(e)
                            
                            # Safely extract error dict if available
                            error_dict = {}
                            if e.args and isinstance(e.args[0], dict):
                                error_dict = e.args[0]
                            elif hasattr(e, '__dict__'):
                                error_dict = e.__dict__
                            
                            # Check if it's a function not found error (PGRST202)
                            if 'PGRST202' in error_str or (error_dict.get('code') == 'PGRST202'):
                                logger.error(
                                    f"RPC function 'search_motorcycle_parts' not found. Please ensure the function is created in Supabase. "
                                    f"Run the SQL function from your schema file in Supabase SQL Editor, "
                                    f"then refresh the schema cache in Settings → API → Reload Schema."
                                )
                                logger.debug(f"Full error: {rpc_error}")
                            else:
                                logger.error(f"Error querying spare parts database for {part_name}: {rpc_error}")
                        
                        # Add recommendations for this damaged part
                        spare_part_recommendations.append({
                            "damaged_part": damaged_part,
                            "matching_spare_parts": matched_parts,
                            "recommendations_count": len(matched_parts),
                            "error": str(rpc_error) if not matched_parts and rpc_error else None,
                            "error_code": error_dict.get('code') if not matched_parts and rpc_error else None
                        })
                        
                        if not matched_parts and rpc_error:
                            logger.warning(f"No matching spare parts found for {part_name}. RPC function may need to be created.")
                            
                    except Exception as e:
                        logger.error(f"Error processing damaged part {idx}: {e}")
                        # Continue with other parts even if one fails
                        spare_part_recommendations.append({
                            "damaged_part": damaged_part,
                            "matching_spare_parts": [],
                            "recommendations_count": 0,
                            "error": str(e)
                        })
                
                logger.info(f"Completed spare part recommendations for {len(damaged_parts)} damaged parts")
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "analysis": parsed_response,
                        "spare_part_recommendations": spare_part_recommendations,
                        "summary": {
                            "total_damaged_parts": len(damaged_parts),
                            "parts_with_recommendations": sum(1 for rec in spare_part_recommendations if rec.get("recommendations_count", 0) > 0),
                            "total_recommendations": sum(rec.get("recommendations_count", 0) for rec in spare_part_recommendations)
                        },
                        "usage": result.get("usage"),
                        "message": "Spare part recommendation analysis completed successfully"
                    }
                )
                
            except json.JSONDecodeError as e:
                # Handle case where response_text might not be defined (e.g., when using test data)
                response_text_local = result.get("response", "")
                response_preview = response_text_local[:1000] if response_text_local and len(response_text_local) > 1000 else response_text_local
                logger.error(f"Failed to parse Claude response as JSON: {e}")
                if response_preview:
                    logger.error(f"Response content (first 1000 chars): {response_preview}")
                
                # Return raw text response if JSON parsing fails
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "raw_response": response_text_local,
                        "usage": result.get("usage"),
                        "warning": "Response could not be parsed as JSON. Returning raw text.",
                        "message": "Spare part recommendation analysis completed, but response format may be invalid"
                    }
                )
            except Exception as e:
                # Handle case where response_text might not be defined (e.g., when using test data)
                response_text_local = result.get("response", "")
                response_preview = response_text_local[:1000] if response_text_local and len(response_text_local) > 1000 else response_text_local
                logger.error(f"Failed to process Claude response: {e}")
                if response_preview:
                    logger.error(f"Response content (first 1000 chars): {response_preview}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process analysis results: {str(e)}"
                )
        else:
            logger.error(f"Claude spare part recommendation analysis failed: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=f"Claude analysis failed: {result['error']}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during spare part recommendation analysis")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during analysis: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the spare part recommendation API.
    
    Returns:
        JSON response with API status and Claude client information
    """
    try:
        client_info = claude_client.get_client_info()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "spare_part_recommendation_api",
                "api_key_connected": bool(settings.anthropic_api_key),
                "claude_client": client_info,
                "endpoints": {
                    "analyze": "POST /spare/part/recommendation/spare-part-recommendation - Analyze image file(s) and return recommendation spare parts for repair",
                    "health": "GET /spare/part/recommendation/health - Health check"
                }
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )
