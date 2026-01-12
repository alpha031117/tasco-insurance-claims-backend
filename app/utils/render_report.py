"""
Report rendering utilities for generating PDF reports from analysis data.
"""
import json
import httpx
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from app.utils.logger import logger
import numpy as np
import pandas as pd

def convert_to_native_types(obj):
    """
    Convert numpy/pandas types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy/pandas types
        
    Returns:
        Object with all numpy/pandas types converted to native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def parse_claude_json_response(response_text: str) -> dict:
    """
    Parse Claude's JSON response, handling various formats including markdown code blocks.
    
    Args:
        response_text: Raw response text from Claude
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    import json
    import re
    
    # Clean up the response
    response_text = response_text.strip()
    
    # Handle escaped newlines
    response_text = response_text.replace('\\n', '\n')
    
    # Try to find JSON within code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)
    elif response_text.startswith('{') and response_text.endswith('}'):
        # Pure JSON response
        pass
    else:
        # Try to find JSON object in the text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            response_text = response_text[json_start:json_end]
    
    # Clean up any remaining formatting issues
    response_text = response_text.strip()
    
    # Parse JSON
    return json.loads(response_text)

def render_police_report_pdf(
    analysis_data: dict,
    filename: str = "police_report_summary.pdf",
    base_dir: Path = None
) -> JSONResponse:
    """
    Render a police report analysis report to PDF format and return JSON with download link.
    
    Args:
        analysis_data: The analysis data from Claude police report analysis
        filename: Filename for the generated PDF
        base_dir: Base directory path (defaults to app directory)
        
    Returns:
        JSONResponse with download_link, filename, and file_info
        
    Raises:
        HTTPException: If PDF generation fails
    """
    try:
        # Set default base directory if not provided
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
        
        logger.info(f"Latest Analysis Data: {json.dumps(analysis_data, indent=4)}")
        logger.info(f"Rendering PDF for police report: {filename}")
        
        # Use production webhook URL
        n8n_webhook_url = "http://178.128.124.9:5678/webhook/99b2cb63-5e81-43b6-a143-29dbb2bbb0f1"
        # n8n_webhook_url = "http://178.128.124.9:5678/webhook-test/99b2cb63-5e81-43b6-a143-29dbb2bbb0f1"
        n8n_webhook_payload = {
            "data": analysis_data
        }
        
        # Send request to n8n webhook
        n8n_webhook_response = httpx.post(n8n_webhook_url, json=n8n_webhook_payload, timeout=30.0)
        logger.info(f"N8N webhook response: {n8n_webhook_response.text}")

        # Parse webhook response to get download link
        try:
            resp_json = n8n_webhook_response.json()
            # Expected format: a list of files; take the first
            file_info = resp_json[0] if isinstance(resp_json, list) and resp_json else resp_json
            
            # Extract direct properties
            justification_road_act = file_info.get("justification_road_act")
            suspect_vehicle_registration = file_info.get("suspect_vehicle_registration")
            download_link = file_info.get("download_link")
            
            # Extract nested insurer properties
            insurer = file_info.get("insurer", {})
            insurer_policy_holder = insurer.get("policy_holder") if isinstance(insurer, dict) else None
            insurer_vehicle_registration = insurer.get("vehicle_registration") if isinstance(insurer, dict) else None
            insurer_age = insurer.get("age") if isinstance(insurer, dict) else None
            insurer_gender = insurer.get("gender") if isinstance(insurer, dict) else None
            insurer_occupational = insurer.get("occupational") if isinstance(insurer, dict) else None
            
            # Extract nested accident_info properties
            accident_info = file_info.get("accident_info", {})
            accident_info_date = accident_info.get("date") if isinstance(accident_info, dict) else None
            accident_info_time = accident_info.get("time") if isinstance(accident_info, dict) else None
            accident_info_location = accident_info.get("location") if isinstance(accident_info, dict) else None
            
            if not download_link:
                raise ValueError("download_link not found in webhook response")

            # Return JSON with download_link
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "download_link": download_link,
                    "justification_road_act": justification_road_act,
                    "insurer": {
                        "policy_holder": insurer_policy_holder,
                        "vehicle_registration": insurer_vehicle_registration,
                        "age": insurer_age,
                        "gender": insurer_gender,
                        "occupational": insurer_occupational,
                    },
                    "accident_info": {
                        "date": accident_info_date,
                        "time": accident_info_time,
                        "location": accident_info_location,
                    },
                    "suspect_vehicle_registration": suspect_vehicle_registration,
                    "filename": filename,
                    "file_info": file_info
                    }
                )
            
        except Exception as e:
            logger.exception("Failed to retrieve download link from webhook response")
            raise HTTPException(status_code=502, detail="Unable to retrieve download link from remote service")
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception("Error rendering police report PDF")
        raise HTTPException(status_code=500, detail=f"Unexpected error while rendering PDF: {str(e)}")