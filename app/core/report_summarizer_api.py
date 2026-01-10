"""
Police Report API endpoints for police report analysis using Claude AI.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64
import json
from typing import List
from app.services.claude_client import claude_client
from app.utils.logger import logger
from app.utils.render_report import render_police_report_pdf, parse_claude_json_response
from app.config import settings

# Create router for police report endpoints
router = APIRouter(prefix="/police/report", tags=["Police Report"])

@router.post("/analyze")
async def analyze_police_report_and_render_pdf(
    files: List[UploadFile] = File(..., description="PDF files to analyze"),
    download: bool = True
):
    """
    Analyze police reports and generate PDF report.
    
    This endpoint accepts PDF files, analyzes them using Claude AI, and returns
    a JSON response with a download link to the generated PDF report.
    
    Args:
        files: List of PDF files to analyze
        download: (Deprecated) No longer used, kept for backward compatibility
        
    Returns:
        JSON response with download_link, filename, and file_info
    """
    try:
        filename = "police_report_summary.pdf"
        # Validate file types
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a PDF file. Only PDF files are supported."
                )
        
        # Check if files are provided
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No files provided. Please upload at least one PDF file."
            )
        
        logger.info(f"Processing {len(files)} PDF files for police report analysis and PDF generation")
        
        # Convert PDF files to base64
        base64_data_list = []
        for file in files:
            try:
                # Read file content
                file_content = await file.read()
                
                # Convert to base64
                base64_data = base64.b64encode(file_content).decode('utf-8')
                base64_data_list.append(base64_data)
                
                logger.info(f"Successfully encoded {file.filename} to base64")
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing file {file.filename}: {str(e)}"
                )
        
        # Add Road Act and Compendium PDF as Base64 (only once, outside the loop)
        try:
            road_act_file_path = "app/templates/Vietnam Road Traffic Law.pdf"
            road_act_file_content = open(road_act_file_path, "rb").read()
            road_act_base64 = base64.b64encode(road_act_file_content).decode('utf-8')
            base64_data_list.append(road_act_base64)
            logger.info("Successfully added Vietnam Road Traffic Law PDF to analysis")

            compendium_file_path = "app/templates/Vehicle_Accident_Compendium.pdf"
            compendium_file_content = open(compendium_file_path, "rb").read()
            compendium_base64 = base64.b64encode(compendium_file_content).decode('utf-8')
            base64_data_list.append(compendium_base64)
            logger.info("Successfully added Vehicle Accident Compendium PDF to analysis")
        except Exception as e:
            logger.error(f"Error loading Road Act & Vehicle Accident Compendium PDF: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading Road Act & Vehicle Accident Compendium PDF: {str(e)}"
            )
        
        # Process with Claude client
        logger.info("Sending files to Claude for police report analysis")
        result = claude_client.analyze_police_report(base64_data_list)
        
        if result["success"]:
            logger.info("Claude police report analysis completed successfully")
            
            # Check if response exists and is not empty
            response_text = result.get("response")
            if not response_text:
                logger.error("Claude returned empty response")
                raise HTTPException(
                    status_code=500,
                    detail="Claude returned an empty response. Please try again."
                )
            
            # Log a preview of the response for debugginG
            response_preview = response_text
            logger.info(f"Claude response preview: {response_preview}")
            
            # Try to parse the JSON response
            try:
                parsed_response = parse_claude_json_response(response_text)
                
                # Generate PDF report
                logger.info("Generating PDF report from analysis data")
                json_response = render_police_report_pdf(
                    analysis_data=parsed_response,
                    filename=filename
                )
                
                return json_response
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Claude response as JSON: {e}")
                logger.error(f"Response content (first 1000 chars): {response_text[:1000]}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse Claude response as JSON: {str(e)}. Response may not be valid JSON."
                )
            except Exception as e:
                logger.error(f"Failed to parse Claude response or generate PDF: {e}")
                logger.error(f"Response content (first 1000 chars): {response_text[:1000]}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process analysis results: {str(e)}"
                )
        else:
            logger.error(f"Claude police report analysis failed: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=f"Claude analysis failed: {result['error']}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during police report analysis and PDF generation")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during analysis and PDF generation: {str(e)}"
        )

# Temporary endpoint to test report generator
@router.post("/test-report-generator")
async def test_report_generator():
    """
    Test report generator endpoint.
    """
    # Load analysis data from report_json.json
    with open("report_json.json", "r", encoding="utf-8") as f:
        analysis_data = json.load(f)
    
    return render_police_report_pdf(analysis_data=analysis_data, filename="test_report.pdf")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the police report API.
    
    Returns:
        JSON response with API status and Claude client information
    """
    try:
        client_info = claude_client.get_client_info()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "police_report_api",
                "api_key_connected": bool(settings.anthropic_api_key),
                "claude_client": client_info,
                "endpoints": {
                    "analyze": "POST /police/report/analyze - Analyze PDF files and generate PDF report",
                    "health": "GET /police/report/health - Health check"
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
