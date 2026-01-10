"""
Spare Parts Upload API endpoint for uploading motorcycle spare parts from Excel to Supabase.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
from app.utils.spare_parts_uploader import SparePartsUploader
from app.utils.logger import logger
from app.services.supabase_client import supabase_vector_client
from app.services.voyage_client import voyage_client


# Create router for spare parts upload endpoints
router = APIRouter(prefix="/spare-parts", tags=["Spare Parts"])


@router.post("/upload")
async def upload_spare_parts_excel(
    file: UploadFile = File(..., description="Excel file (.xlsx) containing spare parts data"),
    batch_size: Optional[int] = Form(50, description="Batch size for processing (default: 50, max: 128)"),
    table_name: Optional[str] = Form("motorcycle_spareparts", description="Supabase table name")
):
    """
    Upload motorcycle spare parts from Excel file to Supabase with embeddings.
    
    This endpoint:
    1. Reads the Excel file
    2. Generates embeddings using Voyage AI (1024-dim vectors)
    3. Uploads to Supabase with vector embeddings for semantic search
    
    **Excel File Format:**
    The Excel file should contain columns such as:
    - Part Name / Name / part_name (required)
    - Description / Part Description / part_description
    - Brand / Manufacturer / brand
    - Model / Vehicle Model / model
    - Hanoi / HoChiMinh / DaNang (regional prices)
    - Repair Hours / labor_hours
    - Categories / Category / damage_categories
    
    **Note:** Column names are flexible and will be matched case-insensitively.
    
    Args:
        file: Excel file (.xlsx) containing spare parts data
        batch_size: Number of items to process per batch (default: 50, max: 128)
        table_name: Name of the Supabase table (default: "motorcycle_spareparts")
        
    Returns:
        JSON response with upload statistics and results
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided"
            )
        
        if not file.filename.lower().endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only Excel files (.xlsx, .xls, .csv) are supported."
            )
        
        # Validate batch size
        if batch_size and (batch_size < 1 or batch_size > 128):
            raise HTTPException(
                status_code=400,
                detail="Batch size must be between 1 and 128 (Voyage AI limit)"
            )
        
        # Check if required clients are initialized
        if voyage_client is None:
            raise HTTPException(
                status_code=500,
                detail="Voyage AI client not initialized. Please configure VOYAGE_API_KEY."
            )
        
        if supabase_vector_client is None:
            raise HTTPException(
                status_code=500,
                detail="Supabase client not initialized. Please configure SUPABASE_URL and SUPABASE_KEY."
            )
        
        logger.info(f"Processing Excel file: {file.filename}")
        
        # Read file content
        try:
            file_content = await file.read()
        except Exception as e:
            logger.error(f"Error reading file {file.filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file: {str(e)}"
            )
        
        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        # Initialize uploader
        try:
            uploader = SparePartsUploader()
        except ValueError as e:
            logger.error(f"Failed to initialize uploader: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
        
        # Process and upload
        logger.info(f"Starting upload process for {file.filename}")
        result = uploader.process_excel_and_upload(
            file_content=file_content,
            batch_size=batch_size or 50,
            table_name=table_name or "motorcycle_spareparts"
        )
        
        if result['success']:
            stats = result['stats']
            logger.info(f"Upload completed successfully: {stats.get('uploaded', 0)} parts uploaded")
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Spare parts uploaded successfully",
                    "filename": file.filename,
                    "statistics": {
                        "total_rows_in_excel": stats.get('total_rows_in_excel', 0),
                        "parts_prepared": stats.get('total_parts', 0),
                        "parts_uploaded": stats.get('uploaded', 0),
                        "parts_failed": stats.get('failed', 0),
                        "rows_skipped": len(stats.get('skipped_rows', [])),
                        "total_in_database": stats.get('total_in_database'),
                        "errors": stats.get('errors')
                    }
                }
            )
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            logger.error(f"Upload failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Upload failed: {error_msg}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during spare parts upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during upload: {str(e)}"
        )


@router.get("/stats")
async def get_upload_stats(table_name: Optional[str] = "motorcycle_spareparts"):
    """
    Get statistics about uploaded spare parts.
    
    Args:
        table_name: Name of the Supabase table (default: "motorcycle_spareparts")
        
    Returns:
        JSON response with database statistics
    """
    try:
        if supabase_vector_client is None:
            raise HTTPException(
                status_code=500,
                detail="Supabase client not initialized"
            )
        
        # Get total count
        result = supabase_vector_client.client.table(table_name).select('id', count='exact').limit(1).execute()
        total_count = result.count if hasattr(result, 'count') else 0
        
        # Get sample records to show structure
        sample_result = supabase_vector_client.client.table(table_name).select('*').limit(5).execute()
        sample_data = sample_result.data if hasattr(sample_result, 'data') else []
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "table_name": table_name,
                "total_records": total_count,
                "sample_records_count": len(sample_data),
                "sample_record": sample_data[0] if sample_data else None
            }
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the spare parts upload API.
    
    Returns:
        JSON response with API status and client information
    """
    try:
        voyage_ready = voyage_client is not None
        supabase_ready = supabase_vector_client is not None
        
        health_status = "healthy" if (voyage_ready and supabase_ready) else "degraded"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": health_status,
                "service": "spare_parts_upload_api",
                "voyage_ai_ready": voyage_ready,
                "supabase_ready": supabase_ready,
                "voyage_client_info": voyage_client.get_client_info() if voyage_client else None,
                "supabase_client_info": supabase_vector_client.get_client_info() if supabase_vector_client else None,
                "endpoints": {
                    "upload": "POST /spare-parts/upload - Upload Excel file with spare parts",
                    "stats": "GET /spare-parts/stats - Get upload statistics",
                    "health": "GET /spare-parts/health - Health check"
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