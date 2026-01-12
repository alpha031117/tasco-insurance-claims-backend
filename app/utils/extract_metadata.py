"""
Metadata extraction utilities for PDF documents and images.
"""
import json
import io
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader
from pikepdf import Pdf
import base64

from app.services.claude_client import claude_client
from app.utils.logger import logger

def extract_images_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract images from PDF file.
    
    Args:
        pdf_bytes: PDF file content as bytes
    
    Returns:
        List of dictionaries containing image data with page number, name, and image bytes
    """
    images = []
    try:
        pdf = Pdf.open(io.BytesIO(pdf_bytes))
        for i, page in enumerate(pdf.pages):
            for name, raw in page.images.items():
                image_bytes = raw.obj.extract_to(stream=True)
                images.append({
                    "page": i + 1,
                    "name": str(name),
                    "image_bytes": image_bytes
                })
        logger.info(f"Extracted {len(images)} images from PDF")
    except Exception as e:
        logger.warning(f"Error extracting images from PDF: {e}")
    
    return images


def extract_claim_metadata_from_pdf(
    pdf_base64: str,
    image_base64_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Extract claim metadata from PDF using Claude AI.
    
    This function uses Claude to extract structured claim information including:
    - Accident time
    - Location (street, city, GPS coordinates if available)
    - Reported damages
    - Narrative/description
    
    Args:
        pdf_base64: Base64 encoded PDF content
        image_base64_list: Optional list of base64 encoded images extracted from PDF
    
    Returns:
        Dictionary containing extracted claim metadata
    """
    try:
        content = []
        
        # Add PDF document
        content.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf_base64
            }
        })

        # Add images to content
        if image_base64_list:
            for idx, image_base64 in enumerate(image_base64_list):
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    },
                    "name": f"Image {idx+1}"
                })

        response_dict = claude_client.metadata_extraction(content)
        response_text = response_dict["response"]
        # Parse JSON response
        try:
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            extracted_metadata = json.loads(response_text)
            logger.info("Successfully extracted claim metadata from PDF")
            return extracted_metadata
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.error(f"Response: {response_text[:500]}")
            # Return empty structure on parse error
            return {
                "reported_time": None,
                "reported_location": {
                    "street": None,
                    "city": None,
                    "lat": None,
                    "lng": None
                },
                "reported_damages": [],
                "narrative": None
            }
            
    except Exception as e:
        logger.error(f"Error extracting claim metadata from PDF: {e}")
        raise
