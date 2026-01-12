"""
Fraud Assessment API endpoints using heuristic checks and Claude reasoning.
"""
import base64
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from app.utils.fraud_detection import (
    FraudCheckRequest,
    Location,
    FraudCheckResponse,
    perform_fraud_check,
)
from app.utils.extract_metadata import (
    extract_images_from_pdf,
    extract_claim_metadata_from_pdf,
)
from app.services.claude_client import claude_client
from app.services.voyage_client import voyage_client
from app.services.supabase_client import supabase_vector_client
from app.utils.logger import logger


router = APIRouter(prefix="/fraud", tags=["Fraud Assessment"])


class FraudAssessmentResponse(FraudCheckResponse):
    """Fraud assessment response with Claude reasoning."""
    claude_reasoning: Optional[str] = None
    extracted_metadata: Optional[dict] = None


class StoreCaseResponse(BaseModel):
    """Response model for storing accident case."""
    id: str
    case_id: str
    message: str


class StoreCaseWithFraudRequest(BaseModel):
    """Request model for storing case with fraud detection results."""
    metadata: Dict[str, Any]
    is_fraud: bool = False
    fraud_score: float = 0.0
    fraud_reasons: List[str] = []


async def _store_case_helper(
    metadata: Dict[str, Any],
    is_fraud: bool = False,
    fraud_score: float = 0.0,
    fraud_reasons: List[str] = []
) -> StoreCaseResponse:
    """
    Internal helper function to store accident case with fraud results.
    
    Args:
        metadata: Accident case metadata
        is_fraud: Whether fraud was detected
        fraud_score: Fraud detection score
        fraud_reasons: List of fraud alerts/reasons
    
    Returns:
        StoreCaseResponse with id, case_id, and message
    
    Raises:
        HTTPException: If storage fails
    """
    # Validate required clients
    if supabase_vector_client is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase client not initialized. Please configure SUPABASE_URL and SUPABASE_KEY."
        )
    
    if voyage_client is None:
        raise HTTPException(
            status_code=500,
            detail="Voyage AI client not initialized. Please configure VOYAGE_API_KEY."
        )
    
    logger.info(f"Storing accident case - is_fraud: {is_fraud}, score: {fraud_score}")
    
    # Step 1: Generate case_id
    case_id = None
    if metadata.get("document_metadata") and metadata["document_metadata"].get("document_number"):
        doc_number = metadata["document_metadata"]["document_number"]
        if doc_number and doc_number not in ["Not provided", "Not specified in document", None]:
            case_id = doc_number
        else:
            case_id = f"CASE_{uuid.uuid4().hex[:12].upper()}"
    else:
        case_id = f"CASE_{uuid.uuid4().hex[:12].upper()}"
    
    logger.info(f"Generated case_id: {case_id}")
    
    # Step 2: Generate summary text from narrative or key fields
    summary_parts = []
    
    # Add primary narrative if available
    if metadata.get("incident_narrative"):
        narrative = metadata["incident_narrative"]
        if isinstance(narrative, dict):
            primary_desc = narrative.get("primary_description") or narrative.get("reporting_person_statement")
            if primary_desc and primary_desc not in ["Not provided", None]:
                summary_parts.append(primary_desc)
        elif isinstance(narrative, str) and narrative != "Not provided":
            summary_parts.append(narrative)
    
    # Add location information
    if metadata.get("accident_information") and metadata["accident_information"].get("location"):
        loc = metadata["accident_information"]["location"]
        if isinstance(loc, dict):
            full_addr = loc.get("full_address") or loc.get("street_address")
            if full_addr and full_addr != "Not provided":
                summary_parts.append(f"Location: {full_addr}")
    
    # Add temporal information
    if metadata.get("accident_information"):
        acc_info = metadata["accident_information"]
        if isinstance(acc_info, dict):
            accident_datetime = acc_info.get("accident_datetime_full") or f"{acc_info.get('accident_date')} {acc_info.get('accident_time')}"
            if accident_datetime and "None" not in accident_datetime:
                summary_parts.append(f"Accident occurred on: {accident_datetime}")
    
    # Add damages summary - extract all damage parts from all vehicles
    if metadata.get("reported_damages"):
        damages = metadata["reported_damages"]
        if isinstance(damages, dict) and damages.get("vehicle_damages"):
            vehicle_damages = damages["vehicle_damages"]
            if vehicle_damages and len(vehicle_damages) > 0:
                all_damage_parts = []
                for vehicle_damage in vehicle_damages:
                    damaged_parts = vehicle_damage.get("damaged_parts", [])
                    if damaged_parts:
                        for part in damaged_parts:
                            if isinstance(part, dict):
                                part_name = part.get("part_name") or part.get("damage_description")
                                if part_name:
                                    all_damage_parts.append(part_name)
                            elif isinstance(part, str):
                                all_damage_parts.append(part)
                if all_damage_parts:
                    summary_parts.append(f"Damages: {', '.join(all_damage_parts)}")
    
    # Combine summary parts
    summary_text = " | ".join(summary_parts) if summary_parts else "Accident case - details in metadata"
    
    # If summary is too short, add more context
    if len(summary_text) < 100:
        if metadata.get("parties_involved"):
            parties = metadata["parties_involved"]
            if isinstance(parties, list) and len(parties) > 0:
                party_names = [
                    p.get("full_name", "Unknown") 
                    for p in parties[:2] 
                    if p.get("full_name") and p.get("full_name") not in ["Not provided", "unidentified male", "unidentified female"]
                ]
                if party_names:
                    summary_text += f" | Parties: {', '.join(party_names)}"
    
    logger.info(f"Generated summary text (length: {len(summary_text)})")
    
    # Step 3: Generate embedding using Voyage AI
    logger.info("Generating embedding using Voyage AI...")
    embedding = voyage_client.generate_embedding(summary_text)
    logger.info(f"Generated embedding of dimension {len(embedding)}")
    
    # Step 4: Extract flattened fields from metadata
    accident_date = None
    if metadata.get("accident_information"):
        acc_info = metadata["accident_information"]
        accident_date = acc_info.get("accident_date")
    
    accident_city = None
    if metadata.get("accident_information") and metadata["accident_information"].get("location"):
        loc = metadata["accident_information"]["location"]
        accident_city = loc.get("city_province") or loc.get("city")
    
    vehicle_plate = None
    other_vehicle_plate = None
    if metadata.get("parties_involved") and isinstance(metadata["parties_involved"], list):
        parties = metadata["parties_involved"]
        if len(parties) > 0 and parties[0].get("vehicle_information"):
            vehicle_plate = parties[0]["vehicle_information"].get("registration_number")
        if len(parties) > 1 and parties[1].get("vehicle_information"):
            other_vehicle_plate = parties[1]["vehicle_information"].get("registration_number")
    
    damage_count = 0
    if metadata.get("reported_damages") and isinstance(metadata["reported_damages"], dict):
        damages = metadata["reported_damages"]
        if damages.get("vehicle_damages") and isinstance(damages["vehicle_damages"], list):
            vehicle_damages = damages["vehicle_damages"]
            if len(vehicle_damages) > 0 and vehicle_damages[0].get("damaged_parts"):
                damaged_parts = vehicle_damages[0]["damaged_parts"]
                damage_count = len(damaged_parts) if isinstance(damaged_parts, list) else 0
    
    # Step 5: Prepare data for insertion with fraud results
    insert_data = {
        "case_id": case_id,
        "metadata": metadata,
        "summary_text": summary_text,
        "embedding": embedding,
        "is_fraud": is_fraud,
        "fraud_score": fraud_score,
        "fraud_reasons": fraud_reasons,
        "accident_date": accident_date,
        "accident_city": accident_city,
        "vehicle_plate": vehicle_plate,
        "other_vehicle_plate": other_vehicle_plate,
        "damage_count": damage_count
    }
    
    logger.info(
        f"Extracted fields - Date: {accident_date}, City: {accident_city}, "
        f"Plate: {vehicle_plate}, Other Plate: {other_vehicle_plate}, "
        f"Damage Count: {damage_count}, Is Fraud: {is_fraud}"
    )
    
    # Step 6: Insert into accident_cases table
    logger.info(f"Inserting case into accident_cases table...")
    result = supabase_vector_client.client.table("accident_cases").insert(insert_data).execute()
    
    if result.data and len(result.data) > 0:
        inserted_record = result.data[0]
        record_id = inserted_record.get("id")
        logger.info(
            f"Successfully stored accident case - "
            f"id: {record_id}, case_id: {case_id}, is_fraud: {is_fraud}"
        )
        
        return StoreCaseResponse(
            id=str(record_id),
            case_id=case_id,
            message=f"Accident case stored successfully"
        )
    else:
        logger.error("Failed to store accident case: No data returned from Supabase")
        raise HTTPException(
            status_code=500,
            detail="Failed to store accident case: No data returned"
        )


@router.post("/assessment", response_model=FraudAssessmentResponse)
async def assess_fraud(
    file: UploadFile = File(..., description="PDF file containing accident report"),
    include_reasoning: Optional[bool] = True
):
    """
    Perform fraud assessment on uploaded PDF report.
    
    Flow:
    1. User uploads PDF (may contain images)
    2. Extract images from PDF if present
    3. Extract claim metadata (time, location, damages, narrative) using Claude
    4. Perform case-based fraud checking
    5. Generate Claude reasoning based on check results
    6. Return comprehensive fraud assessment
    
    Args:
        file: PDF file containing the accident report
        include_reasoning: Whether to include Claude reasoning (default: True)
    
    Returns:
        FraudAssessmentResponse with fraud score, alerts, status, and reasoning
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF. Only PDF files are supported."
            )
        
        logger.info(f"Processing PDF file: {file.filename} for fraud assessment")
        
        # Read PDF content
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail="PDF file is empty"
            )
        
        # Step 1: Extract images from PDF if present
        logger.info("Extracting images from PDF...")
        images = extract_images_from_pdf(pdf_bytes)
        image_base64_list = []
        
        if images:
            logger.info(f"Found {len(images)} images in PDF, encoding to base64...")
            for img in images:
                try:
                    img_base64 = base64.b64encode(img["image_bytes"]).decode('utf-8')
                    image_base64_list.append(img_base64)
                except Exception as e:
                    logger.warning(f"Failed to encode image {img.get('name', 'unknown')}: {e}")
        
        # Step 2: Extract claim metadata from PDF using Claude
        logger.info("Extracting claim metadata from PDF using Claude...")
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        try:
            extracted_metadata = extract_claim_metadata_from_pdf(
                pdf_base64=pdf_base64,
                image_base64_list=image_base64_list if image_base64_list else None
            )
            logger.info(f"Extracted metadata: {extracted_metadata}")
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract metadata from PDF: {str(e)}"
            )
        
        # Step 3: Convert extracted metadata to FraudCheckRequest (handles old/new schemas)
        # Time
        reported_time = extracted_metadata.get("reported_time")
        if not reported_time:
            accident_info = extracted_metadata.get("accident_information", {})
            accident_date = accident_info.get("accident_date")
            accident_time = accident_info.get("accident_time")
            accident_dt_full = accident_info.get("accident_datetime_full") or accident_info.get("accident_datetime_description")
            if accident_dt_full:
                reported_time = accident_dt_full
            elif accident_date and accident_time:
                reported_time = f"{accident_date}T{accident_time}"
            elif accident_date:
                reported_time = accident_date
        
        # Location
        reported_location = None
        if extracted_metadata.get("reported_location"):
            loc_data = extracted_metadata["reported_location"]
            reported_location = Location(
                street=loc_data.get("street"),
                city=loc_data.get("city"),
                lat=loc_data.get("lat"),
                lng=loc_data.get("lng")
            )
        else:
            accident_info = extracted_metadata.get("accident_information", {})
            loc_data = accident_info.get("location", {}) if isinstance(accident_info, dict) else {}
            gps = loc_data.get("gps_coordinates", {}) if isinstance(loc_data, dict) else {}
            street_val = loc_data.get("street_address") or loc_data.get("street_road")
            city_val = loc_data.get("city_province") or loc_data.get("city")
            lat_val = gps.get("latitude") if isinstance(gps, dict) else None
            lng_val = gps.get("longitude") if isinstance(gps, dict) else None
            if any([street_val, city_val, lat_val, lng_val]):
                reported_location = Location(
                    street=street_val,
                    city=city_val,
                    lat=lat_val,
                    lng=lng_val
                )
        
        # Damages
        reported_damages = extracted_metadata.get("reported_damages") or []
        if isinstance(reported_damages, dict):
            # new structure: vehicle_damages -> damaged_parts[]
            damages_list = []
            vehicle_damages = reported_damages.get("vehicle_damages") or []
            for vehicle_damage in vehicle_damages:
                parts = vehicle_damage.get("damaged_parts") or vehicle_damage.get("damage_list") or []
                if parts:
                    for part in parts:
                        if isinstance(part, dict):
                            name = part.get("part_name") or part.get("damage_description")
                            if name:
                                damages_list.append(name)
                        elif isinstance(part, str):
                            damages_list.append(part)
            reported_damages = damages_list
        elif not reported_damages:
            reported_damages = []
        
        # Narrative
        narrative = extracted_metadata.get("narrative")
        if not narrative:
            incident = extracted_metadata.get("incident_narrative", {})
            if isinstance(incident, dict):
                narrative = (
                    incident.get("primary_description")
                    or incident.get("reporting_person_statement")
                    or incident.get("sequence_of_events")
                )
        
        # Extract vehicle plate and accident date for duplicate detection
        vehicle_plate_for_fraud = None
        accident_date_for_fraud = None
        
        if extracted_metadata.get("parties_involved") and isinstance(extracted_metadata["parties_involved"], list):
            parties = extracted_metadata["parties_involved"]
            if len(parties) > 0 and parties[0].get("vehicle_information"):
                vehicle_plate_for_fraud = parties[0]["vehicle_information"].get("registration_number")
        
        if extracted_metadata.get("accident_information"):
            acc_info = extracted_metadata["accident_information"]
            accident_date_for_fraud = acc_info.get("accident_date")
        
        fraud_check_request = FraudCheckRequest(
            reported_time=reported_time,
            reported_location=reported_location,
            reported_damages=reported_damages,
            narrative=narrative,
            vehicle_plate=vehicle_plate_for_fraud,
            accident_date=accident_date_for_fraud
        )
        
        # Step 4: Perform case-based fraud checking
        logger.info("Performing fraud check...")
        try:
            result = perform_fraud_check(fraud_check_request)
            logger.info(f"Fraud check completed. Score: {result.fraud_score}, Status: {result.status}, Alerts: {result.alerts}")
        except Exception as e:
            logger.error(f"Error during fraud assessment: {e}")
            raise HTTPException(status_code=500, detail="Fraud assessment failed") from e
        
        # Step 5: Store case based on fraud detection result
        try:
            # Determine if fraud based on score threshold
            is_fraud = result.fraud_score >= 0.3
            
            # Store the case using helper function
            store_response = await _store_case_helper(
                metadata=extracted_metadata,
                is_fraud=is_fraud,
                fraud_score=result.fraud_score,
                fraud_reasons=result.alerts
            )
            logger.info(
                f"Case stored successfully - case_id: {store_response.case_id}, "
                f"is_fraud: {is_fraud}, score: {result.fraud_score}"
            )
        except HTTPException as e:
            # If storage fails due to duplicate, log but continue
            if e.status_code == 409:
                logger.warning(f"Case already exists in database: {e.detail}")
            else:
                logger.error(f"Failed to store case: {e.detail}")
                # Don't fail the fraud assessment if storage fails
        except Exception as e:
            logger.error(f"Unexpected error storing case: {e}")
            # Don't fail the fraud assessment if storage fails
        
        # Step 6: Return comprehensive response
        response_data = result.dict()
        response_data["extracted_metadata"] = extracted_metadata
        
        return JSONResponse(status_code=200, content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during fraud assessment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during fraud assessment: {str(e)}"
        )


@router.post("/store-case", response_model=StoreCaseResponse)
async def store_accident_case(
    metadata: Dict[str, Any] = Body(..., description="Accident case metadata extracted from PDF")
):
    """
    Store accident case metadata into Supabase pgvector database.
    
    This endpoint:
    1. Accepts extracted PDF metadata (document_metadata, accident_temporal_info, etc.)
    2. Generates a summary text from the narrative or key fields
    3. Generates a case_id (from report_number or UUID)
    4. Generates vector embedding using Voyage AI
    5. Stores in accident_cases table with metadata, summary_text, and embedding
    
    Args:
        metadata: Dictionary containing accident case metadata with structure:
            - document_metadata: Report information
            - accident_temporal_info: Time/date information
            - accident_location: Location details
            - parties_involved: List of parties
            - reported_damages: Damage information
            - incident_narrative: Narrative description
            - environmental_conditions: Weather/road conditions
            - preliminary_findings: Officer findings
            - supporting_documentation: Photos/documents
            - additional_notes: Additional information
    
    Returns:
        StoreCaseResponse with id, case_id, and success message
    """
    try:
        # Validate required clients
        if supabase_vector_client is None:
            raise HTTPException(
                status_code=500,
                detail="Supabase client not initialized. Please configure SUPABASE_URL and SUPABASE_KEY."
            )
        
        if voyage_client is None:
            raise HTTPException(
                status_code=500,
                detail="Voyage AI client not initialized. Please configure VOYAGE_API_KEY."
            )
        
        logger.info("Storing accident case metadata in Supabase...")
        
        # Step 1: Generate case_id
        # Try to use report_number from document_metadata, otherwise generate UUID
        case_id = None
        if metadata.get("document_metadata") and metadata["document_metadata"].get("report_number"):
            report_number = metadata["document_metadata"]["report_number"]
            if report_number and report_number != "Not provided":
                case_id = report_number
            else:
                case_id = f"CASE_{uuid.uuid4().hex[:12].upper()}"
        else:
            case_id = f"CASE_{uuid.uuid4().hex[:12].upper()}"
        
        logger.info(f"Generated case_id: {case_id}")
        
        # Step 2: Generate summary text from narrative or key fields
        summary_parts = []
        
        # Add primary narrative if available
        if metadata.get("incident_narrative"):
            narrative = metadata["incident_narrative"]
            if isinstance(narrative, dict):
                primary_desc = narrative.get("primary_description") or narrative.get("reporting_person_statement")
                if primary_desc and primary_desc != "Not provided":
                    summary_parts.append(primary_desc)
            elif isinstance(narrative, str) and narrative != "Not provided":
                summary_parts.append(narrative)
        
        # Add location information
        if metadata.get("accident_location"):
            loc = metadata["accident_location"]
            if isinstance(loc, dict):
                full_addr = loc.get("full_address")
                if full_addr and full_addr != "Not provided":
                    summary_parts.append(f"Location: {full_addr}")
        
        # Add temporal information
        if metadata.get("accident_temporal_info"):
            temp_info = metadata["accident_temporal_info"]
            if isinstance(temp_info, dict):
                accident_datetime = temp_info.get("accident_datetime_full")
                if accident_datetime and accident_datetime != "Not provided":
                    summary_parts.append(f"Accident occurred on: {accident_datetime}")
        
        # Add damages summary - extract all damage parts from all vehicles
        if metadata.get("reported_damages"):
            damages = metadata["reported_damages"]
            if isinstance(damages, dict) and damages.get("vehicle_damages"):
                vehicle_damages = damages["vehicle_damages"]
                if vehicle_damages and len(vehicle_damages) > 0:
                    # Collect all damage parts from all vehicles
                    all_damage_parts = []
                    for vehicle_damage in vehicle_damages:
                        damage_list = vehicle_damage.get("damage_list", [])
                        if damage_list:
                            all_damage_parts.extend(damage_list)
                    if all_damage_parts:
                        summary_parts.append(f"Damages: {', '.join(all_damage_parts)}")
        
        # Combine summary parts
        summary_text = " | ".join(summary_parts) if summary_parts else "Accident case - details in metadata"
        
        # If summary is too short, add more context
        if len(summary_text) < 100:
            # Add parties information
            if metadata.get("parties_involved"):
                parties = metadata["parties_involved"]
                if isinstance(parties, list) and len(parties) > 0:
                    party_names = [p.get("full_name", "Unknown") for p in parties[:2] if p.get("full_name") and p.get("full_name") != "Not provided"]
                    if party_names:
                        summary_text += f" | Parties: {', '.join(party_names)}"
        
        logger.info(f"Generated summary text (length: {len(summary_text)})")
        
        # Step 3: Generate embedding using Voyage AI
        logger.info("Generating embedding using Voyage AI...")
        try:
            embedding = voyage_client.generate_embedding(summary_text)
            logger.info(f"Generated embedding of dimension {len(embedding)}")
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding: {str(e)}"
            )
        
        # Step 4: Extract flattened fields from metadata
        # Extract accident_date
        accident_date = None
        if metadata.get("accident_information"):
            acc_info = metadata["accident_information"]
            accident_date = acc_info.get("accident_date")
        
        # Extract accident_city
        accident_city = None
        if metadata.get("accident_information") and metadata["accident_information"].get("location"):
            loc = metadata["accident_information"]["location"]
            accident_city = loc.get("city_province") or loc.get("city")
        
        # Extract vehicle plates
        vehicle_plate = None
        other_vehicle_plate = None
        if metadata.get("parties_involved") and isinstance(metadata["parties_involved"], list):
            parties = metadata["parties_involved"]
            if len(parties) > 0 and parties[0].get("vehicle_information"):
                vehicle_plate = parties[0]["vehicle_information"].get("registration_number")
            if len(parties) > 1 and parties[1].get("vehicle_information"):
                other_vehicle_plate = parties[1]["vehicle_information"].get("registration_number")
        
        # Extract damage_count
        damage_count = 0
        if metadata.get("reported_damages") and isinstance(metadata["reported_damages"], dict):
            damages = metadata["reported_damages"]
            if damages.get("vehicle_damages") and isinstance(damages["vehicle_damages"], list):
                vehicle_damages = damages["vehicle_damages"]
                if len(vehicle_damages) > 0 and vehicle_damages[0].get("damaged_parts"):
                    damaged_parts = vehicle_damages[0]["damaged_parts"]
                    damage_count = len(damaged_parts) if isinstance(damaged_parts, list) else 0
        
        # Step 5: Prepare data for insertion with fraud fields (defaults)
        insert_data = {
            "case_id": case_id,
            "metadata": metadata,
            "summary_text": summary_text,
            "embedding": embedding,
            "is_fraud": False,
            "fraud_score": 0.0,
            "fraud_reasons": [],
            "accident_date": accident_date,
            "accident_city": accident_city,
            "vehicle_plate": vehicle_plate,
            "other_vehicle_plate": other_vehicle_plate,
            "damage_count": damage_count
        }
        
        logger.info(
            f"Extracted fields - Date: {accident_date}, City: {accident_city}, "
            f"Plate: {vehicle_plate}, Other Plate: {other_vehicle_plate}, "
            f"Damage Count: {damage_count}"
        )
        
        # Step 6: Insert into accident_cases table
        logger.info(f"Inserting case into accident_cases table...")
        try:
            result = supabase_vector_client.client.table("accident_cases").insert(insert_data).execute()
            
            if result.data and len(result.data) > 0:
                inserted_record = result.data[0]
                record_id = inserted_record.get("id")
                logger.info(f"Successfully stored accident case with id: {record_id}, case_id: {case_id}")
                
                return StoreCaseResponse(
                    id=str(record_id),
                    case_id=case_id,
                    message=f"Accident case stored successfully"
                )
            else:
                logger.error("Failed to store accident case: No data returned from Supabase")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to store accident case: No data returned"
                )
                
        except Exception as e:
            logger.error(f"Error inserting into Supabase: {e}")
            # Check for unique constraint violation (case_id already exists)
            if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                raise HTTPException(
                    status_code=409,
                    detail=f"Case with case_id '{case_id}' already exists"
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store accident case: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error storing accident case: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error storing accident case: {str(e)}"
        )


@router.post("/store-case-with-fraud", response_model=StoreCaseResponse)
async def store_accident_case_with_fraud(
    request: StoreCaseWithFraudRequest = Body(..., description="Accident case with fraud detection results")
):
    """
    Store accident case metadata with fraud detection results into Supabase.
    
    This endpoint accepts metadata along with fraud detection results (is_fraud, fraud_score, fraud_reasons)
    and stores them in the accident_cases table.
    
    Use this endpoint after performing fraud detection to store the complete assessment.
    
    Args:
        request: StoreCaseWithFraudRequest containing metadata and fraud results
    
    Returns:
        StoreCaseResponse with id, case_id, and success message
    """
    try:
        return await _store_case_helper(
            metadata=request.metadata,
            is_fraud=request.is_fraud,
            fraud_score=request.fraud_score,
            fraud_reasons=request.fraud_reasons
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error storing accident case with fraud results: {e}")
        # Check for duplicate key
        if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
            raise HTTPException(
                status_code=409,
                detail=f"Case already exists in database"
            )
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error storing accident case with fraud results: {str(e)}"
        )