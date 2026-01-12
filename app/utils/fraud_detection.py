"""
Fraud detection utilities for insurance claims validation.
"""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from math import radians, sin, cos, atan2, sqrt
from pydantic import BaseModel

from app.utils.logger import logger
from app.services.supabase_client import supabase_vector_client
from app.services.voyage_client import voyage_client


# --- 1) LOAD CASE METADATA (Supabase first, file fallback, then in-code fallback) ---

def _load_case_metadata_from_file(metadata_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load case metadata from JSON file.
    
    Args:
        metadata_path: Optional path to metadata JSON file. 
                       Defaults to presaved_fraud_metadata.json in project root.
    
    Returns:
        Dictionary containing case metadata
    
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if metadata_path is None:
        # Default to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        metadata_path = project_root / "presaved_fraud_metadata.json"
    else:
        metadata_path = Path(metadata_path)
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"Loaded case metadata from {metadata_path}")
        return metadata
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file: {e}")
        raise


def load_case_metadata(
    metadata_path: Optional[str] = None,
    case_id: Optional[str] = None,
    vehicle_plate: Optional[str] = None,
    accident_city: Optional[str] = None,
    accident_date: Optional[str] = None,
    supabase_table: str = "accident_cases",
) -> Dict[str, Any]:
    """
    Load case metadata with priority:
    1) Supabase accident_cases table (pgvector) with optional filters
    2) JSON file
    3) In-code fallback metadata
    
    Args:
        metadata_path: Optional path to JSON file (fallback)
        case_id: Filter by case_id
        vehicle_plate: Filter by vehicle registration number
        accident_city: Filter by accident city
        accident_date: Filter by accident date (YYYY-MM-DD)
        supabase_table: Table name (default: "accident_cases")
    
    Returns:
        Dictionary containing case metadata
    """
    # 1) Supabase
    if supabase_vector_client is not None:
        try:
            query = (
                supabase_vector_client.client.table(supabase_table)
                .select("metadata, case_id, accident_date, accident_city, vehicle_plate, created_at")
                .order("created_at", desc=True)
                .limit(1)
            )
            
            # Apply filters if provided
            if case_id:
                query = query.eq("case_id", case_id)
            if vehicle_plate:
                query = query.eq("vehicle_plate", vehicle_plate)
            if accident_city:
                query = query.ilike("accident_city", f"%{accident_city}%")
            if accident_date:
                query = query.eq("accident_date", accident_date)
            
            result = query.execute()
            if result.data:
                record = result.data[0]
                meta = record.get("metadata") or record
                logger.info(
                    f"Loaded case metadata from Supabase table '{supabase_table}' "
                    f"(case_id={record.get('case_id')}, city={record.get('accident_city')}, "
                    f"plate={record.get('vehicle_plate')})"
                )
                return meta
            else:
                # Supabase is available but no records found - assume new/empty table
                filters_desc = ", ".join([
                    f"case_id={case_id}" if case_id else "",
                    f"vehicle_plate={vehicle_plate}" if vehicle_plate else "",
                    f"accident_city={accident_city}" if accident_city else "",
                    f"accident_date={accident_date}" if accident_date else ""
                ]).strip(", ") or "latest"
                logger.info(
                    f"No metadata found in Supabase table '{supabase_table}' "
                    f"(filters: {filters_desc}). Assuming new/empty table - not using fallback."
                )
                # Return None or empty dict to indicate no case metadata available
                return None
        except Exception as e:
            # Supabase connection/query failed - fall back to file/in-code
            logger.warning(f"Supabase metadata load failed, will fall back to file/in-code. Error: {e}")
    else:
        # Supabase client not initialized - fall back to file/in-code
        logger.info("Supabase client not initialized, using file/in-code fallback")

    # 2) JSON file (only if Supabase unavailable or errored)
    try:
        return _load_case_metadata_from_file(metadata_path)
    except Exception as e:
        logger.warning(f"File metadata load failed, will fall back to in-code default. Error: {e}")

    # 3) In-code fallback (only if Supabase unavailable/errored AND file failed)
    logger.warning("Using in-code fallback metadata.")
    return FALLBACK_METADATA

# Fallback metadata aligned to the new extraction schema (for development/testing)
FALLBACK_METADATA = {
    "case_id": "VN_SYNTH_001",
    "document_metadata": {
        "document_type": "Official Road Traffic Accident Report",
        "document_number": "Not specified in document",
        "issuing_authority": "Da Nang City Police Department - Traffic Police Division",
        "report_date": "2024-03-12",
        "document_language": "English",
        "total_pages": 5,
        "has_official_stamp": False,
        "document_status": "Preliminary",
    },
    "accident_information": {
        "accident_date": "2024-03-12",
        "accident_time": "16:45",
        "accident_datetime_description": "approximately 16:45",
        "location": {
            "street_address": "Nguyen Huu Tho Street",
            "intersection": "near Nguyen Tri Phuong Bridge",
            "landmark": "Nguyen Tri Phuong Bridge",
            "ward_commune": "Not specified",
            "district": "Cam Le District (heading toward)",
            "city_province": "Da Nang City",
            "country": "Vietnam",
            "gps_coordinates": {
                "latitude": None,
                "longitude": None,
            },
            "location_description": "area near Nguyen Tri Phuong Bridge on Nguyen Huu Tho Street",
            "location_type": "highway",
        },
        "environmental_conditions": {
            "weather": "clear",
            "road_surface": "dry",
            "lighting": "daylight",
            "traffic_density": "light",
            "visibility": "good",
        },
    },
    "parties_involved": [
        {
            "party_role": "Reporting Person/Claimant",
            "full_name": "Nguyen Minh Hoang",
            "national_id": "079203004812",
            "date_of_birth": "1987-03-14",
            "gender": "Male",
            "nationality": "Vietnamese",
            "occupation": "Mechanical Technician",
            "contact_information": {
                "phone": "+84 909 215 682",
                "email": "Not provided",
                "address": "128 Tran Dai Nghia Street, Ngu Hanh Son District, Da Nang City",
            },
            "driver_license": {
                "license_number": "Not provided",
                "license_type": "Not provided",
                "issue_date": "Not provided",
                "expiry_date": "Not provided",
            },
            "insurance_information": {
                "insurance_company": "Not provided",
                "policy_number": "Not provided",
                "policy_type": "Not provided",
                "policy_expiry": "Not provided",
            },
            "vehicle_information": {
                "vehicle_type": "motorcycle",
                "make_brand": "Yamaha",
                "model": "Exciter",
                "year": "Not specified",
                "color": "Not specified",
                "registration_number": "43D1-672.91",
                "vin_chassis_number": "Not provided",
                "engine_number": "Not provided",
            },
            "injuries": {
                "injury_status": "Minor injury",
                "injury_description": "minor abrasions on left hand",
                "medical_treatment": "Not specified",
            },
        },
        {
            "party_role": "Other Driver",
            "full_name": "unidentified male",
            "national_id": "Not provided",
            "date_of_birth": "Not provided",
            "gender": "Male",
            "nationality": "Not provided",
            "occupation": "Not provided",
            "contact_information": {
                "phone": "Not provided",
                "email": "Not provided",
                "address": "Not provided",
            },
            "driver_license": {
                "license_number": "Not provided",
                "license_type": "Not provided",
                "issue_date": "Not provided",
                "expiry_date": "Not provided",
            },
            "insurance_information": {
                "insurance_company": "Not provided",
                "policy_number": "Not provided",
                "policy_type": "Not provided",
                "policy_expiry": "Not provided",
            },
            "vehicle_information": {
                "vehicle_type": "sedan",
                "make_brand": "Not specified",
                "model": "Not specified",
                "year": "Not specified",
                "color": "white",
                "registration_number": "92A-458.21",
                "vin_chassis_number": "Not provided",
                "engine_number": "Not provided",
            },
            "injuries": {
                "injury_status": "Not reported",
                "injury_description": "Not provided",
                "medical_treatment": "Not provided",
            },
        },
    ],
    "reported_damages": {
        "vehicle_damages": [
            {
                "vehicle_registration": "43D1-672.91",
                "owner_name": "Nguyen Minh Hoang",
                "damage_summary": "scratches and minor deformation from motorcycle sliding and falling",
                "damaged_parts": [
                    {
                        "part_name": "Front brake lever",
                        "damage_type": "bent",
                        "damage_severity": "Minor",
                        "damage_description": "Front brake lever bent",
                    },
                    {
                        "part_name": "Front fairing",
                        "damage_type": "scratches",
                        "damage_severity": "Minor",
                        "damage_description": "Scratches on front fairing",
                    },
                    {
                        "part_name": "Fuel tank",
                        "damage_type": "abrasion",
                        "damage_severity": "Minor",
                        "damage_description": "Fuel tank surface abrasion",
                    },
                    {
                        "part_name": "Front fork",
                        "damage_type": "bent",
                        "damage_severity": "Minor",
                        "damage_description": "Slightly bent front fork",
                    },
                    {
                        "part_name": "Right side mirror",
                        "damage_type": "broken",
                        "damage_severity": "Moderate",
                        "damage_description": "Right side mirror broken",
                    },
                ],
                "estimated_repair_cost": {
                    "amount": None,
                    "currency": None,
                    "cost_range": "Not provided",
                },
                "vehicle_condition": "Not specified",
            }
        ],
        "property_damages": [],
        "total_estimated_damages": {
            "amount": None,
            "currency": None,
            "notes": "Not provided",
        },
    },
    "incident_narrative": {
        "reporting_person_statement": (
            "At approximately 16:45 on 12 March 2024, I was operating my motorcycle, "
            "a Yamaha Exciter with registration number 43D1-672.91, traveling on Nguyen Huu Tho Street, "
            "heading toward Cam Le District. When approaching the area near Nguyen Tri Phuong Bridge, "
            "a white sedan with registration number 92A-458.21, driven by an unidentified male "
            "(estimated 35–40 years old), suddenly changed lanes from right to left without signaling. "
            "Due to the sudden maneuver, I was unable to react in time, resulting in a collision with the "
            "right side of the sedan. The motorcycle slid and fell, causing scratches and minor deformation. "
            "I sustained minor abrasions on my left hand. The driver briefly stopped but left the scene without "
            "providing information. I request the authorities to identify the driver and take action according "
            "to Vietnamese traffic laws."
        ),
        "other_driver_statement": "Driver left scene without providing statement",
        "witness_statements": [
            {
                "witness_name": "nearby shop owners",
                "witness_contact": "Not provided",
                "statement": "Witness statements collected but not detailed in report",
            }
        ],
        "police_officer_observations": "Not provided in detail",
        "sequence_of_events": (
            "Motorcycle traveling straight on Nguyen Huu Tho Street toward Cam Le District. "
            "White sedan suddenly changed lanes from right to left without signaling near Nguyen Tri Phuong Bridge. "
            "Motorcycle unable to avoid collision with right side of sedan. Motorcycle slid and fell. "
            "White sedan driver briefly stopped then fled scene."
        ),
        "contributing_factors": ["sudden lane change without signaling", "hit and run"],
        "preliminary_cause": "sudden lane change without signaling by white sedan driver",
        "additional_notes": "Driver fled scene after briefly stopping",
    },
    "official_actions": {
        "responding_officer": {
            "name": "Tran Quoc Thinh",
            "rank": "Senior Lieutenant",
            "badge_number": "Not provided",
            "department": "Traffic Police Division, Da Nang City Police",
        },
        "actions_taken": [
            "Collected witness statements from nearby shop owners",
            "Retrieved CCTV footage from a convenience store facing the roadway",
            "Verified vehicle registration number 92A-458.21 in the police database",
            "Inspected and documented the condition of the damaged motorcycle",
        ],
        "citations_issued": [],
        "evidence_collected": [
            "witness statements",
            "CCTV footage",
            "vehicle registration verification",
            "motorcycle damage documentation",
            "photographs of scene and damage",
        ],
        "investigation_status": "Under Investigation",
    },
    "supporting_documentation": {
        "photos_attached": {
            "count": 4,
            "descriptions": [
                "Overview of Nguyen Huu Tho Street: Three-lane roadway, clear weather, light traffic",
                "Motorcycle on roadside after fall: Yamaha Exciter lying on its side with visible scratches",
                "Damaged right mirror & handlebar",
                "Abrasion on fuel tank area",
            ],
        },
        "diagrams_included": True,
        "diagram_description": (
            "Accident sketch diagram showing three-lane roadway with lanes A, B, C, white sedan "
            "changing from lane B to C, motorcycle in lane C, collision point marked with X, "
            "direction toward Cam Le District and Nguyen Tri Phuong Bridge marked"
        ),
        "video_evidence": True,
        "cctv_footage": {
            "available": True,
            "source": "convenience store facing the roadway",
            "description": "CCTV footage retrieved from convenience store",
        },
        "other_documents": ["witness statements"],
    },
    "extracted_images": [
        {
            "image_number": 1,
            "image_type": "Scene Overview",
            "description": (
                "Three-lane roadway view of Nguyen Huu Tho Street showing clear weather conditions, "
                "light traffic, road signage indicating direction, with white sedan visible in right lane"
            ),
            "relevance": "Shows accident location and environmental conditions at time of incident",
        },
        {
            "image_number": 2,
            "image_type": "Damage Photo",
            "description": (
                "Yamaha Exciter motorcycle lying on roadside showing visible scratches and damage to fairing, "
                "positioned upright for documentation"
            ),
            "relevance": (
                "Documents overall condition and visible damage to reporting person's motorcycle"
            ),
        },
        {
            "image_number": 3,
            "image_type": "Damage Photo",
            "description": (
                "Close-up view of damaged right side mirror (broken/cracked) and handlebar area of the motorcycle"
            ),
            "relevance": (
                "Specific documentation of broken right side mirror and handlebar damage as listed in damage assessment"
            ),
        },
        {
            "image_number": 4,
            "image_type": "Damage Photo",
            "description": (
                "Close-up view of fuel tank showing surface abrasion/scratches on the black painted surface"
            ),
            "relevance": (
                "Documents fuel tank surface abrasion mentioned in damage assessment"
            ),
        },
        {
            "image_number": 5,
            "image_type": "Diagram",
            "description": (
                "Hand-drawn accident scene diagram showing three lanes (A, B, C), white sedan's lane change "
                "from B to C, motorcycle position, collision point marked with X, directional arrows showing "
                "traffic flow toward Cam Le District and Nguyen Tri Phuong Bridge"
            ),
            "relevance": (
                "Visual representation of accident sequence and vehicle positions at time of collision"
            ),
        },
    ],
    "data_quality_notes": {
        "missing_information": [
            "Report number/reference number",
            "Driver license information for reporting person",
            "Insurance information for both parties",
            "Exact cost estimates for repairs",
            "Complete contact information for other driver",
            "Detailed witness statements",
            "Vehicle VIN/chassis numbers",
            "Official stamp or seal",
        ],
        "unclear_information": [
            "Age estimate of other driver (35-40 years old)",
            "Exact time described as 'approximately 16:45'",
        ],
        "inconsistencies": [
            "Document labeled as 'Synthetic Example' suggesting this may be a template or practice document"
        ],
        "handwritten_sections": [
            "Signature of reporting person (Nguyen Minh Hoang)"
        ],
        "document_quality": "Good",
        "extraction_confidence": "High",
        "notes": (
            "Document appears to be a synthetic/example report based on header notation. "
            "All key information clearly presented and legible. Photos provide good documentation "
            "of damage. Accident diagram clearly illustrates sequence of events."
        ),
    },
    # Additional normalized pattern used in fraud checks
    "scene_pattern": {
        "impact_side": "right",
        "collision_type": "side impact during lane change",
    },
}

# Load metadata (with fallback)
try:
    CASE_METADATA = load_case_metadata()
    if CASE_METADATA is None:
        logger.info("No case metadata in Supabase (new table). Fraud checks will require explicit case_metadata parameter.")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.warning(f"Metadata loading failed: {e}. Using fallback metadata.")
    CASE_METADATA = FALLBACK_METADATA


# --- 2) REQUEST / RESPONSE SCHEMAS ---

class Location(BaseModel):
    """Location information model."""
    street: Optional[str] = None
    nearby: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


class FraudCheckRequest(BaseModel):
    """Request model for fraud check."""
    case_id: Optional[str] = None
    reported_time: Optional[str] = None  # ISO string
    reported_location: Optional[Location] = None
    reported_damages: List[str] = []
    narrative: Optional[str] = None
    vehicle_plate: Optional[str] = None  # For duplicate detection
    accident_date: Optional[str] = None  # For duplicate detection (YYYY-MM-DD)


class FraudCheckResponse(BaseModel):
    """Response model for fraud check."""
    fraud_score: float
    alerts: List[str]
    status: str


# --- 3) UTILITY FUNCTIONS ---

def parse_time(ts: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO format timestamp string to datetime object.
    
    Supports formats:
    - ISO 8601: 2024-03-12T16:45:00
    - Date only: 2024-03-12
    - Date + time: 2024-03-12T16:45
    
    Args:
        ts: ISO format timestamp string
    
    Returns:
        datetime object or None if input is None/empty or unparseable
    """
    if not ts:
        return None
    
    # Skip natural language descriptions
    if any(word in str(ts).lower() for word in ["approximately", "around", "about", "near", "close to"]):
        logger.debug(f"Skipping natural language time description: {ts}")
        return None
    
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        # Try common date-only format
        try:
            from datetime import date
            if len(ts) == 10 and ts.count('-') == 2:  # YYYY-MM-DD
                date_obj = datetime.strptime(ts, "%Y-%m-%d")
                return date_obj
        except ValueError:
            pass
        
        # Log only if it looks like it should be parseable but isn't
        if ts and len(ts) > 5 and any(c.isdigit() for c in ts):
            logger.debug(f"Could not parse timestamp: {ts}")
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS coordinates using Haversine formula.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
    
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth's radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# --- 4) MAIN FRAUD CHECK LOGIC ---

def perform_fraud_check(
    payload: FraudCheckRequest,
    case_metadata: Optional[Dict[str, Any]] = None,
    time_tolerance_hours: float = 6.0,
    gps_threshold_km: float = 2.0,
    damage_coverage_threshold: float = 0.4
) -> FraudCheckResponse:
    """
    Perform fraud detection checks on insurance claim data.
    
    Args:
        payload: Fraud check request containing reported claim information
        case_metadata: Optional reference case metadata for comparison checks (defaults to global CASE_METADATA)
        time_tolerance_hours: Maximum time difference in hours before flagging (default: 6.0)
        gps_threshold_km: Maximum GPS distance in km before flagging (default: 2.0)
        damage_coverage_threshold: Minimum ratio of expected damages that must be reported (default: 0.4)
    
    Returns:
        FraudCheckResponse withs fraud score, alerts, and status
    """
    # Use provided case_metadata or fall back to global CASE_METADATA
    if case_metadata is None:
        case_metadata = CASE_METADATA
    
    # Note: case_metadata might be None for new tables
    # We'll still run duplicate/embedding checks, but skip comparison-based checks
    has_reference_data = case_metadata is not None
    
    if not has_reference_data:
        logger.info(
            "No reference case metadata available. "
            "Will run duplicate/embedding checks but skip comparison-based fraud detection."
        )
    
    # --- WEIGHTED FRAUD SCORING SETUP ---
    # Define alert weights (higher = more suspicious)
    ALERT_WEIGHTS = {
        'duplicate_submission': 3.0,      # Same accident submitted multiple times (exact match)
        'embedding_similarity_high': 2.8, # Very similar case found (>0.95 similarity)
        'embedding_similarity_med': 2.0,  # Similar case found (>0.90 similarity)
        'time_mismatch_severe': 2.0,      # >24h difference
        'time_mismatch': 1.0,             # <=24h difference
        'gps_mismatch': 3.0,              # GPS coordinates don't match
        'city_mismatch': 1.5,             # City name mismatch
        'street_mismatch': 1.2,           # Street name mismatch
        'damage_mismatch': 2.0,           # Damage patterns inconsistent
        'damage_suspicious': 0.5,         # Suspicious damage (engine/frame)
        'narrative_mismatch': 1.5,        # Narrative inconsistent with evidence
        'recent_policy': 2.5,             # Policy <30 days before accident
        'high_claim_freq': 2.0,           # 3+ prior claims
    }
    
    alerts: List[str] = []
    total_alert_weight: float = 0.0    # Sum of triggered alert weights
    max_possible_weight: float = 0.0   # Sum of all check weights performed

    # --- DUPLICATE SUBMISSION CHECK ---
    # Weight: 3.0 - High priority (indicates potential fraud pattern)
    if supabase_vector_client is not None and payload.vehicle_plate and payload.accident_date:
        try:
            max_possible_weight += ALERT_WEIGHTS['duplicate_submission']
            logger.info(f"Checking for duplicate submissions: vehicle={payload.vehicle_plate}, date={payload.accident_date}")
            
            result = supabase_vector_client.client.table("accident_cases")\
                .select("case_id, created_at, is_fraud, fraud_score")\
                .eq("vehicle_plate", payload.vehicle_plate)\
                .eq("accident_date", payload.accident_date)\
                .execute()
            
            # Debug: Log what was returned
            logger.info(f"Duplicate check query returned {len(result.data) if result.data else 0} records")
            if result.data:
                logger.info(f"Records found: {[r.get('case_id') for r in result.data]}")
            
            if result.data and len(result.data) > 0:
                # Filter out cases with "Not Specified" or similar invalid case IDs
                valid_duplicates = [
                    r for r in result.data 
                    if r.get("case_id") and r.get("case_id") not in [
                        "Not Specified", "Not specified", "not specified", 
                        "Not Provided", "Not provided", "not provided",
                        "N/A", "n/a", "null", "None"
                    ]
                ]
                
                logger.info(f"After filtering: {len(valid_duplicates)} valid duplicate(s)")
                if result.data and not valid_duplicates:
                    filtered_ids = [r.get("case_id") for r in result.data]
                    logger.info(f"Filtered out invalid case IDs: {filtered_ids}")
                
                if valid_duplicates:
                    duplicate_count = len(valid_duplicates)
                    case_ids = [r.get("case_id") for r in valid_duplicates]
                    weight = ALERT_WEIGHTS['duplicate_submission']
                    alerts.append(
                        f"Duplicate submission detected: {duplicate_count} existing case(s) found "
                        f"for vehicle {payload.vehicle_plate} on {payload.accident_date} "
                        f"(case IDs: {', '.join(case_ids[:3])}) [WEIGHT: {weight}]"
                    )
                    total_alert_weight += weight
                    logger.warning(f"Duplicate submission detected: {duplicate_count} valid cases found")
                else:
                    logger.debug(f"Found {len(result.data)} case(s) but all have invalid/unspecified case IDs")
        except Exception as e:
            logger.warning(f"Error checking for duplicate submissions: {e}")

    # --- EMBEDDING SIMILARITY CHECK (VECTOR SEARCH) ---
    # Weight: 2.8 for high similarity (>0.95), 2.0 for medium (>0.90)
    # Uses pgvector cosine similarity to find semantically similar cases
    if supabase_vector_client is not None and voyage_client is not None and payload.narrative:
        try:
            logger.info("Performing embedding similarity check...")
            
            # Generate embedding for current narrative
            query_embedding = voyage_client.generate_embedding(payload.narrative)
            logger.info(f"Generated query embedding of dimension {len(query_embedding)}")
            
            # Search for similar cases using Supabase RPC
            # This requires a search_similar_cases RPC function in Supabase
            try:
                similar_results = supabase_vector_client.client.rpc(
                    'search_similar_accident_cases',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': 0.85,  # 85% similarity minimum
                        'match_count': 5
                    }
                ).execute()
                
                if similar_results.data and len(similar_results.data) > 0:
                    logger.info(f"Found {len(similar_results.data)} similar cases")
                    
                    # Filter out invalid case IDs
                    valid_similar = [
                        r for r in similar_results.data
                        if r.get('case_id') and r.get('case_id') not in [
                            "Not Specified", "Not specified", "not specified",
                            "Not Provided", "Not provided", "not provided",
                            "N/A", "n/a", "null", "None"
                        ]
                    ]
                    
                    for similar_case in valid_similar:
                        similarity = similar_case.get('similarity', 0.0)
                        similar_case_id = similar_case.get('case_id', 'Unknown')
                        
                        # High similarity (>0.95) - very suspicious
                        if similarity > 0.95:
                            weight = ALERT_WEIGHTS['embedding_similarity_high']
                            max_possible_weight += weight
                            alerts.append(
                                f"Very similar case detected: {similarity:.2%} similarity with case {similar_case_id} "
                                f"(potential duplicate or fabricated claim). [WEIGHT: {weight}]"
                            )
                            total_alert_weight += weight
                            logger.warning(f"High similarity detected: {similarity:.2%} with {similar_case_id}")
                            break  # Only flag the most similar case
                        
                        # Medium similarity (>0.90) - moderately suspicious
                        elif similarity > 0.90:
                            weight = ALERT_WEIGHTS['embedding_similarity_med']
                            max_possible_weight += weight
                            alerts.append(
                                f"Similar case detected: {similarity:.2%} similarity with case {similar_case_id}. [WEIGHT: {weight}]"
                            )
                            total_alert_weight += weight
                            logger.info(f"Medium similarity detected: {similarity:.2%} with {similar_case_id}")
                            break  # Only flag the most similar case
                else:
                    logger.info("No similar cases found in embedding search")
                    
            except Exception as rpc_error:
                # RPC function might not exist yet - provide helpful error
                if "function" in str(rpc_error).lower() and "does not exist" in str(rpc_error).lower():
                    logger.warning(
                        "RPC function 'search_similar_accident_cases' not found. "
                        "Skipping embedding similarity check. See documentation for SQL setup."
                    )
                else:
                    logger.warning(f"Error in embedding similarity search: {rpc_error}")
                    
        except Exception as e:
            logger.warning(f"Error performing embedding similarity check: {e}")

    # Normalize expected fields from case_metadata (supports legacy and new schema)
    def _normalize_expected(meta: Dict[str, Any]) -> Dict[str, Any]:
        expected = {
            "time": None,
            "city": None,
            "street": None,
            "gps": {"lat": None, "lng": None},
            "damages": [],
            "impact_side": None,
        }

        # Time - prioritize structured datetime over description
        acc_info = meta.get("accident_information", {}) if isinstance(meta, dict) else {}
        acc_date = acc_info.get("accident_date")
        acc_time = acc_info.get("accident_time")
        acc_dt_full = acc_info.get("accident_datetime_full")
        
        # Build time from structured data (ignore accident_datetime_description - it's natural language)
        if acc_dt_full and not any(word in str(acc_dt_full).lower() for word in ["approximately", "around", "about"]):
            # Use accident_datetime_full only if it's not descriptive text
            expected["time"] = acc_dt_full
        elif acc_date and acc_time:
            # Combine date and time into ISO format
            expected["time"] = f"{acc_date}T{acc_time}"
        elif acc_date:
            # Use just the date
            expected["time"] = acc_date
        else:
            # Legacy fallback
            expected["time"] = meta.get("accident_time")

        # Location
        loc = acc_info.get("location", {}) if isinstance(acc_info, dict) else {}
        if loc:
            expected["street"] = loc.get("street_address") or loc.get("street") or loc.get("location_description")
            expected["city"] = loc.get("city_province") or loc.get("city")
            gps = loc.get("gps_coordinates", {}) if isinstance(loc, dict) else {}
            expected["gps"]["lat"] = gps.get("latitude")
            expected["gps"]["lng"] = gps.get("longitude")
        else:
            # legacy
            loc_legacy = meta.get("accident_location", {}) if isinstance(meta, dict) else {}
            expected["street"] = loc_legacy.get("street") or loc_legacy.get("location_description")
            expected["city"] = loc_legacy.get("city")
            gps = loc_legacy.get("gps", {}) if isinstance(loc_legacy, dict) else {}
            expected["gps"]["lat"] = gps.get("lat")
            expected["gps"]["lng"] = gps.get("lng")

        # Damages
        damages = meta.get("reported_damages", {})
        dmg_list: List[str] = []
        if isinstance(damages, dict):
            vehicle_damages = damages.get("vehicle_damages") or []
            for vehicle_damage in vehicle_damages:
                parts = vehicle_damage.get("damaged_parts") or vehicle_damage.get("damage_list") or []
                for part in parts:
                    if isinstance(part, dict):
                        name = part.get("part_name") or part.get("damage_description")
                        if name:
                            dmg_list.append(name.lower())
                    elif isinstance(part, str):
                        dmg_list.append(part.lower())
        if not dmg_list:
            # legacy motorcycle damages list
            dmg_list = [d.lower() for d in meta.get("motorcycle", {}).get("damages_reported", [])]
        expected["damages"] = dmg_list

        # Impact side (if available)
        expected["impact_side"] = meta.get("scene_pattern", {}).get("impact_side")
        return expected

    # --- COMPARISON-BASED CHECKS (requires reference case_metadata) ---
    if has_reference_data:
        expected_norm = _normalize_expected(case_metadata)

        # 4.1 Time consistency
        # Weight: 2.0 if >24h, 1.0 if <=24h (severe vs minor mismatch)
        expected_time = parse_time(expected_norm["time"])
        reported_time = parse_time(payload.reported_time)

        if expected_time and reported_time:
            delta_hours = abs((reported_time - expected_time).total_seconds()) / 3600
            if delta_hours > time_tolerance_hours:
                # Determine weight based on severity
                if delta_hours > 24:
                    weight = ALERT_WEIGHTS['time_mismatch_severe']
                    max_possible_weight += weight
                else:
                    weight = ALERT_WEIGHTS['time_mismatch']
                    max_possible_weight += weight
                
                alerts.append(
                    f"Time mismatch: reported accident time differs by {delta_hours:.1f} hours "
                    f"from official accident time. [WEIGHT: {weight}]"
                )
                total_alert_weight += weight
            else:
                # No alert, but this check was performed
                weight = ALERT_WEIGHTS['time_mismatch_severe'] if delta_hours > 24 else ALERT_WEIGHTS['time_mismatch']
                max_possible_weight += weight

        # 4.2 Location consistency
        # Use fuzzy matching for city/street comparison (fallback to exact match)
        try:
            from rapidfuzz import fuzz
            fuzzy_available = True
        except ImportError:
            fuzzy_available = False
            logger.debug("rapidfuzz not available, using exact matching")
        
        expected_city = (expected_norm["city"] or "").lower() if expected_norm["city"] else ""
        expected_street = (expected_norm["street"] or "").lower() if expected_norm["street"] else ""
        expected_gps = expected_norm.get("gps", {})
        reported_loc = payload.reported_location

        if reported_loc:
            # 4.2.1 City match with fuzzy logic
            # Weight: 1.5 - City mismatch indicates different location
            if reported_loc.city and expected_city:
                max_possible_weight += ALERT_WEIGHTS['city_mismatch']
                city_match = False
                
                if fuzzy_available:
                    similarity = fuzz.ratio(reported_loc.city.lower(), expected_city)
                    city_match = similarity >= 80  # 80% similarity threshold
                else:
                    city_match = reported_loc.city.lower() == expected_city
                
                if not city_match:
                    weight = ALERT_WEIGHTS['city_mismatch']
                    alerts.append(
                        f"Location mismatch: reported city '{reported_loc.city}' "
                        f"differs from official city '{expected_norm.get('city', 'unknown')}'. [WEIGHT: {weight}]"
                    )
                    total_alert_weight += weight

            # 4.2.2 Street-level check with fuzzy matching
            # Weight: 1.2 - Street mismatch less critical than city
            if reported_loc.street and expected_street:
                max_possible_weight += ALERT_WEIGHTS['street_mismatch']
                street_match = False
                
                if fuzzy_available:
                    similarity = fuzz.partial_ratio(reported_loc.street.lower(), expected_street)
                    street_match = similarity >= 70  # 70% similarity threshold
                else:
                    street_match = expected_street in reported_loc.street.lower()
                
                if not city_match:
                    weight = ALERT_WEIGHTS['street_mismatch']
                    alerts.append(
                        f"Street mismatch: reported street '{reported_loc.street}' "
                        f"does not match official street '{expected_norm.get('street', 'unknown')}'. [WEIGHT: {weight}]"
                    )
                    total_alert_weight += weight

            # 4.2.3 GPS distance check (most reliable)
            # Weight: 3.0 - GPS mismatch is strong indicator of fraud
            if reported_loc.lat is not None and reported_loc.lng is not None:
                if expected_gps.get("lat") is not None and expected_gps.get("lng") is not None:
                    max_possible_weight += ALERT_WEIGHTS['gps_mismatch']
                    dist_km = haversine_km(
                        reported_loc.lat, reported_loc.lng,
                        expected_gps["lat"], expected_gps["lng"]
                    )
                    if dist_km > gps_threshold_km:
                        weight = ALERT_WEIGHTS['gps_mismatch']
                        alerts.append(
                            f"GPS mismatch: reported coordinates are {dist_km:.2f} km "
                            f"away from documented accident location. [WEIGHT: {weight}]"
                        )
                        total_alert_weight += weight

        # 4.3 Damage pattern consistency
        # Weight: 2.0 base + 0.5 if suspicious high-value parts (engine/frame) mentioned
        expected_damages = expected_norm.get("damages", [])
        reported_damages = [d.lower().strip() for d in payload.reported_damages]

        if reported_damages:
            max_possible_weight += ALERT_WEIGHTS['damage_mismatch']
            
            # Check overlap between reported and expected damages
            normalized_expected = {d.lower() for d in expected_damages}
            overlap = normalized_expected.intersection(set(reported_damages))
            coverage_ratio = len(overlap) / max(1, len(normalized_expected))

            if coverage_ratio < damage_coverage_threshold:
                weight = ALERT_WEIGHTS['damage_mismatch']
                expected_damages_str = ", ".join(expected_damages)
                alerts.append(
                    f"Damage pattern mismatch: reported damaged parts are largely "
                    f"inconsistent with documented damages ({expected_damages_str}). [WEIGHT: {weight}]"
                )
                total_alert_weight += weight
            
            # Check for suspicious high-value damage not in expected
            # Weight: 0.5 bonus - engine/frame damage often indicates fraudulent inflation
            suspicious_parts = {'engine', 'frame', 'transmission', 'axle'}
            reported_set = set(reported_damages)
            expected_set = set(expected_damages)
            
            suspicious_claimed = suspicious_parts.intersection(reported_set) - suspicious_parts.intersection(expected_set)
            if suspicious_claimed:
                max_possible_weight += ALERT_WEIGHTS['damage_suspicious']
                weight = ALERT_WEIGHTS['damage_suspicious']
                alerts.append(
                    f"Suspicious damage claimed: high-value parts ({', '.join(suspicious_claimed)}) "
                    f"reported but not documented in official report. [WEIGHT: {weight}]"
                )
                total_alert_weight += weight

        # 4.4 Narrative vs scene pattern
        # Weight: 1.5 - Narrative inconsistency suggests fabrication
        if payload.narrative and expected_norm.get("impact_side"):
            max_possible_weight += ALERT_WEIGHTS['narrative_mismatch']
            narrative = payload.narrative.lower()
            expected_impact_side = expected_norm.get("impact_side", "")

            if "rear" in narrative and expected_impact_side == "right":
                weight = ALERT_WEIGHTS['narrative_mismatch']
                alerts.append(
                    "Narrative mismatch: user describes a rear-end collision, but "
                    f"official report describes a right-side impact during lane change. [WEIGHT: {weight}]"
                )
                total_alert_weight += weight

        # 4.5 Recent policy check (NEW)
        # Weight: 2.5 - Policy started <30 days before accident (common fraud pattern)
        policy_data = case_metadata.get("policy", {}) if isinstance(case_metadata, dict) else {}
        policy_start_date_str = policy_data.get("start_date")
        
        if policy_start_date_str and payload.accident_date:
            try:
                max_possible_weight += ALERT_WEIGHTS['recent_policy']
                policy_start = parse_time(policy_start_date_str)
                accident_date = parse_time(payload.accident_date)
                
                if policy_start and accident_date:
                    days_difference = (accident_date - policy_start).days
                    if 0 <= days_difference < 30:
                        weight = ALERT_WEIGHTS['recent_policy']
                        alerts.append(
                            f"Recent policy: insurance policy started only {days_difference} days "
                            f"before accident (common fraud indicator). [WEIGHT: {weight}]"
                        )
                        total_alert_weight += weight
                        logger.warning(f"Recent policy detected: {days_difference} days between policy start and accident")
            except Exception as e:
                logger.debug(f"Could not check recent policy: {e}")

        # 4.6 High claim frequency check (NEW)
        # Weight: 2.0 - Multiple prior claims (3+) suggests fraud pattern
        claim_history = case_metadata.get("claim_history", {}) if isinstance(case_metadata, dict) else {}
        prior_claims_count = claim_history.get("count", 0)
        
        if prior_claims_count is not None:
            max_possible_weight += ALERT_WEIGHTS['high_claim_freq']
            if prior_claims_count >= 3:
                weight = ALERT_WEIGHTS['high_claim_freq']
                alerts.append(
                    f"High claim frequency: {prior_claims_count} prior claims on record "
                    f"(suggests possible fraud pattern). [WEIGHT: {weight}]"
                )
                total_alert_weight += weight
                logger.warning(f"High claim frequency detected: {prior_claims_count} prior claims")
    else:
        # No reference data - comparison checks skipped
        logger.info("Comparison-based fraud checks skipped (no reference case metadata)")

    # --- WEIGHTED FRAUD SCORE CALCULATION ---
    # Formula: (sum of triggered alert weights) / (sum of all check weights performed)
    # This gives a score between 0.0 and 1.0 that accounts for alert severity
    if max_possible_weight == 0.0:
        # No checks were possible (missing data)
        fraud_score = 0.0
        logger.info("No fraud checks could be performed (insufficient data)")
    else:
        fraud_score = total_alert_weight / max_possible_weight
        logger.info(
            f"Fraud score calculated: {fraud_score:.3f} "
            f"(alert weight: {total_alert_weight:.1f} / max weight: {max_possible_weight:.1f})"
        )

    # Status thresholds remain unchanged
    status = (
        "High risk – manual review required" if fraud_score >= 0.6
        else "Medium risk – review recommended" if fraud_score >= 0.3
        else "Low risk – no major inconsistencies detected"
    )

    return FraudCheckResponse(
        fraud_score=round(fraud_score, 2),
        alerts=alerts,
        status=status
    )
