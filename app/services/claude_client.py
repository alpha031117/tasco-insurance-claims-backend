"""
Claude API client for police report analysis.
"""
from anthropic import Anthropic
from app.config import settings
from app.utils.logger import logger


class ClaudeClient:
    """Client for interacting with Claude API."""
    
    def __init__(self):
        """Initialize Claude client with API key from environment."""
        try:
            self.client = Anthropic(api_key=settings.anthropic_api_key)
            logger.info("Claude client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise

    def analyze_police_report(self, base64_data_list: list) -> dict:
        """
        Analyze police report using Claude AI with PDF documents.
        
        Args:
            base64_data_list: List of base64 encoded PDF documents
            
        Returns:
            Analysis results from Claude
        """
        try: 
            # Add documents to content
            content = []
            
            for base64_data in base64_data_list:
                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_data
                    }
                })
            
            content.append({
                    "type": "text",
                    "text": "Please analyze the attached Vietnamese traffic accident report and provide a comprehensive insurance claims assessment.\n\nUsing the provided documents:\n1. **Accident Report**: Contains the incident details, damage description, and witness statements\n2. **Vietnam Road Traffic Law (Law No. 36/2024/QH15)**: Use this to determine legal liability\n3. **Vehicle Damage Compendium**: Use this to estimate repair costs\n\nI need you to:\n\n1. **Extract all relevant information** from the accident report including parties involved, vehicles, location, and circumstances\n\n2. **Determine liability** by analyzing the incident against Vietnamese traffic law:\n   - Identify which party violated traffic regulations\n   - Cite specific articles, chapters, and clauses from the Road Traffic Law\n   - Explain how the violations led to the accident\n   - Assign fault percentages\n\n3. **Assess claimable damages**:\n   - Match the reported motorcycle damage to items in the compendium table\n   - Provide cost estimates for each damaged component\n   - Calculate total estimated repair costs (minimum and maximum range)\n\n4. **Return the analysis as a properly formatted JSON object** following the structure specified in the system prompt\n\nPlease be thorough and ensure all legal citations are accurate and specific. The assessment will be used for insurance claim processing, so precision is critical."
                })
            
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=20000,
                temperature=1,
                system="You are an expert insurance claims assessor specializing in Vietnamese road traffic accidents. Your role is to analyze accident reports and provide comprehensive liability assessments based on the Law on Road Traffic Order and Safety (Law No. 36/2024/QH15).\n\nWhen analyzing accident reports, you must:\n\n1. **Extract Key Information**: Identify all parties involved, vehicles, location, time, and circumstances of the accident.\n\n2. **Apply Vietnamese Traffic Law**: Reference specific articles from the Law on Road Traffic Order and Safety to determine fault and liability. Pay particular attention to:\n   - Road traffic rules (Chapter II, Articles 10-33)\n   - Lane usage and overtaking regulations\n   - Right-of-way rules\n   - Signal compliance requirements\n   - Prohibited actions (Article 9)\n\n3. **Assess Liability**: Provide a clear determination of fault with detailed legal justification citing specific article numbers and clauses.\n\n4. **Estimate Damages**: Match reported damage to the vehicle damage compendium, providing cost ranges in VND for each claimable item.\n\n5. **Output Format**: Return your analysis as a JSON object with the following structure:\n```json\n{\n  \"insurer\": {\n    \"policy_holder\": \"string\",\n    \"vehicle_registration\": \"string\",\n    \"age\": \"int\",\n    \"gender\": \"string\",\n   \"occupational\": \"string\"\n  },\n  \"accident_info\": {\n    \"date\": \"string\",\n    \"time\": \"string\",\n    \"location\": \"string\",\n    \"weather_conditions\": \"string\",\n    \"traffic_conditions\": \"string\",\n    \"parties_involved\": [\n      {\n        \"party_type\": \"string (e.g., 'Reporting Person', 'Other Driver')\",\n        \"name\": \"string\",\n        \"vehicle_type\": \"string\",\n        \"vehicle_registration\": \"string\",\n        \"description\": \"string\"\n      }\n    ],\n    \"incident_summary\": \"string (brief description of what happened)\"\n  },\n  \"liability_assessment\": {\n    \"primary_fault_party\": \"string\",\n    \"fault_percentage\": \"string (e.g., '100%', '70-30%')\",\n    \"legal_justification\": \"string (detailed explanation with specific article citations)\",\n    \"violated_articles\": [\n      {\n        \"article_number\": \"string\",\n        \"article_title\": \"string\",\n        \"clause\": \"string\",\n        \"violation_description\": \"string\"\n      }\n    ],\n    \"mitigating_factors\": \"string (if any)\",\n    \"conclusion\": \"string\"\n  },\n  \"claimable_damages\": [\n    {\n      \"item\": \"string\",\n      \"damage_category\": \"string (Minor/Moderate/Severe)\",\n      \"cost_range_vnd\": \"string (e.g., '1,000,000 – 3,750,000')\",\n      \"justification\": \"string (how this matches the accident report)\"\n    }\n  ],\n  \"total_estimated_cost\": {\n    \"minimum_vnd\": \"number\",\n    \"maximum_vnd\": \"number\",\n    \"range_display\": \"string\"\n  }\n}\n```\n\nBe thorough, precise, and base all assessments strictly on the provided documents. If information is missing or unclear, note this in your response.",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            
            logger.info("Claude Report Summarizer Completed Successfully")
            return {
                "success": True,
                "response": message.content[0].text,
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error during Claude Report Summarizer: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None,
                "usage": None
            }

    def analyze_spare_part_recommendation(self, base64_data_list: list) -> dict:
        """
        Analyze spare part recommendation using Claude AI with image file(s).
        
        Args:
            base64_data_list: List of base64 encoded image file(s)
            
        Returns:
            Analysis results from Claude
        """
        try: 
            # Add documents to content
            content = []
            
            for idx, base64_data in enumerate(base64_data_list):
                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        # Accepts common image types: jpeg, png, jpg
                        "media_type": "image/png",  # Most Claude APIs use mime type per input, so we use "image/jpeg" for jpg/jpeg, but can be "image/c" if needed.
                        "data": base64_data
                    },
                    "name": f"Image {idx+1}"
                })
            
            content.append({
                    "type": "text",
                    "text": "Please analyze the provided vehicle damage images and provide a comprehensive damage assessment report.\n\n**Your Task**:\n\nExamine all images carefully and identify:\n\n1. **Vehicle Information**:\n   - What type of vehicle is this? (motorcycle/car/truck/etc.)\n   - Can you identify the brand and model?\n   - What is the vehicle's color?\n   - Is the license plate visible? If so, what is the number?\n   - Any other identifying features?\n\n2. **Damage Assessment**:\n   - What parts are damaged?\n   - What type of damage does each part have? (scratch/crack/dent/broken/shattered/deformed/missing)\n   - What is the severity of each damage? (Minor/Moderate/Severe)\n   - Where exactly is each damaged part located on the vehicle?\n   - Should each part be repaired or replaced?\n\n3. **Impact Analysis**:\n   - Where did the primary impact occur?\n   - What direction did the impact come from?\n   - How severe was the impact?\n   - What type of collision does the damage pattern suggest?\n\n4. **Safety Concerns**:\n   - Is there any structural damage?\n   - Are any safety-critical components damaged?\n   - Are there areas that may have hidden damage requiring inspection?\n\n5. **Additional Observations**:\n   - Are there any signs of pre-existing damage unrelated to this incident?\n   - What do the damage patterns tell us about how the accident occurred?\n   - What parts should be inspected even if no visible damage is apparent?\n\n**Output Requirements**:\n- Provide your assessment in the JSON format specified in the system prompt\n- Be thorough and precise in identifying all damaged parts\n- Use proper automotive terminology\n- Clearly distinguish between different severity levels\n- Flag all safety-critical damage\n- Note any limitations due to image angles or quality\n\nPlease provide a detailed, professional assessment suitable for insurance claim processing and repair estimation."
                })
            
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=20000,
                temperature=1,
                system="You are an expert vehicle damage assessment specialist with extensive experience in analyzing accident damage for insurance claims and repair estimates. Your role is to carefully examine vehicle damage images and provide detailed, accurate assessments of all visible damage.\n\nWhen analyzing vehicle damage images, you must:\n\n1. **Vehicle Identification**: Determine and document:\n   - Vehicle type (car/motorcycle/.etc)\n   - Brand/manufacturer (if identifiable from logos, badges, or design)\n   - Model (if identifiable)\n   - Color and paint condition\n   - License plate number (if visible and legible)\n   - Any distinguishing features or modifications\n\n2. **Damage Assessment**: For each damaged component, identify:\n   - Exact part name using proper automotive terminology\n   - Type of damage (scratch, crack, dent, break, deformation, missing, shattered)\n   - Severity level (Minor/Moderate/Severe)\n   - Specific location on vehicle (front/rear/left/right side, upper/lower, specific area)\n   - Whether repair is possible or replacement is required\n\n3. **Impact Analysis**: Determine:\n   - Primary impact location and direction\n   - Secondary damage areas (collateral damage from primary impact)\n   - Estimated force/severity of impact based on damage patterns\n   - Type of collision (side impact, rear-end, frontal, rollover, slide/fall for motorcycles)\n\n4. **Safety and Structural Assessment**:\n   - Identify any structural damage that may affect vehicle safety\n   - Flag damage to safety-critical components (frame, suspension, brakes, steering)\n   - Note any damage that may not be immediately visible but should be inspected\n\n5. **Output Format**: Return your analysis as a JSON object with the following structure:\n```json\n{\n  \"vehicle_identification\": {\n    \"vehicle_type\": \"string (motorcycle/car/truck/suv/etc.)\",\n    \"brand\": \"string (manufacturer name if identifiable, or 'Not identifiable')\",\n    \"model\": \"string (model name if identifiable, or 'Not identifiable')\",\n    \"color\": \"string (primary color)\",\n    \"license_plate\": \"string (plate number if visible, or 'Not visible')\",\n    \"additional_details\": \"string (any other identifying features)\"\n  },\n  \"overall_assessment\": {\n    \"overall_damage_severity\": \"string (Minor/Moderate/Severe/Total Loss)\",\n    \"estimated_impact_type\": \"string (e.g., side collision, frontal impact, slide/fall, rear-end)\",\n    \"primary_impact_location\": \"string (main point of impact)\",\n    \"impact_direction\": \"string (e.g., from left side, from front-right, downward fall)\",\n    \"estimated_impact_force\": \"string (Low/Medium/High based on damage extent)\",\n    \"damage_summary\": \"string (2-3 sentence overview of all damage)\"\n  },\n  \"damaged_parts\": [\n    {\n      \"part_name\": \"string (specific part name, e.g., 'Right Side Mirror', 'Fuel Tank', 'Front Brake Lever')\",\n      \"part_category\": \"string (e.g., Body Panel, Mechanical, Electrical, Glass, Frame)\",\n      \"damage_type\": \"string (scratch/crack/dent/broken/shattered/deformed/missing)\",\n      \"severity\": \"string (Minor/Moderate/Severe)\",\n      \"location\": \"string (precise location on vehicle)\",\n      \"damage_description\": \"string (detailed description of the specific damage)\",\n      \"repair_or_replace\": \"string (Repair/Replace/Inspect)\",\n      \"repair_feasibility\": \"string (explanation of why repair or replacement is recommended)\",\n      \"safety_critical\": \"boolean (true if part is safety-critical)\",\n      \"visible_in_images\": [\"string (image numbers where this damage is visible, e.g., 'Image 1', 'Image 3')\"]\n    }\n  ],\n  \"structural_concerns\": {\n    \"frame_damage\": \"string (description or 'None visible')\",\n    \"alignment_issues\": \"string (description or 'None visible')\",\n    \"hidden_damage_risk\": \"string (areas that may have hidden damage requiring inspection)\",\n    \"safety_implications\": \"string (any safety concerns from visible damage)\"\n  },\n  \"additional_observations\": {\n    \"pre_existing_damage\": [\"string (any damage that appears old/unrelated to current incident)\"],\n    \"environmental_factors\": \"string (road conditions, weather indicators visible in images)\",\n    \"impact_markers\": [\"string (specific visual clues about how impact occurred)\"],\n    \"parts_requiring_inspection\": [\"string (parts that should be professionally inspected even if damage not visible)\"]\n  },\n  \"image_quality_notes\": {\n    \"overall_clarity\": \"string (Good/Fair/Poor)\",\n    \"angles_provided\": [\"string (angles shown in images, e.g., 'front-left', 'close-up damage', 'overall view')\"],\n    \"missing_angles\": [\"string (important angles not shown that would help assessment)\"],\n    \"limitations\": \"string (any limitations in assessment due to image quality or angles)\"\n  }\n}\n```\n\n**Analysis Guidelines**:\n- Be thorough and systematic in examining every visible part of the vehicle\n- Use precise automotive terminology for all parts\n- Consider both visible damage and potential hidden damage\n- Distinguish between primary impact damage and secondary/collateral damage\n- Note any damage that appears inconsistent with the primary impact (may indicate multiple impacts or pre-existing damage)\n- Be conservative with severity ratings - when in doubt, rate higher and recommend inspection\n- Always flag safety-critical damage clearly\n- If multiple images show the same damage from different angles, reference all relevant images\n- If image quality limits your assessment, state this clearly\n\n**Important Notes**:\n- Focus only on objective, visible damage - do not speculate beyond what can be seen\n- If license plates or identifying information are partially obscured, note what is visible\n- Distinguish between dirt/debris and actual damage\n- Consider the vehicle's apparent age and condition when assessing damage\n- Be especially careful with structural components and safety systems",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            
            logger.info("Claude Spare Part Recommendation Completed Successfully")
            return {
                "success": True,
                "response": message.content[0].text,
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error during Claude Spare Part Recommendation: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None,
                "usage": None
            }

    def metadata_extraction(self, content: list) -> dict:
        """
        Extract metadata from the provided base64 encoded data.
        """
        try:
            content.append({
                "type": "text",
                "text": "Please extract all claim metadata and structured information from the provided accident report document.\n"
            })
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=20000,
                temperature=1,
                system="You are an expert insurance claims processor specializing in extracting and structuring information from accident reports and insurance claim documents. Your role is to carefully analyze claim documents (in PDF format with accompanying images) and extract all relevant metadata in a structured, accurate format.\n\nWhen processing claim documents, you must:\n\n1. **Document Analysis**: Thoroughly examine the entire document including:\n   - Text content (typed and handwritten if present)\n   - Embedded images, diagrams, and photographs\n   - Tables, forms, and structured data\n   - Signatures, stamps, and official markings\n   - Any annotations or notes\n\n2. **Information Extraction**: Extract and structure the following key information:\n   \n   **Temporal Information**:\n   - Date of accident (exact date)\n   - Time of accident (exact time or time range)\n   - Date of report filing\n   - Any other relevant dates/timestamps\n\n   **Location Information**:\n   - Specific street address or road name\n   - Intersection or landmark details\n   - District/ward/commune\n   - City/province\n   - Country\n   - GPS coordinates (if provided)\n   - Location description or context\n\n   **Parties Involved**:\n   - Names of all parties (claimants, witnesses, other drivers)\n   - Contact information (phone, address, email)\n   - National ID/identification numbers\n   - Vehicle registration numbers\n   - Insurance policy numbers\n   - Driver's license information\n\n   **Vehicle Information**:\n   - Make, model, and year\n   - Registration/license plate numbers\n   - Vehicle type (motorcycle/car/truck/etc.)\n   - Color\n   - VIN/chassis number (if available)\n\n   **Damage Information**:\n   - List of damaged parts/components\n   - Damage descriptions for each part\n   - Severity assessments\n   - Preliminary cost estimates (if provided)\n\n   **Incident Narrative**:\n   - Detailed description of how accident occurred\n   - Sequence of events\n   - Contributing factors (weather, road conditions, visibility)\n   - Witness statements\n   - Police observations or findings\n\n   **Official Information**:\n   - Report number/reference number\n   - Reporting authority (police station, insurance company)\n   - Officer names and badge numbers\n   - Official stamps or seals\n   - Case status\n\n   **Supporting Evidence**:\n   - Number and description of attached photos\n   - Diagrams or accident scene sketches\n   - CCTV or video evidence mentioned\n   - Other supporting documents referenced\n\n3. **Data Validation and Quality**:\n   - Distinguish between confirmed facts and statements/claims\n   - Note any inconsistencies in the document\n   - Flag missing critical information\n   - Identify handwritten vs. printed information\n   - Note document language (if not in English)\n\n4. **Output Format**: Return your extraction as a JSON object with the following structure:\n```json\n{\n  \"document_metadata\": {\n    \"document_type\": \"string (e.g., 'Police Traffic Accident Report', 'Insurance Claim Form')\",\n    \"document_number\": \"string (report/claim number)\",\n    \"issuing_authority\": \"string (police department, insurance company, etc.)\",\n    \"report_date\": \"string (YYYY-MM-DD format)\",\n    \"document_language\": \"string (primary language of document)\",\n    \"total_pages\": \"number\",\n    \"has_official_stamp\": \"boolean\",\n    \"document_status\": \"string (e.g., 'Preliminary', 'Final', 'Under Investigation')\"\n  },\n  \"accident_information\": {\n    \"accident_date\": \"string (YYYY-MM-DD format)\",\n    \"accident_time\": \"string (HH:MM format, 24-hour)\",\n    \"accident_datetime_description\": \"string (any additional context about timing)\",\n    \"location\": {\n      \"street_address\": \"string (specific street name and number if available)\",\n      \"intersection\": \"string (if accident occurred at intersection)\",\n      \"landmark\": \"string (nearby landmarks for reference)\",\n      \"ward_commune\": \"string\",\n      \"district\": \"string\",\n      \"city_province\": \"string\",\n      \"country\": \"string\",\n      \"gps_coordinates\": {\n        \"latitude\": \"string (if available, otherwise null)\",\n        \"longitude\": \"string (if available, otherwise null)\"\n      },\n      \"location_description\": \"string (additional context about location)\",\n      \"location_type\": \"string (e.g., 'highway', 'intersection', 'residential street', 'parking lot')\"\n    },\n    \"environmental_conditions\": {\n      \"weather\": \"string (e.g., 'clear', 'rainy', 'foggy')\",\n      \"road_surface\": \"string (e.g., 'dry', 'wet', 'icy')\",\n      \"lighting\": \"string (e.g., 'daylight', 'dark', 'street lit')\",\n      \"traffic_density\": \"string (e.g., 'light', 'moderate', 'heavy')\",\n      \"visibility\": \"string (e.g., 'good', 'poor', 'limited')\"\n    }\n  },\n  \"parties_involved\": [\n    {\n      \"party_role\": \"string (e.g., 'Reporting Person', 'Claimant', 'Other Driver', 'Witness', 'Pedestrian')\",\n      \"full_name\": \"string\",\n      \"national_id\": \"string\",\n      \"date_of_birth\": \"string (YYYY-MM-DD or as written)\",\n      \"gender\": \"string\",\n      \"nationality\": \"string\",\n      \"occupation\": \"string\",\n      \"contact_information\": {\n        \"phone\": \"string\",\n        \"email\": \"string\",\n        \"address\": \"string\"\n      },\n      \"driver_license\": {\n        \"license_number\": \"string\",\n        \"license_type\": \"string (e.g., 'B', 'A1', 'C')\",\n        \"issue_date\": \"string\",\n        \"expiry_date\": \"string\"\n      },\n      \"insurance_information\": {\n        \"insurance_company\": \"string\",\n        \"policy_number\": \"string\",\n        \"policy_type\": \"string\",\n        \"policy_expiry\": \"string\"\n      },\n      \"vehicle_information\": {\n        \"vehicle_type\": \"string (motorcycle/car/truck/etc.)\",\n        \"make_brand\": \"string\",\n        \"model\": \"string\",\n        \"year\": \"string\",\n        \"color\": \"string\",\n        \"registration_number\": \"string\",\n        \"vin_chassis_number\": \"string\",\n        \"engine_number\": \"string\"\n      },\n      \"injuries\": {\n        \"injury_status\": \"string (e.g., 'No injury', 'Minor injury', 'Serious injury')\",\n        \"injury_description\": \"string\",\n        \"medical_treatment\": \"string (e.g., 'On-scene first aid', 'Hospitalized', 'None')\"\n      }\n    }\n  ],\n  \"reported_damages\": {\n    \"vehicle_damages\": [\n      {\n        \"vehicle_registration\": \"string\",\n        \"owner_name\": \"string\",\n        \"damage_summary\": \"string (overall description)\",\n        \"damaged_parts\": [\n          {\n            \"part_name\": \"string\",\n            \"damage_type\": \"string (scratch/dent/crack/broken/etc.)\",\n            \"damage_severity\": \"string (Minor/Moderate/Severe)\",\n            \"damage_description\": \"string (detailed description)\"\n          }\n        ],\n        \"estimated_repair_cost\": {\n          \"amount\": \"number (if provided)\",\n          \"currency\": \"string\",\n          \"cost_range\": \"string (if range provided instead of exact amount)\"\n        },\n        \"vehicle_condition\": \"string (e.g., 'Drivable', 'Not drivable', 'Towed')\"\n      }\n    ],\n    \"property_damages\": [\n      {\n        \"property_type\": \"string (e.g., 'Road barrier', 'Street sign', 'Building')\",\n        \"property_owner\": \"string\",\n        \"damage_description\": \"string\",\n        \"estimated_cost\": \"string\"\n      }\n    ],\n    \"total_estimated_damages\": {\n      \"amount\": \"number (if total provided)\",\n      \"currency\": \"string\",\n      \"notes\": \"string\"\n    }\n  },\n  \"incident_narrative\": {\n    \"reporting_person_statement\": \"string (detailed account from person filing report)\",\n    \"other_driver_statement\": \"string (if available)\",\n    \"witness_statements\": [\n      {\n        \"witness_name\": \"string\",\n        \"witness_contact\": \"string\",\n        \"statement\": \"string\"\n      }\n    ],\n    \"police_officer_observations\": \"string (official observations)\",\n    \"sequence_of_events\": \"string (chronological description of what happened)\",\n    \"contributing_factors\": [\"string (factors that contributed to accident)\"],\n    \"preliminary_cause\": \"string (initial determination of cause)\",\n    \"additional_notes\": \"string (any other relevant information)\"\n  },\n  \"official_actions\": {\n    \"responding_officer\": {\n      \"name\": \"string\",\n      \"rank\": \"string\",\n      \"badge_number\": \"string\",\n      \"department\": \"string\"\n    },\n    \"actions_taken\": [\"string (list of actions taken by authorities)\"],\n    \"citations_issued\": [\n      {\n        \"issued_to\": \"string\",\n        \"violation\": \"string\",\n        \"citation_number\": \"string\"\n      }\n    ],\n    \"evidence_collected\": [\"string (list of evidence collected)\"],\n    \"investigation_status\": \"string (e.g., 'Ongoing', 'Closed', 'Under Review')\"\n  },\n  \"supporting_documentation\": {\n    \"photos_attached\": {\n      \"count\": \"number\",\n      \"descriptions\": [\"string (brief description of each photo)\"]\n    },\n    \"diagrams_included\": \"boolean\",\n    \"diagram_description\": \"string (if diagram present)\",\n    \"video_evidence\": \"boolean\",\n    \"cctv_footage\": {\n      \"available\": \"boolean\",\n      \"source\": \"string\",\n      \"description\": \"string\"\n    },\n    \"other_documents\": [\"string (list of other attached documents)\"]\n  },\n  \"extracted_images\": [\n    {\n      \"image_number\": \"number\",\n      \"image_type\": \"string (e.g., 'Damage Photo', 'Scene Overview', 'Diagram', 'ID Document')\",\n      \"description\": \"string (what the image shows)\",\n      \"relevance\": \"string (how this image relates to the claim)\"\n    }\n  ],\n  \"data_quality_notes\": {\n    \"missing_information\": [\"string (list of critical information not found in document)\"],\n    \"unclear_information\": [\"string (information that is ambiguous or illegible)\"],\n    \"inconsistencies\": [\"string (any contradictions or inconsistencies found)\"],\n    \"handwritten_sections\": [\"string (sections with handwritten content)\"],\n    \"document_quality\": \"string (Good/Fair/Poor)\",\n    \"extraction_confidence\": \"string (High/Medium/Low)\",\n    \"notes\": \"string (any additional notes about data quality)\"\n  }\n}\n```\n\n**Extraction Guidelines**:\n- Extract information EXACTLY as written in the document - do not paraphrase unless necessary for clarity\n- For dates and times, standardize format but note original format if significantly different\n- Preserve all identifying numbers exactly (report numbers, license plates, IDs)\n- If information is handwritten, note this and transcribe as accurately as possible\n- Distinguish between information stated by parties vs. official findings\n- If information is missing or illegible, explicitly state this rather than leaving blank\n- For images extracted from PDF, analyze and describe their content in the \"extracted_images\" section\n- Maintain context - don't extract information out of context\n- Flag any information that seems inconsistent or contradictory\n- If document is in a language other than English, provide translations but also note original language text for critical fields\n\n**Important Notes**:\n- Accuracy is paramount - if unsure about information, note the uncertainty\n- Some fields may not be applicable to all documents - use null or \"Not applicable\" as appropriate\n- Legal and insurance documents may have specific technical terms - preserve these exactly\n- Timestamps and dates are critical - pay special attention to these\n- Personal information should be extracted accurately for processing but handled according to privacy regulations",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            return {
                "success": True,
                "response": message.content[0].text,
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None,
                "usage": None
            }

    def get_client_info(self) -> dict:
        """Get information about the Claude client configuration."""
        return {
            "client_type": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key_configured": bool(settings.anthropic_api_key)
        }

# Create a global instance
claude_client = ClaudeClient()
