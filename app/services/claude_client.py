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

    def get_client_info(self) -> dict:
        """Get information about the Claude client configuration."""
        return {
            "client_type": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key_configured": bool(settings.anthropic_api_key)
        }

# Create a global instance
claude_client = ClaudeClient()
