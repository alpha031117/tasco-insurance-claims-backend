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

    def get_client_info(self) -> dict:
        """Get information about the Claude client configuration."""
        return {
            "client_type": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key_configured": bool(settings.anthropic_api_key)
        }

# Create a global instance
claude_client = ClaudeClient()
