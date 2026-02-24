import json
import logging
import time
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import ValidationError

from models.validation import (
    CanonicalFieldSelectionResponse,
    IntentClassificationResponse,
    RequestBodyGenerationResponse,
    TextResponse
)
from repository.chat_repository import save_chat
from repository.error_repository import log_error_entry
from service.yaml_service import get_prompt_by_type
from utils.helpers.secrets_manager import get_secret

load_dotenv()

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



async def llm_call(call_type: str, user_id: str, user_query: str, user_history: list, **kwargs) -> Dict[str, Any]:
    """Process LLM request with call_type and context parameters using AWS Bedrock
    Sequence: LLM 9.1 to 9.48
    """
    try:
        system_prompt_data = get_prompt_by_type(call_type)
        system_prompt = system_prompt_data.get('prompt_body', '')
        
        if call_type == "canonical_field_selection":
            context = f"Query: {user_query}\nRole: {kwargs.get('user_role', '')}\nFields: {kwargs.get('filtered_fields', [])}"
        elif call_type == "request_body_generation":
            context = f"Query: {user_query}\nSelected Fields: {kwargs.get('selected_canonical_fields', [])}"
        elif call_type == "response_generation":
            context = f"Query: {user_query}\nData: {kwargs.get('encompass_response', {})}"
        elif call_type == "follow_up_question":
            context = f"Query: {user_query}\nError: {kwargs.get('error_response', '')}"
        else:
            context = user_query
        
        aws_credentials, bedrock_client = get_secret()
        
        request_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": [{"type": "text", "text": system_prompt}],
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": context}]
            }],
            "max_tokens": 4000
        }
        
        start_time = time.time()
        response = bedrock_client.invoke_model(
            modelId=aws_credentials['MODEL_ARN'],
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_payload)
        )
        processing_time = int((time.time() - start_time) * 1000)
        
        response_body = json.loads(response.get("body").read().decode("utf-8"))
        llm_response = response_body['content'][0].get("text", "")
        
        # Log LLM response for debugging
        logger.info(f"LLM Response for {call_type}: {llm_response}")
        
        validation_result = _validate_response(call_type, llm_response)
        
        if not validation_result['is_valid']:
            return {"status": "error", "response": validation_result['error'], "processing_time_ms": processing_time}
        
        if user_id:
            await save_conversation(call_type, user_id, {"user_query": user_query, "call_type": call_type}, validation_result['parsed_response'])
        return {
            "status": "success",
            "response": validation_result['parsed_response'],
            "processing_time_ms": processing_time
        }
            
    except Exception as e:
        error_message = str(e)
        
        if "ResourceNotFoundException" in error_message and "use case details" in error_message:
            error_message = "Model access not configured. Please submit the required use case form in AWS Bedrock console or use a different model."
        
        log_error_entry({
            "service_name": f"llm_service_{call_type}",
            "severity_text": "ERROR",
            "message": f"LLM call failed for {call_type}: {error_message}",
            "user_id": user_id,
            "details_json": {"call_type": call_type, "user_query": user_query[:100] if user_query else None}
        })
        raise

def _validate_response(call_type: str, response: str) -> Dict:
    """Validate LLM response JSON structure using Pydantic models based on call_type
    Sequence: LLM 9.24 to 9.39
    """
    try:
        if call_type == "intent_classification":
            # Validate intent classification response
            intent_data = {"intent": response.strip().lower()}
            validated = IntentClassificationResponse(**intent_data)
            return {"is_valid": True, "parsed_response": validated.intent}
                
        elif call_type == "canonical_field_selection":
            # Parse and validate canonical field selection JSON
            json_data = json.loads(response.strip())
            validated = CanonicalFieldSelectionResponse(fields=json_data)
            return {"is_valid": True, "parsed_response": response.strip()}
            
        elif call_type == "request_body_generation":
            # Parse and validate request body generation JSON
            json_data = json.loads(response.strip())
            validated = RequestBodyGenerationResponse(**json_data)
            return {"is_valid": True, "parsed_response": response.strip()}
            
        elif call_type in ["response_generation", "follow_up_question", "greeting_response"]:
            # Validate text response
            text_data = {"response": response.strip()}
            validated = TextResponse(**text_data)
            return {"is_valid": True, "parsed_response": validated.response}
            
        else:
            return {"is_valid": True, "parsed_response": response}
        
    except ValidationError as e:
        return {"is_valid": False, "error": f"Pydantic validation error: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"is_valid": False, "error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return {"is_valid": False, "error": f"Validation error: {str(e)}"}

async def save_conversation(call_type: str, user_id: str, context_params: Dict, response: str):
    """Save chat history with user_id, conversation_history, and metadata_json
    Sequence: LLM 9.24 to 9.25
    """
    try:
        if call_type == "intent_classification":
            # Only log error if intent is irrelevant, don't save chat
            if response.lower() == "invalid":
                log_error_entry({
                    "service_name": "intent_classifier",
                    "severity_text": "INFO",
                    "message": "Irrelevant query detected",
                    "user_id": user_id,
                    "details_json": {"query": context_params.get('user_query'), "intent": response}
                })
            return
        
        elif call_type in ["canonical_field_selection", "request_body_generation"]:
            # Update metadata_json field
            chat_data = {
                "user_id": user_id,
                "prompt_type": call_type,
                "metadata_json": {
                    "call_type": call_type,
                    "response": response,
                    "context": context_params
                }
            }
            
        elif call_type in ["greeting_response", "response_generation", "follow_up_question"]:
            # Update conversation_history field
            chat_data = {
                "user_id": user_id,
                "prompt_type": call_type,
                "conversation_entry": {
                    "user_query": context_params.get('user_query', ''),
                    "model_response": response
                }
            }
        
        else:
            return  # Don't save for other call types
        
        save_chat(chat_data)
        
    except Exception as e:
        log_error_entry({
            "service_name": "llm_service_chat_save",
            "severity_text": "ERROR",
            "message": f"Failed to save chat for {call_type}: {str(e)}",
            "user_id": user_id,
            "details_json": {"call_type": call_type, "error": str(e)}
        })