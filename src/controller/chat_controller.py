import logging
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models.validation import ChatRequest, ChatResponse
from repository.error_repository import log_error_entry
from service.intent_classifier_service import classify_intent
from utils.helpers.jwt_utils import validate_jwt_token, authorize_request
####

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/bi", tags=["Loan Insights"])
security = HTTPBearer()

@router.post("/loan_insights", response_model=ChatResponse)
async def process_loan_insights(
    request_payload: ChatRequest, 
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> ChatResponse:
    """Orchestrate complete loan insights workflow with JWT validation and intent classification

    Sequence: Main 1.10 to 1.35  
    """
    try:
        # Extract user_id, user_query, user_role, user_history from request_payload
        user_id = str(request_payload.user_id)
        user_query = request_payload.query
        user_role = request_payload.role
        
        # Validate JWT token from request headers
        token = credentials.credentials
        jwt_data = validate_jwt_token(token)
        
        if not jwt_data.get("is_valid"):
            raise HTTPException(status_code=jwt_data.get("code", 401), detail=jwt_data.get("error", "Invalid token"))
        
        # Authorize request
        authorize_request(request_payload, jwt_data)
        
        # Check if user_query is None or empty
        if not user_query or not user_query.strip():
            raise HTTPException(status_code=404, detail="Missing user query")
        
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        logger.info(
            "Processing loan insight request",
            extra={"request_id": request_id, "user_id": user_id, "role": user_role, "endpoint": "/loan_insights"}
        )
        
        # Call classify_intent to determine user's intention
        intent_result = await classify_intent(request_payload)
        
        return ChatResponse(
            user_query=user_query,
            model_response=intent_result.get("response", "Processing completed"),
            status="success",
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log controller-level processing failure
        log_error_entry({
            "service_name": "LoanInsightsController",
            "severity_text": "ERROR",
            "message": f"Controller processing failed: {str(e)}",
            "user_id": getattr(request_payload, 'user_id', None),
            "details_json": {"error": str(e), "endpoint": "/loan_insights"}
        })
        logger.error(f"Error in loan_insights endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")