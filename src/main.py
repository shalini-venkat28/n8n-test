import asyncio
import logging
import threading
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from controller.chat_controller import router as loan_insights_router
from service.yaml_service import initialize_prompt_cache
from utils.database.database import create_db

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AHM BI Assistant", version="1.0.0")

# Health check router
health_router = APIRouter(prefix="/api/v1/bi", tags=["Health"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions and return standardized error response"""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred"
        }
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

def initialize_prompts() -> None:
    """Initialize prompt cache in separate thread to avoid blocking startup"""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(initialize_prompt_cache())
        loop.close()
        
    except Exception as e:
        logger.error(f"Failed to initialize prompt cache: {str(e)}")

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize database and start prompt cache loading on application startup"""
    try:
        await create_db()        
        # Start the prompt initialization in a separate thread
        prompt_init_thread = threading.Thread(target=initialize_prompts, daemon=True)
        prompt_init_thread.start()
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@health_router.get("/health_check")
async def health_check() -> Dict[str, str]:
    """Return application health status for monitoring"""
    return {"status": "healthy"}

app.include_router(health_router)
app.include_router(loan_insights_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)