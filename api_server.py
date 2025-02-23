import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import os
import uuid
from datetime import datetime
import asyncio
from train_thermal import ThermalClassifier
from predict_thermal import load_model, predict
from alert_system import AlertSystem
import logging
import shutil
from pathlib import Path

# Initialize logging
logging.basicConfig(
    filename='api_server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Thermal Threat Detection API",
    description="API for real-time thermal threat detection and alert generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our models and systems
thermal_model = None
alert_system = None

class ThreatRequest(BaseModel):
    message: str
    location: str
    additional_details: Optional[Dict] = None
    alert_platforms: Optional[List[str]] = ["email", "slack"]

class ThreatResponse(BaseModel):
    request_id: str
    timestamp: str
    threat_level: str
    confidence: float
    location: str
    message: str
    alerts_sent: Dict[str, bool]
    image_path: str

@app.on_event("startup")
async def startup_event():
    """Initialize models and systems on startup"""
    global thermal_model, alert_system
    try:
        # Initialize thermal model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        thermal_model = load_model("best_thermal_model.pth", num_classes=3, device=device)
        
        # Initialize alert system
        alert_system = AlertSystem()
        
        logger.info("Successfully initialized models and systems")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

def cleanup_old_files():
    """Clean up old uploaded files (older than 24 hours)"""
    try:
        current_time = datetime.now().timestamp()
        for file_path in UPLOAD_DIR.glob("*"):
            if current_time - file_path.stat().st_mtime > 86400:  # 24 hours
                file_path.unlink()
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

async def process_threat_detection(
    image_path: str,
    request: ThreatRequest
) -> ThreatResponse:
    """Process threat detection and generate alerts"""
    try:
        # Get thermal model prediction
        pred_class, confidence = predict(thermal_model, image_path)
        
        # Map prediction class to threat level
        threat_levels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        threat_level = threat_levels.get(pred_class, "UNKNOWN")
        
        # Prepare additional details
        details = request.additional_details or {}
        details.update({
            "Thermal Class": str(pred_class),
            "Confidence": f"{confidence:.2%}",
            "Detection Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Send alerts if threat level is MEDIUM or higher
        alerts_sent = {}
        if threat_level in ["MEDIUM", "HIGH"]:
            alerts_sent = alert_system.send_alert(
                threat_level=threat_level,
                location=request.location,
                message=request.message,
                details=details,
                platforms=request.alert_platforms
            )
        
        return ThreatResponse(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            threat_level=threat_level,
            confidence=confidence,
            location=request.location,
            message=request.message,
            alerts_sent=alerts_sent,
            image_path=image_path
        )
    
    except Exception as e:
        logger.error(f"Error processing threat detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/", response_model=ThreatResponse)
async def detect_threat(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: ThreatRequest = None
):
    """
    Detect threats from thermal images and generate alerts
    
    - **file**: Thermal image file
    - **message**: Description or context of the situation
    - **location**: Location where the image was captured
    - **additional_details**: Optional additional context
    - **alert_platforms**: Platforms to send alerts to (email, slack, telegram)
    """
    try:
        # Generate unique filename
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process threat detection
        response = await process_threat_detection(
            str(file_path),
            request
        )
        
        # Schedule cleanup of old files
        background_tasks.add_task(cleanup_old_files)
        
        return response
    
    except Exception as e:
        logger.error(f"Error in detect_threat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the API and all its components are healthy"""
    try:
        # Check if models are loaded
        if thermal_model is None or alert_system is None:
            raise HTTPException(
                status_code=503,
                detail="Models not initialized"
            )
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": True,
            "upload_dir_exists": UPLOAD_DIR.exists()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )
