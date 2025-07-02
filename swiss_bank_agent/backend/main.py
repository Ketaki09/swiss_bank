# backend/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
from typing import Optional, List
import json
import os

from models.complaint_models import ComplaintCreate, ComplaintResponse
from services.database_service import DatabaseService
from services.llm_service import LLMService
from services.investigation_service import InvestigationService
from services.email_service import EmailService

app = FastAPI(title="Swiss Bank Complaint Bot API", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
db_service = DatabaseService()
llm_service = LLMService()
investigation_service = InvestigationService()
email_service = EmailService()

@app.on_event("startup")
async def startup_event():
    """Initialize database connections and services"""
    await db_service.connect()
    print("✅ Database connected successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections"""
    await db_service.disconnect()

@app.get("/")
async def root():
    return {"message": "Wells Fargo Complaint Bot API is running"}

@app.post("/api/complaints/submit", response_model=ComplaintResponse)
async def submit_complaint(
    complaint_text: str = Form(...),
    customer_id: str = Form(...),
    channel: str = Form(default="web"),
    files: List[UploadFile] = File(default=[])
):
    """
    Submit a new complaint with optional file attachments
    """
    try:
        # Process uploaded files
        attachments = []
        for file in files:
            if file.filename:
                file_path = await save_uploaded_file(file)
                attachments.append(file_path)
        
        # Get customer context
        customer = await db_service.get_customer(customer_id)
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Process complaint with LLM
        processed_complaint = await llm_service.process_complaint(
            text=complaint_text,
            customer_context=customer,
            attachments=attachments
        )
        
        # Save to database
        complaint_id = await db_service.save_complaint(processed_complaint)
        
        # Start investigation process (async)
        investigation_service.start_investigation(complaint_id, processed_complaint)
        
        # Send confirmation email to customer
        await email_service.send_confirmation_email(
            customer["email"], 
            complaint_id, 
            processed_complaint["theme"]
        )
        
        return ComplaintResponse(
            complaint_id=complaint_id,
            status="received",
            message="Complaint received and investigation started",
            estimated_resolution_time=processed_complaint.get("resolution_time_expected", "2-3 business days")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing complaint: {str(e)}")

@app.get("/api/complaints/{complaint_id}")
async def get_complaint(complaint_id: str):
    """Get complaint details by ID"""
    complaint = await db_service.get_complaint(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return complaint

@app.get("/api/customers/{customer_id}/history")
async def get_customer_history(customer_id: str):
    """Get customer's complaint history"""
    history = await db_service.get_customer_complaint_history(customer_id)
    return {"customer_id": customer_id, "complaints": history}

@app.post("/api/chat/message")
async def chat_message(
    message: str = Form(...),
    customer_id: str = Form(...),
    session_id: str = Form(...)
):
    """
    Handle chat messages from the bot interface
    """
    try:
        # Get customer context
        customer = await db_service.get_customer(customer_id)
        
        # Process with LLM
        response = await llm_service.chat_response(
            message=message,
            customer_context=customer,
            session_id=session_id
        )
        
        return {"response": response, "session_id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/api/dashboard/complaints")
async def get_dashboard_complaints(
    status: Optional[str] = None,
    limit: int = 50
):
    """Get complaints for dashboard view"""
    complaints = await db_service.get_complaints_for_dashboard(status, limit)
    return {"complaints": complaints}

@app.put("/api/complaints/{complaint_id}/status")
async def update_complaint_status(
    complaint_id: str,
    status: str = Form(...),
    notes: Optional[str] = Form(None)
):
    """Update complaint status (for agent dashboard)"""
    success = await db_service.update_complaint_status(complaint_id, status, notes)
    if not success:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return {"message": "Status updated successfully"}

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return file path"""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return file_path

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)