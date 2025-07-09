# backend/main.py 
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse

import uvicorn
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import os
import smtplib
import redis
from twilio.rest import Client

from models.complaint_models import ComplaintResponse, ComplaintStatus
from services.database_service import DatabaseService
from services.llm_service import LLMService
from services.investigation_service import InvestigationService
from services.email_service import EmailService
from services.auth_controller import AuthController
from services.auth_service import AuthService
from services.auth_utils import AuthUtils

# Global services dictionary
services = {}

# Security scheme
security = HTTPBearer()

def test_smtp_connection():
    try:
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        email_user = os.getenv("SMTP_USERNAME")
        email_password = os.getenv("SMTP_PASSWORD")
        
        if not email_user or not email_password:
            print("❌ SMTP credentials not configured")
            return False
        
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(email_user, email_password)
            return True
            
    except Exception as e:
        print(f"❌ SMTP connection failed: {e}")
        return False

def test_twilio_connection():
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        
        if not account_sid or not auth_token:
            print("❌ Twilio credentials not configured")
            return False
        
        client = Client(account_sid, auth_token)
        # Test by fetching account info
        account = client.api.accounts(account_sid).fetch()
        print(f"✅ Twilio connection successful - Account: {account.friendly_name}")
        return True
        
    except Exception as e:
        print(f"❌ Twilio connection failed: {e}")
        return False

def test_redis_connection():
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(
            redis_url,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        # Test connection
        redis_client.ping()
        print(f"✅ Redis connection successful")
        redis_client.close()
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events"""
    try:
        print("🚀 Starting Swiss Bank Complaint Bot API...")
        
        # Test external service connections
        print("\n📡 Testing external service connections...")
        smtp_ok = test_smtp_connection()
        twilio_ok = test_twilio_connection()
        redis_ok = test_redis_connection()
        
        # Initialize services
        print("\n🔧 Initializing services...")
        services["db"] = DatabaseService()
        services["llm"] = LLMService()
        services["investigation"] = InvestigationService()
        services["email"] = EmailService()
        services["auth_controller"] = AuthController()
        services["auth_controller"].auth_service = AuthService()
        services["auth_service"].auth_utils = AuthUtils()

        # Connect to database
        await services["db"].connect()
        print("✅ Database connected successfully")
        
        # Initialize auth service
        await services["auth_controller"].auth_service.initialize()
        print("✅ Authentication service initialized")
        
        print("\n📊 Service Status Summary:")
        print(f"  Database: ✅ Connected")
        print(f"  SMTP: {'✅ Connected' if smtp_ok else '❌ Failed'}")
        print(f"  Twilio: {'✅ Connected' if twilio_ok else '❌ Failed'}")
        print(f"  Redis: {'✅ Connected' if redis_ok else '❌ Failed'}")
        
        print("\n🎉 All services initialized successfully")
        print("🌐 API is ready to serve requests")
        
        yield
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise
    finally:
        # Cleanup resources
        print("\n🧹 Cleaning up resources...")
        if "db" in services:
            await services["db"].disconnect()
            print("✅ Database disconnected")
        
        if "auth_controller" in services:
            await services["auth_controller"].auth_service.cleanup_and_disconnect()
            print("✅ Auth service disconnected")
        
        print("✅ Resources cleaned up successfully")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Swiss Bank Complaint Bot API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8001"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get services
def get_db_service() -> DatabaseService:
    return services["db"]

def get_llm_service() -> LLMService:
    return services["llm"]

def get_investigation_service() -> InvestigationService:
    return services["investigation"]

def get_email_service() -> EmailService:
    return services["email"]

def get_auth_controller() -> AuthController:
    return services["auth_controller"]

# Authentication dependency
async def get_current_user(
    token: HTTPAuthorizationCredentials = Depends(security),
    auth_controller: AuthController = Depends(get_auth_controller)
) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from session token.
    Token should be the session_id from authentication flow.
    """
    try:
        # Extract session_id from token
        session_id = token.credentials
        
        # Get session status
        session_status = await auth_controller.get_session_status(session_id)
        
        if not session_status["success"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not session_status["authenticated"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "session_id": session_id,
            "session_data": session_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Optional authentication dependency (for endpoints that can work with or without auth)
async def get_current_user_optional(
    token: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_controller: AuthController = Depends(get_auth_controller)
) -> Optional[Dict[str, Any]]:
    """Optional authentication - returns None if not authenticated"""
    if not token:
        return None
    
    try:
        return await get_current_user(token, auth_controller)
    except HTTPException:
        return None

# ==================== BASIC ENDPOINTS ====================

@app.get("/")
async def root():
    return {"message": "Swiss Bank Complaint Bot API is running"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected" if services.get("db") else "disconnected",
            "auth": "available" if services.get("auth_controller") else "unavailable"
        }
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with external service status"""
    smtp_status = test_smtp_connection()
    twilio_status = test_twilio_connection()
    redis_status = test_redis_connection()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected" if services.get("db") else "disconnected",
            "auth": "available" if services.get("auth_controller") else "unavailable",
            "smtp": "connected" if smtp_status else "failed",
            "twilio": "connected" if twilio_status else "failed",
            "redis": "connected" if redis_status else "failed"
        }
    }

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/api/auth/session")
async def create_auth_session(
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Create a new authentication session"""
    try:
        result = await auth_controller.create_auth_session(ip_address, user_agent)
        return result
    except Exception as e:
        print(f"❌ Error creating auth session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create authentication session"
        )

@app.post("/api/auth/verify-contact")
async def verify_contact(
    session_id: str = Form(...),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    preferred_otp_method: Optional[str] = Form(None),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Verify customer contact details (email or phone)"""
    try:
        result = await auth_controller.verify_contact_details(
            session_id, email, phone, preferred_otp_method
        )
        return result
    except Exception as e:
        print(f"❌ Error verifying contact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify contact details"
        )

@app.post("/api/auth/initiate-otp")
async def initiate_otp(
    session_id: str = Form(...),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Initiate OTP verification"""
    try:
        result = await auth_controller.initiate_otp_verification(session_id)
        return result
    except Exception as e:
        print(f"❌ Error initiating OTP: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate OTP verification"
        )

@app.post("/api/auth/verify-otp")
async def verify_otp(
    session_id: str = Form(...),
    otp: str = Form(...),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Verify OTP code"""
    try:
        result = await auth_controller.verify_otp(session_id, otp)
        return result
    except Exception as e:
        print(f"❌ Error verifying OTP: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify OTP"
        )

@app.post("/api/auth/resend-otp")
async def resend_otp(
    session_id: str = Form(...),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Resend OTP code"""
    try:
        result = await auth_controller.resend_otp(session_id)
        return result
    except Exception as e:
        print(f"❌ Error resending OTP: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resend OTP"
        )

@app.get("/api/auth/session/{session_id}")
async def get_session_status(
    session_id: str,
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Get authentication session status"""
    try:
        result = await auth_controller.get_session_status(session_id)
        return result
    except Exception as e:
        print(f"❌ Error getting session status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session status"
        )

# ==================== COMPLAINT ENDPOINTS (PROTECTED) ====================

@app.post("/api/complaints/submit", response_model=ComplaintResponse)
async def submit_complaint(
    complaint_text: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service),
    llm_service: LLMService = Depends(get_llm_service),
    investigation_service: InvestigationService = Depends(get_investigation_service),
    email_service: EmailService = Depends(get_email_service)
):
    """
    Submit a new complaint with authentication required
    """
    try:
        # Get customer ID from authenticated session
        session_data = current_user["session_data"]
        customer_data = session_data.get("customer_data", {})
        customer_id = customer_data.get("customer_id")
        
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Customer ID not found in session"
            )
        
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
        
        print(f"✅ Complaint {complaint_id} submitted successfully")
        
        return ComplaintResponse(
            complaint_id=complaint_id,
            status=ComplaintStatus.RECEIVED,  
            message="Complaint received and investigation started",
            estimated_resolution_time=processed_complaint.get("resolution_time_expected", "2-3 business days")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing complaint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing complaint: {str(e)}"
        )

@app.get("/api/complaints/{complaint_id}")
async def get_complaint(
    complaint_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service)
):
    """Get complaint details by ID (authenticated)"""
    try:
        complaint = await db_service.get_complaint(complaint_id)
        if not complaint:
            raise HTTPException(status_code=404, detail="Complaint not found")
        
        # Verify customer owns this complaint
        session_data = current_user["session_data"]
        customer_data = session_data.get("customer_data", {})
        customer_id = customer_data.get("customer_id")
        
        if complaint.get("customer_id") != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this complaint"
            )
        
        return complaint
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting complaint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve complaint"
        )

@app.get("/api/customers/history")
async def get_customer_history(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service)
):
    """Get authenticated customer's complaint history"""
    try:
        session_data = current_user["session_data"]
        customer_data = session_data.get("customer_data", {})
        customer_id = customer_data.get("customer_id")
        
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Customer ID not found in session"
            )
        
        history = await db_service.get_customer_complaint_history(customer_id)
        return {"customer_id": customer_id, "complaints": history}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting customer history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve customer history"
        )

@app.post("/api/chat/message")
async def chat_message(
    message: str = Form(...),
    session_id: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Handle chat messages from the bot interface (authenticated)
    """
    try:
        # Verify session_id matches current user's session
        if session_id != current_user["session_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session ID mismatch"
            )
        
        session_data = current_user["session_data"]
        customer_data = session_data.get("customer_data", {})
        customer_id = customer_data.get("customer_id")
        
        # Get customer context
        customer = await db_service.get_customer(customer_id)
        
        # Ensure customer exists before passing to LLM
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Process with LLM
        response = await llm_service.chat_response(
            message=message,
            customer_context=customer,
            session_id=session_id
        )
        
        return {"response": response, "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing chat message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

# ==================== ADMIN ENDPOINTS (PROTECTED) ====================

@app.get("/api/dashboard/complaints")
async def get_dashboard_complaints(
    complaint_status: Optional[str] = None,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service)
):
    """Get complaints for dashboard view (requires authentication)"""
    try:
        # TODO: Add admin role check here
        complaints = await db_service.get_complaints_for_dashboard(complaint_status, limit)
        return {"complaints": complaints}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting dashboard complaints: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard complaints"
        )

@app.put("/api/complaints/{complaint_id}/status")
async def update_complaint_status(
    complaint_id: str,
    new_status: str = Form(...),
    notes: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service)
):
    """Update complaint status (for agent dashboard, requires authentication)"""
    try:
        # TODO: Add admin/agent role check here
        success = await db_service.update_complaint_status(complaint_id, new_status, notes)
        if not success:
            raise HTTPException(status_code=404, detail="Complaint not found")
        return {"message": "Status updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating complaint status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update complaint status"
        )

# ==================== UTILITY FUNCTIONS ====================

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return file path"""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return file_path

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": "HTTP_ERROR",
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    print(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)

# To run the application, use the command:
# uvicorn backend.main:app --reload
# python main.py
# This will start the FastAPI server on http://localhost:8000