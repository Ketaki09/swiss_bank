# backend/main.py - Updated with shared configuration
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
import logging
from twilio.rest import Client

from models.complaint_models import ComplaintResponse, ComplaintStatus
from services.database_service import DatabaseService
from services.llm_service import LLMService
from services.investigation_service import InvestigationService
from services.email_service import EmailService
from services.auth_controller import AuthController
from services.auth_service import AuthService
from services.auth_utils import AuthUtils
from dotenv import load_dotenv

# Create email message using shared config
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
            
# Load environment variables
load_dotenv()

# Configure logging to reduce verbosity
logging.getLogger("twilio.http_client").setLevel(logging.WARNING)
logging.getLogger("services.auth_service").setLevel(logging.WARNING)

# Global services dictionary
services = {}

# Global shared configuration
shared_config = {
    "smtp": {
        "server": None,
        "port": 587,
        "username": None,
        "password": None,
        "connection_pool": None,
        "initialized": False
    },
    "redis": {
        "client": None,
        "url": None,
        "connection_pool": None,
        "initialized": False
    },
    "twilio": {
        "client": None,
        "account_sid": None,
        "auth_token": None,
        "phone_number": None,
        "initialized": False
    },
    "mongodb": {
        "connection": None,
        "database": None,
        "initialized": False
    }
}


# Security scheme
security = HTTPBearer()

def initialize_smtp_config():
    """Initialize SMTP configuration and test connection"""
    try:
        smtp_config = shared_config["smtp"]
        smtp_config["server"] = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_config["port"] = int(os.getenv("SMTP_PORT", "587"))
        smtp_config["username"] = os.getenv("SMTP_USERNAME")
        smtp_config["password"] = os.getenv("SMTP_PASSWORD")
        
        if not smtp_config["username"] or not smtp_config["password"]:
            print("❌ SMTP credentials not configured")
            return False
        
        # Test connection
        with smtplib.SMTP(smtp_config["server"], smtp_config["port"], timeout=10) as server:
            server.starttls()
            server.login(smtp_config["username"], smtp_config["password"])
            
        smtp_config["initialized"] = True
        print("✅ SMTP configuration initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ SMTP configuration failed: {e}")
        shared_config["smtp"]["initialized"] = False
        return False

def initialize_redis_config():
    """Initialize Redis configuration and connection"""
    try:
        redis_config = shared_config["redis"]
        redis_config["url"] = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Create Redis client with connection pooling
        redis_config["client"] = redis.from_url(
            redis_config["url"],
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
            max_connections=20,  # Connection pool size
            decode_responses=False  # Keep as bytes for consistency
        )
        
        # Test connection
        redis_config["client"].ping()
        redis_config["initialized"] = True
        print("✅ Redis configuration initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Redis configuration failed: {e}")
        shared_config["redis"]["initialized"] = False
        shared_config["redis"]["client"] = None
        return False

def initialize_twilio_config():
    """Initialize Twilio configuration and test connection"""
    try:
        twilio_config = shared_config["twilio"]
        twilio_config["account_sid"] = os.getenv("TWILIO_ACCOUNT_SID")
        twilio_config["auth_token"] = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_config["phone_number"] = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not twilio_config["account_sid"] or not twilio_config["auth_token"]:
            print("❌ Twilio credentials not configured")
            return False
        
        # Create Twilio client
        twilio_config["client"] = Client(
            twilio_config["account_sid"], 
            twilio_config["auth_token"]
        )
        
        # Test connection by fetching account info
        account = twilio_config["client"].api.accounts(twilio_config["account_sid"]).fetch()
        twilio_config["initialized"] = True
        print(f"✅ Twilio configuration initialized successfully - Account: {account.friendly_name}")
        return True
        
    except Exception as e:
        print(f"❌ Twilio configuration failed: {e}")
        shared_config["twilio"]["initialized"] = False
        shared_config["twilio"]["client"] = None
        return False

def get_smtp_config():
    """Get SMTP configuration"""
    return shared_config["smtp"]

def get_redis_client():
    """Get Redis client (thread-safe)"""
    if shared_config["redis"]["initialized"]:
        return shared_config["redis"]["client"]
    return None

def get_twilio_client():
    """Get Twilio client"""
    if shared_config["twilio"]["initialized"]:
        return shared_config["twilio"]["client"]
    return None

def is_service_available(service_name: str) -> bool:
    """Check if a service is available and initialized"""
    return shared_config.get(service_name, {}).get("initialized", False)

def get_service_status():
    """Get status of all configured services"""
    return {
        "smtp": shared_config["smtp"]["initialized"],
        "redis": shared_config["redis"]["initialized"],
        "twilio": shared_config["twilio"]["initialized"],
        "mongodb": shared_config["mongodb"]["initialized"]
    }
def validate_shared_config():
    """Validate that required shared configurations are available"""
    required_services = ["smtp", "redis", "twilio", "mongodb"]
    missing_services = []
    
    for service in required_services:
        if not shared_config.get(service, {}).get("initialized", False):
            missing_services.append(service)
    
    if missing_services:
        print(f"⚠️  Warning: Some services are not initialized: {missing_services}")
        return False
    
    return True

def get_available_services():
    """Get list of available services"""
    available = []
    for service_name, config in shared_config.items():
        if config.get("initialized", False):
            available.append(service_name)
    return available

# Enhanced service availability check
def require_service(service_name: str):
    """Decorator to require a specific service"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_service_available(service_name):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Service {service_name} is not available"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def initialize_mongodb_config():
    """Initialize MongoDB configuration"""
    try:
        # This should be integrated with your DatabaseService
        shared_config["mongodb"]["initialized"] = True
        return True
        
    except Exception as e:
        print(f"❌ MongoDB configuration failed: {e}")
        shared_config["mongodb"]["initialized"] = False
        return False

def test_all_connections():
    """Test all external service connections"""
    print("\n📡 Testing external service connections...")
    
    results = {
        "smtp": initialize_smtp_config(),
        "redis": initialize_redis_config(),
        "twilio": initialize_twilio_config(),
        "mongodb": initialize_mongodb_config()  
    }
    
    return results

def cleanup_shared_resources():
    """Cleanup shared resources"""
    try:
        # Cleanup Redis
        if shared_config["redis"]["client"]:
            shared_config["redis"]["client"].close()
            shared_config["redis"]["client"] = None
            shared_config["redis"]["initialized"] = False
            print("✅ Redis connection closed")
        
        # Reset other configurations
        shared_config["smtp"]["initialized"] = False
        shared_config["twilio"]["initialized"] = False
        shared_config["twilio"]["client"] = None
        
        print("✅ Shared resources cleaned up successfully")
        
    except Exception as e:
        print(f"❌ Error cleaning up shared resources: {e}")

# Custom AuthService that uses shared configuration
class SharedConfigAuthService(AuthService):
    """Enhanced AuthService that uses shared configuration"""
    
    def __init__(self):
        # Initialize parent class but override connection methods
        super().__init__(shared_config_getter=lambda: shared_config)
        
        # Override configurations with shared ones
        self.use_shared_config = True
        
    def _init_redis(self):
        """Override Redis initialization to use shared config"""
        if is_service_available("redis"):
            self.redis_client = get_redis_client()
            self.use_redis = True
            print("✅ AuthService using shared Redis connection")
        else:
            self.redis_client = None
            self.use_redis = False
            print("❌ Redis not available, falling back to MongoDB storage")
    
    def get_smtp_config(self):
        """Get SMTP configuration from shared config"""
        if is_service_available("smtp"):
            smtp_config = get_smtp_config()
            return {
                "server": smtp_config["server"],
                "port": smtp_config["port"],
                "username": smtp_config["username"],
                "password": smtp_config["password"]
            }
        return None
    
    def get_twilio_client(self):
        """Get Twilio client from shared config"""
        if is_service_available("twilio"):
            return get_twilio_client()
        return None
    
    async def send_otp_email(self, email: str, otp: str, customer_name: str = "Valued Customer") -> Dict[str, Any]:
        """Enhanced send_otp_email using shared SMTP config"""
        try:
            smtp_config = self.get_smtp_config()
            if not smtp_config:
                print("SMTP service not available")
                return AuthUtils.create_error_response(
                    "Email service not configured",
                    "SERVICE_ERROR",
                    technical_error=True
                )
        
            msg = MIMEMultipart()
            msg['From'] = smtp_config["username"]
            msg['To'] = email
            msg['Subject'] = "Swiss Bank - Authentication Code"
            
            # Load and render template
            template_content = self.load_email_template("otp_email.html")
            html_body = self.render_template(
                template_content,
                customer_name=customer_name,
                otp=otp,
                expiry_minutes=str(self.otp_expiry_minutes)
            )
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email with shared SMTP config
            try:
                with smtplib.SMTP(smtp_config["server"], smtp_config["port"], timeout=15) as server:
                    server.starttls()
                    server.login(smtp_config["username"], smtp_config["password"])
                    server.send_message(msg)
                
                return AuthUtils.create_success_response(
                    "OTP email sent successfully",
                    data={
                        "sent_to": AuthUtils.mask_email(email),
                        "method": "email"
                    }
                )
                
            except Exception as smtp_error:
                print(f"SMTP error with shared config: {smtp_error}")
                return AuthUtils.create_error_response(
                    "Email sending failed",
                    "SEND_FAILED",
                    retry_allowed=True,
                    technical_error=True
                )
                
        except Exception as e:
            print(f"Error sending OTP email with shared config: {e}")
            return AuthUtils.create_error_response(
                "Failed to send OTP email",
                "SEND_FAILED",
                retry_allowed=True,
                technical_error=True
            )
    
    async def send_otp_sms(self, phone: str, otp: str) -> Dict[str, Any]:
        """Enhanced send_otp_sms using shared Twilio config"""
        try:
            twilio_client = self.get_twilio_client()
            if not twilio_client:
                print("Twilio service not available")
                return AuthUtils.create_error_response(
                    "SMS service not configured",
                    "SERVICE_ERROR",
                    technical_error=True
                )
            
            formatted_phone = AuthUtils.format_phone(phone)
            twilio_phone = shared_config["twilio"]["phone_number"]
            
            message = twilio_client.messages.create(
                body=f"Your Swiss Bank verification code is: {otp}. This code expires in {self.otp_expiry_minutes} minutes. Do not share this code with anyone.",
                from_=twilio_phone,
                to=formatted_phone
            )
            
            return AuthUtils.create_success_response(
                "OTP SMS sent successfully",
                data={
                    "sent_to": AuthUtils.mask_phone(phone),
                    "method": "sms",
                    "message_sid": message.sid
                }
            )
            
        except Exception as e:
            print(f"Error sending OTP SMS with shared config: {e}")
            return AuthUtils.create_error_response(
                "Failed to send OTP SMS",
                "SEND_FAILED",
                retry_allowed=True,
                technical_error=True
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events"""
    try:
        print("\n🚀 Starting Swiss Bank Complaint Bot API...")

        # Initialize shared configurations first
        print("\n🔧 Initializing shared service configurations...")
        connection_results = test_all_connections()

        # Initialize basic services
        print("\n🔧 Initializing services...")
        services["db"] = DatabaseService()
        services["llm"] = LLMService()
        services["investigation"] = InvestigationService()
        services["email"] = EmailService()
        
        # Connect to database first
        await services["db"].connect()
        shared_config["mongodb"]["initialized"] = True
        print("✅ Database connected successfully")
        
        # Initialize auth services with shared config
        services["auth_service"] = SharedConfigAuthService()
        await services["auth_service"].initialize()
        print("✅ Authentication service initialized with shared config")
        
        services["auth_controller"] = AuthController()
        services["auth_controller"].auth_service = services["auth_service"]
        print("✅ Authentication controller initialized")
        
        # Update connection results with actual MongoDB status
        connection_results["mongodb"] = shared_config["mongodb"]["initialized"]
        
        print("\n📊 Service Status Summary:")
        print(f"  Database: {'✅ Connected' if connection_results['mongodb'] else '❌ Failed'}")
        print(f"  Authentication: ✅ Initialized")
        print(f"  SMTP: {'✅ Connected' if connection_results['smtp'] else '❌ Failed'}")
        print(f"  Twilio: {'✅ Connected' if connection_results['twilio'] else '❌ Failed'}")
        print(f"  Redis: {'✅ Connected' if connection_results['redis'] else '❌ Failed'}")
        
        print("\n🎉 All services initialized successfully")
        print(f"  Redis connection pooling: {'✅ Enabled' if connection_results['redis'] else '❌ Disabled'}")
        print(f"  Shared configuration: ✅ Active")
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
        
        if "auth_service" in services:
            await services["auth_service"].cleanup_and_disconnect()
            print("✅ Auth service disconnected")
        
        # Cleanup shared resources
        cleanup_shared_resources()
        print("✅ Shared resources cleaned up successfully")

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

def get_auth_service() -> SharedConfigAuthService:
    return services["auth_service"]

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
    service_status = get_service_status()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected" if service_status["mongodb"] else "disconnected",
            "auth": "available" if services.get("auth_controller") else "unavailable",
            "smtp": "connected" if service_status["smtp"] else "failed",
            "twilio": "connected" if service_status["twilio"] else "failed",
            "redis": "connected" if service_status["redis"] else "failed"
        },
        "shared_config": {
            "redis_pooling": service_status["redis"],
            "smtp_ready": service_status["smtp"],
            "twilio_ready": service_status["twilio"]
        }
    }

@app.get("/health/config")
async def config_health_check():
    """Configuration health check endpoint"""
    return {
        "shared_config_status": get_service_status(),
        "redis_client_active": get_redis_client() is not None,
        "twilio_client_active": get_twilio_client() is not None,
        "smtp_config_active": get_smtp_config() is not None
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
    auth_service: SharedConfigAuthService = Depends(get_auth_service)
):
    """Initiate OTP verification"""
    try:
        result = await auth_service.initiate_otp_verification(session_id)
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
    try:
        result = await auth_controller.resend_otp(session_id)
        
        # Check if the result indicates an error
        if not result.get("success", True):
            error_code = result.get("error_code")
            
            # Map error codes to appropriate HTTP status codes
            if error_code in ["INVALID_SESSION", "SESSION_EXPIRED"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=result.get("message", "Invalid session")
                )
            elif error_code == "INVALID_STATE":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.get("message", "Invalid session state")
                )
            elif error_code == "OTP_NOT_INITIATED":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.get("message", "OTP not initiated")
                )
            elif error_code in ["SERVICE_ERROR", "RESEND_FAILED"]:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.get("message", "Service temporarily unavailable")
                )
            else:
                # Generic error handling
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.get("message", "Request failed")
                )
        
        return result
        
    except HTTPException:
        raise
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