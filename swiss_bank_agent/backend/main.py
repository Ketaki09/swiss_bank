# backend/main.py - UPDATED VERSION with hardcoded categories/constraints
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse

import uvicorn
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import asyncio
from fastapi import WebSocket
import os
import smtplib
import redis
import uuid
import logging
from twilio.rest import Client

from models.complaint_models import ComplaintResponse, ComplaintStatus
from services.database_service import DatabaseService
from services.email_service import EmailService
from services.auth_controller import AuthController
from services.auth_service import AuthService
from services.auth_utils import AuthUtils
from dotenv import load_dotenv

# Create email message using shared config
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Agent services - FIXED IMPORT ORDER
from services.eva_agent_service import EvaAgentService
from services.triage_agent_service import TriageAgentService
from services.banking_policy_service import BankingPolicyService
            
# Load environment variables
load_dotenv()

# Configure logging to reduce verbosity
logging.getLogger("twilio.http_client").setLevel(logging.WARNING)
logging.getLogger("services.auth_service").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global services dictionary
services = {}

# Global shared configuration (UNCHANGED)
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

# [ALL CONFIGURATION FUNCTIONS REMAIN UNCHANGED - keeping original implementation]
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

# [ALL OTHER CONFIGURATION HELPER FUNCTIONS REMAIN UNCHANGED]
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

# Custom AuthService that uses shared configuration (UNCHANGED)
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
    """UPDATED Application lifespan handler - removed config dependencies"""
    try:
        print("\n🚀 Starting Swiss Bank Complaint Bot API...")

        # STEP 1: Initialize shared configurations first
        print("\n🔧 Initializing shared service configurations...")
        connection_results = test_all_connections()

        # STEP 2: Initialize basic services (but don't connect yet)
        print("\n🔧 Initializing services...")
        services["db"] = DatabaseService()
        services["email"] = EmailService()
        
        # STEP 3: Connect to database FIRST
        print("\n📊 Connecting to database...")
        await services["db"].connect()
        shared_config["mongodb"]["initialized"] = True
        print("✅ Database connected successfully")
        
        # STEP 4: Create configuration indexes (only for timelines)
        print("\n⚙️ Setting up database configuration system...")
        try:
            await services["db"].create_realistic_timelines_indexes()
            print("✅ Timelines configuration indexes created")
        except Exception as e:
            print(f"⚠️ Configuration indexes creation failed: {e}")
            
        # STEP 5: NOW initialize Eva with connected database (hardcoded categories/constraints)
        print("\n🤖 Initializing Eva Agent with connected database...")
        services["eva"] = EvaAgentService(database_service=services["db"], triage_service=None)

        # STEP 5.1: Initialize Eva's async components
        print("\n⚙️ Initializing Eva async components...")
        try:
            eva_init_success = await services["eva"].initialize_async_components()
            if eva_init_success:
                print("✅ Eva async components initialized successfully")
            else:
                print("⚠️ Eva async components initialization had warnings")
        except Exception as e:
            print(f"⚠️ Eva async initialization error: {e}")

        # STEP 6: Wait for Eva configuration to load (only timelines from DB)
        print("\n⚙️ Loading Eva configuration...")
        try:
            eva_config_status = await services["eva"].get_configuration_status()
            if eva_config_status.get("configuration_complete"):
                print("✅ Eva configuration loaded successfully")
            else:
                print("⚠️ Eva configuration incomplete, using fallback configurations")
                
        except Exception as e:
            print(f"⚠️ Eva configuration loading error: {e}")

        # STEP 7: Test Eva's database integration
        eva_health = await services["eva"].check_database_integration()
        
        # STEP 8: Initialize Triage Agent Service
        print("\n🎯 Initializing Triage Agent...")
        services["triage"] = TriageAgentService(
            database_service=services["db"],
            eva_agent_service=services["eva"]
        )

        # STEP 9: Link Triage service to Eva
        services["eva"].triage_service = services["triage"]
        print("✅ Eva and Triage services fully integrated")

        triage_health = await services["triage"].health_check()
        if triage_health["status"] == "healthy":
            print("✅ Triage Agent service initialized successfully")
        else:
            print(f"⚠️ Triage Agent initialization warning: {triage_health.get('warnings', [])}")

        # STEP 10: Initialize Banking Policy Service
        print("\n🏛️  Initializing Banking Policy Service...")
        services["banking_policy"] = BankingPolicyService()
        print("✅ Banking Policy Service initialized")

        # STEP 11: Initialize auth services with shared config
        print("\n🔐 Initializing authentication services...")
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
        print(f"  Eva Agent: {'✅ Initialized with DB' if eva_health['success'] else '⚠️ Initialized (DB issues)'}")
        print(f"  Eva Configuration: {'✅ Hardcoded + DB timelines' if eva_config_status.get('configuration_complete') else '⚠️ Fallback mode'}")
        print(f"  SMTP: {'✅ Connected' if connection_results['smtp'] else '❌ Failed'}")
        print(f"  Twilio: {'✅ Connected' if connection_results['twilio'] else '❌ Failed'}")
        print(f"  Redis: {'✅ Connected' if connection_results['redis'] else '❌ Failed'}")
        
        print("\n🎉 All services initialized successfully")
        print(f"  Redis connection pooling: {'✅ Enabled' if connection_results['redis'] else '❌ Disabled'}")
        print(f"  Shared configuration: ✅ Active")
        print(f"  Eva database integration: {'✅ Active' if eva_health['success'] else '⚠️ Limited'}")
        print(f"  Configuration system: {'✅ Hardcoded categories/constraints + DB timelines' if eva_config_status.get('configuration_complete') else '⚠️ Fallback mode'}")
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
        
        if "eva" in services:
            await services["eva"].cleanup()
            print("✅ Eva agent cleaned up")
        
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
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://localhost:8001"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get services
def get_db_service() -> DatabaseService:
    return services["db"]

def get_eva_service() -> EvaAgentService:
    eva_service = services["eva"]
    # Ensure triage service is linked
    if not eva_service.triage_service and "triage" in services:
        eva_service.triage_service = services["triage"]
    return eva_service

def get_email_service() -> EmailService:
    return services["email"]

def get_auth_controller() -> AuthController:
    return services["auth_controller"]

def get_auth_service() -> SharedConfigAuthService:
    return services["auth_service"]

def get_triage_service() -> TriageAgentService:
    return services["triage"]

async def get_current_user(
    token: HTTPAuthorizationCredentials = Depends(security),
    auth_controller: AuthController = Depends(get_auth_controller)
) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from session token.
    Token should be the session_id from authentication flow
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
        session_data = session_status.get("data", {})
        if not session_data.get("authenticated"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "session_id": session_id,
            "session_data": session_data
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

# ==================== BASIC ENDPOINTS (UNCHANGED) ====================
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
            "auth": "available" if services.get("auth_controller") else "unavailable",
            "eva": "available" if services.get("eva") else "unavailable"
        }
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with triage agent status"""
    service_status = get_service_status()
    
    # Check Eva health
    eva_status = "unavailable"
    if services.get("eva"):
        try:
            eva_health = await services["eva"].check_database_integration()
            eva_status = "healthy" if eva_health["success"] else "degraded"
        except:
            eva_status = "error"
    
    # Check Triage health
    triage_status = "unavailable"
    if services.get("triage"):
        try:
            triage_health = await services["triage"].health_check()
            triage_status = triage_health["status"]
        except:
            triage_status = "error"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected" if service_status["mongodb"] else "disconnected",
            "auth": "available" if services.get("auth_controller") else "unavailable",
            "eva": eva_status,
            "triage": triage_status,  # NEW: Triage status
            "smtp": "connected" if service_status["smtp"] else "failed",
            "twilio": "connected" if service_status["twilio"] else "failed",
            "redis": "connected" if service_status["redis"] else "failed"
        },
        "agent_capabilities": {
            "eva_conversation_memory": eva_status == "healthy",
            "eva_learning_system": eva_status == "healthy",
            "triage_classification": triage_status == "healthy",
            "triage_followup_detection": triage_status == "healthy",
            "triage_new_theme_detection": triage_status == "healthy",
            "orchestrator_alerts": triage_status == "healthy"
        },
        "shared_config": {
            "redis_pooling": service_status["redis"],
            "smtp_ready": service_status["smtp"],
            "twilio_ready": service_status["twilio"]
        }
    }

# ==================== UPDATED CONFIGURATION ENDPOINTS ====================

@app.get("/api/config/status")
async def get_configuration_status( current_user: Dict[str, Any] = Depends(get_current_user), 
                                   eva_service: EvaAgentService = Depends(get_eva_service),
                                   db_service: DatabaseService = Depends(get_db_service)
):
    """Get current configuration status for Eva agent"""
    try:
        # TODO: Add admin role check here
        
        eva_config_status = await eva_service.get_configuration_status()
        db_config_status = await db_service.get_realistic_timelines_status()
        
        return {
            "eva_configuration": eva_config_status,
            "database_configuration": db_config_status,
            "overall_status": "healthy" if eva_config_status.get("configuration_complete") else "degraded",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting configuration status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get configuration status"
        )

@app.post("/api/config/refresh")
async def refresh_configuration(
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """Refresh Eva configuration from database (only realistic timelines)"""
    try:
        # TODO: Add admin role check here
        
        success = await eva_service.refresh_timelines_configuration()
        
        if success:
            return {
                "success": True,
                "message": "Realistic timelines configuration refreshed successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to refresh configuration"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error refreshing configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh configuration"
        )

@app.get("/api/config/complaint-categories")
async def get_complaint_categories(
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """Get current complaint categories (hardcoded)"""
    try:
        categories = eva_service.complaint_categories
        
        return {
            "categories": categories,
            "total_count": len(categories),
            "source": "hardcoded",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting complaint categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get complaint categories"
        )

@app.get("/api/config/realistic-timelines")
async def get_realistic_timelines(
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """Get current realistic timelines configuration from database"""
    try:
        timelines = eva_service.realistic_timelines
        
        return {
            "timelines": timelines,
            "categories_count": len(timelines),
            "source": "database",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting realistic timelines: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get realistic timelines"
        )

@app.get("/api/config/banking-constraints")
async def get_banking_constraints(
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """Get current banking constraints (hardcoded)"""
    try:
        constraints = eva_service.banking_constraints
        
        return {
            "constraints": constraints,
            "constraints_count": len(constraints),
            "source": "hardcoded",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting banking constraints: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get banking constraints"
        )
    
@app.put("/api/config/realistic-timelines")
async def update_realistic_timelines(
    timelines: str = Form(...),  
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """Update realistic timelines configuration (admin only)"""
    try:
        # TODO: Add admin role check here

        try:
            timelines_data = json.loads(timelines)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format for timelines"
            )
        
        new_config_data = {
            "timelines": timelines_data
        }
        
        success = await db_service.update_realistic_timelines_configuration(new_config_data)
        
        if success:
            # Refresh Eva's configuration
            await eva_service.refresh_timelines_configuration()
            
            return {
                "success": True,
                "message": "Realistic timelines updated successfully",
                "updated_categories": len(timelines_data),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update realistic timelines"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating realistic timelines: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update realistic timelines"
        )

@app.get("/api/config/health")
async def configuration_health_check():
    """Health check for configuration system (no auth required)"""
    try:
        db_service = services.get("db")
        eva_service = services.get("eva")
        
        if not db_service or not eva_service:
            return {
                "status": "unhealthy",
                "error": "Required services not available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Check database configuration status (only timelines)
        db_config_status = await db_service.get_realistic_timelines_status()
        eva_config_status = await eva_service.get_configuration_status()
        
        overall_healthy = (
            db_config_status.get("status") == "active" and
            eva_config_status.get("configuration_complete", False)
        )
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "database_config": db_config_status,
            "eva_config": eva_config_status,
            "configuration_loaded": eva_config_status.get("configuration_complete", False),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
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

# ==================== AUTHENTICATION ENDPOINTS (UNCHANGED) ====================

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

@app.get("/api/auth/otp-status/{session_id}")
async def get_otp_status(
    session_id: str,
    auth_service: SharedConfigAuthService = Depends(get_auth_service)
):
    """Get real-time OTP status including countdown information"""
    try:
        # Retrieve session data
        session_key = f"auth_session:{session_id}"
        session_data = await auth_service._retrieve_data(session_key)
        
        if not session_data:
            return {
                "success": False,
                "message": "Invalid session",
                "error_code": "INVALID_SESSION"
            }
        
        # Check if OTP is active
        otp_auth_key = session_data.get("otp_auth_key")
        if not otp_auth_key:
            return {
                "success": True,
                "data": {
                    "otp_active": False,
                    "otp_initiated": False,
                    "expires_at": None,
                    "remaining_seconds": 0,
                    "method": session_data.get("preferred_otp_method", "email"),
                    "masked_contact": ""
                }
            }
        
        # Get OTP data
        otp_data = await auth_service._retrieve_data(otp_auth_key)
        if not otp_data:
            return {
                "success": True,
                "data": {
                    "otp_active": False,
                    "otp_initiated": True,
                    "expires_at": None,
                    "remaining_seconds": 0,
                    "method": session_data.get("preferred_otp_method", "email"),
                    "masked_contact": ""
                }
            }
        
        # Calculate remaining time
        expiry_time = otp_data["expiry"]
        if isinstance(expiry_time, str):
            expiry_time = datetime.fromisoformat(expiry_time)
        
        now = datetime.now()
        remaining_seconds = max(0, int((expiry_time - now).total_seconds()))
        
        # Get masked contact
        contact = otp_data["contact"]
        method = otp_data["method"]
        masked_contact = (
            AuthUtils.mask_email(contact) if method == 'email' 
            else AuthUtils.mask_phone(contact)
        )
        
        return {
            "success": True,
            "data": {
                "otp_active": remaining_seconds > 0,
                "otp_initiated": True,
                "expires_at": expiry_time.isoformat(),
                "remaining_seconds": remaining_seconds,
                "method": method,
                "masked_contact": masked_contact,
                "attempts_used": otp_data.get("attempts", 0),
                "max_attempts": auth_service.max_otp_attempts
            }
        }
        
    except Exception as e:
        print(f"❌ Error getting OTP status: {e}")
        return {
            "success": False,
            "message": "Failed to get OTP status",
            "error_code": "SERVICE_ERROR"
        }

@app.post("/api/auth/refresh-otp-status")
async def refresh_otp_status(
    session_id: str = Form(...),
    auth_service: SharedConfigAuthService = Depends(get_auth_service)
):
    """Refresh OTP status and cleanup expired OTPs"""
    try:
        # Get current OTP status
        status_response = await get_otp_status(session_id, auth_service)
        
        if not status_response["success"]:
            return status_response
        
        otp_data = status_response["data"]
        
        # If OTP is expired, clean up the session
        if otp_data["otp_initiated"] and not otp_data["otp_active"]:
            session_key = f"auth_session:{session_id}"
            session_data = await auth_service._retrieve_data(session_key)
            
            if session_data and session_data.get("otp_auth_key"):
                # Remove expired OTP key from session
                session_data.pop("otp_auth_key", None)
                await auth_service._store_data(session_key, session_data, 30 * 60)
        
        return status_response
        
    except Exception as e:
        print(f"❌ Error refreshing OTP status: {e}")
        return {
            "success": False,
            "message": "Failed to refresh OTP status",
            "error_code": "SERVICE_ERROR"
        }

# Enhanced initiate-otp endpoint with better response
@app.post("/api/auth/initiate-otp-enhanced")
async def initiate_otp_enhanced(
    session_id: str = Form(...),
    auth_service: SharedConfigAuthService = Depends(get_auth_service)
):
    """Enhanced OTP initiation with detailed timing information"""
    try:
        result = await auth_service.initiate_otp_verification(session_id)
        
        if result.get("success"):
            # Add precise timing information
            current_time = datetime.now()
            expiry_time = current_time + timedelta(minutes=auth_service.otp_expiry_minutes)
            
            # Enhanced response with timing details
            enhanced_result = {
                **result,
                "data": {
                    **result.get("data", {}),
                    "initiated_at": current_time.isoformat(),
                    "expires_at": expiry_time.isoformat(),
                    "expiry_minutes": auth_service.otp_expiry_minutes,
                    "total_seconds": auth_service.otp_expiry_minutes * 60,
                    "server_time": current_time.isoformat()
                }
            }
            
            return enhanced_result
        
        return result
        
    except Exception as e:
        print(f"❌ Error initiating enhanced OTP: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate OTP verification"
        )

# WebSocket endpoint for real-time OTP updates (optional)
@app.websocket("/ws/otp-status/{session_id}")
async def websocket_otp_status(
    websocket: WebSocket,
    session_id: str,
    auth_service: SharedConfigAuthService = Depends(get_auth_service)
):
    """WebSocket endpoint for real-time OTP status updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send current OTP status
            status_response = await get_otp_status(session_id, auth_service)
            await websocket.send_json(status_response)
            
            # Wait 1 second before next update
            await asyncio.sleep(1)
            
            # If OTP is not active, reduce update frequency
            if (status_response.get("success") and 
                not status_response.get("data", {}).get("otp_active", False)):
                await asyncio.sleep(4)  # Update every 5 seconds when inactive
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

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

# ==================== COMPLAINT ENDPOINTS (UNCHANGED) ====================

@app.post("/api/complaints/submit", response_model=ComplaintResponse)
async def submit_complaint(
    complaint_text: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service),
    triage_service: TriageAgentService = Depends(get_triage_service),  # FIXED: Use Triage, not Eva
    eva_service: EvaAgentService = Depends(get_eva_service),
    email_service: EmailService = Depends(get_email_service)
):
    """
    Submit a new complaint with PROPER Triage Agent processing
    FIXED: Triage does classification, Eva handles conversation
    """
    try:
        start_time = datetime.now()
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
                attachments.append({
                    "filename": file.filename,
                    "filepath": file_path,
                    "content_type": file.content_type,
                    "size": file.size
                })
        
        # Get customer context
        customer = await db_service.get_customer(customer_id)
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Prepare complaint data for Triage Agent
        complaint_data = {
            "customer_id": customer_id,
            "complaint_text": complaint_text,
            "customer_context": customer,
            "submission_timestamp": datetime.now().isoformat(),
            "submission_method": "web",
            "attachments": attachments
        }
        
        # FIXED: Process with Triage Agent (not Eva)
        print(f"🎯 Processing complaint through Triage Agent for customer {customer_id}")
        triage_result = await triage_service.process_complaint(complaint_data)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Handle different complaint types from triage
        if triage_result.get("complaint_type") == "new_theme":
            # New theme detected - special handling
            await _handle_new_theme_complaint(triage_result, customer, email_service)
            
            return ComplaintResponse(
                complaint_id="NEW_THEME_" + str(uuid.uuid4())[:8],
                status=ComplaintStatus.RECEIVED,
                message="Your inquiry involves a new type of issue. Our senior team has been immediately notified and will contact you within 1 hour.",
                estimated_resolution_time="1 hour for initial contact"
            )
            
        elif triage_result.get("complaint_type") == "followup":
            # Follow-up complaint - return status update
            related_complaint_id = triage_result.get("related_complaint_id")
            current_status = triage_result.get("current_status", {})
            
            return ComplaintResponse(
                complaint_id=related_complaint_id or "UNKNOWN",
                status=ComplaintStatus.IN_PROGRESS,
                message=f"This appears to be a follow-up on your existing case {related_complaint_id}. " + 
                       current_status.get("message", "Your case is being actively processed."),
                estimated_resolution_time=current_status.get("resolution_estimate", "2-3 business days")
            )
            
        elif triage_result.get("complaint_type") == "additional_context":
            # Additional context provided
            related_complaint_id = triage_result.get("related_complaint_id")
            
            return ComplaintResponse(
                complaint_id=related_complaint_id or "UNKNOWN",
                status=ComplaintStatus.IN_PROGRESS,
                message=f"Thank you for the additional information on case {related_complaint_id}. " +
                       "Our team has been updated and will review this new information.",
                estimated_resolution_time="24-48 hours for review"
            )
            
        else:
            # New complaint - save to database with triage results
            processed_complaint = await _create_complaint_from_triage(
                triage_result, complaint_data, customer_id
            )
            
            # Save to database
            complaint_id = await db_service.save_complaint(processed_complaint)
            
            # Log triage processing
            await db_service.log_triage_processing({
                "complaint_id": complaint_id,
                "customer_id": customer_id,
                "complaint_type": triage_result.get("complaint_type"),
                "classification_result": triage_result.get("triage_analysis", {}).get("primary_category"),
                "confidence_score": max(triage_result.get("triage_analysis", {}).get("confidence_scores", {}).values(), default=0),
                "processing_time_ms": processing_time,
                "new_theme_detected": False,
                "orchestrator_alert_sent": triage_result.get("orchestrator_alert_sent", False),
                "error_occurred": False
            })
            
            # Send confirmation email to customer
            await email_service.send_confirmation_email(
                customer["email"], 
                complaint_id, 
                triage_result.get("triage_analysis", {}).get("primary_category", "General Inquiry")
            )
            
            print(f"✅ Complaint {complaint_id} processed successfully via Triage Agent")
            
            return ComplaintResponse(
                complaint_id=complaint_id,
                status=ComplaintStatus.RECEIVED,  
                message="Complaint received and our specialist team has been notified. " +
                       f"Priority level: {triage_result.get('routing_package', {}).get('priority_level', 'Medium')}",
                estimated_resolution_time=triage_result.get("triage_analysis", {}).get("estimated_resolution_time", "2-3 business days")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing complaint via Triage Agent: {e}")
        
        # Log error
        try:
            await db_service.log_triage_processing({
                "complaint_id": None,
                "customer_id": customer_id,
                "complaint_type": "error",
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "error_occurred": True,
                "error_message": str(e)
            })
        except:
            pass
        
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
    """Get complaint details by ID (authenticated) with enhanced error handling"""
    try:
        # Validate complaint_id format (optional)
        if not complaint_id or len(complaint_id) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid complaint ID format"
            )
        
        complaint = await db_service.get_complaint(complaint_id)
        if not complaint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found"
            )
        
        # Verify customer owns this complaint
        session_data = current_user["session_data"]
        customer_data = session_data.get("customer_data", {})
        customer_id = customer_data.get("customer_id")
        
        if complaint.get("customer_id") != customer_id:
            print(f"⚠️ Access denied: Customer {customer_id} tried to access complaint {complaint_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this complaint"
            )
        
        # Optionally filter sensitive internal data before returning
        if "internal_notes" in complaint:
            # Remove internal notes from customer view
            complaint_copy = complaint.copy()
            complaint_copy.pop("internal_notes", None)
            complaint_copy.pop("agent_comments", None)
            return complaint_copy
        
        return complaint
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting complaint {complaint_id}: {e}")
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

# ==================== EVA CHAT ENDPOINTS (UNCHANGED) ====================

@app.post("/api/eva/chat")
async def eva_chat_enhanced(
    message: str = Form(...),
    session_id: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service),
    db_service: DatabaseService = Depends(get_db_service)
):
    pass

@app.post("/api/eva/confirm-classification")
async def confirm_complaint_classification(
    complaint_id: str = Form(...),
    customer_feedback: str = Form(...),
    original_classification: str = Form(...),  # JSON string
    session_id: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """
    Process customer confirmation/correction of complaint classification
    This enables reinforcement learning for continuous improvement
    """
    try:
        # Verify session
        if session_id != current_user["session_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session ID mismatch"
            )
        
        # Parse original classification
        try:
            classification_data = json.loads(original_classification)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid classification data"
            )
        
        # Process customer feedback for learning
        feedback_result = await eva_service.process_customer_feedback(
            complaint_id=complaint_id,
            customer_feedback=customer_feedback,
            original_classification=classification_data
        )
        
        # Generate Eva's follow-up response based on feedback
        session_data = current_user["session_data"]
        customer_data = session_data.get("customer_data", {})
        
        followup_response = await eva_service._generate_followup_response(
            customer_feedback, feedback_result, customer_data
        )
        
        return {
            "feedback_processed": feedback_result["feedback_processed"],
            "feedback_type": feedback_result["feedback_type"],
            "learning_applied": feedback_result["learning_applied"],
            "followup_response": followup_response,
            "next_steps": await eva_service._get_next_steps_after_confirmation(
                feedback_result["feedback_type"], classification_data
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing classification feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process classification feedback"
        )

@app.post("/api/eva/chat-natural")
async def eva_chat_natural_flow(
    message: str = Form(...),
    session_id: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service),
    db_service: DatabaseService = Depends(get_db_service)
):
    """Eva chat with natural triage flow and banking policy compliance"""

    try:
        # Verify session matches current user
        if session_id != current_user["session_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session ID mismatch"
            )
        
        # Get customer context
        session_data = current_user["session_data"]
        customer_data = session_data.get("customer_data", {})
        customer_id = customer_data.get("customer_id")
        
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Customer ID not found in session"
            )
        
        # Get full customer context from database
        customer_context = await db_service.get_customer(customer_id)
        if not customer_context:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Use enhanced Eva with natural flow
        eva_response = await eva_service.eva_chat_response_with_natural_flow(
            message=message,
            customer_context=customer_context,
            conversation_id=session_id
        )
        
        print(f"🎯 NATURAL FLOW RESPONSE: {eva_response.get('stage', 'no-stage')}")
        
        # Save chat messages to database
        await db_service.save_chat_message(session_id, customer_id, message, is_bot=False)
        await db_service.save_chat_message(session_id, customer_id, eva_response["response"], is_bot=True)
        
        # UPDATED: Return response with additional metadata for frontend handling
        response_data = {
            "response": eva_response["response"],
            "conversation_id": eva_response["conversation_id"],
            "stage": eva_response.get("stage"),
            "emotional_state": eva_response.get("emotional_state"),
            "background_processing": eva_response.get("background_processing", False),
            "next_action": eva_response.get("next_action"),
            "retry_in_seconds": eva_response.get("retry_in_seconds"),
            "needs_first_question": eva_response.get("needs_first_question", False),  # NEW
            "question_number": eva_response.get("question_number"),  # NEW
            "ready_for_normal_chat": eva_response.get("ready_for_normal_chat", False)  # NEW
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in Eva natural flow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Eva natural flow error: {str(e)}"
        )


@app.get("/api/eva/triage-status/{conversation_id}")
async def get_triage_status(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """Get triage processing status without triggering Eva chat"""
    try:
        if conversation_id != current_user["session_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session ID mismatch"
            )
        
        # Check conversation state without processing
        conversation_state = eva_service.conversation_states.get(conversation_id, {"stage": "initial"})
        
        return {
            "conversation_id": conversation_id,
            "stage": conversation_state.get("stage"),
            "triage_results_ready": conversation_state.get("stage") in [
                "triage_results_ready", 
                "triage_confirmation_pending",
                "triage_confirmation_needed"
            ],
            "analysis_complete": conversation_state.get("background_analysis_completed", False),
            "triage_results": conversation_state.get("triage_results") if conversation_state.get("stage") == "triage_results_ready" else None
        }
        
    except Exception as e:
        print(f"❌ Error getting triage status: {e}")
        return {
            "conversation_id": conversation_id,
            "stage": "error",
            "triage_results_ready": False,
            "error": str(e)
        }

@app.get("/api/eva/conversation-history/{conversation_id}")
async def get_eva_conversation_history(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """
    Get Eva conversation history (Requirement 1: Conversation Memory)
    """
    try:
        # Verify user owns this conversation
        if conversation_id != current_user["session_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation"
            )
        
        # Get conversation context
        context = eva_service.conversation_contexts.get(conversation_id)
        
        if not context:
            return {
                "conversation_id": conversation_id,
                "messages": [],
                "customer_name": "valued customer",
                "ongoing_issues": []
            }
        
        return {
            "conversation_id": context.conversation_id,
            "customer_name": context.customer_name,
            "messages": context.messages,
            "ongoing_issues": context.ongoing_issues,
            "specialist_assignments": context.specialist_assignments,
            "emotional_state": context.emotional_state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )

@app.get("/api/eva/learning-metrics")
async def get_eva_learning_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """
    Get Eva's learning performance metrics (for admin/analytics)
    """
    try:
        # TODO: Add admin role check here
        
        total_feedback = len(eva_service.feedback_history)
        
        if total_feedback == 0:
            return {
                "total_feedback_received": 0,
                "accuracy_rate": 0.0,
                "learning_active": True,
                "categories_improved": [],
                "feedback_breakdown": {}
            }
        
        # Calculate feedback breakdown
        feedback_types = {}
        for feedback in eva_service.feedback_history:
            feedback_type = feedback.feedback_type
            feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
        
        # Calculate accuracy (confirmed + partial corrections)
        accurate_feedback = feedback_types.get("confirmed", 0) + feedback_types.get("partial_correction", 0)
        accuracy_rate = accurate_feedback / total_feedback if total_feedback > 0 else 0
        
        return {
            "total_feedback_received": total_feedback,
            "accuracy_rate": accuracy_rate,
            "learning_active": True,
            "feedback_breakdown": feedback_types,
            "categories_improved": list(eva_service.classification_weights.keys()),
            "classification_weights": eva_service.classification_weights,
            "learning_version": "reinforcement_learning_v1.0"
        }
        
    except Exception as e:
        print(f"❌ Error getting learning metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve learning metrics"
        )

@app.post("/api/eva/test-greeting")
async def test_eva_greeting(
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Test Eva's contextual greeting system (Requirement 4)
    """
    try:
        session_data = current_user["session_data"]
        customer_data = session_data.get("customer_data", {})
        customer_id = customer_data.get("customer_id")
        
        customer_context = await db_service.get_customer(customer_id)
        if not customer_context:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Generate contextual greeting
        greeting = await eva_service._generate_contextual_greeting(customer_context)
        
        return {
            "greeting": greeting,
            "customer_name": customer_context.get("name", "valued customer"),
            "current_time": datetime.now().isoformat(),
            "greeting_type": "contextual_time_based"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error testing greeting: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate test greeting"
        )

@app.get("/api/eva/integration-status")
async def get_eva_integration_status():
    """Test Eva integration with all services"""
    try:
        eva_service = services.get("eva")
        triage_service = services.get("triage")
        banking_policy = services.get("banking_policy")
        
        return {
            "eva_available": eva_service is not None,
            "triage_linked": eva_service.triage_service is not None if eva_service else False,
            "banking_policy_available": banking_policy is not None,
            "conversation_states_initialized": hasattr(eva_service, 'conversation_states') if eva_service else False,
            "natural_flow_method_available": hasattr(eva_service, 'eva_chat_response_with_natural_flow') if eva_service else False,
            "banking_constraints_loaded": hasattr(eva_service, 'banking_constraints') if eva_service else False,
            "realistic_timelines_loaded": hasattr(eva_service, 'realistic_timelines') if eva_service else False
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/eva/specialist-assignments")
async def get_specialist_assignments(
    current_user: Dict[str, Any] = Depends(get_current_user),
    eva_service: EvaAgentService = Depends(get_eva_service)
):
    """
    Get available specialist assignments (Requirement 5: Human Names)
    """
    try:
        return {
            "specialist_categories": list(eva_service.specialist_names.keys()),
            "specialist_details": eva_service.specialist_names,
            "assignment_method": "consistent_hash_based",
            "total_specialists": sum(len(specialists) for specialists in eva_service.specialist_names.values())
        }
        
    except Exception as e:
        print(f"❌ Error getting specialist assignments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve specialist assignments"
        )

# ==================== EVA SYSTEM STATUS ENDPOINTS ====================

@app.get("/api/eva/status")
async def eva_system_status():
    """Get Eva system status and capabilities"""
    try:
        eva_service = services.get("eva")
        
        if not eva_service:
            return {
                "status": "not_initialized",
                "error": "Eva service not available"
            }
        
        return {
            "status": "active",
            "version": "v2.0_complete",
            "capabilities": {
                "conversation_memory": True,
                "bullet_point_responses": True, 
                "emotional_intelligence": True,
                "contextual_greetings": True,
                "human_specialist_names": True,
                "reinforcement_learning": True
            },
            "learning_stats": {
                "total_conversations": len(eva_service.conversation_contexts),
                "feedback_received": len(eva_service.feedback_history),
                "categories_tracked": len(eva_service.classification_weights)
            },
            "anthropic_integration": "claude-sonnet-4",
            "database_integration": services.get("db") is not None
        }
        
    except Exception as e:
        print(f"❌ Error getting Eva status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# ==================== TRIAGE AGENT ENDPOINTS (UNCHANGED) ====================
@app.post("/api/triage/process-complaint")
async def process_complaint_triage(
    complaint_text: str = Form(...),
    customer_context: str = Form(default="{}"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    triage_service: TriageAgentService = Depends(get_triage_service)
):
    """Process complaint through triage agent (direct API)"""
    try:
        customer_data = json.loads(customer_context) if customer_context != "{}" else {}
        session_data = current_user["session_data"]
        customer_id = session_data.get("customer_data", {}).get("customer_id")
        
        complaint_data = {
            "customer_id": customer_id,
            "complaint_text": complaint_text,
            "customer_context": customer_data,
            "submission_timestamp": datetime.now().isoformat(),
            "submission_method": "api"
        }
        
        result = await triage_service.process_complaint(complaint_data)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triage processing error: {str(e)}"
        )

@app.get("/api/triage/health")
async def triage_health_check(
    triage_service: TriageAgentService = Depends(get_triage_service)
):
    """Get triage agent health status"""
    try:
        health_status = await triage_service.health_check()
        return health_status
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "components": {
                "anthropic_api": "unknown",
                "database": "unknown", 
                "eva_integration": "unknown",
                "new_theme_detector": "unknown"
            }
        }

@app.get("/api/triage/analytics")
async def get_triage_analytics(
    days: int = 30,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service)
):
    """Get triage processing analytics"""
    try:
        # TODO: Add admin role check
        analytics = await db_service.get_triage_analytics(days)
        return analytics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get triage analytics: {str(e)}"
        )

# ==================== ORCHESTRATOR ALERT ENDPOINTS (UNCHANGED) ====================

@app.get("/api/orchestrator/alerts")
async def get_orchestrator_alerts(
    alert_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service)
):
    """Get pending orchestrator alerts"""
    try:
        # TODO: Add orchestrator role check
        alerts = await db_service.get_pending_orchestrator_alerts(alert_type)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orchestrator alerts: {str(e)}"
        )

@app.post("/api/orchestrator/alerts/process")
async def process_orchestrator_alerts(
    alert_ids: List[str] = Form(...),
    processed_by: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service),
    triage_service: TriageAgentService = Depends(get_triage_service)
):
    """Mark orchestrator alerts as processed"""
    try:
        # TODO: Add orchestrator role check
        
        # Mark in database
        db_success = await db_service.mark_orchestrator_alerts_processed(alert_ids, processed_by)
        
        # Clear from triage service
        triage_success = await triage_service.clear_processed_alerts(alert_ids)
        
        return {
            "success": db_success and triage_success,
            "processed_count": len(alert_ids),
            "processed_by": processed_by,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process alerts: {str(e)}"
        )

@app.get("/api/orchestrator/alerts/statistics")
async def get_orchestrator_alert_statistics(
    days: int = 7,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service)
):
    """Get orchestrator alert statistics"""
    try:
        # TODO: Add admin role check
        stats = await db_service.get_orchestrator_alert_statistics(days)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alert statistics: {str(e)}"
        )

@app.get("/api/orchestrator/alerts/new-themes")
async def get_new_theme_alerts(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_db_service)
):
    """Get new theme alerts specifically"""
    try:
        # TODO: Add senior management role check
        new_theme_alerts = await db_service.get_pending_orchestrator_alerts("NEW_THEME_DETECTED")
        return {
            "new_theme_alerts": new_theme_alerts,
            "count": len(new_theme_alerts),
            "requires_immediate_attention": len(new_theme_alerts) > 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get new theme alerts: {str(e)}"
        )

# ==================== HELPER FUNCTIONS (UNCHANGED) ====================

async def _handle_new_theme_complaint(triage_result: Dict[str, Any], 
                                     customer: Dict[str, Any], 
                                     email_service) -> None:
    """Handle new theme complaint with special processing"""
    try:
        # Send priority email to customer
        await email_service.send_new_theme_notification_email(
            customer["email"],
            customer["name"],
            triage_result.get("new_theme_alert", {}).get("detection_reason", "Novel issue type")
        )
        
        # Log as high priority
        print(f"🚨 NEW THEME DETECTED for customer {customer['customer_id']}: {triage_result.get('new_theme_alert', {}).get('detection_reason')}")
        
    except Exception as e:
        print(f"❌ Error handling new theme complaint: {e}")

async def _create_complaint_from_triage(triage_result: Dict[str, Any], 
                                       complaint_data: Dict[str, Any], 
                                       customer_id: str) -> Dict[str, Any]:
    """Create complaint document from triage results"""
    
    original_complaint = triage_result.get("original_complaint", {})
    triage_analysis = triage_result.get("triage_analysis", {})
    routing_package = triage_result.get("routing_package", {})
    
    return {
        "customer_id": customer_id,
        "theme": triage_analysis.get("primary_category", "General Inquiry"),
        "title": f"Customer Complaint - {triage_analysis.get('primary_category', 'General')}",
        "description": original_complaint.get("complaint_text", ""),
        "channel": original_complaint.get("submission_method", "web"),
        "severity": triage_analysis.get("urgency_level", "medium"),
        "submission_date": datetime.now(),
        "status": "received",
        "attachments": original_complaint.get("attachments", []),
        "related_transactions": triage_analysis.get("key_entities", []),
        "customer_sentiment": triage_analysis.get("emotional_state", "neutral"),
        "urgency_keywords": triage_analysis.get("escalation_triggers", []),
        "resolution_time_expected": triage_analysis.get("estimated_resolution_time", "2-3 business days"),
        "financial_impact": triage_analysis.get("financial_impact", False),
        "estimated_financial_amount": triage_analysis.get("estimated_financial_amount"),
        "compliance_flags": triage_analysis.get("compliance_flags", []),
        "relationship_risk": triage_analysis.get("relationship_risk", "low"),
        "resolution_complexity": triage_analysis.get("resolution_complexity", "moderate"),
        "specialist_assignment": routing_package.get("specialist_assignment", {}),
        "priority_level": routing_package.get("priority_level", "P3_MEDIUM"),
        "sla_targets": routing_package.get("sla_targets", {}),
        "orchestrator_instructions": routing_package.get("orchestrator_instructions", []),
        "triage_metadata": {
            "processing_timestamp": triage_analysis.get("processing_timestamp"),
            "confidence_scores": triage_analysis.get("confidence_scores", {}),
            "triage_version": triage_analysis.get("triage_version", "v1.0"),
            "orchestrator_alert_sent": triage_result.get("orchestrator_alert_sent", False)
        },
        "processed_content": {
            "triage_analysis": triage_analysis,
            "routing_package": routing_package
        }
    }

# ==================== ADMIN ENDPOINTS (UNCHANGED) ====================

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

# ==================== UTILITY FUNCTIONS (UNCHANGED) ====================
    
async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return file path"""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return file_path

# ==================== ERROR HANDLERS (UNCHANGED) ====================

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