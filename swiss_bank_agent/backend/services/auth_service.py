# backend/services/auth_service.py - Updated with shared configuration support
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import os
import json
from pathlib import Path
from twilio.rest import Client
from .database_service import DatabaseService
from .auth_utils import AuthUtils
import redis
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AuthService:
    def __init__(self, shared_config_getter: Optional[Callable] = None):
        """
        Initialize AuthService with optional shared configuration support
        
        Args:
            shared_config_getter: Optional callable that returns shared configuration
        """
        self.db_service = DatabaseService()
        self._db_connected = False
        
        # Shared configuration support
        self.shared_config_getter = shared_config_getter
        self.use_shared_config = shared_config_getter is not None
        
        # Email configuration - with shared config fallback
        if self.use_shared_config:
            self.smtp_server = None  # Will be set via shared config
            self.smtp_port = None
            self.email_user = None
            self.email_password = None
        else:
            # Fallback to environment variables
            self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
            self.email_user = os.getenv("SMTP_USERNAME")
            self.email_password = os.getenv("SMTP_PASSWORD")
        
        # Twilio configuration - with shared config fallback
        if self.use_shared_config:
            self.twilio_account_sid = None  # Will be set via shared config
            self.twilio_auth_token = None
            self.twilio_phone_number = None
        else:
            # Fallback to environment variables
            self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            self.twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        # Redis configuration with shared config support
        self.redis_client = None
        self.use_redis = False
        if not self.use_shared_config:
            self._init_redis()  # Only init if not using shared config
        
        # Template path
        self.template_path = Path(__file__).parent.parent / "templates" / "emails"
        
        # OTP configuration
        self.otp_length = 6
        self.otp_expiry_minutes = 5
        self.max_otp_attempts = 3
        self.otp_cooldown_seconds = 60
        
        # Contact verification attempts configuration
        self.max_contact_attempts = 3
        self.contact_lockout_minutes = 15
        
        # Fallback storage (only used if both Redis and MongoDB fail)
        self.memory_storage = {}

        # Technical error codes that should trigger retries
        self.technical_error_codes = {
            "DATABASE_ERROR", "NETWORK_ERROR", "TIMEOUT_ERROR", 
            "SERVICE_ERROR", "SEND_FAILED", "RESEND_FAILED"
        }

    def _get_shared_config(self) -> Optional[Dict[str, Any]]:
        """Get shared configuration if available"""
        if self.shared_config_getter:
            try:
                return self.shared_config_getter()
            except Exception as e:
                print(f"Error getting shared config: {e}")
                return None
        return None

    def _is_service_available(self, service_name: str) -> bool:
        """Check if a shared service is available"""
        if not self.use_shared_config:
            return True  # Assume available for legacy mode
        
        shared_config = self._get_shared_config()
        if not shared_config:
            return False
        
        return shared_config.get(service_name, {}).get("initialized", False)

    def _get_smtp_config(self) -> Optional[Dict[str, Any]]:
        """Get SMTP configuration from shared config or environment"""
        if self.use_shared_config:
            shared_config = self._get_shared_config()
            if shared_config and self._is_service_available("smtp"):
                smtp_config = shared_config["smtp"]
                return {
                    "server": smtp_config["server"],
                    "port": smtp_config["port"],
                    "username": smtp_config["username"],
                    "password": smtp_config["password"]
                }
            return None
        else:
            # Fallback to instance variables
            if self.email_user and self.email_password:
                return {
                    "server": self.smtp_server,
                    "port": self.smtp_port,
                    "username": self.email_user,
                    "password": self.email_password
                }
            return None

    def _get_redis_client(self):
        """Get Redis client from shared config or instance"""
        if self.use_shared_config:
            shared_config = self._get_shared_config()
            if shared_config and self._is_service_available("redis"):
                return shared_config["redis"]["client"]
            return None
        else:
            return self.redis_client

    def _get_twilio_client(self):
        """Get Twilio client from shared config or create new one"""
        if self.use_shared_config:
            shared_config = self._get_shared_config()
            if shared_config and self._is_service_available("twilio"):
                return shared_config["twilio"]["client"]
            return None
        else:
            # Create client from instance variables
            if self.twilio_account_sid and self.twilio_auth_token:
                return Client(self.twilio_account_sid, self.twilio_auth_token)
            return None

    def _get_twilio_phone_number(self) -> Optional[str]:
        """Get Twilio phone number from shared config or instance"""
        if self.use_shared_config:
            shared_config = self._get_shared_config()
            if shared_config and self._is_service_available("twilio"):
                return shared_config["twilio"]["phone_number"]
            return None
        else:
            return self.twilio_phone_number

    async def initialize(self):
        """Initialize the AuthService - must be called before using other methods"""
        try:
            await self.db_service.connect()
            self._db_connected = True
            
            # Initialize Redis based on configuration mode
            if self.use_shared_config:
                self._init_redis_from_shared_config()
            else:
                self._init_redis()  # Legacy method
                
        except Exception as e:
            self._db_connected = False
            print(f"Failed to connect to the database: {e}")
            raise e

    def _init_redis_from_shared_config(self):
        """Initialize Redis connection from shared configuration"""
        try:
            if not self.use_shared_config:
                print("Not using shared config, skipping Redis initialization")
                return
                
            shared_config = self._get_shared_config()
            if not shared_config:
                print("❌ Shared config not available")
                self.redis_client = None
                self.use_redis = False
                return
                
            if not shared_config.get("redis", {}).get("initialized", False):
                print("❌ Redis not initialized in shared config")
                self.redis_client = None
                self.use_redis = False
                return
                
            redis_client = shared_config["redis"]["client"]
            if redis_client:
                self.redis_client = redis_client
                self.use_redis = True
                print("✅ AuthService using shared Redis connection")
            else:
                self.redis_client = None
                self.use_redis = False
                print("❌ Redis client not available from shared config")
                
        except Exception as e:
            print(f"Error initializing Redis from shared config: {e}")
            self.redis_client = None
            self.use_redis = False

    async def ensure_db_connection(self):
        """Ensure database connection is established"""
        if not self._db_connected:
            try:
                await self.db_service.connect()
                self._db_connected = True
            except Exception as e:
                print(f"Failed to establish database connection: {e}")
                raise ConnectionError("Database connection failed")

    def _init_redis(self):
        """Initialize Redis connection with error handling (legacy method)"""
        if self.use_shared_config:
            # Skip legacy Redis initialization if using shared config
            return
        
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(
                redis_url,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            self.use_redis = True
        except Exception as e:
            print(f"Redis connection failed: {e}. Falling back to MongoDB storage")
            self.redis_client = None
            self.use_redis = False

    async def _store_data(self, key: str, data: Dict[str, Any], expiry_seconds: int = 180):
        """Store data with Redis primary, MongoDB fallback"""
        try:
            # Convert datetime objects to ISO format for JSON serialization
            def datetime_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            
            serialized_data = json.dumps(data, default=datetime_serializer)
            
            # Try Redis first (with shared config support)
            redis_client = self._get_redis_client()
            if redis_client and (self.use_redis or self.use_shared_config):
                try:
                    redis_client.setex(key, expiry_seconds, serialized_data)
                    return True
                except Exception as e:
                    print(f"Redis storage failed: {e}. Falling back to MongoDB")
                    if not self.use_shared_config:
                        self.use_redis = False
            
            # MongoDB fallback
            await self.ensure_db_connection()
            expiry_time = datetime.now() + timedelta(seconds=expiry_seconds)
            await self.db_service.store_temp_data({
                "_id": key,
                "data": serialized_data,
                "expires_at": expiry_time,
                "created_at": datetime.now()
            })
            return True
            
        except Exception as e:
            print(f"Both Redis and MongoDB storage failed: {e}")
            # Final fallback to memory (not recommended for production)
            self.memory_storage[key] = {
                "data": data,
                "expires_at": datetime.now() + timedelta(seconds=expiry_seconds)
            }
            return True

    async def _retrieve_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data with Redis primary, MongoDB fallback"""
        try:
            # Try Redis first (with shared config support)
            redis_client = self._get_redis_client()
            if redis_client and (self.use_redis or self.use_shared_config):
                try:
                    value = redis_client.get(key)
                    if value:  # Only process if value exists
                        # Decode and parse JSON
                        if isinstance(value, bytes):
                            data = json.loads(value.decode("utf-8"))
                        elif isinstance(value, str):
                            data = json.loads(value)
                        else:
                            print(f"Unexpected Redis value type: {type(value)}")
                            return None
                        
                        # Convert ISO format strings back to datetime objects
                        return self._deserialize_datetime_fields(data)

                except Exception as e:
                    print(f"Redis retrieval failed: {e}. Trying MongoDB")
                    if not self.use_shared_config:
                        self.use_redis = False
            
            # MongoDB fallback
            await self.ensure_db_connection()
            temp_data = await self.db_service.get_temp_data(key)
            if temp_data:
                # Check if expired
                if datetime.now() > temp_data["expires_at"]:
                    await self.db_service.delete_temp_data(key)
                    return None
                data = json.loads(temp_data["data"])
                return self._deserialize_datetime_fields(data)
            
            # Memory fallback
            if key in self.memory_storage:
                stored = self.memory_storage[key]
                if datetime.now() > stored["expires_at"]:
                    del self.memory_storage[key]
                    return None
                # Parse JSON and deserialize datetime fields
                if isinstance(stored["data"], str):
                    data = json.loads(stored["data"])
                    return self._deserialize_datetime_fields(data)
                return stored["data"]
            
            return None
        
        except Exception as e:
            print(f"Data retrieval failed: {e}")
            return None

    async def _delete_data(self, key: str):
        """Delete data from all storage systems"""
        try:
            # Try Redis first (with shared config support)
            redis_client = self._get_redis_client()
            if redis_client and (self.use_redis or self.use_shared_config):
                try:
                    redis_client.delete(key)
                except Exception as e:
                    print(f"Redis deletion failed: {e}")
            
            # MongoDB cleanup
            await self.ensure_db_connection()
            await self.db_service.delete_temp_data(key)
            
            # Memory cleanup
            if key in self.memory_storage:
                del self.memory_storage[key]
                
        except Exception as e:
            print(f"Data deletion failed: {e}")

    def _deserialize_datetime_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ISO format datetime strings back to datetime objects"""
        datetime_fields = [
            'created_at', 'last_activity', 'expiry', 'locked_at', 
            'contact_verified_at', 'otp_initiated_at', 'otp_resent_at', 
            'authenticated_at', 'expires_at'
        ]
        
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except ValueError:
                    # If it's not a valid ISO format, leave as string
                    pass
        
        return data

    def load_email_template(self, template_name: str) -> str:
        """Load email template from file with improved error handling"""
        try:
            print(f"Loading email template: {template_name}")
            template_file = self.template_path / template_name
            with open(template_file, 'r', encoding='utf-8') as file:
                content = file.read()
                return content
        except FileNotFoundError:
            print(f"Template {template_name} not found at {template_file}")
            return self._get_simple_fallback_template()
        except Exception as e:
            print("Using fallback template")
            return self._get_simple_fallback_template()

    def _get_simple_fallback_template(self) -> str:
        """Simple fallback template as last resort"""
        return """
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <h2 style="color: #1a472a;">Swiss Bank - Authentication Code</h2>
            <p>Dear {customer_name},</p>
            <p>Your verification code is:</p>
            <div style="font-size: 24px; font-weight: bold; color: #1a472a; text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin: 20px 0;">
                {otp}
            </div>
            <p>This code will expire in <strong>{expiry_minutes} minutes</strong>.</p>
            <p><strong>Important:</strong> Do not share this code with anyone.</p>
            <p>Best regards,<br>Swiss Bank Security Team</p>
        </body>
        </html>
        """

    def render_template(self, template_content: str, **kwargs) -> str:
        """Render template with improved error handling - using replace method like test_email.py"""
        try:
            # First try the string replacement method (more reliable)
            rendered_content = template_content
            for key, value in kwargs.items():
                placeholder = "{" + key + "}"
                rendered_content = rendered_content.replace(placeholder, str(value))
            
            # Check if all placeholders were replaced
            if "{" in rendered_content and "}" in rendered_content:
                print("Some template placeholders may not have been replaced")
            return rendered_content
            
        except Exception as e:
            print(f"Error rendering template with replace method: {e}")
            try:
                # Fallback to format method
                return template_content.format(**kwargs)
            except Exception as format_error:
                print(f"Error rendering template with format method: {format_error}")
                # Return simple fallback
                return self._get_simple_fallback_template().replace("{customer_name}", str(kwargs.get("customer_name", "Valued Customer"))).replace("{otp}", str(kwargs.get("otp", "000000"))).replace("{expiry_minutes}", str(kwargs.get("expiry_minutes", "5")))

    def determine_otp_method(self, email: Optional[str] = None, phone: Optional[str] = None, 
                           preferred_method: Optional[str] = None) -> str:
        """Determine the OTP method based on available contact info and preference"""
        if preferred_method and preferred_method in ['email', 'sms']:
            if preferred_method == 'email' and email:
                return 'email'
            elif preferred_method == 'sms' and phone:
                return 'sms'
        
        # Auto-determine based on available contact info
        if email and not phone:
            return 'email'
        elif phone and not email:
            return 'sms'
        else:
            return 'email'  # Default fallback

    async def check_customer_exists(self, email: Optional[str] = None, phone: Optional[str] = None) -> Dict[str, Any]:
        """Check if customer exists in database - returns standardized response"""
        try:
            # Ensure database connection
            await self.ensure_db_connection()
            
            query = {}
            if email:
                query["email"] = email.lower().strip()
            if phone:
                formatted_phone = AuthUtils.format_phone(phone)
                query["phone"] = formatted_phone
            
            customer = await self.db_service.find_customer(query)
            
            return AuthUtils.create_success_response(
                "Customer lookup completed",
                data={
                    "exists": bool(customer),
                    "customer_data": customer
                }
            )
                
        except Exception as e:
            print(f"Error checking customer existence: {e}")
            return AuthUtils.create_error_response(
                "Customer lookup failed",
                "DATABASE_ERROR",
                retry_allowed=True,
                technical_error=True
            )
    async def initiate_otp_verification(self, session_id: str) -> Dict[str, Any]:
        """Initiate OTP verification - moved from auth_controller.py"""
        try:
            # Retrieve and validate session
            session_key = f"auth_session:{session_id}"
            session_data = await self._retrieve_data(session_key)
            
            if not session_data:
                return AuthUtils.create_error_response(
                    "Invalid or expired session. Please start again.",
                    "INVALID_SESSION",
                    action_required="restart"
                )
            
            # Check if session is in correct state
            if session_data.get("state") != "otp_verification":
                return AuthUtils.create_error_response(
                    "Invalid session state. Contact verification required first.",
                    "INVALID_STATE"
                )
            
            # Check if session is expired
            if AuthUtils.is_session_expired(session_data, 30):  
                await self._delete_data(session_key)
                return AuthUtils.create_error_response(
                    "Session expired. Please start again.",
                    "SESSION_EXPIRED",
                    action_required="restart"
                )
            
            # Check if contact is verified
            if not session_data.get("contact_verified"):
                return AuthUtils.create_error_response(
                    "Contact verification required before OTP generation.",
                    "CONTACT_NOT_VERIFIED"
                )
            
            # Generate and send OTP
            otp_result = await self._generate_and_send_otp(session_data)
            
            if not otp_result.get("success"):
                return otp_result
            
            # Update session with OTP data
            session_data.update({
                "otp_auth_key": otp_result["data"]["auth_key"],
                "otp_initiated_at": datetime.now(),
                "last_activity": datetime.now()
            })
            
            # Store updated session
            await self._store_data(session_key, session_data, 30 * 60)  # 30 minutes
            
            return AuthUtils.create_success_response(
                otp_result["data"]["message"],
                masked_contact=otp_result["data"]["masked_contact"],
                expires_in=otp_result["data"]["expires_in"],
                otp_method=session_data["preferred_otp_method"],
                state="otp_verification"
            )
            
        except Exception as e:
            print(f"Error initiating OTP verification: {e}")
            return AuthUtils.create_error_response(
                "OTP service temporarily unavailable. Please try again.",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def _generate_and_send_otp(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and send OTP - moved from auth_controller.py"""
        try:
            preferred_method = session_data["preferred_otp_method"]
            email = session_data["contact_email"]
            phone = session_data["contact_phone"]

            # Validate required data
            if not preferred_method:
                return AuthUtils.create_error_response(
                    "OTP method not specified",
                    "INVALID_STATE",
                    retry_allowed=False
                )
            if preferred_method == 'email' and not email:
                return AuthUtils.create_error_response(
                    "Email address required for email OTP",
                    "INVALID_STATE",
                    retry_allowed=False
                )
            
            if preferred_method == 'sms' and not phone:
                return AuthUtils.create_error_response(
                    "Phone number required for SMS OTP",
                    "INVALID_STATE",
                    retry_allowed=False
                )

            # Generate OTP
            contact = email if preferred_method == 'email' else phone
            otp_result = await self.generate_otp(contact, preferred_method)
            
            if not otp_result or not otp_result.get("success"):
                return otp_result or AuthUtils.create_error_response(
                    "OTP generation failed",
                    "SERVICE_ERROR",
                    retry_allowed=True,
                    technical_error=True
                )
            
            # Validate OTP result structure
            if not otp_result.get("data") or not otp_result["data"].get("otp"):
                return AuthUtils.create_error_response(
                    "Invalid OTP generation response",
                    "SERVICE_ERROR",
                    retry_allowed=True,
                    technical_error=True
                )
            
            # Send OTP
            if preferred_method == 'email':
                send_result = await self.send_otp_email(
                    email, 
                    otp_result["data"]["otp"],
                    session_data.get("customer_data", {}).get("name", "Valued Customer")
                )
            else:
                send_result = await self.send_otp_sms(
                    phone,
                    otp_result["data"]["otp"]
                )
            
            if not send_result or not send_result.get("success"):
                return send_result or AuthUtils.create_error_response(
                    "OTP sending failed",
                    "SEND_FAILED",
                    retry_allowed=True,
                    technical_error=True
                )
            
            # Prepare success response
            masked_contact = (
                AuthUtils.mask_email(email) if preferred_method == 'email' 
                else AuthUtils.mask_phone(phone)
            )
            
            return AuthUtils.create_success_response(
                f"OTP sent successfully via {preferred_method}",
                data={
                    "auth_key": otp_result["data"]["auth_key"],
                    "message": f"OTP sent to {masked_contact}",
                    "masked_contact": masked_contact,
                    "expires_in": otp_result["data"]["expires_in"]
                }
            )
            
        except Exception as e:
            print(f"OTP generation/sending error: {e}")
            return AuthUtils.create_error_response(
                "Failed to generate or send OTP",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )
        
    async def generate_otp(self, contact: str, method: str) -> Dict[str, Any]:
        """Generate OTP and create auth session - returns standardized response"""
        try:
            otp = ''.join(random.choices(string.digits, k=self.otp_length))
            auth_key = f"otp:{method}:{contact}:{datetime.now().timestamp()}"
            
            otp_data = {
                "otp": otp,
                "contact": contact,
                "method": method,
                "expiry": datetime.now() + timedelta(minutes=self.otp_expiry_minutes),
                "attempts": 0,
                "created_at": datetime.now()
            }
            
            await self._store_data(auth_key, otp_data, self.otp_expiry_minutes * 60)
            
            return AuthUtils.create_success_response(
                "OTP generated successfully",
                data={
                    "otp": otp,
                    "auth_key": auth_key,
                    "expires_in": self.otp_expiry_minutes
                }
            )
            
        except Exception as e:
            print(f"Error generating OTP: {e}")
            return AuthUtils.create_error_response(
                "OTP generation failed",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def send_otp_email(self, email: str, otp: str, customer_name: str = "Valued Customer") -> Dict[str, Any]:
        """Send OTP via email - updated with shared config support"""
        try:
            # Get SMTP configuration from shared config or environment
            smtp_config = self._get_smtp_config()
            if not smtp_config:
                print("Email credentials not configured")
                return AuthUtils.create_error_response(
                    "Email service not configured",
                    "SERVICE_ERROR",
                    technical_error=True
                )
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = smtp_config["username"]
            msg['To'] = email
            msg['Subject'] = "Swiss Bank - Authentication Code"
            
            # Load and render template
            template_content = self.load_email_template("otp_email.html")
            
            # Render template using the improved method
            html_body = self.render_template(
                template_content,
                customer_name=customer_name,
                otp=otp,
                expiry_minutes=str(self.otp_expiry_minutes)
            )
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email with shared config
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
                
            except smtplib.SMTPAuthenticationError as e:
                print(f"SMTP authentication failed: {e}")
                return AuthUtils.create_error_response(
                    "Email authentication failed",
                    "SEND_FAILED",
                    retry_allowed=True,
                    technical_error=True
                )
            except smtplib.SMTPConnectError as e:
                print(f"SMTP connection failed: {e}")
                return AuthUtils.create_error_response(
                    "Email server connection failed",
                    "SEND_FAILED",
                    retry_allowed=True,
                    technical_error=True
                )
            except smtplib.SMTPException as e:
                print(f"SMTP error: {e}")
                return AuthUtils.create_error_response(
                    "Email sending failed",
                    "SEND_FAILED",
                    retry_allowed=True,
                    technical_error=True
                )
            
        except Exception as e:
            print(f"Unexpected error sending OTP email: {e}")
            return AuthUtils.create_error_response(
                "Failed to send OTP email",
                "SEND_FAILED",
                retry_allowed=True,
                technical_error=True
            )

    async def send_otp_sms(self, phone: str, otp: str) -> Dict[str, Any]:
        """Send OTP via SMS - updated with shared config support"""
        try:
            # Get Twilio client and phone number from shared config or environment
            twilio_client = self._get_twilio_client()
            twilio_phone = self._get_twilio_phone_number()
            
            if not twilio_client or not twilio_phone:
                print("Twilio credentials not configured")
                return AuthUtils.create_error_response(
                    "SMS service not configured",
                    "SERVICE_ERROR",
                    technical_error=True
                )
            
            formatted_phone = AuthUtils.format_phone(phone)
            
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
            print(f"Error sending OTP SMS: {e}")
            return AuthUtils.create_error_response(
                "Failed to send OTP SMS",
                "SEND_FAILED",
                retry_allowed=True,
                technical_error=True
            )

    async def get_otp_status_detailed(self, session_id: str) -> Dict[str, Any]:
        """Get detailed OTP status for real-time tracking"""
        try:
            # Retrieve session data
            session_key = f"auth_session:{session_id}"
            session_data = await self._retrieve_data(session_key)
            
            if not session_data:
                return AuthUtils.create_error_response(
                    "Invalid session",
                    "INVALID_SESSION"
                )
            
            # Check if OTP is active
            otp_auth_key = session_data.get("otp_auth_key")
            if not otp_auth_key:
                return AuthUtils.create_success_response(
                    "No active OTP session",
                    data={
                        "otp_active": False,
                        "otp_initiated": False,
                        "expires_at": None,
                        "remaining_seconds": 0,
                        "remaining_minutes": 0,
                        "method": session_data.get("preferred_otp_method", "email"),
                        "masked_contact": "",
                        "progress_percentage": 0,
                        "status": "not_initiated"
                    }
                )
            
            # Get OTP data
            otp_data = await self._retrieve_data(otp_auth_key)
            if not otp_data:
                return AuthUtils.create_success_response(
                    "OTP session expired",
                    data={
                        "otp_active": False,
                        "otp_initiated": True,
                        "expires_at": None,
                        "remaining_seconds": 0,
                        "remaining_minutes": 0,
                        "method": session_data.get("preferred_otp_method", "email"),
                        "masked_contact": "",
                        "progress_percentage": 0,
                        "status": "expired"
                    }
                )
            
            # Calculate timing details
            expiry_time = otp_data["expiry"]
            if isinstance(expiry_time, str):
                expiry_time = datetime.fromisoformat(expiry_time)
            
            created_time = otp_data.get("created_at")
            if isinstance(created_time, str):
                created_time = datetime.fromisoformat(created_time)
            
            now = datetime.now()
            remaining_seconds = max(0, int((expiry_time - now).total_seconds()))
            remaining_minutes = remaining_seconds / 60
            
            # Calculate progress percentage (100% = full time, 0% = expired)
            total_duration = self.otp_expiry_minutes * 60  # in seconds
            progress_percentage = min(100, max(0, (remaining_seconds / total_duration) * 100))
            
            # Determine status
            if remaining_seconds <= 0:
                status = "expired"
            elif remaining_seconds <= 60:
                status = "expiring_soon"
            elif remaining_seconds <= 180:
                status = "active_warning"
            else:
                status = "active"
            
            # Get contact information
            contact = otp_data["contact"]
            method = otp_data["method"]
            masked_contact = (
                AuthUtils.mask_email(contact) if method == 'email' 
                else AuthUtils.mask_phone(contact)
            )
            
            return AuthUtils.create_success_response(
                "OTP status retrieved",
                data={
                    "otp_active": remaining_seconds > 0,
                    "otp_initiated": True,
                    "expires_at": expiry_time.isoformat(),
                    "created_at": created_time.isoformat() if created_time else None,
                    "remaining_seconds": remaining_seconds,
                    "remaining_minutes": round(remaining_minutes, 1),
                    "method": method,
                    "masked_contact": masked_contact,
                    "attempts_used": otp_data.get("attempts", 0),
                    "max_attempts": self.max_otp_attempts,
                    "remaining_attempts": self.max_otp_attempts - otp_data.get("attempts", 0),
                    "progress_percentage": round(progress_percentage, 1),
                    "status": status,
                    "server_time": now.isoformat(),
                    "expiry_minutes": self.otp_expiry_minutes
                }
            )
            
        except Exception as e:
            print(f"Error getting detailed OTP status: {e}")
            return AuthUtils.create_error_response(
                "Failed to get OTP status",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def cleanup_expired_otp_sessions(self) -> int:
        """Clean up expired OTP sessions and return count of cleaned items"""
        try:
            cleaned_count = 0
            
            # This would require iterating through stored sessions
            # For now, rely on TTL and manual cleanup
            
            # Clean up MongoDB temp data
            mongo_cleaned = await self.db_service.cleanup_expired_temp_data()
            cleaned_count += mongo_cleaned
            
            print(f"Cleaned up {cleaned_count} expired OTP sessions")
            return cleaned_count
            
        except Exception as e:
            print(f"Error during OTP cleanup: {e}")
            return 0

    def format_otp_time_remaining(self, seconds: int) -> str:
        """Format remaining time in human-readable format"""
        if seconds <= 0:
            return "Expired"
        
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        
        if minutes > 0:
            return f"{minutes}m {remaining_seconds}s"
        else:
            return f"{remaining_seconds}s"

    def get_otp_urgency_level(self, remaining_seconds: int) -> str:
        """Get urgency level based on remaining time"""
        if remaining_seconds <= 0:
            return "expired"
        elif remaining_seconds <= 30:
            return "critical"
        elif remaining_seconds <= 60:
            return "urgent"
        elif remaining_seconds <= 180:
            return "warning"
        else:
            return "normal"

    def get_otp_color_code(self, remaining_seconds: int) -> str:
        """Get color code for UI based on remaining time"""
        urgency = self.get_otp_urgency_level(remaining_seconds)
        
        color_map = {
            "expired": "#DC2626",    # red-600
            "critical": "#DC2626",   # red-600  
            "urgent": "#F59E0B",     # amber-500
            "warning": "#EAB308",    # yellow-500
            "normal": "#10B981"      # emerald-500
        }
        
        return color_map.get(urgency, "#6B7280")  # gray-500 as fallback
    async def verify_otp(self, auth_key: str, provided_otp: str) -> Dict[str, Any]:
        """Verify the provided OTP - returns standardized response"""
        try:
            stored_data = await self._retrieve_data(auth_key)
            if not stored_data:
                return AuthUtils.create_error_response(
                    "Invalid or expired authentication session",
                    "INVALID_SESSION"
                )
            
            # Check if OTP is expired
            expiry_time = stored_data["expiry"]
            if isinstance(expiry_time, str):
                expiry_time = datetime.fromisoformat(expiry_time)
            
            if datetime.now() > expiry_time:
                await self._delete_data(auth_key)
                return AuthUtils.create_error_response(
                    "Verification code has expired. Please request a new one.",
                    "OTP_EXPIRED"
                )
            
            # Check attempts
            if stored_data["attempts"] >= self.max_otp_attempts:
                await self._delete_data(auth_key)
                return AuthUtils.create_error_response(
                    "Maximum verification attempts exceeded. Please request a new code.",
                    "MAX_ATTEMPTS_EXCEEDED"
                )
            
            # Verify OTP
            if provided_otp == stored_data["otp"]:
                await self._delete_data(auth_key)
                
                # Get customer data for successful verification
                contact = stored_data["contact"]
                method = stored_data["method"]
                
                # Fetch customer data
                customer_query = {}
                if method == "email":
                    customer_query["email"] = contact.lower()
                else:
                    customer_query["phone"] = AuthUtils.format_phone(contact)
                
                customer = await self.db_service.find_customer(customer_query)
                
                return AuthUtils.create_success_response(
                    "Authentication successful",
                    data={
                        "customer_data": customer,
                        "authenticated": True
                    }
                )
            else:
                # Increment attempts
                stored_data["attempts"] += 1
                remaining_attempts = self.max_otp_attempts - stored_data["attempts"]
                
                # Update stored data
                await self._store_data(auth_key, stored_data, self.otp_expiry_minutes * 60)
                
                return AuthUtils.create_error_response(
                    f"Invalid verification code. {remaining_attempts} attempts remaining.",
                    "INVALID_OTP",
                    remaining_attempts=remaining_attempts
                )
                
        except Exception as e:
            print(f"Error verifying OTP: {e}")
            return AuthUtils.create_error_response(
                "Verification service temporarily unavailable. Please try again.",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def resend_otp(self, auth_key: str) -> Dict[str, Any]:
        """Resend OTP to the user - returns standardized response"""
        try:
            stored_data = await self._retrieve_data(auth_key)
            if not stored_data:
                return AuthUtils.create_error_response(
                    "Invalid authentication session",
                    "INVALID_SESSION"
                )
            
            # Generate new OTP
            new_otp = ''.join(random.choices(string.digits, k=self.otp_length))
            
            # Update stored data
            stored_data["otp"] = new_otp
            stored_data["expiry"] = datetime.now() + timedelta(minutes=self.otp_expiry_minutes)
            stored_data["attempts"] = 0
            
            await self._store_data(auth_key, stored_data, self.otp_expiry_minutes * 60)
            
            # Send new OTP using the stored method
            contact = stored_data["contact"]
            method = stored_data["method"]
            
            if method == "email":
                # Get customer name for email
                customer_query = {"email": contact.lower()}
                customer = await self.db_service.find_customer(customer_query)
                customer_name = customer.get("name", "Valued Customer") if customer else "Valued Customer"
                
                send_result = await self.send_otp_email(contact, new_otp, customer_name)
            else:
                send_result = await self.send_otp_sms(contact, new_otp)
            
            if send_result.get("success"):
                return AuthUtils.create_success_response(
                    f"New verification code sent to your {method}",
                    data={
                        "expires_in": self.otp_expiry_minutes,
                        "sent_to": send_result["data"]["sent_to"]
                    }
                )
            else:
                return AuthUtils.create_error_response(
                    f"Failed to resend verification code via {method}. Please try again.",
                    "RESEND_FAILED",
                    retry_allowed=True,
                    technical_error=True
                )
                
        except Exception as e:
            print(f"Error resending OTP: {e}")
            return AuthUtils.create_error_response(
                "Resend service temporarily unavailable. Please try again.",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def cleanup_expired_sessions(self, session_timeout_minutes: int = 30) -> int:
        """Clean up expired sessions - returns count of cleaned sessions"""
        try:
            cleaned_count = 0
            current_time = datetime.now()
            
            # Clean up memory storage
            expired_keys = []
            for key, data in self.memory_storage.items():
                if current_time > data["expires_at"]:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_storage[key]
            
            cleaned_count += len(expired_keys)
            
            # Clean up MongoDB temp data
            cutoff_time = current_time - timedelta(minutes=session_timeout_minutes)
            mongo_cleaned = await self.db_service.cleanup_expired_temp_data()
            cleaned_count += mongo_cleaned
            
            print(f"Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
            
        except Exception as e:
            print(f"Error during session cleanup: {e}")
            return 0

    async def get_auth_status(self, auth_key: str) -> Dict[str, Any]:
        """Get authentication status"""
        try:
            stored_data = await self._retrieve_data(auth_key)
            if not stored_data:
                return AuthUtils.create_success_response(
                    "Authentication session not found",
                    data={
                        "exists": False,
                        "expired": True
                    }
                )
            
            expiry_time = stored_data["expiry"]
            if isinstance(expiry_time, str):
                expiry_time = datetime.fromisoformat(expiry_time)
            
            is_expired = datetime.now() > expiry_time
            
            return AuthUtils.create_success_response(
                "Authentication status retrieved",
                data={
                    "exists": True,
                    "expired": is_expired,
                    "attempts": stored_data["attempts"],
                    "max_attempts": self.max_otp_attempts,
                    "remaining_attempts": self.max_otp_attempts - stored_data["attempts"],
                    "otp_method": stored_data.get("method", "email")
                }
            )
            
        except Exception as e:
            print(f"Error getting auth status: {e}")
            return AuthUtils.create_error_response(
                "Unable to retrieve authentication status",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    def get_contact_lockout_info(self) -> Dict[str, Any]:
        """Get contact lockout configuration info"""
        return {
            "max_attempts": self.max_contact_attempts,
            "lockout_minutes": self.contact_lockout_minutes
        }

    def get_otp_config(self) -> Dict[str, Any]:
        """Get OTP configuration info"""
        return {
            "otp_length": self.otp_length,
            "expiry_minutes": self.otp_expiry_minutes,
            "max_attempts": self.max_otp_attempts
        }
    
    async def cleanup_and_disconnect(self):
        """Cleanup resources and disconnect"""
        try:
            if self.db_service:
                # Redis not available from shared config, falling back to MongoDB storage
                await self.db_service.disconnect()
                self._db_connected = False
            
            if self.redis_client:
                self.redis_client.close()
                self.redis_client = None
            
            print("AuthService disconnected successfully")
        except Exception as e:
            print(f"Error during AuthService cleanup: {e}")


