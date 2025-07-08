# backend/services/auth_service.py - Fixed version
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path
from twilio.rest import Client
from .database_service import DatabaseService
from .auth_utils import AuthUtils
import redis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self):
        self.db_service = DatabaseService()
        self._db_connected = False
        
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("SMTP_USERNAME")
        self.email_password = os.getenv("SMTP_PASSWORD")
        
        # Twilio configuration for SMS
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        # Redis configuration with fallback
        self.redis_client = None
        self.use_redis = False
        self._init_redis()
        
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

    async def initialize(self):
        """Initialize the AuthService - must be called before using other methods"""
        try:
            await self.db_service.connect()
            self._db_connected = True
            logger.info("AuthService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AuthService: {e}")
            self._db_connected = False
            raise e

    async def ensure_db_connection(self):
        """Ensure database connection is established"""
        if not self._db_connected:
            try:
                await self.db_service.connect()
                self._db_connected = True
            except Exception as e:
                logger.error(f"Failed to establish database connection: {e}")
                raise ConnectionError("Database connection failed")


    def _init_redis(self):
        """Initialize Redis connection with error handling"""
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
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to MongoDB storage")
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
            
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.setex(key, expiry_seconds, serialized_data)
                    return True
                except Exception as e:
                    logger.warning(f"Redis storage failed: {e}. Falling back to MongoDB")
                    self.use_redis = False
            
            # MongoDB fallback
            await self.ensure_db_connection()  # Ensure DB connection is established
            expiry_time = datetime.now() + timedelta(seconds=expiry_seconds)
            await self.db_service.store_temp_data({
                "_id": key,
                "data": serialized_data,
                "expires_at": expiry_time,
                "created_at": datetime.now()
            })
            return True
            
        except Exception as e:
            logger.error(f"Both Redis and MongoDB storage failed: {e}")
            # Final fallback to memory (not recommended for production)
            self.memory_storage[key] = {
                "data": data,
                "expires_at": datetime.now() + timedelta(seconds=expiry_seconds)
            }
            return True

    async def _retrieve_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data with Redis primary, MongoDB fallback"""
        try:
            if self.use_redis and self.redis_client:
                try:
                    value = self.redis_client.get(key)
                    if value:
                        # Decode and parse JSON
                        if isinstance(value, bytes):
                            data = json.loads(value.decode("utf-8"))
                        elif isinstance(value, str):
                            data = json.loads(value)
                        else:
                            logger.warning(f"Unexpected Redis value type: {type(value)}")
                            return None
                        
                        # Convert ISO format strings back to datetime objects
                        return self._deserialize_datetime_fields(data)
                except Exception as e:
                    logger.warning(f"Redis retrieval failed: {e}. Trying MongoDB")
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
            logger.error(f"Data retrieval failed: {e}")
            return None

    async def _delete_data(self, key: str):
        """Delete data from all storage systems"""
        try:
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Redis deletion failed: {e}")
            
            # MongoDB cleanup
            await self.ensure_db_connection() 
            await self.db_service.delete_temp_data(key)
            
            # Memory cleanup
            if key in self.memory_storage:
                del self.memory_storage[key]
                
        except Exception as e:
            logger.error(f"Data deletion failed: {e}")

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
                logger.info(f"Successfully loaded template: {template_name}")
                return content
        except FileNotFoundError:
            logger.warning(f"Template {template_name} not found at {template_file}")
            logger.info("Using fallback template")
            return self._get_fallback_template()
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            logger.info("Using fallback template")
            return self._get_fallback_template()

    def _get_fallback_template(self) -> str:
        """Improved fallback template matching test_email.py style"""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Swiss Bank - Authentication Code</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            background-color: #1a472a;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .content {
            padding: 30px;
        }
        .otp {
            font-size: 32px;
            font-weight: bold;
            color: #1a472a;
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 20px 0;
            letter-spacing: 3px;
        }
        .footer {
            background-color: #f8f9fa;
            padding: 20px;
            font-size: 14px;
            color: #6c757d;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Swiss Bank</h1>
            <h2>Authentication Code</h2>
        </div>
        <div class="content">
            <p>Dear {customer_name},</p>
            <p>Your verification code is:</p>
            <div class="otp">{otp}</div>
            <p>This code will expire in <strong>{expiry_minutes} minutes</strong>.</p>
            <p><strong>Important:</strong> Do not share this code with anyone. Swiss Bank will never ask for this code over the phone or email.</p>
        </div>
        <div class="footer">
            <p><strong>Best regards,</strong><br>Swiss Bank Security Team</p>
            <p>If you did not request this code, please contact us immediately.</p>
        </div>
    </div>
</body>
</html>"""

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
                logger.warning("Some template placeholders may not have been replaced")
            
            return rendered_content
            
        except Exception as e:
            logger.error(f"Error rendering template with replace method: {e}")
            try:
                # Fallback to format method
                return template_content.format(**kwargs)
            except Exception as format_error:
                logger.error(f"Error rendering template with format method: {format_error}")
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
            logger.error(f"Error checking customer existence: {e}")
            return AuthUtils.create_error_response(
                "Customer lookup failed",
                "DATABASE_ERROR",
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
            logger.error(f"Error generating OTP: {e}")
            return AuthUtils.create_error_response(
                "OTP generation failed",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def send_otp_email(self, email: str, otp: str, customer_name: str = "Valued Customer") -> Dict[str, Any]:
        """Send OTP via email - improved version based on test_email.py"""
        try:
            # Check email configuration first
            if not self.email_user or not self.email_password:
                logger.error("Email credentials not configured")
                return AuthUtils.create_error_response(
                    "Email service not configured",
                    "SERVICE_ERROR",
                    technical_error=True
                )
            
            logger.info(f"Attempting to send OTP email to: {AuthUtils.mask_email(email)}")
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_user
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
            
            # Send email with detailed logging
            try:
                logger.info(f"Connecting to SMTP server: {self.smtp_server}:{self.smtp_port}")
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    logger.info("Starting TLS...")
                    server.starttls()
                    logger.info("Logging in to SMTP server...")
                    server.login(self.email_user, self.email_password)
                    logger.info("Sending email...")
                    server.send_message(msg)
                    logger.info("Email sent successfully!")
                
                return AuthUtils.create_success_response(
                    "OTP email sent successfully",
                    data={
                        "sent_to": AuthUtils.mask_email(email),
                        "method": "email"
                    }
                )
                
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"SMTP authentication failed: {e}")
                return AuthUtils.create_error_response(
                    "Email authentication failed",
                    "SEND_FAILED",
                    retry_allowed=True,
                    technical_error=True
                )
            except smtplib.SMTPConnectError as e:
                logger.error(f"SMTP connection failed: {e}")
                return AuthUtils.create_error_response(
                    "Email server connection failed",
                    "SEND_FAILED",
                    retry_allowed=True,
                    technical_error=True
                )
            except smtplib.SMTPException as e:
                logger.error(f"SMTP error: {e}")
                return AuthUtils.create_error_response(
                    "Email sending failed",
                    "SEND_FAILED",
                    retry_allowed=True,
                    technical_error=True
                )
            
        except Exception as e:
            logger.error(f"Unexpected error sending OTP email: {e}")
            return AuthUtils.create_error_response(
                "Failed to send OTP email",
                "SEND_FAILED",
                retry_allowed=True,
                technical_error=True
            )

    async def send_otp_sms(self, phone: str, otp: str) -> Dict[str, Any]:
        """Send OTP via SMS - returns standardized response"""
        try:
            if not self.twilio_account_sid or not self.twilio_auth_token:
                logger.error("Twilio credentials not configured")
                return AuthUtils.create_error_response(
                    "SMS service not configured",
                    "SERVICE_ERROR",
                    technical_error=True
                )
            
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            formatted_phone = AuthUtils.format_phone(phone)
            
            message = client.messages.create(
                body=f"Your Swiss Bank verification code is: {otp}. This code expires in {self.otp_expiry_minutes} minutes. Do not share this code with anyone.",
                from_=self.twilio_phone_number,
                to=formatted_phone
            )
            
            logger.info(f"SMS sent successfully. SID: {message.sid}")
            
            return AuthUtils.create_success_response(
                "OTP SMS sent successfully",
                data={
                    "sent_to": AuthUtils.mask_phone(phone),
                    "method": "sms",
                    "message_sid": message.sid
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending OTP SMS: {e}")
            return AuthUtils.create_error_response(
                "Failed to send OTP SMS",
                "SEND_FAILED",
                retry_allowed=True,
                technical_error=True
            )

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
            logger.error(f"Error verifying OTP: {e}")
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
            logger.error(f"Error resending OTP: {e}")
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
            # Fix: Pass cutoff_time as a keyword argument
            mongo_cleaned = await self.db_service.cleanup_expired_temp_data()
            cleaned_count += mongo_cleaned
            
            logger.info(f"Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            return 0

    async def get_auth_status(self, auth_key: str) -> Dict[str, Any]:
        """Get authentication status - returns standardized response"""
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
            logger.error(f"Error getting auth status: {e}")
            return AuthUtils.create_error_response(
                "Unable to retrieve authentication status",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    # Legacy method for backward compatibility - kept minimal
    async def initiate_authentication(self, email: Optional[str] = None, phone: Optional[str] = None, 
                                    session_id: Optional[str] = None, ip_address: Optional[str] = None,
                                    preferred_otp_method: Optional[str] = None) -> Dict[str, Any]:
        """Legacy method - use controller methods instead"""
        logger.warning("Using deprecated initiate_authentication method. Use AuthController instead.")
        
        # Basic validation
        if not email and not phone:
            return AuthUtils.create_error_response(
                "Please provide either email or phone number",
                "INVALID_INPUT"
            )
        
        return AuthUtils.create_error_response(
            "Please use the new authentication flow through AuthController",
            "DEPRECATED_METHOD"
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
                await self.db_service.disconnect()
                self._db_connected = False
            
            if self.redis_client:
                self.redis_client.close()
                self.redis_client = None
            
            logger.info("AuthService disconnected successfully")
        except Exception as e:
            logger.error(f"Error during AuthService cleanup: {e}")
