# backend/services/auth_service.py
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import os
import re
import json
from pathlib import Path
from twilio.rest import Client
from database_service import DatabaseService
import redis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self):
        self.db_service = DatabaseService()
        
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
        self.otp_expiry_minutes = 3
        self.max_otp_attempts = 3
        
        # Contact verification attempts configuration
        self.max_contact_attempts = 3
        self.contact_lockout_minutes = 15
        
        # Fallback storage (only used if both Redis and MongoDB fail)
        self.memory_storage = {}

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
            serialized_data = json.dumps(data, default=str)
            
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.setex(key, expiry_seconds, serialized_data)
                    return True
                except Exception as e:
                    logger.warning(f"Redis storage failed: {e}. Falling back to MongoDB")
                    self.use_redis = False
            
            # MongoDB fallback
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
                        return json.loads(value.decode("utf-8"))
                except Exception as e:
                    logger.warning(f"Redis retrieval failed: {e}. Trying MongoDB")
                    self.use_redis = False
            
            # MongoDB fallback
            temp_data = await self.db_service.get_temp_data(key)
            if temp_data:
                # Check if expired
                if datetime.now() > temp_data["expires_at"]:
                    await self.db_service.delete_temp_data(key)
                    return None
                return json.loads(temp_data["data"])
            
            # Memory fallback
            if key in self.memory_storage:
                stored = self.memory_storage[key]
                if datetime.now() > stored["expires_at"]:
                    del self.memory_storage[key]
                    return None
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
            await self.db_service.delete_temp_data(key)
            
            # Memory cleanup
            if key in self.memory_storage:
                del self.memory_storage[key]
                
        except Exception as e:
            logger.error(f"Data deletion failed: {e}")

    def load_email_template(self, template_name: str) -> str:
        """Load email template from file"""
        try:
            template_file = self.template_path / template_name
            with open(template_file, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            logger.warning(f"Template {template_name} not found at {template_file}")
            return self._get_fallback_template()
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            return self._get_fallback_template()

    def _get_fallback_template(self) -> str:
        """Fallback template if file loading fails"""
        return """
        <html>
        <body>
            <h2>Swiss Bank - Authentication Code</h2>
            <p>Dear {customer_name},</p>
            <p>Your verification code is: <strong>{otp}</strong></p>
            <p>This code will expire in {expiry_minutes} minutes.</p>
            <p>Best regards,<br>Swiss Bank Security Team</p>
        </body>
        </html>
        """

    def render_template(self, template_content: str, **kwargs) -> str:
        """Render template with provided variables"""
        try:
            return template_content.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return template_content
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return template_content

    def generate_otp(self) -> str:
        """Generate a random OTP"""
        return ''.join(random.choices(string.digits, k=self.otp_length))

    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        clean_phone = re.sub(r'\D', '', phone)
        return len(clean_phone) >= 10 and len(clean_phone) <= 15

    def format_phone(self, phone: str) -> str:
        """Format phone number for international use"""
        clean_phone = re.sub(r'\D', '', phone)
        if len(clean_phone) == 10:
            return f"+1{clean_phone}"
        elif not clean_phone.startswith('+'):
            return f"+{clean_phone}"
        return clean_phone

    async def _check_contact_attempts(self, identifier: str, ip_address: str = None) -> Tuple[bool, int]:
        """Check if contact verification attempts are within limits"""
        attempt_key = f"contact_attempts:{identifier}"
        if ip_address:
            attempt_key += f":{ip_address}"
        
        attempts_data = await self._retrieve_data(attempt_key)
        if not attempts_data:
            return True, 0
        
        attempts = attempts_data.get("attempts", 0)
        return attempts < self.max_contact_attempts, attempts

    async def _increment_contact_attempts(self, identifier: str, ip_address: str = None):
        """Increment contact verification attempts"""
        attempt_key = f"contact_attempts:{identifier}"
        if ip_address:
            attempt_key += f":{ip_address}"
        
        attempts_data = await self._retrieve_data(attempt_key)
        if not attempts_data:
            attempts_data = {"attempts": 0, "first_attempt": datetime.now()}
        
        attempts_data["attempts"] += 1
        attempts_data["last_attempt"] = datetime.now()
        
        # Store with lockout duration
        await self._store_data(attempt_key, attempts_data, self.contact_lockout_minutes * 60)

    async def check_customer_exists(self, email: Optional[str] = None, phone: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if customer exists in database by email or phone"""
        try:
            query = {}
            if email:
                query["email"] = email.lower()
            if phone:
                formatted_phone = self.format_phone(phone)
                query["phone"] = formatted_phone
            
            customer = await self.db_service.find_customer(query)
            
            if customer:
                return True, customer
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Error checking customer existence: {e}")
            return False, None

    async def send_otp_email(self, email: str, otp: str, customer_name: str = "Valued Customer") -> bool:
        """Send OTP via email"""
        try:
            if not self.email_user or not self.email_password:
                logger.error("Email credentials not configured")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = email
            msg['Subject'] = "Swiss Bank - Authentication Code"
            
            template_content = self.load_email_template("otp_email.html")
            html_body = self.render_template(
                template_content,
                customer_name=customer_name,
                otp=otp,
                expiry_minutes=self.otp_expiry_minutes
            )
            
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending OTP email: {e}")
            return False

    async def send_otp_sms(self, phone: str, otp: str) -> bool:
        """Send OTP via SMS using Twilio"""
        try:
            if not self.twilio_account_sid or not self.twilio_auth_token:
                logger.error("Twilio credentials not configured")
                return False
            
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            formatted_phone = self.format_phone(phone)
            
            message = client.messages.create(
                body=f"Your Swiss Bank verification code is: {otp}. This code expires in {self.otp_expiry_minutes} minutes. Do not share this code with anyone.",
                from_=self.twilio_phone_number,
                to=formatted_phone
            )
            
            logger.info(f"SMS sent successfully. SID: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending OTP SMS: {e}")
            return False

    async def initiate_authentication(self, email: Optional[str] = None, phone: Optional[str] = None, 
                                    session_id: Optional[str] = None, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """Initiate authentication process with contact attempt tracking"""
        try:
            # Validate input
            if not email and not phone:
                return {
                    "success": False,
                    "message": "Please provide either email or phone number",
                    "error_code": "INVALID_INPUT"
                }
            
            if email and not self.validate_email(email):
                return {
                    "success": False,
                    "message": "Please provide a valid email address",
                    "error_code": "INVALID_EMAIL"
                }
            
            if phone and not self.validate_phone(phone):
                return {
                    "success": False,
                    "message": "Please provide a valid phone number",
                    "error_code": "INVALID_PHONE"
                }
            
            # Check contact attempts
            contact_identifier = email or phone
            can_attempt, attempts = await self._check_contact_attempts(contact_identifier, ip_address)
            
            if not can_attempt:
                remaining_time = self.contact_lockout_minutes
                return {
                    "success": False,
                    "message": f"Too many failed attempts. Please try again after {remaining_time} minutes.",
                    "error_code": "TOO_MANY_ATTEMPTS",
                    "retry_after_minutes": remaining_time
                }
            
            # Check if customer exists
            customer_exists, customer_data = await self.check_customer_exists(email, phone)
            
            if not customer_exists:
                # Increment failed contact attempts
                await self._increment_contact_attempts(contact_identifier, ip_address)
                
                remaining_attempts = self.max_contact_attempts - attempts - 1
                
                if remaining_attempts > 0:
                    if email and phone:
                        message = f"The email or phone number doesn't match our records. {remaining_attempts} attempts remaining."
                    elif email:
                        message = f"The email address doesn't match our records. {remaining_attempts} attempts remaining."
                    else:
                        message = f"The phone number doesn't match our records. {remaining_attempts} attempts remaining."
                else:
                    message = f"Account not found. You've reached the maximum attempts. Please try again after {self.contact_lockout_minutes} minutes."
                
                return {
                    "success": False,
                    "message": message,
                    "error_code": "CUSTOMER_NOT_FOUND",
                    "remaining_attempts": remaining_attempts
                }
            
            # Generate and store OTP
            otp = self.generate_otp()
            auth_key = f"otp:{session_id}_{email or phone}"
            
            otp_data = {
                "otp": otp,
                "expiry": datetime.now() + timedelta(minutes=self.otp_expiry_minutes),
                "attempts": 0,
                "customer_data": customer_data,
                "email": email,
                "phone": phone,
                "created_at": datetime.now()
            }
            
            await self._store_data(auth_key, otp_data, self.otp_expiry_minutes * 60)
            
            # Send OTP
            otp_sent = False
            send_method = ""
            
            if email:
                customer_name = customer_data.get("name", "Valued Customer") if customer_data else "Valued Customer"
                otp_sent = await self.send_otp_email(email, otp, customer_name)
                send_method = "email"
            elif phone:
                otp_sent = await self.send_otp_sms(phone, otp)
                send_method = "SMS"
            
            if otp_sent:
                # Reset contact attempts on successful OTP send
                await self._delete_data(f"contact_attempts:{contact_identifier}")
                
                # Mask contact for security
                masked_contact = ""
                if email:
                    masked_contact = f"{email[:3]}***@{email.split('@')[1]}"
                elif phone:
                    masked_phone = self.format_phone(phone)
                    masked_contact = f"***-***-{masked_phone[-4:]}"
                
                return {
                    "success": True,
                    "message": f"Verification code sent to your {send_method}",
                    "masked_contact": masked_contact,
                    "auth_key": auth_key,
                    "expires_in": self.otp_expiry_minutes
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to send verification code via {send_method}. Please try again.",
                    "error_code": "SEND_FAILED"
                }
                
        except Exception as e:
            logger.error(f"Error initiating authentication: {e}")
            return {
                "success": False,
                "message": "Authentication service temporarily unavailable. Please try again.",
                "error_code": "SERVICE_ERROR"
            }

    async def verify_otp(self, auth_key: str, provided_otp: str) -> Dict[str, Any]:
        """Verify the provided OTP"""
        try:
            stored_data = await self._retrieve_data(auth_key)
            if not stored_data:
                return {
                    "success": False,
                    "message": "Invalid or expired authentication session",
                    "error_code": "INVALID_SESSION"
                }
            
            # Check if OTP is expired
            expiry_time = stored_data["expiry"]
            if isinstance(expiry_time, str):
                expiry_time = datetime.fromisoformat(expiry_time)
            
            if datetime.now() > expiry_time:
                await self._delete_data(auth_key)
                return {
                    "success": False,
                    "message": "Verification code has expired. Please request a new one.",
                    "error_code": "OTP_EXPIRED"
                }
            
            # Check attempts
            if stored_data["attempts"] >= self.max_otp_attempts:
                await self._delete_data(auth_key)
                return {
                    "success": False,
                    "message": "Maximum verification attempts exceeded. Please request a new code.",
                    "error_code": "MAX_ATTEMPTS_EXCEEDED"
                }
            
            # Verify OTP
            if provided_otp == stored_data["otp"]:
                customer_data = stored_data["customer_data"]
                await self._delete_data(auth_key)
                
                return {
                    "success": True,
                    "message": "Authentication successful",
                    "customer_data": customer_data,
                    "authenticated": True
                }
            else:
                # Increment attempts
                stored_data["attempts"] += 1
                remaining_attempts = self.max_otp_attempts - stored_data["attempts"]
                
                # Update stored data
                await self._store_data(auth_key, stored_data, self.otp_expiry_minutes * 60)
                
                return {
                    "success": False,
                    "message": f"Invalid verification code. {remaining_attempts} attempts remaining.",
                    "error_code": "INVALID_OTP",
                    "remaining_attempts": remaining_attempts
                }
                
        except Exception as e:
            logger.error(f"Error verifying OTP: {e}")
            return {
                "success": False,
                "message": "Verification service temporarily unavailable. Please try again.",
                "error_code": "SERVICE_ERROR"
            }

    async def resend_otp(self, auth_key: str) -> Dict[str, Any]:
        """Resend OTP to the user"""
        try:
            stored_data = await self._retrieve_data(auth_key)
            if not stored_data:
                return {
                    "success": False,
                    "message": "Invalid authentication session",
                    "error_code": "INVALID_SESSION"
                }
            
            # Generate new OTP
            new_otp = self.generate_otp()
            
            # Update stored data
            stored_data["otp"] = new_otp
            stored_data["expiry"] = datetime.now() + timedelta(minutes=self.otp_expiry_minutes)
            stored_data["attempts"] = 0
            
            await self._store_data(auth_key, stored_data, self.otp_expiry_minutes * 60)
            
            # Send new OTP
            otp_sent = False
            send_method = ""
            
            if stored_data["email"]:
                customer_name = stored_data["customer_data"].get("name", "Valued Customer")
                otp_sent = await self.send_otp_email(stored_data["email"], new_otp, customer_name)
                send_method = "email"
            elif stored_data["phone"]:
                otp_sent = await self.send_otp_sms(stored_data["phone"], new_otp)
                send_method = "SMS"
            
            if otp_sent:
                return {
                    "success": True,
                    "message": f"New verification code sent to your {send_method}",
                    "expires_in": self.otp_expiry_minutes
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to resend verification code via {send_method}. Please try again.",
                    "error_code": "RESEND_FAILED"
                }
                
        except Exception as e:
            logger.error(f"Error resending OTP: {e}")
            return {
                "success": False,
                "message": "Resend service temporarily unavailable. Please try again.",
                "error_code": "SERVICE_ERROR"
            }

    async def cleanup_expired_data(self):
        """Clean up expired data from memory storage"""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for key, data in self.memory_storage.items():
                if current_time > data["expires_at"]:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_storage[key]
            
            logger.info(f"Cleaned up {len(expired_keys)} expired entries from memory storage")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_auth_status(self, auth_key: str) -> Dict[str, Any]:
        """Get authentication status"""
        try:
            stored_data = await self._retrieve_data(auth_key)
            if not stored_data:
                return {
                    "exists": False,
                    "expired": True
                }
            
            expiry_time = stored_data["expiry"]
            if isinstance(expiry_time, str):
                expiry_time = datetime.fromisoformat(expiry_time)
            
            is_expired = datetime.now() > expiry_time
            
            return {
                "exists": True,
                "expired": is_expired,
                "attempts": stored_data["attempts"],
                "max_attempts": self.max_otp_attempts,
                "remaining_attempts": self.max_otp_attempts - stored_data["attempts"]
            }
            
        except Exception as e:
            logger.error(f"Error getting auth status: {e}")
            return {
                "exists": False,
                "expired": True
            }