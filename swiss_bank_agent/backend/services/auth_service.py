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
from pathlib import Path
from twilio.rest import Client
from database_service import DatabaseService
import redis

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
        
        # Redis configuration
        self.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))

        # Template path
        self.template_path = Path(__file__).parent.parent / "templates" / "emails"
        
        # OTP storage (in production, use Redis or database)
        self.otp_storage = {}
        
        # OTP configuration
        self.otp_length = 6
        self.otp_expiry_minutes = 3
        self.max_attempts = 3

    def load_email_template(self, template_name: str) -> str:
        """Load email template from file"""
        try:
            template_file = self.template_path / template_name
            with open(template_file, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Template {template_name} not found at {template_file}")
            return self._get_fallback_template()
        except Exception as e:
            print(f"Error loading template {template_name}: {e}")
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
            print(f"Missing template variable: {e}")
            return template_content
        except Exception as e:
            print(f"Error rendering template: {e}")
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
        # Remove all non-digit characters
        clean_phone = re.sub(r'\D', '', phone)
        # Check if it's a valid length (10-15 digits)
        return len(clean_phone) >= 10 and len(clean_phone) <= 15

    def format_phone(self, phone: str) -> str:
        """Format phone number for international use"""
        clean_phone = re.sub(r'\D', '', phone)
        if len(clean_phone) == 10:
            return f"+1{clean_phone}"  # Assume US number
        elif not clean_phone.startswith('+'):
            return f"+{clean_phone}"
        return clean_phone

    async def check_customer_exists(self, email: Optional[str] = None, phone: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if customer exists in database by email or phone"""
        try:
            query = {}
            if email:
                query["email"] = email.lower()
            if phone:
                formatted_phone = self.format_phone(phone)
                query["phone"] = formatted_phone
            
            # Query database for customer
            customer = await self.db_service.find_customer(query)
            
            if customer:
                return True, customer
            else:
                return False, None
                
        except Exception as e:
            print(f"Error checking customer existence: {e}")
            return False, None

    async def send_otp_email(self, email: str, otp: str, customer_name: str = "Valued Customer") -> bool:
        """Send OTP via email"""
        try:
            # Check if email credentials are available
            if not self.email_user or not self.email_password:
                print("Error: Email credentials not configured")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = email
            msg['Subject'] = "Swiss Bank - Authentication Code"
            
            # Load and render email template
            template_content = self.load_email_template("otp_email.html")
            html_body = self.render_template(
                template_content,
                customer_name=customer_name,
                otp=otp,
                expiry_minutes=self.otp_expiry_minutes
            )
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Error sending OTP email: {e}")
            return False

    async def send_otp_sms(self, phone: str, otp: str) -> bool:
        """Send OTP via SMS using Twilio"""
        try:
            if not self.twilio_account_sid or not self.twilio_auth_token:
                print("Twilio credentials not configured")
                return False
            
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            formatted_phone = self.format_phone(phone)
            
            message = client.messages.create(
                body=f"Your Swiss Bank verification code is: {otp}. This code expires in {self.otp_expiry_minutes} minutes. Do not share this code with anyone.",
                from_=self.twilio_phone_number,
                to=formatted_phone
            )
            
            print(f"SMS sent successfully. SID: {message.sid}")
            return True
            
        except Exception as e:
            print(f"Error sending OTP SMS: {e}")
            return False

    async def initiate_authentication(self, email: Optional[str] = None, phone: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Initiate authentication process"""
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
            
            # Check if customer exists
            customer_exists, customer_data = await self.check_customer_exists(email, phone)
            
            if not customer_exists:
                if email and phone:
                    return {
                        "success": False,
                        "message": "Oops! The email or phone number you entered doesn't seem to be linked to a Swiss Bank account. Please double-check your details and try again.",
                        "error_code": "CUSTOMER_NOT_FOUND"
                    }
                elif email:
                    return {
                        "success": False,
                        "message": "Oops! The email you entered doesn't seem to be linked to a Swiss Bank account. Please double-check your details and try again.",
                        "error_code": "CUSTOMER_NOT_FOUND_EMAIL"
                    }
                elif phone:
                    return {
                        "success": False,
                        "message": "Oops! The phone number you entered doesn't seem to be linked to a Swiss Bank account. Please double-check your details and try again",
                        "error_code": "CUSTOMER_NOT_FOUND_PHONE"
                    }
            
            # Generate OTP
            otp = self.generate_otp()
            
            # Store OTP with expiry
            auth_key = f"{session_id}_{email or phone}"
            self.otp_storage[auth_key] = {
                "otp": otp,
                "expiry": datetime.now() + timedelta(minutes=self.otp_expiry_minutes),
                "attempts": 0,
                "customer_data": customer_data,
                "email": email,
                "phone": phone
            }
            
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
                # Mask email/phone for security
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
            print(f"Error initiating authentication: {e}")
            return {
                "success": False,
                "message": "Authentication service temporarily unavailable. Please try again.",
                "error_code": "SERVICE_ERROR"
            }

    async def verify_otp(self, auth_key: str, provided_otp: str) -> Dict[str, Any]:
        """Verify the provided OTP"""
        try:
            if auth_key not in self.otp_storage:
                return {
                    "success": False,
                    "message": "Invalid or expired authentication session",
                    "error_code": "INVALID_SESSION"
                }
            
            stored_data = self.otp_storage[auth_key]
            
            # Check if OTP is expired
            if datetime.now() > stored_data["expiry"]:
                del self.otp_storage[auth_key]
                return {
                    "success": False,
                    "message": "Verification code has expired. Please request a new one.",
                    "error_code": "OTP_EXPIRED"
                }
            
            # Check attempts
            if stored_data["attempts"] >= self.max_attempts:
                del self.otp_storage[auth_key]
                return {
                    "success": False,
                    "message": "Maximum verification attempts exceeded. Please request a new code.",
                    "error_code": "MAX_ATTEMPTS_EXCEEDED"
                }
            
            # Verify OTP
            if provided_otp == stored_data["otp"]:
                customer_data = stored_data["customer_data"]
                
                # Clean up OTP storage
                del self.otp_storage[auth_key]
                
                return {
                    "success": True,
                    "message": "Authentication successful",
                    "customer_data": customer_data,
                    "authenticated": True
                }
            else:
                # Increment attempts
                stored_data["attempts"] += 1
                remaining_attempts = self.max_attempts - stored_data["attempts"]
                
                return {
                    "success": False,
                    "message": f"Invalid verification code. {remaining_attempts} attempts remaining.",
                    "error_code": "INVALID_OTP",
                    "remaining_attempts": remaining_attempts
                }
                
        except Exception as e:
            print(f"Error verifying OTP: {e}")
            return {
                "success": False,
                "message": "Verification service temporarily unavailable. Please try again after sometime.",
                "error_code": "SERVICE_ERROR"
            }

    async def resend_otp(self, auth_key: str) -> Dict[str, Any]:
        """Resend OTP to the user"""
        try:
            if auth_key not in self.otp_storage:
                return {
                    "success": False,
                    "message": "Invalid authentication session",
                    "error_code": "INVALID_SESSION"
                }
            
            stored_data = self.otp_storage[auth_key]
            
            # Generate new OTP
            new_otp = self.generate_otp()
            
            # Update stored data
            stored_data["otp"] = new_otp
            stored_data["expiry"] = datetime.now() + timedelta(minutes=self.otp_expiry_minutes)
            stored_data["attempts"] = 0
            
            # Send new OTP
            otp_sent = False
            send_method = ""
            
            if stored_data["email"]:
                otp_sent = await self.send_otp_email(
                    stored_data["email"], 
                    new_otp, 
                    stored_data["customer_data"].get("name", "Valued Customer")
                )
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
                    "message": f"Failed to resend verification code via {send_method}. Please try again, after sometime.",
                    "error_code": "RESEND_FAILED"
                }
                
        except Exception as e:
            print(f"Error resending OTP: {e}")
            return {
                "success": False,
                "message": "Resend service temporarily unavailable. Please try again, after sometime.",
                "error_code": "SERVICE_ERROR"
            }

    def cleanup_expired_otps(self):
        """Clean up expired OTPs from storage"""
        current_time = datetime.now()
        expired_keys = [
            key for key, data in self.otp_storage.items()
            if current_time > data["expiry"]
        ]
        
        for key in expired_keys:
            del self.otp_storage[key]
        
        print(f"Cleaned up {len(expired_keys)} expired OTPs")

    def get_auth_status(self, auth_key: str) -> Dict[str, Any]:
        """Get authentication status"""
        if auth_key not in self.otp_storage:
            return {
                "exists": False,
                "expired": True
            }
        
        stored_data = self.otp_storage[auth_key]
        is_expired = datetime.now() > stored_data["expiry"]
        
        return {
            "exists": True,
            "expired": is_expired,
            "attempts": stored_data["attempts"],
            "max_attempts": self.max_attempts,
            "remaining_attempts": self.max_attempts - stored_data["attempts"]
        }
    
