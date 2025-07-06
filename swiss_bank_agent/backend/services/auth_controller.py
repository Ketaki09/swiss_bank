# backend/services/auth_controller.py - Enhanced version with fixed imports
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging
from .auth_service import AuthService
from .database_service import DatabaseService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthController:
    """
    Enhanced authentication controller with technical error handling
    and proper OTP source selection.
    """
    
    def __init__(self):
        self.auth_service = AuthService()
        self.db_service = DatabaseService()
        
        # Configuration for contact verification attempts
        self.max_contact_verification_attempts = 3
        self.contact_lockout_minutes = 300 
        self.session_timeout_minutes = 15
        
        # Technical error retry configuration
        self.max_technical_retries = 5
        self.technical_error_codes = {
            "SERVICE_ERROR", "DATABASE_ERROR", "NETWORK_ERROR", 
            "TIMEOUT_ERROR", "SEND_FAILED", "RESEND_FAILED"
        }
        
        # Session states
        self.SESSION_STATES = {
            "CONTACT_VERIFICATION": "contact_verification",
            "OTP_VERIFICATION": "otp_verification", 
            "AUTHENTICATED": "authenticated",
            "LOCKED": "locked",
            "EXPIRED": "expired"
        }

    def _is_technical_error(self, error_code: str) -> bool:
        """Check if error is technical (system) vs user input error"""
        return error_code in self.technical_error_codes

    async def create_auth_session(self, ip_address: Optional[str] = None, 
                                user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Create a new authentication session"""
        try:
            session_id = str(uuid.uuid4())
            
            session_data = {
                "session_id": session_id,
                "state": self.SESSION_STATES["CONTACT_VERIFICATION"],
                "contact_attempts": 0,
                "technical_retry_count": 0,
                "max_technical_retries": 5,  
                "technical_retry_reset_time": None, 
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "contact_verified": False,
                "authenticated": False,
                "customer_data": None,
                "contact_email": None,
                "contact_phone": None,
                "preferred_otp_method": None  # 'email' or 'sms'
            }
            
            # Store session
            session_key = f"auth_session:{session_id}"
            await self.auth_service._store_data(
                session_key, 
                session_data, 
                self.session_timeout_minutes * 60
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "state": self.SESSION_STATES["CONTACT_VERIFICATION"],
                "message": "Please provide your email or phone number.",
                "max_attempts": self.max_contact_verification_attempts,
                "expires_in_minutes": self.session_timeout_minutes
            }
            
        except Exception as e:
            logger.error(f"Error creating auth session: {e}")
            return {
                "success": False,
                "message": "Failed to create authentication session. Please contact support@swissbank.com",
                "error_code": "SESSION_CREATION_FAILED"
            }

    async def verify_contact_details(self, session_id: str, email: Optional[str] = None, 
                                   phone: Optional[str] = None, 
                                   preferred_otp_method: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced contact verification with technical error handling and OTP method selection.
        """
        try:
            # Get session data
            session_key = f"auth_session:{session_id}"
            session_data = await self.auth_service._retrieve_data(session_key)
            
            if not session_data:
                return {
                    "success": False,
                    "message": "Invalid or expired session. Please start again.",
                    "error_code": "INVALID_SESSION",
                    "action_required": "restart"
                }
            
            # Check session state
            if session_data["state"] != self.SESSION_STATES["CONTACT_VERIFICATION"]:
                return {
                    "success": False,
                    "message": f"Invalid session state. Expected contact verification, got {session_data['state']}",
                    "error_code": "INVALID_STATE"
                }
            
            # Check if session is locked
            if session_data["state"] == self.SESSION_STATES["LOCKED"]:
                lockout_remaining = self._get_lockout_remaining_time(session_data)
                return {
                    "success": False,
                    "message": f"Session locked due to too many failed attempts. Try again in {lockout_remaining} minutes.",
                    "error_code": "SESSION_LOCKED",
                    "retry_after_minutes": lockout_remaining
                }
            
            # Validate input
            if email and phone:
                return {
                    "success": False,
                    "message": "Please provide either email or phone number, not both.",
                    "error_code": "INVALID_INPUT"
                }
            
            if not email and not phone:
                return {
                    "success": False,
                    "message": "Please provide email or phone number.",
                    "error_code": "INVALID_INPUT",
                    "current_attempt": session_data["contact_attempts"] + 1,
                    "max_attempts": self.max_contact_verification_attempts
                }
            
            # Validate OTP method preference
            if preferred_otp_method and preferred_otp_method not in ['email', 'sms']:
                return {
                    "success": False,
                    "message": "Invalid OTP method. Choose 'email' or 'sms'.",
                    "error_code": "INVALID_OTP_METHOD"
                }
            
            # Auto-determine OTP method if not specified
            if not preferred_otp_method:
                if email and not phone:
                    preferred_otp_method = 'email'
                else:
                    preferred_otp_method = 'sms'
                
            
            # Validate that preferred method matches provided contact info
            if preferred_otp_method == 'email' and not email:
                return {
                    "success": False,
                    "message": "Email address required for email OTP.",
                    "error_code": "EMAIL_REQUIRED"
                }
            
            if preferred_otp_method == 'sms' and not phone:
                return {
                    "success": False,
                    "message": "Phone number required for SMS OTP.",
                    "error_code": "PHONE_REQUIRED"
                }
            
            # Validate formats - these are USER errors, not technical
            if email and not self.auth_service.validate_email(email):
                session_data["contact_attempts"] += 1
                session_data["last_activity"] = datetime.now()
                await self.auth_service._store_data(
                    session_key, 
                    session_data, 
                    self.session_timeout_minutes * 60
                )
                
                remaining_attempts = self.max_contact_verification_attempts - session_data["contact_attempts"]
                
                return {
                    "success": False,
                    "message": f"Invalid email format. Please provide a valid email address.",
                    "error_code": "INVALID_EMAIL_FORMAT",
                    "current_attempt": session_data["contact_attempts"],
                    "remaining_attempts": remaining_attempts,
                    "max_attempts": self.max_contact_verification_attempts
                }
            
            if phone and not self.auth_service.validate_phone(phone):
                session_data["contact_attempts"] += 1
                session_data["last_activity"] = datetime.now()
                await self.auth_service._store_data(
                    session_key, 
                    session_data, 
                    self.session_timeout_minutes * 60
                )
                
                remaining_attempts = self.max_contact_verification_attempts - session_data["contact_attempts"]
                
                return {
                    "success": False,
                    "message": f"Invalid phone number format. Please provide a valid phone number.",
                    "error_code": "INVALID_PHONE_FORMAT",
                    "current_attempt": session_data["contact_attempts"],
                    "remaining_attempts": remaining_attempts,
                    "max_attempts": self.max_contact_verification_attempts
                }
            
            # Check if customer exists - with retry logic for technical errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    customer_exists, customer_data = await self.auth_service.check_customer_exists(email, phone)
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        session_data["technical_retry_count"] += 1
                        await self.auth_service._store_data(session_key, session_data, self.session_timeout_minutes * 60)
                        
                        logger.error(f"Customer lookup failed after {max_retries} attempts: {e}")
                        return {
                            "success": False,
                            "message": "Customer verification service temporarily unavailable. Please try again.",
                            "error_code": "SERVICE_ERROR",
                            "retry_allowed": True,
                            "technical_error": True
                        }
                    # Wait before retry
                    await asyncio.sleep(1)
                    continue
            
            if not customer_exists:
                # This is a USER error - increment attempts
                session_data["contact_attempts"] += 1
                session_data["last_activity"] = datetime.now()
                
                # Check if max attempts reached
                if session_data["contact_attempts"] >= self.max_contact_verification_attempts:
                    # Lock the session
                    session_data["state"] = self.SESSION_STATES["LOCKED"]
                    session_data["locked_at"] = datetime.now()
                    await self.auth_service._store_data(
                        session_key, 
                        session_data, 
                        self.contact_lockout_minutes * 60
                    )
                    
                    return {
                        "success": False,
                        "message": f"Maximum contact verification attempts exceeded. Session locked for {self.contact_lockout_minutes} minutes.",
                        "error_code": "MAX_ATTEMPTS_EXCEEDED",
                        "retry_after_minutes": self.contact_lockout_minutes
                    }
                
                await self.auth_service._store_data(
                    session_key, 
                    session_data, 
                    self.session_timeout_minutes * 60
                )
                
                remaining_attempts = self.max_contact_verification_attempts - session_data["contact_attempts"]
                
                if email and phone:
                    message = f"No account found with the provided email or phone number. Please verify your details and try again."
                elif email:
                    message = f"No account found with email '{email}'. Please check your email address and try again."
                else:
                    message = f"No account found with phone number '{phone}'. Please check your phone number and try again."
                
                message += f" ({remaining_attempts} attempts remaining)"
                
                return {
                    "success": False,
                    "message": message,
                    "error_code": "CUSTOMER_NOT_FOUND",
                    "current_attempt": session_data["contact_attempts"],
                    "remaining_attempts": remaining_attempts,
                    "max_attempts": self.max_contact_verification_attempts,
                    "suggestions": [
                        "Double-check your email address or phone number",
                        "Try using the alternate contact method (email vs phone)",
                        "Contact customer support if you're unsure about your details"
                    ]
                }
            
            # Contact verification successful
            session_data["contact_verified"] = True
            session_data["customer_data"] = customer_data
            session_data["contact_email"] = email
            session_data["contact_phone"] = phone
            session_data["preferred_otp_method"] = preferred_otp_method
            session_data["state"] = self.SESSION_STATES["OTP_VERIFICATION"]
            session_data["last_activity"] = datetime.now()
            session_data["contact_verified_at"] = datetime.now()
            
            await self.auth_service._store_data(
                session_key, 
                session_data, 
                self.session_timeout_minutes * 60
            )
            
            return {
                "success": True,
                "message": "Contact details verified successfully. Proceeding to OTP verification.",
                "state": self.SESSION_STATES["OTP_VERIFICATION"],
                "customer_name": customer_data.get("name", "Valued Customer"),
                "otp_method": preferred_otp_method,
                "masked_email": self._mask_email(email) if email else None,
                "masked_phone": self._mask_phone(phone) if phone else None
            }
            
        except Exception as e:
            logger.error(f"Error verifying contact details: {e}")
            return {
                "success": False,
                "message": "Contact verification service temporarily unavailable. Please try again later.",
                "error_code": "SERVICE_ERROR",
                "retry_allowed": True,
                "technical_error": True
            }

    async def initiate_otp_verification(self, session_id: str) -> Dict[str, Any]:
        """
        Enhanced OTP initiation with method-specific sending and technical error handling.
        """
        try:
            session_key = f"auth_session:{session_id}"
            session_data = await self.auth_service._retrieve_data(session_key)
            
            if not session_data:
                return {
                    "success": False,
                    "message": "Invalid or expired session.",
                    "error_code": "INVALID_SESSION"
                }
            
            # Check session state
            if session_data["state"] != self.SESSION_STATES["OTP_VERIFICATION"]:
                return {
                    "success": False,
                    "message": "Invalid session state. Contact verification required first.",
                    "error_code": "INVALID_STATE"
                }
            
            if not session_data["contact_verified"]:
                return {
                    "success": False,
                    "message": "Contact verification required before OTP generation.",
                    "error_code": "CONTACT_NOT_VERIFIED"
                }
            
            # Enhanced OTP sending with method-specific logic
            preferred_method = session_data["preferred_otp_method"]
            email = session_data["contact_email"]
            phone = session_data["contact_phone"]
            
            # Use method-specific OTP sending with retry logic
            max_retries = 3
            result = None
            
            for attempt in range(max_retries):
                try:
                    if preferred_method == 'email':
                        result = await self.auth_service.initiate_authentication(
                            email=email,
                            phone=None,  # Explicitly set to None
                            session_id=session_id,
                            ip_address=session_data.get("ip_address")
                        )
                    elif preferred_method == 'sms':
                        result = await self.auth_service.initiate_authentication(
                            email=None,  # Explicitly set to None
                            phone=phone,
                            session_id=session_id,
                            ip_address=session_data.get("ip_address")
                        )
                    
                    # Check if it's a technical error
                    if not result["success"] and self._is_technical_error(result.get("error_code", "")):
                        if attempt == max_retries - 1:
                            return {
                                "success": False,
                                "message": f"OTP service temporarily unavailable. Please try again.",
                                "error_code": "SERVICE_ERROR",
                                "retry_allowed": True,
                                "technical_error": True
                            }
                        await asyncio.sleep(1)  # Wait before retry
                        continue
                    
                    # Success or user error - break out of retry loop
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        session_data["technical_retry_count"] += 1
                        await self.auth_service._store_data(session_key, session_data, self.session_timeout_minutes * 60)
        
                        logger.error(f"OTP initiation failed after {max_retries} attempts: {e}")
                        return {
                            "success": False,
                            "message": "OTP service temporarily unavailable. Please try again.",
                            "error_code": "SERVICE_ERROR",
                            "retry_allowed": True,
                            "technical_error": True
                        }
                    await asyncio.sleep(1)
                    continue
            
            if result and result["success"]:
                # Update session with OTP auth key
                session_data["otp_auth_key"] = result["auth_key"]
                session_data["otp_initiated_at"] = datetime.now()
                session_data["last_activity"] = datetime.now()
                
                await self.auth_service._store_data(
                    session_key, 
                    session_data, 
                    self.session_timeout_minutes * 60
                )
                
                return {
                    "success": True,
                    "message": result["message"],
                    "masked_contact": result["masked_contact"],
                    "expires_in": result["expires_in"],
                    "otp_method": preferred_method,
                    "state": self.SESSION_STATES["OTP_VERIFICATION"]
                }
            else:
                return result if result else {
                    "success": False,
                    "message": "Failed to initiate OTP verification.",
                    "error_code": "OTP_INITIATION_FAILED"
                }
                
        except Exception as e:
            logger.error(f"Error initiating OTP verification: {e}")
            return {
                "success": False,
                "message": "OTP service temporarily unavailable. Please try again after sometime.",
                "error_code": "SERVICE_ERROR",
                "retry_allowed": True,
                "technical_error": True
            }

    async def verify_otp(self, session_id: str, otp: str) -> Dict[str, Any]:
        """Enhanced OTP verification with technical error handling"""
        try:
            session_key = f"auth_session:{session_id}"
            session_data = await self.auth_service._retrieve_data(session_key)
            
            if not session_data:
                return {
                    "success": False,
                    "message": "Invalid or expired session.",
                    "error_code": "INVALID_SESSION"
                }
            
            if session_data["state"] != self.SESSION_STATES["OTP_VERIFICATION"]:
                return {
                    "success": False,
                    "message": "Invalid session state for OTP verification.",
                    "error_code": "INVALID_STATE"
                }
            
            if not session_data.get("otp_auth_key"):
                return {
                    "success": False,
                    "message": "OTP not initiated. Please request OTP first.",
                    "error_code": "OTP_NOT_INITIATED"
                }
            
            # Verify OTP with retry logic for technical errors
            max_retries = 3
            result = None
            
            for attempt in range(max_retries):
                try:
                    result = await self.auth_service.verify_otp(
                        session_data["otp_auth_key"], 
                        otp
                    )
                    
                    # Check if it's a technical error
                    if not result["success"] and self._is_technical_error(result.get("error_code", "")):
                        if attempt == max_retries - 1:
                            return {
                                "success": False,
                                "message": "OTP verification service temporarily unavailable. Please try again.",
                                "error_code": "SERVICE_ERROR",
                                "retry_allowed": True,
                                "technical_error": True
                            }
                        await asyncio.sleep(1)
                        continue
                    
                    # Success or user error - break out of retry loop
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        session_data["technical_retry_count"] += 1
                        await self.auth_service._store_data(session_key, session_data, self.session_timeout_minutes * 60)
        
                        logger.error(f"OTP verification failed after {max_retries} attempts: {e}")
                        return {
                            "success": False,
                            "message": "OTP verification service temporarily unavailable. Please try again.",
                            "error_code": "SERVICE_ERROR",
                            "retry_allowed": True,
                            "technical_error": True
                        }
                    await asyncio.sleep(1)
                    continue
            
            if result and result["success"]:
                # Update session to authenticated state
                session_data["state"] = self.SESSION_STATES["AUTHENTICATED"]
                session_data["authenticated"] = True
                session_data["authenticated_at"] = datetime.now()
                session_data["last_activity"] = datetime.now()
                
                await self.auth_service._store_data(
                    session_key, 
                    session_data, 
                    self.session_timeout_minutes * 60
                )
                
                return {
                    "success": True,
                    "message": "Authentication successful!",
                    "state": self.SESSION_STATES["AUTHENTICATED"],
                    "customer_data": result["customer_data"],
                    "session_id": session_id
                }
            else:
                # Update last activity for user errors
                session_data["last_activity"] = datetime.now()
                await self.auth_service._store_data(
                    session_key, 
                    session_data, 
                    self.session_timeout_minutes * 60
                )
                
                return result if result else {
                    "success": False,
                    "message": "OTP verification failed.",
                    "error_code": "VERIFICATION_FAILED"
                }
                
        except Exception as e:
            logger.error(f"Error verifying OTP: {e}")
            return {
                "success": False,
                "message": "OTP verification service temporarily unavailable. Please try again.",
                "error_code": "SERVICE_ERROR",
                "retry_allowed": True,
                "technical_error": True
            }

    async def resend_otp(self, session_id: str) -> Dict[str, Any]:
        """Enhanced OTP resend with technical error handling"""
        try:
            session_key = f"auth_session:{session_id}"
            session_data = await self.auth_service._retrieve_data(session_key)
            
            if not session_data:
                return {
                    "success": False,
                    "message": "Invalid or expired session.",
                    "error_code": "INVALID_SESSION"
                }
            
            if session_data["state"] != self.SESSION_STATES["OTP_VERIFICATION"]:
                return {
                    "success": False,
                    "message": "Invalid session state for OTP resend.",
                    "error_code": "INVALID_STATE"
                }
            
            if not session_data.get("otp_auth_key"):
                return {
                    "success": False,
                    "message": "OTP not initiated. Please request OTP first.",
                    "error_code": "OTP_NOT_INITIATED"
                }
            
            # Resend OTP with retry logic for technical errors
            max_retries = 3
            result = None
            
            for attempt in range(max_retries):
                try:
                    result = await self.auth_service.resend_otp(session_data["otp_auth_key"])
                    
                    # Check if it's a technical error
                    if not result["success"] and self._is_technical_error(result.get("error_code", "")):
                        if attempt == max_retries - 1:
                            return {
                                "success": False,
                                "message": "OTP resend service temporarily unavailable. Please try again.",
                                "error_code": "SERVICE_ERROR",
                                "retry_allowed": True,
                                "technical_error": True
                            }
                        await asyncio.sleep(1)
                        continue
                    
                    # Success or user error - break out of retry loop
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        session_data["technical_retry_count"] += 1
                        await self.auth_service._store_data(session_key, session_data, self.session_timeout_minutes * 60)
        
                        logger.error(f"OTP resend failed after {max_retries} attempts: {e}")
                        return {
                            "success": False,
                            "message": "OTP resend service temporarily unavailable. Please try again.",
                            "error_code": "SERVICE_ERROR",
                            "retry_allowed": True,
                            "technical_error": True
                        }
                    await asyncio.sleep(1)
                    continue
            
            if result and result["success"]:
                # Update session activity
                session_data["last_activity"] = datetime.now()
                session_data["otp_resent_at"] = datetime.now()
                
                await self.auth_service._store_data(
                    session_key, 
                    session_data, 
                    self.session_timeout_minutes * 60
                )
            
            return result if result else {
                "success": False,
                "message": "OTP resend failed.",
                "error_code": "RESEND_FAILED"
            }
            
        except Exception as e:
            logger.error(f"Error resending OTP: {e}")
            return {
                "success": False,
                "message": "OTP resend service temporarily unavailable. Please try again.",
                "error_code": "SERVICE_ERROR",
                "retry_allowed": True,
                "technical_error": True
            }
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current session status"""
        try:
            session_key = f"auth_session:{session_id}"
            session_data = await self.auth_service._retrieve_data(session_key)
            
            if not session_data:
                return {
                    "success": False,
                    "message": "Session not found or expired.",
                    "error_code": "SESSION_NOT_FOUND"
                }
            
            # Check for expired session
            if self._is_session_expired(session_data):
                await self.auth_service._delete_data(session_key)
                return {
                    "success": False,
                    "message": "Session expired.",
                    "error_code": "SESSION_EXPIRED"
                }
            
            return {
                "success": True,
                "session_id": session_id,
                "state": session_data["state"],
                "contact_verified": session_data["contact_verified"],
                "authenticated": session_data["authenticated"],
                "contact_attempts": session_data["contact_attempts"],
                "max_contact_attempts": self.max_contact_verification_attempts,
                "remaining_contact_attempts": self.max_contact_verification_attempts - session_data["contact_attempts"],
                "preferred_otp_method": session_data.get("preferred_otp_method"),
                "created_at": session_data["created_at"],
                "last_activity": session_data["last_activity"],
                "customer_data": session_data.get("customer_data")
            }
            
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return {
                "success": False,
                "message": "Unable to retrieve session status.",
                "error_code": "SERVICE_ERROR"
            }

    def _mask_email(self, email: str) -> str:
        """Mask email for security"""
        if not email or "@" not in email:
            return email
        
        local, domain = email.split("@", 1)
        if len(local) <= 3:
            masked_local = local[0] + "*" * (len(local) - 1)
        else:
            masked_local = local[:2] + "*" * (len(local) - 2)
        
        return f"{masked_local}@{domain}"

    def _mask_phone(self, phone: str) -> str:
        """Mask phone number for security"""
        if not phone:
            return phone
        
        formatted_phone = self.auth_service.format_phone(phone)
        if len(formatted_phone) >= 4:
            return f"***-***-{formatted_phone[-4:]}"
        return "***-***-****"

    def _is_session_expired(self, session_data: Dict[str, Any]) -> bool:
        """Check if session is expired"""
        try:
            last_activity = session_data["last_activity"]
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity)
            
            timeout_duration = timedelta(minutes=self.session_timeout_minutes)
            return datetime.now() > (last_activity + timeout_duration)
            
        except Exception:
            return True

    def _get_lockout_remaining_time(self, session_data: Dict[str, Any]) -> int:
        """Get remaining lockout time in minutes"""
        try:
            locked_at = session_data.get("locked_at")
            if not locked_at:
                return 0
            
            if isinstance(locked_at, str):
                locked_at = datetime.fromisoformat(locked_at)
            
            lockout_duration = timedelta(minutes=self.contact_lockout_minutes)
            unlock_time = locked_at + lockout_duration
            
            if datetime.now() >= unlock_time:
                return 0
            
            remaining = unlock_time - datetime.now()
            return max(0, int(remaining.total_seconds() / 60))
        except Exception:
            return 0
        
