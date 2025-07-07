# backend/services/auth_controller.py - Refactored version
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging
from .auth_service import AuthService
from .database_service import DatabaseService
from .auth_utils import AuthUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthController:
    """
    Refactored authentication controller with improved separation of concerns,
    standardized error handling, and eliminated code duplication.
    """
    
    def __init__(self):
        self.auth_service = AuthService()
        self.db_service = DatabaseService()
        
        # Configuration for contact verification attempts
        self.max_contact_verification_attempts = 3
        self.contact_lockout_minutes = 3 
        self.session_timeout_minutes = 30
        
        # Technical error retry configuration
        self.max_technical_retries = 3
        self.technical_retry_delay = 1  # seconds
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

    async def _execute_with_technical_retry(self, operation, *args, **kwargs):
        """Execute operation with technical error retry logic"""
        for attempt in range(self.max_technical_retries):
            try:
                result = await operation(*args, **kwargs)
                
                # Check if it's a technical error that should be retried
                if (not result.get("success") and 
                    self._is_technical_error(result.get("error_code", ""))):
                    
                    if attempt == self.max_technical_retries - 1:
                        return AuthUtils.create_error_response(
                            "Service temporarily unavailable. Please try again.",
                            "SERVICE_ERROR",
                            retry_allowed=True,
                            technical_error=True
                        )
                    await asyncio.sleep(self.technical_retry_delay)
                    continue
                
                return result
                
            except Exception as e:
                if attempt == self.max_technical_retries - 1:
                    logger.error(f"Operation failed after {self.max_technical_retries} attempts: {e}")
                    return AuthUtils.create_error_response(
                        "Service temporarily unavailable. Please try again.",
                        "SERVICE_ERROR",
                        retry_allowed=True,
                        technical_error=True
                    )
                await asyncio.sleep(self.technical_retry_delay)
                continue
        
        return AuthUtils.create_error_response(
            "Service temporarily unavailable. Please try again.",
            "SERVICE_ERROR",
            retry_allowed=True,
            technical_error=True
        )

    async def _validate_session(self, session_id: str, 
                               expected_state: Optional[str] = None) -> Tuple[bool, Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Validate session and return (is_valid, session_data, error_response)
        Returns (False, {}, error_response) if invalid
        Returns (True, session_data, None) if valid
        """
        try:
            session_key = f"auth_session:{session_id}"
            session_data = await self.auth_service._retrieve_data(session_key)
            
            if not session_data:
                return False, {}, AuthUtils.create_error_response(
                    "Invalid or expired session. Please start again.",
                    "INVALID_SESSION",
                    action_required="restart"
                )
            
            # Check if session is expired
            if AuthUtils.is_session_expired(session_data, self.session_timeout_minutes):
                await self.auth_service._delete_data(session_key)
                return False, {}, AuthUtils.create_error_response(
                    "Session expired. Please start again.",
                    "SESSION_EXPIRED",
                    action_required="restart"
                )
            
            # Check if session is locked
            if session_data["state"] == self.SESSION_STATES["LOCKED"]:
                lockout_remaining = AuthUtils.get_lockout_remaining_time(session_data, self.contact_lockout_minutes)
                if lockout_remaining > 0:
                    return False, {}, AuthUtils.create_error_response(
                        f"Session locked due to too many failed attempts. Try again in {lockout_remaining} minutes.",
                        "SESSION_LOCKED",
                        retry_after_minutes=lockout_remaining
                    )
                else:
                    # Unlock session
                    session_data["state"] = self.SESSION_STATES["CONTACT_VERIFICATION"]
                    session_data["contact_attempts"] = 0
                    await self.auth_service._store_data(session_key, session_data, self.session_timeout_minutes * 60)
            
            # Check expected state
            if expected_state and session_data["state"] != expected_state:
                return False, {}, AuthUtils.create_error_response(
                    f"Invalid session state. Expected {expected_state}, got {session_data['state']}",
                    "INVALID_STATE"
                )
            
            return True, session_data, None
            
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False, {}, AuthUtils.create_error_response(
                "Session validation failed. Please try again.",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def _update_session_activity(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """Update session last activity timestamp"""
        session_data["last_activity"] = datetime.now()
        session_key = f"auth_session:{session_id}"
        await self.auth_service._store_data(
            session_key, 
            session_data, 
            self.session_timeout_minutes * 60
        )

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
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "contact_verified": False,
                "authenticated": False,
                "customer_data": None,
                "contact_email": None,
                "contact_phone": None,
                "preferred_otp_method": None,
                "otp_auth_key": None
            }
            
            # Store session
            session_key = f"auth_session:{session_id}"
            await self.auth_service._store_data(
                session_key, 
                session_data, 
                self.session_timeout_minutes * 60
            )
            
            return AuthUtils.create_success_response(
                "Please provide your email or phone number.",
                session_id=session_id,
                state=self.SESSION_STATES["CONTACT_VERIFICATION"],
                max_attempts=self.max_contact_verification_attempts,
                expires_in_minutes=self.session_timeout_minutes
            )
            
        except Exception as e:
            logger.error(f"Error creating auth session: {e}")
            return AuthUtils.create_error_response(
                "Failed to create authentication session. Please contact support@swissbank.com",
                "SESSION_CREATION_FAILED"
            )

    async def verify_contact_details(self, session_id: str, email: Optional[str] = None, 
                                   phone: Optional[str] = None, 
                                   preferred_otp_method: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced contact verification with improved error handling"""
        try:
            # Validate session
            is_valid, session_data, error_response = await self._validate_session(
                session_id, 
                self.SESSION_STATES["CONTACT_VERIFICATION"]
            )
            if not is_valid:
                return error_response
            
            # Validate input
            input_validation_result = self._validate_contact_input(
                email, phone, preferred_otp_method
            )
            if not input_validation_result["success"]:
                return input_validation_result
            
            # Auto-determine OTP method if not specified
            if not preferred_otp_method:
                preferred_otp_method = 'email' if email else 'sms'
            
            # Validate contact formats
            format_validation_result = await self._validate_contact_formats(
                session_id, session_data, email, phone
            )
            if not format_validation_result["success"]:
                return format_validation_result
            
            # Check customer existence with retry logic
            customer_check_result = await self._execute_with_technical_retry(
                self.auth_service.check_customer_exists,
                email, phone
            )
            
            if not customer_check_result.get("success"):
                return customer_check_result
            
            customer_exists = customer_check_result["data"]["exists"]
            customer_data = customer_check_result["data"]["customer_data"]
            
            if not customer_exists:
                return await self._handle_customer_not_found(
                    session_id, session_data, email, phone
                )
            
            # Contact verification successful
            session_data.update({
                "contact_verified": True,
                "customer_data": customer_data,
                "contact_email": email,
                "contact_phone": phone,
                "preferred_otp_method": preferred_otp_method,
                "state": self.SESSION_STATES["OTP_VERIFICATION"],
                "contact_verified_at": datetime.now()
            })
            
            await self._update_session_activity(session_id, session_data)
            
            return AuthUtils.create_success_response(
                "Contact details verified successfully. Proceeding to OTP verification.",
                state=self.SESSION_STATES["OTP_VERIFICATION"],
                customer_name=customer_data.get("name", "Valued Customer"),
                otp_method=preferred_otp_method,
                masked_email=AuthUtils.mask_email(email) if email else None,
                masked_phone=AuthUtils.mask_phone(phone) if phone else None
           )
            
        except Exception as e:
            logger.error(f"Error verifying contact details: {e}")
            return AuthUtils.create_error_response(
                "Contact verification service temporarily unavailable. Please try again later.",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    def _validate_contact_input(self, email: Optional[str], phone: Optional[str], 
                               preferred_otp_method: Optional[str]) -> Dict[str, Any]:
        """Validate contact input parameters"""
        if email and phone:
            return AuthUtils.create_error_response(
                "Please provide either email or phone number, not both.",
                "INVALID_INPUT"
            )
        
        if not email and not phone:
            return AuthUtils.create_error_response(
                "Please provide email or phone number.",
                "INVALID_INPUT"
            )
        
        if preferred_otp_method and preferred_otp_method not in ['email', 'sms']:
            return AuthUtils.create_error_response(
                "Invalid OTP method. Choose 'email' or 'sms'.",
                "INVALID_OTP_METHOD"
            )
        
        if preferred_otp_method == 'email' and not email:
            return AuthUtils.create_error_response(
                "Email address required for email OTP.",
                "EMAIL_REQUIRED"
            )
        
        if preferred_otp_method == 'sms' and not phone:
            return AuthUtils.create_error_response(
                "Phone number required for SMS OTP.",
                "PHONE_REQUIRED"
            )
        
        return AuthUtils.create_success_response("Input validation passed")

    async def _validate_contact_formats(self, session_id: str, session_data: Dict[str, Any],
                                   email: Optional[str], phone: Optional[str]) -> Dict[str, Any]:
        """Validate email and phone formats"""
        if email and not AuthUtils.validate_email(email):
            session_data["contact_attempts"] += 1
            await self._update_session_activity(session_id, session_data)
            
            remaining_attempts = self.max_contact_verification_attempts - session_data["contact_attempts"]
            
            return AuthUtils.create_error_response(
                "Invalid email format. Please provide a valid email address.",
                "INVALID_EMAIL_FORMAT",
                current_attempt=session_data["contact_attempts"],
                remaining_attempts=remaining_attempts,
                max_attempts=self.max_contact_verification_attempts
            )
        
        if phone and not AuthUtils.validate_phone(phone):
            session_data["contact_attempts"] += 1
            await self._update_session_activity(session_id, session_data)
            
            remaining_attempts = self.max_contact_verification_attempts - session_data["contact_attempts"]
            
            return AuthUtils.create_error_response(
                "Invalid phone number format. Please provide a valid phone number.",
                "INVALID_PHONE_FORMAT",
                current_attempt=session_data["contact_attempts"],
                remaining_attempts=remaining_attempts,
                max_attempts=self.max_contact_verification_attempts
            )
        
        return AuthUtils.create_success_response("Format validation passed")

    async def _handle_customer_not_found(self, session_id: str, session_data: Dict[str, Any],
                                        email: Optional[str], phone: Optional[str]) -> Dict[str, Any]:
        """Handle customer not found scenario"""
        session_data["contact_attempts"] += 1
        
        # Check if max attempts reached
        if session_data["contact_attempts"] >= self.max_contact_verification_attempts:
            session_data["state"] = self.SESSION_STATES["LOCKED"]
            session_data["locked_at"] = datetime.now()
            
            session_key = f"auth_session:{session_id}"
            await self.auth_service._store_data(
                session_key, 
                session_data, 
                self.contact_lockout_minutes * 60
            )
            
            return AuthUtils.create_error_response(
                f"Maximum contact verification attempts exceeded. Session locked for {self.contact_lockout_minutes} minutes.",
                "MAX_ATTEMPTS_EXCEEDED",
                retry_after_minutes=self.contact_lockout_minutes
            )
        
        await self._update_session_activity(session_id, session_data)
        
        remaining_attempts = self.max_contact_verification_attempts - session_data["contact_attempts"]
        
        if email:
            message = f"No account found with email '{email}'. Please check your email address and try again."
        else:
            message = f"No account found with phone number '{phone}'. Please check your phone number and try again."
        
        message += f" ({remaining_attempts} attempts remaining)"
        
        return AuthUtils.create_error_response(
            message,
            "CUSTOMER_NOT_FOUND",
            current_attempt=session_data["contact_attempts"],
            remaining_attempts=remaining_attempts,
            max_attempts=self.max_contact_verification_attempts,
            suggestions=[
                "Double-check your email address or phone number",
                "Try using the alternate contact method (email vs phone)",
                "Contact customer support if you're unsure about your details"
            ]
        )

    async def initiate_otp_verification(self, session_id: str) -> Dict[str, Any]:
        """Initiate OTP verification with improved error handling"""
        try:
            # Validate session
            is_valid, session_data, error_response = await self._validate_session(
                session_id, 
                self.SESSION_STATES["OTP_VERIFICATION"]
            )
            if not is_valid:
                return error_response
            
            if not session_data["contact_verified"]:
                return AuthUtils.create_error_response(
                    "Contact verification required before OTP generation.",
                    "CONTACT_NOT_VERIFIED"
                )
            
            # Generate and send OTP with retry logic
            otp_result = await self._execute_with_technical_retry(
                self._generate_and_send_otp,
                session_data
            )
            
            if not otp_result.get("success"):
                return otp_result
            
            # Update session with OTP data
            session_data.update({
                "otp_auth_key": otp_result["data"]["auth_key"],
                "otp_initiated_at": datetime.now()
            })
            
            await self._update_session_activity(session_id, session_data)
            
            return AuthUtils.create_success_response(
                otp_result["data"]["message"],
                masked_contact=otp_result["data"]["masked_contact"],
                expires_in=otp_result["data"]["expires_in"],
                otp_method=session_data["preferred_otp_method"],
                state=self.SESSION_STATES["OTP_VERIFICATION"]
            )
            
        except Exception as e:
            logger.error(f"Error initiating OTP verification: {e}")
            return AuthUtils.create_error_response(
                "OTP service temporarily unavailable. Please try again.",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def _generate_and_send_otp(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and send OTP using AuthService core methods"""
        try:
            preferred_method = session_data["preferred_otp_method"]
            email = session_data["contact_email"]
            phone = session_data["contact_phone"]
            
            # Generate OTP
            otp_result = await self.auth_service.generate_otp(
                email if preferred_method == 'email' else phone,
                preferred_method
            )
            
            if not otp_result.get("success"):
                return otp_result
            
            # Send OTP
            if preferred_method == 'email':
                send_result = await self.auth_service.send_otp_email(
                    email, 
                    otp_result["data"]["otp"],
                    session_data["customer_data"].get("name", "Valued Customer")
                )
            else:
                send_result = await self.auth_service.send_otp_sms(
                    phone,
                    otp_result["data"]["otp"]
                )
            
            if not send_result.get("success"):
                return send_result
            
            # Prepare success response
            masked_contact = (
                self.auth_service.mask_email(email) if preferred_method == 'email' 
                else self.auth_service.mask_phone(phone)
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
            logger.error(f"OTP generation/sending error: {e}")
            return AuthUtils.create_error_response(
                "Failed to generate or send OTP",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def verify_otp(self, session_id: str, otp: str) -> Dict[str, Any]:
        """Verify OTP with improved error handling"""
        try:
            # Validate session
            is_valid, session_data, error_response = await self._validate_session(
                session_id, 
                self.SESSION_STATES["OTP_VERIFICATION"]
            )
            if not is_valid:
                return error_response
            
            if not session_data.get("otp_auth_key"):
                return AuthUtils.create_error_response(
                    "OTP not initiated. Please request OTP first.",
                    "OTP_NOT_INITIATED"
                )
            
            # Verify OTP with retry logic
            verify_result = await self._execute_with_technical_retry(
                self.auth_service.verify_otp,
                session_data["otp_auth_key"],
                otp
            )
            
            if not verify_result.get("success"):
                await self._update_session_activity(session_id, session_data)
                return verify_result
            
            # Update session to authenticated state
            session_data.update({
                "state": self.SESSION_STATES["AUTHENTICATED"],
                "authenticated": True,
                "authenticated_at": datetime.now()
            })
            
            await self._update_session_activity(session_id, session_data)
            
            return AuthUtils.create_success_response(
                "Authentication successful!",
                state=self.SESSION_STATES["AUTHENTICATED"],
                customer_data=verify_result["data"]["customer_data"],
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"Error verifying OTP: {e}")
            return AuthUtils.create_error_response(
                "OTP verification service temporarily unavailable. Please try again.",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def resend_otp(self, session_id: str) -> Dict[str, Any]:
        """Resend OTP with improved error handling"""
        try:
            # Validate session
            is_valid, session_data, error_response = await self._validate_session(
                session_id, 
                self.SESSION_STATES["OTP_VERIFICATION"]
            )
            if not is_valid:
                return error_response
            
            if not session_data.get("otp_auth_key"):
                return AuthUtils.create_error_response(
                    "OTP not initiated. Please request OTP first.",
                    "OTP_NOT_INITIATED"
                )
            
            # Resend OTP with retry logic
            resend_result = await self._execute_with_technical_retry(
                self.auth_service.resend_otp,
                session_data["otp_auth_key"]
            )
            
            if resend_result.get("success"):
                session_data["otp_resent_at"] = datetime.now()
                await self._update_session_activity(session_id, session_data)
            
            return resend_result
            
        except Exception as e:
            logger.error(f"Error resending OTP: {e}")
            return AuthUtils.create_error_response(
                "OTP resend service temporarily unavailable. Please try again.",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current session status with improved error handling"""
        try:
            # Validate session (but don't check state)
            is_valid, session_data, error_response = await self._validate_session(session_id)
            if not is_valid:
                return error_response
            
            return AuthUtils.create_success_response(
                "Session status retrieved successfully",
                data={
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
            )
            
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return AuthUtils.create_error_response(
                "Unable to retrieve session status.",
                "SERVICE_ERROR",
                retry_allowed=True,
                technical_error=True
            )

    async def cleanup_expired_sessions(self) -> Dict[str, Any]:
        """Cleanup expired sessions - utility method"""
        try:
            # This would typically be called by a background task
            # Implementation depends on your session storage mechanism
            cleaned_count = await self.auth_service.cleanup_expired_sessions(
                self.session_timeout_minutes
            )
            
            return AuthUtils.create_success_response(
                f"Cleaned up {cleaned_count} expired sessions",
                data={"cleaned_sessions": cleaned_count}
            )
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return AuthUtils.create_error_response(
                "Session cleanup failed",
                "SERVICE_ERROR"
            )

    


