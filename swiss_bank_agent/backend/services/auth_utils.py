# backend/services/auth_utils.py
import re
from typing import Dict, Any
from datetime import datetime

class AuthUtils:
    """Shared utilities for authentication system"""
    
    @staticmethod
    def create_error_response(message: str, error_code: str, 
                              retry_allowed: bool = False, 
                              technical_error: bool = False,
                              **kwargs) -> Dict[str, Any]:
        """Create standardized error response"""
        response = {
            "success": False,
            "message": message,
            "error_code": error_code,
            "retry_allowed": retry_allowed,
        }
        
        if technical_error:
            response["technical_error"] = True
            
        response.update(kwargs)
        return response

    @staticmethod
    def create_success_response(message: str, **kwargs) -> Dict[str, Any]:
        """Create standardized success response"""
        response = {
            "success": True,
            "message": message
        }
        response.update(kwargs)
        return response

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        if not email:
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format"""
        if not phone:
            return False
        clean_phone = re.sub(r'\D', '', phone)
        return len(clean_phone) >= 10 and len(clean_phone) <= 15

    @staticmethod
    def format_phone(phone: str) -> str:
        """Format phone number for international use"""
        if not phone:
            return phone
        clean_phone = re.sub(r'\D', '', phone)
        if len(clean_phone) == 10:
            return f"+91{clean_phone}"     # change country code as needed
        elif not clean_phone.startswith('+'):
            return f"+{clean_phone}"
        return clean_phone

    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email for security"""
        if not email or "@" not in email:
            return email
        
        local, domain = email.split("@", 1)
        if len(local) <= 3:
            masked_local = local[0] + "*" * (len(local) - 1)
        else:
            masked_local = local[:2] + "*" * (len(local) - 2)
        
        return f"{masked_local}@{domain}"

    @staticmethod
    def mask_phone(phone: str) -> str:
        """Mask phone number for security"""
        if not phone:
            return phone
        
        formatted_phone = AuthUtils.format_phone(phone)
        if len(formatted_phone) >= 4:
            return f"***-***-{formatted_phone[-4:]}"
        return "***-***-****"

    @staticmethod
    def is_session_expired(session_data: Dict[str, Any], timeout_minutes: int) -> bool:
        """Check if session is expired"""
        try:
            last_activity = session_data["last_activity"]
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity)
            
            from datetime import timedelta
            timeout_duration = timedelta(minutes=timeout_minutes)
            return datetime.now() > (last_activity + timeout_duration)
            
        except Exception:
            return True

    @staticmethod
    def get_lockout_remaining_time(session_data: Dict[str, Any], lockout_minutes: int) -> int:
        """Get remaining lockout time in minutes"""
        try:
            locked_at = session_data.get("locked_at")
            if not locked_at:
                return 0
            
            if isinstance(locked_at, str):
                locked_at = datetime.fromisoformat(locked_at)
            
            from datetime import timedelta
            lockout_duration = timedelta(minutes=lockout_minutes)
            unlock_time = locked_at + lockout_duration
            
            if datetime.now() >= unlock_time:
                return 0
            
            remaining = unlock_time - datetime.now()
            return max(0, int(remaining.total_seconds() / 60))
        except Exception:
            return 0

