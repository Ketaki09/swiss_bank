
"""
Enhanced SMS OTP Debug Script for Swiss Bank API
This script provides deep debugging for SMS OTP initiation failures using Twilio
"""

import requests
import json
from pymongo import MongoClient
from datetime import datetime
import sys
import time
import os
from dotenv import load_dotenv
import redis
from pathlib import Path
from twilio.rest import Client
from twilio.base.exceptions import TwilioException

# Load environment variables
load_dotenv()

# Configuration - make these configurable
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "swiss_bank")
CUSTOMERS_COLLECTION = "customers"

class SMSOTPServiceDebugger:
    def __init__(self):
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.twilio_client = None
        
    def check_environment_variables(self):
        """Check if all required environment variables are set"""
        print("\n🔧 ENVIRONMENT VARIABLES CHECK")
        print("=" * 50)
        
        env_vars = {
            'TWILIO_ACCOUNT_SID': self.twilio_account_sid,
            'TWILIO_AUTH_TOKEN': self.twilio_auth_token,
            'TWILIO_PHONE_NUMBER': self.twilio_phone_number,
            'REDIS_URL': self.redis_url
        }
        
        missing_vars = []
        for var, value in env_vars.items():
            if not value:
                missing_vars.append(var)
                print(f"❌ {var}: Not set")
            else:
                if var == 'TWILIO_AUTH_TOKEN':
                    # Mask the auth token for security
                    masked_value = f"{value[:8]}{'*' * (len(value) - 8)}"
                    print(f"✅ {var}: {masked_value}")
                else:
                    print(f"✅ {var}: {value}")
        
        if missing_vars:
            print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
            return False
        else:
            print("\n✅ All environment variables are set")
            return True
    
    def test_twilio_connection(self):
        """Test Twilio API connection and configuration"""
        print("\n📱 TWILIO CONNECTION TEST")
        print("=" * 50)
        
        if not self.twilio_account_sid or not self.twilio_auth_token:
            print("❌ Twilio credentials not configured")
            return False
        
        try:
            # Initialize Twilio client
            self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            # Test 1: Account validation
            print("Test 1: Account validation")
            account = self.twilio_client.api.accounts(self.twilio_account_sid).fetch()
            print(f"  ✅ Account SID: {account.sid}")
            print(f"  ✅ Account Status: {account.status}")
            print(f"  ✅ Account Type: {account.type}")
            
            # Test 2: Phone number validation
            print("\nTest 2: Phone number validation")
            if self.twilio_phone_number:
                try:
                    phone_numbers = self.twilio_client.incoming_phone_numbers.list(
                        phone_number=self.twilio_phone_number
                    )
                    if phone_numbers:
                        phone_number = phone_numbers[0]
                        print(f"  ✅ Phone number: {phone_number.phone_number}")
                        print(f"  ✅ SMS capability: {phone_number.capabilities.get('sms', False)}")
                    else:
                        print(f"  ❌ Phone number {self.twilio_phone_number} not found in account")
                        return False
                except Exception as e:
                    print(f"  ❌ Error validating phone number: {e}")
                    return False
            else:
                print("  ❌ TWILIO_PHONE_NUMBER not configured")
                return False
            
            # Test 3: Account balance (optional)
            print("\nTest 3: Account balance")
            try:
                balance = self.twilio_client.api.accounts(self.twilio_account_sid).balance.fetch()
                print(f"  ✅ Account balance: {balance.balance} {balance.currency}")
            except Exception as e:
                print(f"  ⚠️  Could not fetch balance: {e}")
            
            print("\n✅ Twilio connection and configuration successful!")
            return True
            
        except TwilioException as e:
            print(f"❌ Twilio API Error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def test_redis_connection(self):
        """Test Redis connection for OTP storage"""
        print("\n🔴 REDIS CONNECTION TEST")
        print("=" * 50)

        try:
            r = redis.from_url(self.redis_url)
            r.ping()
            print("✅ Redis connection successful!")
            
            # Test basic operations
            test_key = "sms_otp_test_key"
            test_value = "test_value"
            r.setex(test_key, 60, test_value)
            retrieved = r.get(test_key)
            
            if retrieved and retrieved.decode() == test_value:
                r.delete(test_key)
                print("✅ Redis read/write operations successful!")
                return True
            else:
                print("❌ Redis read/write operations failed")
                return False
  
        except Exception as e:
            print(f"❌ Redis error: {e}")
            return False
    
    def check_sms_template(self):
        """Check if SMS template exists and is readable"""
        print("\n📄 SMS TEMPLATE CHECK")
        print("=" * 50)
        
        # Check common template locations
        template_paths = [
            Path("templates/sms/otp_sms.txt"),
            Path("templates/otp_sms.txt"),
            Path("otp_sms_template.txt"),
            Path("sms_template.txt")
        ]
        
        for template_path in template_paths:
            if template_path.exists():
                print(f"Found template: {template_path}")
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for required placeholders
                        required_placeholders = ['{customer_name}', '{otp}', '{expiry_minutes}']
                        missing_placeholders = []
                        
                        for placeholder in required_placeholders:
                            if placeholder not in content:
                                missing_placeholders.append(placeholder)
                        
                        if missing_placeholders:
                            print(f"⚠️  Missing placeholders: {', '.join(missing_placeholders)}")
                            print(f"Template content preview: {content[:100]}...")
                            return False
                        else:
                            print("✅ All required placeholders found")
                            print(f"Template content preview: {content[:100]}...")
                        
                        return True
                except Exception as e:
                    print(f"❌ Error reading template: {e}")
                    continue
        
        print("❌ No SMS template found in any expected location")
        print("Expected locations:")
        for path in template_paths:
            print(f"  - {path}")
        return False
    
    def test_api_service_health(self):
        """Test specific API service health endpoints"""
        print("\n🏥 API SERVICE HEALTH CHECK")
        print("=" * 50)
        
        # Define endpoints to test with their expected behavior
        endpoints = [
            {"path": "/health", "description": "Main health check", "priority": "high"},
            {"path": "/", "description": "Root endpoint", "priority": "low"},
            {"path": "/health/detailed", "description": "Detailed health check", "priority": "medium"}
        ]
        
        health_status = False
        passed_endpoints = 0
        total_endpoints = len(endpoints)
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint['path']}", timeout=5)
                
                # Other endpoints should return 200
                success = response.status_code == 200
                status_symbol = "✅" if success else "⚠️"
                
                priority_indicator = f"[{endpoint['priority'].upper()}]"
                print(f"{status_symbol} {endpoint['path']} {priority_indicator}: {response.status_code} - {endpoint['description']}")
                
                if success:
                    passed_endpoints += 1
                    
                # Main health endpoint determines overall health status
                if endpoint['path'] == "/health" and success:
                    health_status = True
                    
            except requests.exceptions.RequestException as e:
                priority_indicator = f"[{endpoint['priority'].upper()}]"
                print(f"❌ {endpoint['path']} {priority_indicator}: {e}")
        
        print(f"\nEndpoint Summary: {passed_endpoints}/{total_endpoints} endpoints accessible")
        
        return health_status
    
    def test_database_connectivity(self):
        """Test database connectivity and customer data"""
        print("\n💾 DATABASE CONNECTIVITY CHECK")
        print("=" * 50)
        
        try:
            client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
            db = client[DATABASE_NAME]
            customers_collection = db[CUSTOMERS_COLLECTION]
            
            # Test connection
            client.admin.command('ping')
            print("✅ Database connection successful!")
            
            # Test specific customer with phone number
            sample_customer = customers_collection.find_one({
                "phone": {"$exists": True, "$ne": None, "$ne": ""}
            })
            
            if sample_customer:
                print("✅ Sample customer found with phone number")
                phone = sample_customer.get('phone', 'N/A')
                # Mask phone number for privacy
                masked_phone = f"{phone[:3]}***{phone[-3:]}" if len(phone) > 6 else "***"
                print(f"   Phone: {masked_phone}")
                return True, sample_customer
            else:
                print("❌ No customers with phone number found")
                return False, None
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False, None
    
    def test_verify_otp_endpoint(self, session_id):
        """Test OTP verification endpoint with various scenarios"""
        print("\n🔐 OTP VERIFICATION ENDPOINT TEST")
        print("=" * 50)
        
        test_results = {
            'invalid_otp': False,
            'missing_session': False,
            'invalid_session': False,
            'endpoint_availability': False
        }
        
        try:
            # Test 1: Invalid OTP
            print("Test 1: Invalid OTP verification")
            invalid_otp_payload = {
                "session_id": session_id,
                "otp": "000000"  # Invalid OTP
            }
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/auth/verify-otp",
                    data=invalid_otp_payload,
                    timeout=10
                )
                
                print(f"  Status: {response.status_code}")
                if response.status_code == 200:
                    response_data = response.json()
                    # Check if it's an error response
                    if not response_data.get('success', True):
                        error_code = response_data.get('error_code')
                        if error_code in ['INVALID_OTP', 'INVALID_SESSION', 'OTP_EXPIRED', 'MAX_ATTEMPTS_EXCEEDED']:
                            print(f"  ✅ Expected error response: {response_data.get('message', 'No message')}")
                            print(f"     Error code: {error_code}")
                            test_results['invalid_otp'] = True
                        else:
                            print(f"  ⚠️  Unexpected error code: {error_code}")
                    else:
                        print("  ⚠️  Unexpected success response for invalid OTP")
                elif response.status_code == 404:
                    print("  ❌ Endpoint not found!")
                else:
                    print(f"  ⚠️  Unexpected status code: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error testing invalid OTP: {e}")
            
            # Test 2: Missing session ID
            print("\nTest 2: Missing session ID")
            missing_session_payload = {
                "otp": "123456"
            }
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/auth/verify-otp",
                    data=missing_session_payload,
                    timeout=10
                )
                
                print(f"  Status: {response.status_code}")
                if response.status_code == 422:
                    response_data = response.json()
                    print(f"  ✅ Expected validation error: {response_data.get('message', response_data.get('detail', 'No message'))}")
                    test_results['missing_session'] = True
                elif response.status_code == 200:
                    response_data = response.json()
                    if not response_data.get('success', True):
                        print(f"  ✅ Expected error response: {response_data.get('message', 'No message')}")
                        test_results['missing_session'] = True
                    else:
                        print("  ⚠️  Unexpected success response for missing session")
                else:
                    print(f"  ⚠️  Unexpected status code: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error testing missing session: {e}")
            
            # Test 3: Invalid session ID
            print("\nTest 3: Invalid session ID")
            invalid_session_payload = {
                "session_id": "invalid_sms_session_12345",
                "otp": "123456"
            }
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/auth/verify-otp",
                    data=invalid_session_payload,
                    timeout=10
                )
                
                print(f"  Status: {response.status_code}")
                if response.status_code == 200:
                    response_data = response.json()
                    if not response_data.get('success', True):
                        error_code = response_data.get('error_code')
                        if error_code == 'INVALID_SESSION':
                            print(f"  ✅ Expected session error: {response_data.get('message', 'No message')}")
                            print(f"     Error code: {error_code}")
                            test_results['invalid_session'] = True
                        else:
                            print(f"  ⚠️  Unexpected error code: {error_code}")
                    else:
                        print("  ⚠️  Unexpected success response for invalid session")
                else:
                    print(f"  ⚠️  Unexpected status code: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error testing invalid session: {e}")
            
            # Test 4: Endpoint availability
            test_results['endpoint_availability'] = any([
                test_results['invalid_otp'],
                test_results['missing_session'],
                test_results['invalid_session']
            ])
            
            if test_results['endpoint_availability']:
                print("\n✅ Verify OTP endpoint is available and responding correctly")
            else:
                print("\n❌ Verify OTP endpoint may not be working properly")
                
        except Exception as e:
            print(f"❌ Verify OTP test failed: {e}")
            
        return test_results['endpoint_availability']
    
    def test_resend_otp_endpoint(self, session_id):
        """Test OTP resend endpoint with various scenarios"""
        print("\n🔄 OTP RESEND ENDPOINT TEST")
        print("=" * 50)
        
        test_results = {
            'valid_session': False,
            'invalid_session': False,
            'missing_session': False,
            'endpoint_availability': False
        }
        
        try:
            # Test 1: Valid session ID
            print("Test 1: Valid session resend")
            valid_payload = {
                "session_id": session_id
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{BASE_URL}/api/auth/resend-otp",
                    data=valid_payload,
                    timeout=30
                )
                end_time = time.time()
                
                duration = end_time - start_time
                print(f"  Status: {response.status_code} (took {duration:.2f}s)")
                
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('success'):
                        print("  ✅ SMS OTP resent successfully!")
                        print(f"     Message: {response_data.get('message')}")
                        test_results['valid_session'] = True
                    else:
                        print(f"  ⚠️  Resend failed: {response_data.get('message')}")
                        # Still counts as endpoint working if it returns proper error
                        test_results['valid_session'] = True
                elif response.status_code == 404:
                    print("  ❌ Endpoint not found!")
                else:
                    print(f"  ⚠️  Unexpected status: {response.status_code}")
                    print(f"     Response: {response.text[:200]}...")
                    
            except requests.exceptions.Timeout:
                print("  ❌ Request timeout - server may be overloaded")
            except Exception as e:
                print(f"  ❌ Error testing valid session: {e}")
            
            # Test 2: Invalid session ID
            print("\nTest 2: Invalid session resend")
            invalid_payload = {
                "session_id": "invalid_sms_session_id_12345"
            }
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/auth/resend-otp",
                    data=invalid_payload,
                    timeout=10
                )
                
                print(f"  Status: {response.status_code}")
                if response.status_code in [400, 401, 404]:
                    response_data = response.json()
                    print(f"  ✅ Expected session error: {response_data.get('message', 'No message')}")
                    test_results['invalid_session'] = True
                else:
                    print(f"  ⚠️  Unexpected status code: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error testing invalid session: {e}")
            
            # Test 3: Missing session ID
            print("\nTest 3: Missing session ID")
            missing_payload = {}
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/auth/resend-otp",
                    data=missing_payload,
                    timeout=10
                )
                
                print(f"  Status: {response.status_code}")
                if response.status_code in [400, 422]:
                    response_data = response.json()
                    print(f"  ✅ Expected validation error: {response_data.get('message', 'No message')}")
                    test_results['missing_session'] = True
                else:
                    print(f"  ⚠️  Unexpected status code: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error testing missing session: {e}")
            
            # Test 4: Endpoint availability
            test_results['endpoint_availability'] = any([
                test_results['valid_session'],
                test_results['invalid_session'],
                test_results['missing_session']
            ])
            
            if test_results['endpoint_availability']:
                print("\n✅ Resend SMS OTP endpoint is available and responding")
            else:
                print("\n❌ Resend SMS OTP endpoint may not be working properly")
                
        except Exception as e:
            print(f"❌ Resend SMS OTP test failed: {e}")
            
        return test_results['endpoint_availability']
    
    def test_sms_otp_functionality(self, session_id):
        """Test SMS OTP functionality with real session"""
        print("\n🧪 SMS OTP FUNCTIONALITY TEST")
        print("=" * 50)
        
        try:
            # Get session details
            session_response = requests.get(f"{BASE_URL}/api/auth/session/{session_id}")
            
            if session_response.status_code != 200:
                print(f"❌ Cannot get session details: {session_response.status_code}")
                return False
            
            session_data = session_response.json().get('data', {})
            
            # Display session information
            print("Session information:")
            print(f"  State: {session_data.get('state')}")
            print(f"  Contact Verified: {session_data.get('contact_verified')}")
            print(f"  OTP Method: {session_data.get('preferred_otp_method')}")
            customer_phone = session_data.get('customer_data', {}).get('phone')
            if customer_phone:
                masked_phone = f"{customer_phone[:3]}***{customer_phone[-3:]}" if len(customer_phone) > 6 else "***"
                print(f"  Customer Phone: {masked_phone}")
            
            otp_payload = {"session_id": session_id}
            
            try:
                start_time = time.time()
                otp_response = requests.post(
                    f"{BASE_URL}/api/auth/initiate-otp",
                    data=otp_payload,
                    timeout=30
                )
                end_time = time.time()
                
                duration = end_time - start_time
                # Warn about slow response time
                if duration > 5.0:
                    print(f"⚠️  Slow response time ({duration:.2f}s) - SMS delivery may be delayed")
                
                print(f"Response status: {otp_response.status_code}")
                
                if otp_response.status_code == 200:
                    otp_data = otp_response.json()
                    
                    if otp_data.get('success'):
                        print("✅ SMS OTP sent successfully!")
                        print(f"   Message: {otp_data.get('message')}")
                        print(f"   OTP Method: {otp_data.get('otp_method')}")
                        
                        # Test additional OTP endpoints
                        verify_result = self.test_verify_otp_endpoint(session_id)
                        resend_result = self.test_resend_otp_endpoint(session_id)
                        
                        # Overall OTP functionality passes if initiate works and other endpoints are available
                        return verify_result and resend_result
                    else:
                        print("❌ SMS OTP initiation failed!")
                        print(f"   Error: {otp_data.get('message')}")
                        error_code = otp_data.get('error_code')
                        technical_error = otp_data.get('technical_error', False)
                        
                        if error_code == "SERVICE_ERROR" and technical_error:
                            print("\n🔍 SERVICE_ERROR detected - Check Twilio configuration")
                            
                        return False
                else:
                    print(f"❌ HTTP Error: {otp_response.status_code}")
                    print(f"Response: {otp_response.text}")
                    return False
                    
            except requests.exceptions.Timeout:
                print("❌ Request timeout - server may be overloaded or SMS service slow")
                return False
            except requests.exceptions.ConnectionError:
                print("❌ Connection error - server may be down")
                return False
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        
    def run_comprehensive_debug(self):
        """Run comprehensive debugging suite"""
        print("🔍 COMPREHENSIVE SMS OTP DEBUG SUITE")
        print("=" * 60)
        
        debug_results = {
            'environment': False,
            'twilio': False,
            'redis': False,
            'template': False,
            'api_health': False,
            'database': False,
            'sms_otp_test': False
        }
        
        debug_results['environment'] = self.check_environment_variables()
        debug_results['twilio'] = self.test_twilio_connection()
        debug_results['redis'] = self.test_redis_connection()
        debug_results['template'] = self.check_sms_template()
        debug_results['api_health'] = self.test_api_service_health()
        db_success, sample_customer = self.test_database_connectivity()
        debug_results['database'] = db_success
        
        # SMS OTP functionality test with real session
        if sample_customer:
            try:
                # Create session
                auth_payload = {
                    "ip_address": "127.0.0.1",
                    "user_agent": "sms-debug-client"
                }
                
                response = requests.post(f"{BASE_URL}/api/auth/session", json=auth_payload)
                if response.status_code == 200:
                    session_id = response.json().get('session_id')
                    
                    # Verify contact with phone number
                    verify_payload = {
                        "session_id": session_id,
                        "phone": sample_customer.get('phone')
                    }
                    
                    verify_response = requests.post(f"{BASE_URL}/api/auth/verify-contact", data=verify_payload)
                    if verify_response.status_code == 200 and verify_response.json().get('success'):
                        # Test SMS OTP functionality
                        debug_results['sms_otp_test'] = self.test_sms_otp_functionality(session_id)
                        
            except Exception as e:
                print(f"❌ Debug session creation failed: {e}")
        
        # Summary
        print("\n📊 DEBUG SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for result in debug_results.values() if result)
        total = len(debug_results)
        
        for test, result in debug_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test.upper().replace('_', ' ')}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if debug_results['sms_otp_test']:
            print("\n🎉 SMS OTP service is working correctly!")
        else:
            print("\n❌ SMS OTP service has issues. Check the failed tests above.")
            
        # Additional troubleshooting tips
        if not debug_results['twilio']:
            print("\n💡 TROUBLESHOOTING TIPS:")
            print("- Verify Twilio Account SID and Auth Token")
            print("- Check if Twilio phone number is SMS-enabled")
            print("- Ensure sufficient account balance")
            print("- Verify phone number format (+1234567890)")
                
        return debug_results

def main():
    """Main debugging function"""
    debugger = SMSOTPServiceDebugger()
    results = debugger.run_comprehensive_debug()
    
    # Exit with appropriate code
    if results.get('sms_otp_test', False):
        print("\n✅ SMS OTP debugging completed successfully!")
        return 0
    else:
        print("\n❌ SMS OTP debugging found issues that need to be resolved.")
        return 1

if __name__ == "__main__":
    sys.exit(main())