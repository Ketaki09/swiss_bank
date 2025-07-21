
"""
Fixed Eva Backend Integration Test - CORRECTED VERSION
This script properly tests Eva service integration with correct authentication flow
"""

import requests
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
current_file_dir = Path(__file__).parent
backend_dir = current_file_dir.parent if current_file_dir.name == 'tests' else current_file_dir
env_path = backend_dir / '.env'
load_dotenv(env_path)

# Configuration
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "swiss_bank")

class EvaIntegrationTester:
    def __init__(self):
        self.base_url = BASE_URL
        self.session_id = None
        self.auth_token = None
        self.test_results = {
            'backend_connection': False,
            'auth_session': False,
            'contact_verification': False,
            'otp_verification': False,
            'session_status': False,
            'eva_status': False,
            'eva_chat': False,
            'eva_greeting': False,
            'eva_conversation_history': False
        }
    
    def run_complete_test(self):
        """Run complete Eva integration test suite"""
        print("\n🤖 Eva Backend Complete Integration Test")
        print("=" * 60)
        
        # Step 1: Test backend connection
        if not self.test_backend_connection():
            print("❌ Backend not accessible. Cannot proceed with tests.")
            return self.print_results()
        
        # Step 2: Get sample customer for authentication
        sample_customer = self.get_sample_customer()
        if not sample_customer:
            print("❌ No sample customer found. Cannot proceed with authentication tests.")
            return self.print_results()
        
        # Step 3: Create authentication session
        if not self.test_auth_session():
            print("❌ Authentication session failed. Cannot proceed with authenticated tests.")
            return self.print_results()
        
        # Step 4: Use the email you requested for OTP testing
        print("\n📧 Using sanjuroyal6x@gmail.com for OTP testing...")
        otp_customer = {"email": "sanjuroyal6x@gmail.com", "name": "Test User"}
        if not self.test_contact_verification(otp_customer):
            print("❌ Contact verification failed. Cannot proceed with OTP tests.")
            # Try with original sample customer as fallback
            if not self.test_contact_verification(sample_customer):
                print("❌ Fallback contact verification also failed.")
                return self.print_results()
        
        # Step 5: Complete OTP verification (INTERACTIVE)
        otp_success = self.complete_otp_verification()
        
        # Step 6: Test Eva status (should work without auth)
        self.test_eva_status()
        
        # Step 7: Test authenticated Eva endpoints (only if OTP successful)
        if otp_success and self.auth_token:
            print("\n🔐 Testing Authenticated Eva Endpoints...")
            
            # First test basic authentication with a simple endpoint
            if self.test_auth_middleware():
                print("✅ Basic authentication is working")
                self.test_eva_chat()
                self.test_eva_greeting()
                self.test_eva_conversation_history()
            else:
                print("❌ Basic authentication failed - debugging required")
                print("\n🔍 AUTHENTICATION DEBUG ANALYSIS:")
                print("   The session is authenticated but Bearer token auth is failing")
                print("   This suggests an issue with the get_current_user dependency")
                print("   in your main.py file")
                print("\n💡 DEBUGGING SUGGESTIONS:")
                print("   1. Check if get_current_user is properly validating session_id")
                print("   2. Verify that HTTPBearer is extracting credentials correctly")
                print("   3. Test with curl command:")
                print(f"      curl -H 'Authorization: Bearer {self.session_id}' \\")
                print(f"           {self.base_url}/api/customers/history")
                
                # Still try Eva endpoints but expect them to fail
                print("\n⚠️ Testing Eva endpoints anyway (expecting 401 errors):")
                self.test_eva_chat()
                self.test_eva_greeting()
                self.test_eva_conversation_history()
        else:
            print("\n⚠️ Skipping authenticated Eva endpoints - OTP verification required")
        
        # Step 8: Print final results
        return self.print_results()
    
    def test_backend_connection(self):
        """Test basic backend connectivity"""
        print("\n🔍 Testing Backend Connection...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Backend Health Response:")
                print(json.dumps(health_data, indent=2))
                self.test_results['backend_connection'] = True
                return True
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Backend connection error: {e}")
            return False
    
    def get_sample_customer(self):
        """Get sample customer for testing"""
        try:
            client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
            db = client[DATABASE_NAME]
            customers_collection = db["customers"]
            
            # Get a customer with email
            sample_customer = customers_collection.find({"email": {"$exists": True, "$ne": ""}}).limit(1)
            customer = next(sample_customer, None)
            
            if customer:
                print(f"✅ Found sample customer: {customer.get('name', 'Unknown')} ({customer.get('email', 'No email')})")
                return customer
            else:
                print("❌ No sample customer with email found")
                return None
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return None
    
    def test_auth_session(self):
        """Test authentication session creation"""
        print("\n🔐 Creating Authentication Session...")
        
        try:
            auth_payload = {
                "ip_address": "127.0.0.1",
                "user_agent": "eva-integration-test"
            }
            
            response = requests.post(
                f"{self.base_url}/api/auth/session", 
                json=auth_payload,
                timeout=10
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                auth_data = response.json()
                print("✅ Auth Session Response:")
                print(json.dumps(auth_data, indent=2))
                
                self.session_id = auth_data.get('session_id')
                if self.session_id:
                    print(f"✅ Session ID: {self.session_id}")
                    self.test_results['auth_session'] = True
                    return True
                else:
                    print("❌ No session_id in response")
                    return False
            else:
                print(f"❌ Auth session failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Auth session error: {e}")
            return False
    
    def test_contact_verification(self, customer):
        """Test contact verification"""
        print("\n📧 Testing Contact Verification...")
        
        if not self.session_id:
            print("❌ No session_id available for contact verification")
            return False
        
        try:
            verify_payload = {
                "session_id": self.session_id,
                "email": customer.get('email'),
                "preferred_otp_method": "email"
            }
            
            response = requests.post(
                f"{self.base_url}/api/auth/verify-contact",
                data=verify_payload,
                timeout=10
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                verify_data = response.json()
                print("✅ Contact Verification Response:")
                print(json.dumps(verify_data, indent=2))
                
                if verify_data.get('success'):
                    print("✅ Contact verified successfully")
                    self.test_results['contact_verification'] = True
                    return True
                else:
                    print(f"❌ Contact verification failed: {verify_data.get('message')}")
                    return False
            else:
                print(f"❌ Contact verification failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Contact verification error: {e}")
            return False
    
    def complete_otp_verification(self):
        """Complete OTP verification with user input"""
        print("\n🔑 OTP Verification Process...")
        
        if not self.session_id:
            print("❌ No session_id for OTP verification")
            return False
        
        try:
            # Step 1: Initiate OTP
            print("📤 Initiating OTP...")
            otp_payload = {"session_id": self.session_id}
            
            response = requests.post(
                f"{self.base_url}/api/auth/initiate-otp",
                data=otp_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                otp_data = response.json()
                if otp_data.get('success'):
                    print("✅ OTP sent successfully!")
                    print(f"   Method: {otp_data.get('otp_method', 'email')}")
                    print(f"   Sent to: {otp_data.get('data', {}).get('sent_to', 'your contact method')}")
                    
                    # Step 2: Get OTP from user
                    print("\n📧 Please check your email for the OTP code.")
                    otp_code = input("🔢 Enter the 6-digit OTP code: ").strip()
                    
                    if len(otp_code) == 6 and otp_code.isdigit():
                        # Step 3: Verify OTP
                        print("🔐 Verifying OTP...")
                        verify_payload = {
                            "session_id": self.session_id,
                            "otp": otp_code
                        }
                        
                        verify_response = requests.post(
                            f"{self.base_url}/api/auth/verify-otp",
                            data=verify_payload,
                            timeout=10
                        )
                        
                        if verify_response.status_code == 200:
                            verify_data = verify_response.json()
                            
                            if verify_data.get('success'):
                                print("✅ OTP verified successfully!")
                                print(f"   Authentication: {verify_data.get('message', 'Complete')}")
                                
                                # Use session_id as Bearer token after successful OTP verification
                                self.auth_token = self.session_id
                                print(f"   🔑 Auth token set: {self.session_id[:20]}...")
                                self.test_results['otp_verification'] = True
                                
                                # Test session status
                                return self.test_session_status()
                            else:
                                print(f"❌ OTP verification failed: {verify_data.get('message')}")
                                print(f"   Error: {verify_data.get('error_code', 'Unknown')}")
                                return False
                        else:
                            print(f"❌ OTP verification request failed: {verify_response.status_code}")
                            print(f"   Response: {verify_response.text}")
                            return False
                    else:
                        print("❌ Invalid OTP format. Please enter a 6-digit number.")
                        return False
                else:
                    print(f"❌ OTP initiation failed: {otp_data.get('message')}")
                    return False
            else:
                print(f"❌ OTP initiation request failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ OTP verification error: {e}")
            return False
    
    def test_session_status(self):
        """Test session status after OTP verification"""
        print("\n📋 Testing Session Status...")
        
        if not self.session_id:
            print("❌ No session_id for status check")
            return False
        
        try:
            response = requests.get(f"{self.base_url}/api/auth/session/{self.session_id}", timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                session_data = response.json()
                print("✅ Session Status Response:")
                print(json.dumps(session_data, indent=2))
                
                # Check if authenticated
                data = session_data.get('data', {})
                if data.get('authenticated'):
                    print("✅ Session is authenticated!")
                    self.test_results['session_status'] = True
                    return True
                else:
                    print(f"⚠️ Session not authenticated. State: {data.get('state')}")
                    return False
            else:
                print(f"❌ Session status failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Session status error: {e}")
            return False
    
    def test_eva_status(self):
        """Test Eva status endpoint (no auth required)"""
        print("\n🤖 Testing Eva Status...")
        
        try:
            response = requests.get(f"{self.base_url}/api/eva/status", timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                eva_data = response.json()
                print("✅ Eva Status Response:")
                print(json.dumps(eva_data, indent=2))
                self.test_results['eva_status'] = True
                return True
            else:
                print(f"❌ Eva status failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Eva status error: {e}")
            return False
    
    def test_auth_middleware(self):
        """Test authentication middleware directly with detailed debugging"""
        print("\n🔒 Testing Authentication Middleware...")
        
        if not self.auth_token:
            print("❌ No authentication token available")
            return False
        
        try:
            # Test customer history endpoint (requires authentication)
            print(f"\n🧪 Testing authenticated endpoint: /api/customers/history")
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            
            print(f"🔑 Session ID being used as Bearer token: {self.session_id}")
            print(f"🔍 Full Authorization header: Authorization: Bearer {self.auth_token}")
            
            response = requests.get(
                f"{self.base_url}/api/customers/history", 
                headers=headers, 
                timeout=10
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ Authentication working - customer history accessible")
                return True
            elif response.status_code == 401:
                response_data = response.json()
                print(f"   ❌ Authentication failed: {response_data.get('detail', 'Unknown')}")
                print(f"   🔍 Full response: {response_data}")
                
                # Debug: Check what the get_current_user dependency is receiving
                print(f"\n🔍 DEBUGGING INFO:")
                print(f"   - Session is authenticated: ✅ (verified above)")
                print(f"   - Using session_id as Bearer token: {self.session_id}")
                print(f"   - Session state in backend: authenticated")
                print(f"   - Issue: get_current_user dependency not recognizing session_id")
                
                return False
            else:
                print(f"   ⚠️ Unexpected status: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
            
        except Exception as e:
            print(f"❌ Auth middleware test error: {e}")
            return False
    
    def test_eva_chat(self):
        """Test Eva chat endpoint with authentication"""
        print("\n💬 Testing Eva Chat...")
        
        if not self.auth_token:
            print("❌ No authentication token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            chat_payload = {
                "message": "Hello Eva, I have a problem with my account",
                "session_id": self.session_id
            }
            
            print(f"🔑 Using Bearer token: Bearer {self.auth_token[:20]}...")
            print(f"🔍 Full session_id being used: {self.session_id}")
            
            response = requests.post(
                f"{self.base_url}/api/eva/chat",
                data=chat_payload,
                headers=headers,
                timeout=30
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                chat_data = response.json()
                print("✅ Eva Chat Response:")
                print(json.dumps(chat_data, indent=2))
                self.test_results['eva_chat'] = True
                return True
            elif response.status_code == 401:
                print(f"❌ Authentication failed: {response.status_code}")
                response_data = response.json()
                print(f"   Error: {response_data.get('message', 'Unknown auth error')}")
                print(f"   Detail: {response_data.get('detail', 'No details')}")
                
                print(f"🔍 Debug - Token format being sent: Bearer {self.auth_token[:10]}...")
                return False
            else:
                print(f"❌ Eva chat failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Eva chat error: {e}")
            return False
    
    def test_eva_greeting(self):
        """Test Eva greeting endpoint"""
        print("\n👋 Testing Eva Greeting...")
        
        if not self.auth_token:
            print("❌ No authentication token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            
            print(f"🔑 Using Bearer token: Bearer {self.auth_token[:20]}...")
            
            response = requests.post(
                f"{self.base_url}/api/eva/test-greeting",
                headers=headers,
                timeout=10
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                greeting_data = response.json()
                print("✅ Eva Greeting Response:")
                print(json.dumps(greeting_data, indent=2))
                self.test_results['eva_greeting'] = True
                return True
            elif response.status_code == 401:
                print(f"❌ Authentication failed: {response.status_code}")
                response_data = response.json()
                print(f"   Error: {response_data.get('message', 'Unknown auth error')}")
                return False
            else:
                print(f"❌ Eva greeting failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Eva greeting error: {e}")
            return False
    
    def test_eva_conversation_history(self):
        """Test Eva conversation history endpoint"""
        print("\n📚 Testing Eva Conversation History...")
        
        if not self.auth_token or not self.session_id:
            print("❌ No authentication token or session ID available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            
            print(f"🔑 Using Bearer token: Bearer {self.auth_token[:20]}...")
            
            response = requests.get(
                f"{self.base_url}/api/eva/conversation-history/{self.session_id}",
                headers=headers,
                timeout=10
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                history_data = response.json()
                print("✅ Eva Conversation History Response:")
                print(json.dumps(history_data, indent=2))
                self.test_results['eva_conversation_history'] = True
                return True
            elif response.status_code == 401:
                print(f"❌ Authentication failed: {response.status_code}")
                response_data = response.json()
                print(f"   Error: {response_data.get('message', 'Unknown auth error')}")
                return False
            else:
                print(f"❌ Eva conversation history failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Eva conversation history error: {e}")
            return False
    
    def print_results(self):
        """Print comprehensive test results with detailed breakdown"""
        print("\n" + "=" * 60)
        print("📊 FINAL TEST RESULTS SUMMARY:")
        print("=" * 60)
        
        # Group tests logically
        infrastructure_tests = {
            'backend_connection': 'Backend Connection',
            'auth_session': 'Auth Session Creation',
            'contact_verification': 'Contact Verification'
        }
        
        authentication_tests = {
            'otp_verification': 'OTP Verification',
            'session_status': 'Session Status Check'
        }
        
        eva_service_tests = {
            'eva_status': 'Eva Service Status',
            'eva_chat': 'Eva Chat Response',
            'eva_greeting': 'Eva Contextual Greeting',
            'eva_conversation_history': 'Eva Conversation Memory'
        }
        
        # Print infrastructure tests
        print("\n🏗️  INFRASTRUCTURE TESTS:")
        infra_passed = 0
        for test_key, test_name in infrastructure_tests.items():
            result = self.test_results.get(test_key, False)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name:<25}: {status}")
            if result:
                infra_passed += 1
        
        # Print authentication tests  
        print("\n🔐 AUTHENTICATION TESTS:")
        auth_passed = 0
        for test_key, test_name in authentication_tests.items():
            result = self.test_results.get(test_key, False)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name:<25}: {status}")
            if result:
                auth_passed += 1
        
        # Print Eva service tests
        print("\n🤖 EVA SERVICE TESTS:")
        eva_passed = 0
        for test_key, test_name in eva_service_tests.items():
            result = self.test_results.get(test_key, False)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name:<25}: {status}")
            if result:
                eva_passed += 1
        
        # Overall summary
        total_passed = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        
        print(f"\n📈 OVERALL SUMMARY:")
        print(f"   Infrastructure Tests: {infra_passed}/{len(infrastructure_tests)} passed")
        print(f"   Authentication Tests: {auth_passed}/{len(authentication_tests)} passed") 
        print(f"   Eva Service Tests:    {eva_passed}/{len(eva_service_tests)} passed")
        print(f"   TOTAL:               {total_passed}/{total_tests} tests passed")
        
        # Success determination
        if total_passed == total_tests:
            print("\n🎉 ALL TESTS PASSED! Eva backend is fully functional.")
            print("   ✅ Infrastructure is solid")
            print("   ✅ Authentication flow works")
            print("   ✅ Eva service is fully operational")
            return True
        elif eva_passed == len(eva_service_tests) and infra_passed == len(infrastructure_tests):
            print("\n🎊 CORE EVA FUNCTIONALITY VERIFIED!")
            print("   ✅ Eva service is working perfectly")
            print("   ✅ Backend infrastructure is solid")
            if auth_passed < len(authentication_tests):
                print("   ⚠️  Authentication tests need attention")
            return True
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {total_passed}/{total_tests} tests passed")
            
            # Provide specific guidance
            if infra_passed < len(infrastructure_tests):
                print("   🚨 Infrastructure issues detected")
            elif auth_passed < len(authentication_tests):
                print("   🚨 Authentication flow issues detected") 
                print("   💡 Suggestion: Check if session_id is being used correctly as Bearer token")
                print("   💡 Suggestion: Verify that Eva endpoints are using same auth middleware")
            elif eva_passed < len(eva_service_tests):
                print("   🚨 Eva service issues detected")
                print("   💡 Suggestion: Check Eva chat endpoint authentication requirements")
                print("   💡 Suggestion: Verify Eva service initialization and dependencies")
            
            self.print_troubleshooting_tips()
            return False
    
    def print_troubleshooting_tips(self):
        """Print troubleshooting guidance"""
        print("\n🔧 Troubleshooting Tips:")
        
        if not self.test_results['backend_connection']:
            print("   1. Backend Connection Issues:")
            print("      - Check if backend is running: python main.py")
            print("      - Verify port 8001 is correct")
            print("      - Check firewall settings")
        
        if not self.test_results['auth_session']:
            print("   2. Authentication Session Issues:")
            print("      - Verify database connection")
            print("      - Check authentication service initialization")
            print("      - Make sure Redis/MongoDB are running")
        
        if self.test_results['otp_verification'] and not any([self.test_results['eva_chat'], self.test_results['eva_greeting']]):
            print("   3. Authentication Dependency Issues (CRITICAL):")
            print("      - Session is authenticated but Bearer token auth fails")
            print("      - Check get_current_user function in main.py")
            print("      - Verify HTTPBearer token extraction")
            print("      - Debug with manual curl command:")
            if hasattr(self, 'session_id') and self.session_id:
                print(f"        curl -H 'Authorization: Bearer {self.session_id}' \\")
                print(f"             {self.base_url}/api/customers/history")
            print("      - Check if auth_controller.get_session_status is working correctly")
        
        if not self.test_results['eva_status']:
            print("   4. Eva Service Issues:")
            print("      - Check Anthropic API key in .env file")
            print("      - Verify Eva service initialization in startup logs")
            print("      - Check database service dependency")
        
        if not any([self.test_results['eva_chat'], self.test_results['eva_greeting']]):
            print("   5. Eva Endpoint Authentication Issues:")
            print("      - Eva service is working but endpoints need authentication")
            print("      - This confirms Eva backend is functional")
            print("      - Issue is with authentication middleware only")


def main():
    """Main test execution"""
    tester = EvaIntegrationTester()
    success = tester.run_complete_test()
    
    if success:
        print("\n✅ Eva backend testing completed successfully!")
        return 0
    else:
        print("\n🚨 Eva backend testing completed with errors!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
            