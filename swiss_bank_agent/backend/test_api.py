#!/usr/bin/env python3
"""
Enhanced OTP Debug Script for Swiss Bank API
This script provides deep debugging for OTP initiation failures
"""

import requests
import json
from pymongo import MongoClient
from datetime import datetime
import sys
import time
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import redis
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration - make these configurable
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "swiss_bank")
CUSTOMERS_COLLECTION = "customers"

class OTPServiceDebugger:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("SMTP_USERNAME")
        self.email_password = os.getenv("SMTP_PASSWORD")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
    def check_environment_variables(self):
        
        env_vars = {
            'SMTP_SERVER': self.smtp_server,
            'SMTP_PORT': self.smtp_port,
            'SMTP_USERNAME': self.email_user,
            'SMTP_PASSWORD': self.email_password,
            'REDIS_URL': self.redis_url
        }
        
        missing_vars = []
        for var, value in env_vars.items():
            if not value:
                missing_vars.append(var)
                print(f"❌ {var}: Not set")
        
        if missing_vars:
            print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
            return False
        else:
            print("\n✅ All environment variables are set")
            return True
    
    def test_smtp_connection(self):
        
        if not self.email_user or not self.email_password:
            print("❌ SMTP credentials not configured")
            return False
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                print("✅ SMTP connection successful!")
                return True
            
        except Exception as e:
            print(f"❌ SMTP Error: {e}")
            return False
    
    def test_redis_connection(self):

        try:
            r = redis.from_url(self.redis_url)
            r.ping()
            print("✅ Redis connection successful!")
            
            # Test basic operations
            test_key = "otp_test_key"
            test_value = "test_value"
            r.setex(test_key, 60, test_value)
            retrieved = r.get(test_key)
            
            if retrieved and retrieved.decode() == test_value:
                r.delete(test_key)
                return True
            else:
                print("❌ Redis read/write operations failed")
                return False
  
        except Exception as e:
            print(f"❌ Redis error: {e}")
            return False
    
    def check_email_template(self):
        """Check if email template exists and is readable"""
        print("\n📄 EMAIL TEMPLATE CHECK")
        
        template_path = Path("templates/emails/otp_email.html")
                
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
                    return False
                else:
                    print("✅ All required placeholders found")
                
                return True
        except Exception as e:
            print(f"❌ Error reading template: {e}")
            return False
        
        print("❌ No email template found in any expected location")
        return False
    
    def test_api_service_health(self):
        """Test specific API service health endpoints"""
        print("\n🏥 API SERVICE HEALTH CHECK")
        print("=" * 50)
        
        endpoint = "/health"
        health_status = False
        
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            status_symbol = "✅" if response.status_code == 200 else "⚠️"
            print(f"{status_symbol} {endpoint}: {response.status_code}")
            
            if response.status_code == 200:
                health_status = True
                        
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint}: {e}")
        
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
            
            # Test specific customer
            sample_customer = customers_collection.find().sort("_id", 1).skip(3).limit(1).next()
            
            if sample_customer:
                print("✅ Sample customer found with email")
                return True, sample_customer
            else:
                print("❌ No customers with email found")
                return False, None
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False, None
    
    def test_otp_functionality(self, session_id):
        """Test OTP functionality with real session"""
        print("\n🧪 OTP FUNCTIONALITY TEST")
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
            print(f"  Customer Email: {session_data.get('customer_data', {}).get('email')}")
            
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
                # Warn about slow response
                if duration > 2.0:
                    print(f"⚠️  Slow response time ({duration:.2f}s) - consider investigating server performance")
                
                print(f"Response status: {otp_response.status_code}")
                
                if otp_response.status_code == 200:
                    otp_data = otp_response.json()
                    
                    if otp_data.get('success'):
                        print("✅ OTP sent successfully!")
                        print(f"   Message: {otp_data.get('message')}")
                        print(f"   OTP Method: {otp_data.get('otp_method')}")
                        return True
                    else:
                        print("❌ OTP initiation failed!")
                        print(f"   Error: {otp_data.get('message')}")
                        error_code = otp_data.get('error_code')
                        technical_error = otp_data.get('technical_error', False)
                        
                        if error_code == "SERVICE_ERROR" and technical_error:
                            print("\n🔍 SERVICE_ERROR detected")
                            
                        return False
                else:
                    print(f"❌ HTTP Error: {otp_response.status_code}")
                    print(f"Response: {otp_response.text}")
                    return False
                    
            except requests.exceptions.Timeout:
                print("❌ Request timeout - server may be overloaded")
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
        print("🔍 COMPREHENSIVE OTP DEBUG SUITE")
        print("=" * 60)
        
        debug_results = {
            'environment': False,
            'smtp': False,
            'redis': False,
            'template': False,
            'api_health': False,
            'database': False,
            'otp_test': False
        }
        
        debug_results['environment'] = self.check_environment_variables()
        debug_results['smtp'] = self.test_smtp_connection()
        debug_results['redis'] = self.test_redis_connection()
        debug_results['template'] = self.check_email_template()
        debug_results['api_health'] = self.test_api_service_health()
        db_success, sample_customer = self.test_database_connectivity()
        debug_results['database'] = db_success
        
        # OTP functionality test with real session
        if sample_customer:
            try:
                # Create session
                auth_payload = {
                    "ip_address": "127.0.0.1",
                    "user_agent": "debug-client"
                }
                
                response = requests.post(f"{BASE_URL}/api/auth/session", json=auth_payload)
                if response.status_code == 200:
                    session_id = response.json().get('session_id')
                    
                    # Verify contact
                    verify_payload = {
                        "session_id": session_id,
                        "email": sample_customer.get('email')
                    }
                    
                    verify_response = requests.post(f"{BASE_URL}/api/auth/verify-contact", data=verify_payload)
                    if verify_response.status_code == 200 and verify_response.json().get('success'):
                        # Test OTP functionality
                        debug_results['otp_test'] = self.test_otp_functionality(session_id)
                        
            except Exception as e:
                print(f"❌ Debug session creation failed: {e}")
        
        # Summary
        print("\n📊 DEBUG SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for result in debug_results.values() if result)
        total = len(debug_results)
        
        for test, result in debug_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test.upper()}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if debug_results['otp_test']:
            print("\n🎉 OTP service is working correctly!")
        else:
            print("\n❌ OTP service has issues. Check the failed tests above.")
                
        return debug_results

def main():
    """Main debugging function"""
    debugger = OTPServiceDebugger()
    results = debugger.run_comprehensive_debug()
    
    # Exit with appropriate code
    if results.get('otp_test', False):
        print("\n✅ Debugging completed successfully!")
        return 0
    else:
        print("\n❌ Debugging found issues that need to be resolved.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

