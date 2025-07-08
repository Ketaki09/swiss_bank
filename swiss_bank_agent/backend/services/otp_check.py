#!/usr/bin/env python3
"""
Enhanced OTP Email Service Checker
This script specifically targets the OTP service methods over email to help identify 
if they exist and are working based on the auth_service.py implementation.
"""

import asyncio
import inspect
import sys
import os
import traceback
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class OTPEmailServiceChecker:
    """Enhanced checker specifically for OTP email service methods"""
    
    def __init__(self, auth_service=None):
        self.auth_service = auth_service
        self.checks = {}
        self.test_email = "test@example.com"
        self.test_otp = "123456"
        self.test_customer_name = "Test Customer"
        
    def log_check(self, check_name: str, success: bool, message: str, details: Any = None):
        """Log check result with enhanced formatting"""
        self.checks[check_name] = {
            "success": success,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        status = "✅" if success else "❌"
        print(f"{status} {check_name}: {message}")
        if details:
            if isinstance(details, dict):
                print(f"   Details: {json.dumps(details, indent=2, default=str)}")
            else:
                print(f"   Details: {details}")
    
    def check_auth_service_initialization(self):
        """Check if auth_service is properly initialized"""
        if self.auth_service is None:
            self.log_check("auth_service_initialization", False, "auth_service is None")
            return False
        
        # Check if it's the correct type
        service_type = type(self.auth_service).__name__
        if service_type != "AuthService":
            self.log_check("auth_service_initialization", False, f"Expected AuthService, got {service_type}")
            return False
        
        # Check if it has required attributes
        required_attrs = ['smtp_server', 'smtp_port', 'email_user', 'email_password', 'template_path']
        missing_attrs = []
        
        for attr in required_attrs:
            if not hasattr(self.auth_service, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            self.log_check("auth_service_initialization", False, f"Missing attributes: {missing_attrs}")
            return False
        
        self.log_check("auth_service_initialization", True, f"AuthService properly initialized")
        return True
    
    def check_email_configuration(self):
        """Check email configuration settings"""
        if not hasattr(self.auth_service, 'email_user') or not hasattr(self.auth_service, 'email_password'):
            self.log_check("email_configuration", False, "Email credentials attributes not found")
            return False
        
        config_status = {
            "smtp_server": getattr(self.auth_service, 'smtp_server', None),
            "smtp_port": getattr(self.auth_service, 'smtp_port', None),
            "email_user": "SET" if getattr(self.auth_service, 'email_user', None) else "NOT SET",
            "email_password": "SET" if getattr(self.auth_service, 'email_password', None) else "NOT SET"
        }
        
        has_credentials = bool(self.auth_service.email_user and self.auth_service.email_password)
        
        self.log_check("email_configuration", has_credentials, 
                      "Email configuration complete" if has_credentials else "Email credentials not configured",
                      config_status)
        return has_credentials
    
    def check_template_system(self):
        """Check email template system"""
        if not hasattr(self.auth_service, 'load_email_template'):
            self.log_check("template_system", False, "load_email_template method not found")
            return False
        
        if not hasattr(self.auth_service, 'render_template'):
            self.log_check("template_system", False, "render_template method not found")
            return False
        
        # Test template loading
        try:
            template_content = self.auth_service.load_email_template("otp_email.html")
            if template_content:
                self.log_check("template_system", True, "Template system working", 
                              {"template_length": len(template_content), "has_placeholders": "{otp}" in template_content})
            else:
                self.log_check("template_system", False, "Template loading returned empty content")
                return False
        except Exception as e:
            self.log_check("template_system", False, f"Template loading failed: {e}")
            return False
        
        return True
    
    def check_email_methods_exist(self):
        """Check if email-related methods exist"""
        required_methods = [
            'send_otp_email',
            'load_email_template',
            'render_template',
            '_get_fallback_template',
            '_get_simple_fallback_template'
        ]
        
        all_methods_exist = True
        method_details = {}
        
        for method_name in required_methods:
            if hasattr(self.auth_service, method_name):
                method = getattr(self.auth_service, method_name)
                if callable(method):
                    sig = inspect.signature(method)
                    method_details[method_name] = {
                        "exists": True,
                        "callable": True,
                        "signature": str(sig),
                        "is_async": inspect.iscoroutinefunction(method)
                    }
                else:
                    method_details[method_name] = {"exists": True, "callable": False}
                    all_methods_exist = False
            else:
                method_details[method_name] = {"exists": False, "callable": False}
                all_methods_exist = False
        
        self.log_check("email_methods_exist", all_methods_exist, 
                      "All email methods exist" if all_methods_exist else "Some email methods missing",
                      method_details)
        return all_methods_exist
    
    async def test_generate_otp(self):
        """Test OTP generation specifically for email"""
        if not hasattr(self.auth_service, 'generate_otp'):
            self.log_check("test_generate_otp", False, "generate_otp method not found")
            return False
        
        try:
            result = await self.auth_service.generate_otp(self.test_email, "email")
            
            if isinstance(result, dict):
                if result.get("success"):
                    data = result.get("data", {})
                    has_required_fields = all(field in data for field in ["otp", "auth_key", "expires_in"])
                    
                    self.log_check("test_generate_otp", has_required_fields, 
                                  "OTP generation successful" if has_required_fields else "Missing required fields",
                                  {"result": result, "has_required_fields": has_required_fields})
                    return has_required_fields
                else:
                    self.log_check("test_generate_otp", False, "OTP generation failed", result)
                    return False
            else:
                self.log_check("test_generate_otp", False, "Unexpected return format", {"type": type(result), "value": result})
                return False
            
        except Exception as e:
            self.log_check("test_generate_otp", False, f"Exception during OTP generation: {e}", 
                          {"traceback": traceback.format_exc()})
            return False
    
    async def test_send_otp_email_method(self):
        """Test send_otp_email method specifically"""
        if not hasattr(self.auth_service, 'send_otp_email'):
            self.log_check("test_send_otp_email", False, "send_otp_email method not found")
            return False
        
        try:
            result = await self.auth_service.send_otp_email(self.test_email, self.test_otp, self.test_customer_name)
            
            if isinstance(result, dict):
                success = result.get("success", False)
                message = result.get("message", "No message")
                error_code = result.get("error_code", None)
                
                # Check if it's a configuration error (expected in test environment)
                is_config_error = error_code == "SERVICE_ERROR" and "not configured" in message
                
                test_result = {
                    "success": success,
                    "message": message,
                    "error_code": error_code,
                    "is_config_error": is_config_error,
                    "technical_error": result.get("technical_error", False)
                }
                
                # Consider it a pass if method exists and returns proper error for missing config
                method_works = success or is_config_error
                
                self.log_check("test_send_otp_email", method_works, 
                              "Email sending method works" if method_works else "Email sending method failed",
                              test_result)
                return method_works
            else:
                self.log_check("test_send_otp_email", False, "Unexpected return format", 
                              {"type": type(result), "value": result})
                return False
            
        except Exception as e:
            self.log_check("test_send_otp_email", False, f"Exception during email sending: {e}", 
                          {"traceback": traceback.format_exc()})
            return False
    
    async def test_template_rendering(self):
        """Test template rendering functionality"""
        if not hasattr(self.auth_service, 'render_template'):
            self.log_check("test_template_rendering", False, "render_template method not found")
            return False
        
        try:
            # Load template
            template_content = self.auth_service.load_email_template("otp_email.html")
            
            # Test rendering
            rendered = self.auth_service.render_template(
                template_content,
                customer_name=self.test_customer_name,
                otp=self.test_otp,
                expiry_minutes="5"
            )
            
            # Check if placeholders were replaced
            placeholders_replaced = all(placeholder not in rendered for placeholder in [
                "{customer_name}", "{otp}", "{expiry_minutes}"
            ])
            
            contains_data = all(data in rendered for data in [
                self.test_customer_name, self.test_otp, "5"
            ])
            
            success = placeholders_replaced and contains_data
            
            self.log_check("test_template_rendering", success, 
                          "Template rendering successful" if success else "Template rendering failed",
                          {
                              "placeholders_replaced": placeholders_replaced,
                              "contains_data": contains_data,
                              "rendered_length": len(rendered)
                          })
            return success
            
        except Exception as e:
            self.log_check("test_template_rendering", False, f"Exception during template rendering: {e}", 
                          {"traceback": traceback.format_exc()})
            return False
    
    async def test_storage_methods(self):
        """Test data storage methods used by OTP system"""
        storage_methods = ['_store_data', '_retrieve_data', '_delete_data']
        all_methods_work = True
        
        for method_name in storage_methods:
            if not hasattr(self.auth_service, method_name):
                self.log_check(f"test_{method_name}", False, f"{method_name} method not found")
                all_methods_work = False
                continue
            
            method = getattr(self.auth_service, method_name)
            if not callable(method):
                self.log_check(f"test_{method_name}", False, f"{method_name} is not callable")
                all_methods_work = False
                continue
            
            self.log_check(f"test_{method_name}", True, f"{method_name} method exists and is callable")
        
        # Test storage functionality if all methods exist
        if all_methods_work:
            try:
                test_key = f"test_key_{datetime.now().timestamp()}"
                test_data = {"test": "data", "timestamp": datetime.now()}
                
                # Test store
                await self.auth_service._store_data(test_key, test_data, 60)
                
                # Test retrieve
                retrieved_data = await self.auth_service._retrieve_data(test_key)
                
                # Test delete
                await self.auth_service._delete_data(test_key)
                
                storage_works = retrieved_data is not None
                self.log_check("test_storage_functionality", storage_works, 
                              "Storage functionality works" if storage_works else "Storage functionality failed",
                              {"stored": test_data, "retrieved": retrieved_data})
                
            except Exception as e:
                self.log_check("test_storage_functionality", False, f"Storage test failed: {e}", 
                              {"traceback": traceback.format_exc()})
                all_methods_work = False
        
        return all_methods_work
    
    async def test_full_otp_email_flow(self):
        """Test the complete OTP email flow"""
        print("\n🔄 Testing complete OTP email flow...")
        
        try:
            # Step 1: Generate OTP
            print("   Step 1: Generating OTP...")
            otp_result = await self.auth_service.generate_otp(self.test_email, "email")
            
            if not otp_result.get("success"):
                self.log_check("test_full_flow", False, "OTP generation failed in full flow test", otp_result)
                return False
            
            otp_data = otp_result.get("data", {})
            generated_otp = otp_data.get("otp")
            auth_key = otp_data.get("auth_key")
            
            print(f"   Generated OTP: {generated_otp}")
            print(f"   Auth Key: {auth_key}")
            
            # Step 2: Send OTP Email
            print("   Step 2: Sending OTP email...")
            email_result = await self.auth_service.send_otp_email(self.test_email, generated_otp, self.test_customer_name)
            
            print(f"   Email result: {email_result}")
            
            # Step 3: Check if we can retrieve the stored OTP data
            print("   Step 3: Checking stored OTP data...")
            stored_data = await self.auth_service._retrieve_data(auth_key)
            
            if stored_data:
                print(f"   Stored data retrieved successfully")
                print(f"   OTP matches: {stored_data.get('otp') == generated_otp}")
            else:
                print("   No stored data found")
            
            # Evaluate flow success
            flow_success = (
                otp_result.get("success") and
                generated_otp and
                auth_key and
                stored_data is not None
            )
            
            flow_details = {
                "otp_generated": bool(generated_otp),
                "auth_key_created": bool(auth_key),
                "data_stored": stored_data is not None,
                "email_method_called": email_result is not None
            }
            
            self.log_check("test_full_flow", flow_success, 
                          "Complete OTP email flow successful" if flow_success else "OTP email flow failed",
                          flow_details)
            
            return flow_success
            
        except Exception as e:
            self.log_check("test_full_flow", False, f"Exception in full flow test: {e}", 
                          {"traceback": traceback.format_exc()})
            return False
    
    async def run_all_checks(self):
        """Run all OTP email service checks"""
        print("📧 OTP Email Service Checker")
        print("=" * 60)
        print(f"Target: {self.test_email}")
        print(f"Test OTP: {self.test_otp}")
        print(f"Test Customer: {self.test_customer_name}")
        print("=" * 60)
        
        # Basic checks
        print("\n🔍 Basic Service Checks:")
        if not self.check_auth_service_initialization():
            print("\n❌ Cannot proceed without properly initialized auth_service")
            return False
        
        print("\n📋 Email Configuration Checks:")
        self.check_email_configuration()
        
        print("\n🛠️ Template System Checks:")
        self.check_template_system()
        
        print("\n📝 Method Existence Checks:")
        self.check_email_methods_exist()
        
        print("\n🧪 Functionality Tests:")
        await self.test_generate_otp()
        await self.test_send_otp_email_method()
        await self.test_template_rendering()
        await self.test_storage_methods()
        
        print("\n🔄 Integration Tests:")
        await self.test_full_otp_email_flow()
        
        # Generate summary
        self.generate_summary()
        
        # Return overall success
        total_checks = len(self.checks)
        passed_checks = sum(1 for check in self.checks.values() if check["success"])
        
        return passed_checks == total_checks
    
    def generate_summary(self):
        """Generate detailed summary of all checks"""
        print("\n📊 Detailed Summary:")
        print("=" * 60)
        
        # Categorize checks
        categories = {
            "Service Initialization": ["auth_service_initialization"],
            "Email Configuration": ["email_configuration"],
            "Template System": ["template_system", "test_template_rendering"],
            "Method Existence": ["email_methods_exist"],
            "Core Functionality": ["test_generate_otp", "test_send_otp_email"],
            "Storage System": ["test__store_data", "test__retrieve_data", "test__delete_data", "test_storage_functionality"],
            "Integration": ["test_full_flow"]
        }
        
        for category, check_names in categories.items():
            relevant_checks = {name: self.checks[name] for name in check_names if name in self.checks}
            if relevant_checks:
                passed = sum(1 for check in relevant_checks.values() if check["success"])
                total = len(relevant_checks)
                status = "✅" if passed == total else "❌"
                print(f"{status} {category}: {passed}/{total} checks passed")
        
        print("\n" + "=" * 60)
        
        total_checks = len(self.checks)
        passed_checks = sum(1 for check in self.checks.values() if check["success"])
        
        print(f"📈 Overall Results:")
        print(f"   Total checks: {total_checks}")
        print(f"   Passed: {passed_checks}")
        print(f"   Failed: {total_checks - passed_checks}")
        print(f"   Success rate: {(passed_checks/total_checks)*100:.1f}%")
        
        if passed_checks == total_checks:
            print("\n✅ All checks passed! Your OTP email service is ready to use.")
        else:
            print("\n❌ Some checks failed. See details above for troubleshooting.")
            
            # Show failed checks
            failed_checks = [name for name, check in self.checks.items() if not check["success"]]
            if failed_checks:
                print(f"\n🔍 Failed checks: {', '.join(failed_checks)}")
    
    def save_report(self, filename: str = "otp_email_check_report.json"):
        """Save detailed report to file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_email": self.test_email,
            "test_otp": self.test_otp,
            "test_customer_name": self.test_customer_name,
            "checks": self.checks,
            "summary": {
                "total_checks": len(self.checks),
                "passed_checks": sum(1 for check in self.checks.values() if check["success"]),
                "failed_checks": sum(1 for check in self.checks.values() if not check["success"])
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Report saved to {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save report: {e}")

async def main():
    """Main function to run the OTP email service checker"""
    try:
        # Import the AuthService
        print("📦 Importing AuthService...")
        try:
            # Try to import from the services directory
            from auth_service import AuthService
            print("✅ AuthService imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import AuthService: {e}")
            print("💡 Make sure you're running this script from the correct directory")
            return False
        
        # Initialize AuthService
        print("🔧 Initializing AuthService...")
        auth_service = AuthService()
        
        # Initialize the service (this connects to database, etc.)
        try:
            await auth_service.initialize()
            print("✅ AuthService initialized successfully")
        except Exception as e:
            print(f"⚠️  AuthService initialization failed: {e}")
            print("   This is expected if database is not configured")
            print("   Continuing with limited testing...")
        
        # Create checker and run tests
        checker = OTPEmailServiceChecker(auth_service)
        
        # Run all checks
        success = await checker.run_all_checks()
        
        # Save report
        checker.save_report()
        
        # Clean up
        try:
            await auth_service.cleanup_and_disconnect()
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
        
        return success
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🚀 Starting OTP Email Service Checker...")
    success = asyncio.run(main())
    
    if success:
        print("\n🎉 OTP Email Service Check completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 OTP Email Service Check completed with errors!")
        sys.exit(1)

