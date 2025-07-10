#!/usr/bin/env python3
"""
Health API Test Script for Swiss Bank API
This script tests all health check endpoints and basic API availability
"""

import requests
import json
import sys
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import traceback

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
REQUEST_TIMEOUT = 10

class HealthAPITester:
    def __init__(self):
        self.base_url = BASE_URL
        self.timeout = REQUEST_TIMEOUT
        self.test_results = {}
        
    def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"raw_response": response.text}
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_time": response_time,
                "data": response_data,
                "headers": dict(response.headers),
                "url": url
            }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout",
                "error_type": "TIMEOUT",
                "url": url
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error - server may be down",
                "error_type": "CONNECTION_ERROR",
                "url": url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "UNKNOWN_ERROR",
                "url": url
            }
    
    def test_root_endpoint(self) -> bool:
        """Test GET / endpoint"""
        print("\n🏠 ROOT ENDPOINT TEST")
        print("=" * 50)
        
        result = self.make_request("GET", "/")
        self.test_results["root_endpoint"] = result
        
        if not result["success"]:
            print(f"❌ Request failed: {result['error']}")
            return False
        
        status_code = result["status_code"]
        response_time = result["response_time"]
        data = result["data"]
        
        print(f"Status Code: {status_code}")
        print(f"Response Time: {response_time:.3f}s")
        
        # Check response time
        if response_time > 2.0:
            print(f"⚠️  Slow response time: {response_time:.3f}s")
        
        # Validate response
        if status_code == 200:
            print("✅ Root endpoint responding correctly")
            
            # Check response structure
            if isinstance(data, dict) and "message" in data:
                print(f"   Message: {data['message']}")
                return True
            else:
                print("⚠️  Unexpected response format")
                print(f"   Response: {data}")
                return True  # Still working, just unexpected format
        else:
            print(f"❌ Unexpected status code: {status_code}")
            print(f"   Response: {data}")
            return False
    
    def test_favicon_endpoint(self) -> bool:
        """Test GET /favicon.ico endpoint"""
        print("\n🖼️  FAVICON ENDPOINT TEST")
        print("=" * 50)
        
        result = self.make_request("GET", "/favicon.ico")
        self.test_results["favicon_endpoint"] = result
        
        if not result["success"]:
            print(f"❌ Request failed: {result['error']}")
            return False
        
        status_code = result["status_code"]
        response_time = result["response_time"]
        
        print(f"Status Code: {status_code}")
        print(f"Response Time: {response_time:.3f}s")
        
        # Favicon can return 200 (found) or 404 (not found) - both are acceptable
        if status_code in [200, 404]:
            if status_code == 200:
                print("✅ Favicon available")
                # Check content type
                content_type = result["headers"].get("content-type", "")
                if "image" in content_type.lower():
                    print(f"   Content-Type: {content_type}")
                else:
                    print(f"⚠️  Unexpected content-type: {content_type}")
            else:
                print("✅ Favicon endpoint responding (404 is acceptable)")
            return True
        else:
            print(f"❌ Unexpected status code: {status_code}")
            return False
    
    def test_basic_health_endpoint(self) -> bool:
        """Test GET /health endpoint"""
        print("\n🏥 BASIC HEALTH ENDPOINT TEST")
        print("=" * 50)
        
        result = self.make_request("GET", "/health")
        self.test_results["basic_health"] = result
        
        if not result["success"]:
            print(f"❌ Request failed: {result['error']}")
            return False
        
        status_code = result["status_code"]
        response_time = result["response_time"]
        data = result["data"]
        
        print(f"Status Code: {status_code}")
        print(f"Response Time: {response_time:.3f}s")
        
        # Health check should be fast
        if response_time > 1.0:
            print(f"⚠️  Slow health check: {response_time:.3f}s")
        
        if status_code == 200:
            print("✅ Basic health endpoint responding")
            
            # Validate response structure
            if isinstance(data, dict):
                # Check required fields
                required_fields = ["status", "timestamp", "services"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"⚠️  Missing fields: {missing_fields}")
                else:
                    print("✅ All required fields present")
                
                # Check status
                status = data.get("status")
                if status == "healthy":
                    print("✅ Service status: healthy")
                else:
                    print(f"⚠️  Service status: {status}")
                
                # Check timestamp
                timestamp = data.get("timestamp")
                if timestamp:
                    try:
                        parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        print(f"   Timestamp: {parsed_time}")
                    except:
                        print(f"⚠️  Invalid timestamp format: {timestamp}")
                
                # Check services
                services = data.get("services", {})
                if isinstance(services, dict):
                    print("   Services status:")
                    for service_name, service_status in services.items():
                        status_icon = "✅" if service_status in ["connected", "available"] else "❌"
                        print(f"     {status_icon} {service_name}: {service_status}")
                
                return True
            else:
                print(f"❌ Invalid response format: {data}")
                return False
        else:
            print(f"❌ Health check failed with status: {status_code}")
            print(f"   Response: {data}")
            return False
    
    def test_detailed_health_endpoint(self) -> bool:
        """Test GET /health/detailed endpoint"""
        print("\n🔍 DETAILED HEALTH ENDPOINT TEST")
        print("=" * 50)
        
        result = self.make_request("GET", "/health/detailed")
        self.test_results["detailed_health"] = result
        
        if not result["success"]:
            print(f"❌ Request failed: {result['error']}")
            return False
        
        status_code = result["status_code"]
        response_time = result["response_time"]
        data = result["data"]
        
        print(f"Status Code: {status_code}")
        print(f"Response Time: {response_time:.3f}s")
        
        # Detailed health check can be slower
        if response_time > 5.0:
            print(f"⚠️  Very slow detailed health check: {response_time:.3f}s")
        elif response_time > 2.0:
            print(f"⚠️  Slow detailed health check: {response_time:.3f}s")
        
        if status_code == 200:
            print("✅ Detailed health endpoint responding")
            
            # Validate response structure
            if isinstance(data, dict):
                # Check required fields
                required_fields = ["status", "timestamp", "services", "shared_config"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"⚠️  Missing fields: {missing_fields}")
                else:
                    print("✅ All required fields present")
                
                # Check services
                services = data.get("services", {})
                if isinstance(services, dict):
                    print("   Services status:")
                    for service_name, service_status in services.items():
                        status_icon = "✅" if service_status in ["connected", "available"] else "❌"
                        print(f"     {status_icon} {service_name}: {service_status}")
                
                # Check shared config
                shared_config = data.get("shared_config", {})
                if isinstance(shared_config, dict):
                    print("   Shared config status:")
                    for config_name, config_status in shared_config.items():
                        status_icon = "✅" if config_status else "❌"
                        print(f"     {status_icon} {config_name}: {config_status}")
                
                return True
            else:
                print(f"❌ Invalid response format: {data}")
                return False
        else:
            print(f"❌ Detailed health check failed with status: {status_code}")
            print(f"   Response: {data}")
            return False
    
    def test_config_health_endpoint(self) -> bool:
        """Test GET /health/config endpoint"""
        print("\n⚙️  CONFIG HEALTH ENDPOINT TEST")
        print("=" * 50)
        
        result = self.make_request("GET", "/health/config")
        self.test_results["config_health"] = result
        
        if not result["success"]:
            print(f"❌ Request failed: {result['error']}")
            return False
        
        status_code = result["status_code"]
        response_time = result["response_time"]
        data = result["data"]
        
        print(f"Status Code: {status_code}")
        print(f"Response Time: {response_time:.3f}s")
        
        if status_code == 200:
            print("✅ Config health endpoint responding")
            
            # Validate response structure
            if isinstance(data, dict):
                # Check for expected configuration fields
                expected_fields = ["shared_config_status", "redis_client_active", "twilio_client_active", "smtp_config_active"]
                
                print("   Configuration status:")
                for field in expected_fields:
                    if field in data:
                        status = data[field]
                        if isinstance(status, dict):
                            # Handle nested status objects
                            print(f"     {field}:")
                            for sub_key, sub_value in status.items():
                                status_icon = "✅" if sub_value else "❌"
                                print(f"       {status_icon} {sub_key}: {sub_value}")
                        else:
                            status_icon = "✅" if status else "❌"
                            print(f"     {status_icon} {field}: {status}")
                    else:
                        print(f"     ⚠️  Missing field: {field}")
                
                return True
            else:
                print(f"❌ Invalid response format: {data}")
                return False
        else:
            print(f"❌ Config health check failed with status: {status_code}")
            print(f"   Response: {data}")
            return False
    
    def test_api_connectivity(self) -> bool:
        """Test basic API connectivity"""
        print("\n🌐 API CONNECTIVITY TEST")
        print("=" * 50)
        
        # Test if we can reach the API at all
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code in [200, 404, 500]:  # Any response means connectivity
                print("✅ API server is reachable")
                return True
            else:
                print(f"⚠️  API server responding with status: {response.status_code}")
                return True
        except requests.exceptions.Timeout:
            print("❌ API server timeout - may be overloaded")
            return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API server")
            print(f"   Check if server is running at: {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ Connectivity test failed: {e}")
            return False
    
    def test_response_headers(self) -> bool:
        """Test response headers for security and best practices"""
        print("\n📋 RESPONSE HEADERS TEST")
        print("=" * 50)
        
        result = self.make_request("GET", "/health")
        
        if not result["success"]:
            print(f"❌ Cannot test headers: {result['error']}")
            return False
        
        headers = result["headers"]
        
        # Check important headers
        important_headers = {
            "content-type": "Content type specification",
            "server": "Server information",
            "date": "Response timestamp",
            "content-length": "Content length"
        }
        
        print("   Response headers:")
        for header, description in important_headers.items():
            if header in headers:
                value = headers[header]
                print(f"     ✅ {header}: {value}")
            else:
                print(f"     ⚠️  {header}: Not present ({description})")
        
        # Check for security headers
        security_headers = {
            "x-content-type-options": "MIME type sniffing protection",
            "x-frame-options": "Clickjacking protection",
            "x-xss-protection": "XSS protection"
        }
        
        print("   Security headers:")
        for header, description in security_headers.items():
            if header in headers:
                value = headers[header]
                print(f"     ✅ {header}: {value}")
            else:
                print(f"     ⚠️  {header}: Not present ({description})")
        
        return True
    
    def run_all_health_tests(self) -> Dict[str, bool]:
        """Run all health API tests"""
        print("🏥 HEALTH API TEST SUITE")
        print("=" * 60)
        print(f"Target URL: {self.base_url}")
        print(f"Timeout: {self.timeout}s")
        print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_results = {}
        
        # Test API connectivity first
        test_results["api_connectivity"] = self.test_api_connectivity()
        
        if not test_results["api_connectivity"]:
            print("\n❌ API connectivity failed - skipping remaining tests")
            return test_results
        
        # Run all health endpoint tests
        test_results["root_endpoint"] = self.test_root_endpoint()
        test_results["favicon_endpoint"] = self.test_favicon_endpoint()
        test_results["basic_health"] = self.test_basic_health_endpoint()
        test_results["detailed_health"] = self.test_detailed_health_endpoint()
        test_results["config_health"] = self.test_config_health_endpoint()
        test_results["response_headers"] = self.test_response_headers()
        
        return test_results
    
    def generate_test_report(self, test_results: Dict[str, bool]) -> None:
        """Generate test report"""
        print("\n📊 HEALTH API TEST REPORT")
        print("=" * 60)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nTest Results:")
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            formatted_name = test_name.replace("_", " ").title()
            print(f"  {formatted_name}: {status}")
        
        # Overall status
        if passed_tests == total_tests:
            print("\n🎉 ALL HEALTH TESTS PASSED!")
            print("   API health endpoints are functioning correctly")
        elif passed_tests >= total_tests * 0.8:
            print("\n⚠️  MOST TESTS PASSED")
            print("   API is mostly healthy but has some issues")
        else:
            print("\n❌ MULTIPLE TESTS FAILED")
            print("   API health endpoints have significant issues")
        
        # Performance summary
        if hasattr(self, 'test_results'):
            print("\nPerformance Summary:")
            for test_name, result in self.test_results.items():
                if result.get("success") and "response_time" in result:
                    response_time = result["response_time"]
                    if response_time < 0.5:
                        perf_status = "🚀 Fast"
                    elif response_time < 2.0:
                        perf_status = "✅ Good"
                    else:
                        perf_status = "⚠️  Slow"
                    print(f"  {test_name}: {response_time:.3f}s {perf_status}")

def main():
    """Main function to run health API tests"""
    try:
        tester = HealthAPITester()
        
        # Run all tests
        results = tester.run_all_health_tests()
        
        # Generate report
        tester.generate_test_report(results)
        
        
        # Exit with appropriate code
        if all(results.values()):
            print("\n✅ All health tests completed successfully!")
            return 0
        else:
            print("\n❌ Some health tests failed.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())


