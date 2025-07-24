#!/usr/bin/env python3
"""
Anthropic API Key Test Script
Run this to verify your API key is working correctly
"""

import os
import asyncio
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_anthropic_api():
    """Test Anthropic API key with detailed error handling"""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        return False
    
    print(f"🔑 Testing API key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Initialize client
        client = Anthropic(api_key=api_key)
        
        print("📡 Making test request to Anthropic API...")
        
        # Make a simple test request
        message = client.messages.create(
            model="claude-sonnet-4-20250514",  # Use standard model
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Hello! Just testing the API connection. Please respond with 'API connection successful!'"}
            ]
        )
        
        print("✅ API Test Successful!")
        print(f"Response: {message.content[0].text}")
        print(f"Model used: {message.model}")
        print(f"Usage: {message.usage}")
        return True
        
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        
        # Detailed error analysis
        error_str = str(e)
        if "401" in error_str:
            print("🔍 Error Analysis:")
            print("  - 401 Unauthorized usually means:")
            print("    1. Invalid API key")
            print("    2. No credits/billing setup")
            print("    3. API key doesn't have required permissions")
            print("\n💡 Solutions:")
            print("  1. Check your Anthropic Console for billing setup")
            print("  2. Ensure you have at least $5 in credits")
            print("  3. Regenerate your API key if needed")
            
        elif "429" in error_str:
            print("🔍 Rate limit hit - too many requests")
            
        elif "403" in error_str:
            print("🔍 Permission denied - check API key permissions")
            
        return False

async def test_multiple_requests():
    """Test multiple requests to simulate Eva + Triage scenario"""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ API key not found")
        return
    
    client = Anthropic(api_key=api_key)
    
    print("\n🔄 Testing multiple consecutive requests (Eva + Triage simulation)...")
    
    for i in range(3):
        try:
            print(f"  Request {i+1}...")
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
                messages=[
                    {"role": "user", "content": f"Test request #{i+1}"}
                ]
            )
            print(f"  ✅ Request {i+1} successful")
            
            # Small delay between requests
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"  ❌ Request {i+1} failed: {e}")
            break
    
    print("✅ Multiple request test completed")

def check_environment():
    """Check environment configuration"""
    print("\n🔧 Environment Check:")
    
    # Check .env file
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ .env file found")
        
        with open(env_file, 'r') as f:
            content = f.read()
            if "ANTHROPIC_API_KEY" in content:
                print("✅ ANTHROPIC_API_KEY found in .env")
            else:
                print("❌ ANTHROPIC_API_KEY not found in .env")
    else:
        print("❌ .env file not found")
    
    # Check environment variable
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print(f"✅ ANTHROPIC_API_KEY loaded: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("❌ ANTHROPIC_API_KEY not loaded into environment")

if __name__ == "__main__":
    print("🧪 Anthropic API Key Test Script")
    print("=" * 50)
    
    # Step 1: Check environment
    check_environment()
    
    # Step 2: Test basic API call
    print("\n" + "=" * 50)
    success = test_anthropic_api()
    
    # Step 3: Test multiple requests if basic test passes
    if success:
        print("\n" + "=" * 50)
        asyncio.run(test_multiple_requests())
    
    print("\n🏁 Test completed!")
