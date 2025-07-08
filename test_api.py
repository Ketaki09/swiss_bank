#!/usr/bin/env python3
"""
MongoDB Customer Testing Script for Swiss Bank API
This script will:
1. Connect to MongoDB and list existing customers
2. Test authentication flow with real customer data (automatically uses 4th customer)
3. Provide a complete end-to-end test

Requirements: pip install pymongo requests
"""

import requests
import json
import pymongo
from pymongo import MongoClient
from datetime import datetime
import sys

# Configuration
BASE_URL = "http://127.0.0.1:8001"
MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "swiss_bank"
CUSTOMERS_COLLECTION = "customers"

def test_api_endpoints():
    """Test basic API endpoints"""
    print("\n🚀 Testing API endpoints...")
    
    # Test health
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test root
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Root endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")

def connect_to_mongodb():
    """Connect to MongoDB and return the customers collection"""
    try:
        client = MongoClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        customers_collection = db[CUSTOMERS_COLLECTION]
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        
        return customers_collection
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return None

def list_customers(collection):
    """List all customers in the database"""
    try:
        customers = list(collection.find({}, {
            "_id": 1,
            "customer_id": 1,
            "email": 1,
            "phone": 1,
            "name": 1,
        }).limit(10))  # Get more customers to ensure we have at least 4
        
        print(f"\n📋 Found {collection.count_documents({})} customers in database")
        print("First 10 customers:")
        print("-" * 80)
        
        for i, customer in enumerate(customers, 1):
            print(f"{i}. ID: {customer.get('customer_id', 'N/A')}")
            print(f"   Name: {customer.get('name', 'N/A')}")
            print(f"   Email: {customer.get('email', 'N/A')}")
            print(f"   Phone: {customer.get('phone', 'N/A')}")
            print()
        
        return customers
    except Exception as e:
        print(f"❌ Error listing customers: {e}")
        return []

def test_auth_with_real_customer(customer_email, customer_phone=None):
    """Test authentication flow with real customer data"""
    print(f"\n🔐 Testing authentication with customer: {customer_email}")
    
    # Step 1: Create authentication session
    print("\n1. Creating authentication session...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/session",
            json={
                "ip_address": "127.0.0.1", 
                "user_agent": "test-client"
            }
        )
        
        if response.status_code != 200:
            print(f"❌ Session creation failed: {response.text}")
            return None
        
        session_data = response.json()
        session_id = session_data.get('session_id')
        print(f"✅ Session created: {session_id}")
        
        # Step 2: Verify contact details with email
        print(f"\n2. Verifying contact with email: {customer_email}")
        verify_response = requests.post(
            f"{BASE_URL}/api/auth/verify-contact",
            data={
                "session_id": session_id,
                "email": customer_email
            }
        )
        
        print(f"Status: {verify_response.status_code}")
        verify_data = verify_response.json()
        print(f"Response: {json.dumps(verify_data, indent=2)}")
        
        if verify_data.get('success'):
            print("✅ Contact verification successful!")
            
            # Step 3: Initiate OTP
            print(f"\n3. Initiating OTP verification...")
            otp_response = requests.post(
                f"{BASE_URL}/api/auth/initiate-otp",
                data={"session_id": session_id}
            )
            
            print(f"Status: {otp_response.status_code}")
            otp_data = otp_response.json()
            print(f"Response: {json.dumps(otp_data, indent=2)}")
            
            if otp_data.get('success'):
                print("✅ OTP initiation successful!")
                print(f"📱 OTP sent to: {otp_data.get('sent_to', 'N/A')}")
                
                # For testing, we'll simulate OTP verification
                # In real scenario, user would enter the OTP they received
                print(f"\n4. Simulating OTP verification...")
                print("💡 In real scenario, user would enter OTP received via email/SMS")
                
                # Step 4: Check final session status
                print(f"\n5. Checking session status...")
                status_response = requests.get(f"{BASE_URL}/api/auth/session/{session_id}")
                status_data = status_response.json()
                print(f"Final session status: {json.dumps(status_data, indent=2)}")
                
                return session_id
            else:
                print("❌ OTP initiation failed")
        else:
            print(f"❌ Contact verification failed: {verify_data.get('message', 'Unknown error')}")
            
            # Try with phone if available
            if customer_phone and customer_phone != "N/A":
                print(f"\n2b. Trying with phone number: {customer_phone}")
                phone_response = requests.post(
                    f"{BASE_URL}/api/auth/verify-contact",
                    data={
                        "session_id": session_id,
                        "phone": customer_phone
                    }
                )
                phone_data = phone_response.json()
                print(f"Phone verification response: {json.dumps(phone_data, indent=2)}")
        
        return session_id
        
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return None

def main():
    """Main testing function"""
    print("🏦 Swiss Bank API Testing Script")
    print("=" * 50)
    
    # Test basic API endpoints
    test_api_endpoints()
    
    # Connect to MongoDB
    collection = connect_to_mongodb()
    if collection is None: 
        print("❌ Cannot continue without MongoDB connection")
        return
    
    # List customers
    customers = list_customers(collection)
    if not customers:
        print("❌ No customers found in database")
        print("💡 Make sure you have customer data in MongoDB")
        return
    
    # Automatically test with 4th customer
    if len(customers) >= 4:
        fourth_customer = customers[3]  # Index 3 for 4th customer
        customer_email = fourth_customer.get('email')
        customer_phone = fourth_customer.get('phone')
        
        print(f"\n🎯 Automatically testing with 4th customer:")
        print(f"   Name: {fourth_customer.get('name', 'N/A')}")
        print(f"   Email: {customer_email}")
        print(f"   Phone: {customer_phone}")
        
        if customer_email and customer_email != "N/A":
            test_auth_with_real_customer(customer_email, customer_phone)
        else:
            print("❌ 4th customer doesn't have a valid email")
            print("💡 Falling back to first customer with valid email...")
            
            # Fallback to first customer with valid email
            for i, customer in enumerate(customers):
                email = customer.get('email')
                if email and email != "N/A":
                    print(f"   Using customer #{i+1} instead: {email}")
                    test_auth_with_real_customer(email, customer.get('phone'))
                    break
            else:
                print("❌ No customers found with valid email addresses")
    else:
        print(f"❌ Only {len(customers)} customers found, need at least 4")
        print("💡 Testing with first available customer...")
        
        # Test with first customer if less than 4 customers
        if customers:
            first_customer = customers[0]
            customer_email = first_customer.get('email')
            customer_phone = first_customer.get('phone')
            
            if customer_email and customer_email != "N/A":
                test_auth_with_real_customer(customer_email, customer_phone)
            else:
                print("❌ First customer doesn't have a valid email")
    
    # Interactive mode - let user choose customer
    print(f"\n🎯 Interactive Testing Mode")
    print("Enter the number of the customer you want to test with (1-10), or 'q' to quit:")
    
    while True:
        try:
            choice = input("Choice: ").strip()
            if choice.lower() == 'q':
                break
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(customers):
                customer = customers[choice_num - 1]
                customer_email = customer.get('email')
                customer_phone = customer.get('phone')
                
                if customer_email and customer_email != "N/A":
                    test_auth_with_real_customer(customer_email, customer_phone)
                else:
                    print(f"❌ Customer {choice_num} doesn't have a valid email")
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(customers)} or 'q'")
                
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()

    