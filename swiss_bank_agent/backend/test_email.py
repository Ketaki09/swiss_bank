#!/usr/bin/env python3
"""
Email Test Script - Test SMTP configuration and email sending
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import asyncio
import sys
import textwrap

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class EmailTester:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("SMTP_USERNAME")
        self.email_password = os.getenv("SMTP_PASSWORD")
        
        # Template path (adjust as needed)
        self.template_path = Path(__file__).parent / "templates" / "emails"
        
    def check_environment(self):
        """Check if all required environment variables are set"""
        print("🔍 Checking environment variables...")
        
        missing_vars = []
        if not self.email_user:
            missing_vars.append("SMTP_USERNAME")
        if not self.email_password:
            missing_vars.append("SMTP_PASSWORD")
            
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            return False
            
        print(f"✅ SMTP Server: {self.smtp_server}:{self.smtp_port}")
        print(f"✅ Email User: {self.email_user}")
        print(f"✅ Password: {'*' * len(self.email_password) if self.email_password else 'Not set'}")
        return True
    
    def test_smtp_connection(self):
        """Test SMTP connection"""
        print(f"\n🔗 Testing SMTP connection to {self.smtp_server}:{self.smtp_port}...")
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                print("✅ SMTP connection successful!")
                return True
        except Exception as e:
            print(f"❌ SMTP connection failed: {e}")
            return False
    
    def check_template_file(self):
        """Check if email template file exists"""
        print(f"\n📄 Checking email template...")
        
        template_file = self.template_path / "otp_email.html"
        print(f"Template path: {template_file}")
        
        if template_file.exists():
            print("✅ Template file found!")
            
            # Read and display template
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"Template length: {len(content)} characters")
                    print("Template preview:")
                    print("-" * 50)
                    print(content[:300] + "..." if len(content) > 300 else content)
                    print("-" * 50)
                    return True
            except Exception as e:
                print(f"❌ Error reading template: {e}")
                return False
        else:
            print("❌ Template file not found!")
            print("Creating fallback template...")
            
            # Create directory if it doesn't exist
            self.template_path.mkdir(parents=True, exist_ok=True)
            
            # Create fallback template
            fallback_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Swiss Bank - Authentication Code</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 600px; margin: 0 auto; }
        .header { background-color: #1a472a; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; border: 1px solid #ddd; }
        .otp { font-size: 24px; font-weight: bold; color: #1a472a; text-align: center; padding: 20px; }
        .footer { background-color: #f8f9fa; padding: 15px; font-size: 12px; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Swiss Bank</h1>
            <h2>Authentication Code</h2>
        </div>
        <div class="content">
            <p>Dear {customer_name},</p>
            <p>Your verification code is:</p>
            <div class="otp">{otp}</div>
            <p>This code will expire in <strong>{expiry_minutes} minutes</strong>.</p>
            <p><strong>Important:</strong> Do not share this code with anyone. Swiss Bank will never ask for this code over the phone or email.</p>
        </div>
        <div class="footer">
            <p>Best regards,<br>Swiss Bank Security Team</p>
            <p>If you did not request this code, please contact us immediately.</p>
        </div>
    </div>
</body>
</html>"""
            
            try:
                with open(template_file, 'w', encoding='utf-8') as f:
                    f.write(fallback_template)
                print(f"✅ Fallback template created at: {template_file}")
                return True
            except Exception as e:
                print(f"❌ Error creating fallback template: {e}")
                return False
    
    def send_test_email(self, test_email=None):
        """Send a test OTP email"""
        if not test_email:
            test_email = self.email_user  # Send to self for testing
            
        print(f"\n📧 Sending test OTP email to: {test_email}")
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = test_email
            msg['Subject'] = "Swiss Bank - Test Authentication Code"
            
            # Load template
            template_file = self.template_path / "otp_email.html"
            if template_file.exists():
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template_content = f.read()
                except Exception as e:
                    print(f"⚠️  Error reading template file: {e}")
                    print("Using fallback template...")
                    template_content = self.get_fallback_template()
            else:
                template_content = self.get_fallback_template()
            
            # Render template with test data using string replacement instead of .format()
            try:
                html_body = template_content.replace("{customer_name}", "Test Customer")
                html_body = html_body.replace("{otp}", "123456")
                html_body = html_body.replace("{expiry_minutes}", "5")
            except Exception as e:
                print(f"⚠️  Error formatting template: {e}")
                print("Using simple fallback template...")
                html_body = self.get_simple_fallback_template().format(
                    customer_name="Test Customer",
                    otp="123456",
                    expiry_minutes=5
                )
            
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            print("✅ Test email sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send test email: {e}")
            return False
    
    def get_fallback_template(self):
        """Get fallback template if file doesn't exist"""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Swiss Bank - Authentication Code</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background-color: #1a472a;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .content {{
            padding: 30px;
        }}
        .otp {{
            font-size: 32px;
            font-weight: bold;
            color: #1a472a;
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 20px 0;
            letter-spacing: 3px;
        }}
        .footer {{
            background-color: #f8f9fa;
            padding: 20px;
            font-size: 14px;
            color: #6c757d;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Swiss Bank</h1>
            <h2>Authentication Code</h2>
        </div>
        <div class="content">
            <p>Dear {customer_name},</p>
            <p>Your verification code is:</p>
            <div class="otp">{otp}</div>
            <p>This code will expire in <strong>{expiry_minutes} minutes</strong>.</p>
            <p><strong>Important:</strong> Do not share this code with anyone. Swiss Bank will never ask for this code over the phone or email.</p>
        </div>
        <div class="footer">
            <p><strong>Best regards,</strong><br>Swiss Bank Security Team</p>
            <p>If you did not request this code, please contact us immediately.</p>
        </div>
    </div>
</body>
</html>"""
    
    def get_simple_fallback_template(self):
        """Get simple fallback template as last resort"""
        return """
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <h2 style="color: #1a472a;">Swiss Bank - Authentication Code</h2>
            <p>Dear {customer_name},</p>
            <p>Your verification code is:</p>
            <div style="font-size: 24px; font-weight: bold; color: #1a472a; text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin: 20px 0;">
                {otp}
            </div>
            <p>This code will expire in <strong>{expiry_minutes} minutes</strong>.</p>
            <p><strong>Important:</strong> Do not share this code with anyone.</p>
            <p>Best regards,<br>Swiss Bank Security Team</p>
        </body>
        </html>
        """
    
    def run_full_test(self, test_email=None):
        """Run complete email test suite"""
        print("🏦 Swiss Bank Email Test Suite")
        print("=" * 50)
        
        tests_passed = 0
        total_tests = 4
        
        # Test 1: Environment variables
        if self.check_environment():
            tests_passed += 1
        
        # Test 2: SMTP connection
        if self.test_smtp_connection():
            tests_passed += 1
        
        # Test 3: Template file
        if self.check_template_file():
            tests_passed += 1
        
        # Test 4: Send test email
        if self.send_test_email(test_email):
            tests_passed += 1
        
        if tests_passed == total_tests:
            print("🎉 All tests passed! Email system is working correctly.")
            return True
        else:
            print("❌ Some tests failed. Please check the issues above.")
            return False

def main():
    """Main function"""
    tester = EmailTester()
    
    # Get test email from command line argument or use sender's email
    test_email = sys.argv[1] if len(sys.argv) > 1 else None

    
    success = tester.run_full_test(test_email)
    
    if success:
        print("\n✅ Email system is ready for OTP sending!")
    else:
        print("\n❌ Email system needs configuration. Please fix the issues above.")
    
    return success

if __name__ == "__main__":
    main()


    