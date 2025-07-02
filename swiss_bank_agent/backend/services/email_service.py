# backend/services/email_service.py
import smtplib
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import jinja2
from pathlib import Path

class EmailService:
    def __init__(self):
        # Email configuration
        self.use_sendgrid = os.getenv("USE_SENDGRID", "true").lower() == "true"
        
        # SendGrid configuration
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.sendgrid_client = None
        if self.sendgrid_api_key and self.use_sendgrid:
            self.sendgrid_client = sendgrid.SendGridAPIClient(api_key=self.sendgrid_api_key)
        
        # SMTP configuration (fallback)
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        
        # Default sender information
        self.from_email = os.getenv("FROM_EMAIL", "noreply@wellsfargo.com")
        self.from_name = os.getenv("FROM_NAME", "Wells Fargo Customer Service")
        
        # Initialize template engine
        self.template_env = self._setup_templates()

    def _setup_templates(self) -> jinja2.Environment:
        """Setup Jinja2 template environment"""
        template_dir = Path(__file__).parent.parent / "templates" / "emails"
        template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default templates if they don't exist
        self._create_default_templates(template_dir)
        
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

    def _create_default_templates(self, template_dir: Path):
        """Create default email templates"""
        templates = {
            "complaint_confirmation.html": """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Complaint Confirmation - Wells Fargo</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #d71e2b; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .footer { background-color: #333; color: white; padding: 15px; text-align: center; font-size: 12px; }
        .complaint-box { background: white; padding: 15px; border-left: 4px solid #d71e2b; margin: 15px 0; }
        .button { background-color: #d71e2b; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Wells Fargo</h1>
            <h2>Complaint Confirmation</h2>
        </div>
        <div class="content">
            <p>Dear {{ customer_name }},</p>
            
            <p>Thank you for bringing your concern to our attention. We have received your complaint and want to assure you that we take all customer feedback seriously.</p>
            
            <div class="complaint-box">
                <h3>Complaint Details</h3>
                <p><strong>Complaint ID:</strong> {{ complaint_id }}</p>
                <p><strong>Theme:</strong> {{ theme }}</p>
                <p><strong>Submitted:</strong> {{ submission_date }}</p>
                <p><strong>Expected Resolution:</strong> {{ estimated_resolution_time }}</p>
            </div>
            
            <h3>What Happens Next?</h3>
            <ul>
                <li>Our investigation team will review your complaint within 24 hours</li>
                <li>We will conduct a thorough analysis of your concern</li>
                <li>You will receive updates as our investigation progresses</li>
                <li>We aim to resolve your issue within {{ estimated_resolution_time }}</li>
            </ul>
            
            <p>You can track the status of your complaint using your complaint ID: <strong>{{ complaint_id }}</strong></p>
            
            <p>If you have any additional information or questions, please don't hesitate to contact us.</p>
            
            <p>Thank you for your patience and for being a valued Wells Fargo customer.</p>
            
            <p>Best regards,<br>
            Wells Fargo Customer Service Team</p>
        </div>
        <div class="footer">
            <p>&copy; {{ current_year }} Wells Fargo Bank, N.A. All rights reserved.</p>
            <p>This is an automated message. Please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
            """,
            
            "investigation_update.html": """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Complaint Investigation Update - Wells Fargo</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #d71e2b; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .footer { background-color: #333; color: white; padding: 15px; text-align: center; font-size: 12px; }
        .update-box { background: white; padding: 15px; border-left: 4px solid #2e8b57; margin: 15px 0; }
        .status-badge { background-color: #2e8b57; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Wells Fargo</h1>
            <h2>Investigation Update</h2>
        </div>
        <div class="content">
            <p>Dear {{ customer_name }},</p>
            
            <p>We have an update regarding your complaint investigation.</p>
            
            <div class="update-box">
                <h3>Complaint Status Update</h3>
                <p><strong>Complaint ID:</strong> {{ complaint_id }}</p>
                <p><strong>Current Status:</strong> <span class="status-badge">{{ status }}</span></p>
                <p><strong>Last Updated:</strong> {{ update_date }}</p>
            </div>
            
            <h3>Investigation Progress</h3>
            <p>{{ investigation_summary }}</p>
            
            {% if next_steps %}
            <h3>Next Steps</h3>
            <ul>
            {% for step in next_steps %}
                <li>{{ step }}</li>
            {% endfor %}
            </ul>
            {% endif %}
            
            {% if estimated_completion %}
            <p><strong>Estimated Completion:</strong> {{ estimated_completion }}</p>
            {% endif %}
            
            <p>We appreciate your patience as we work to resolve your concern thoroughly and fairly.</p>
            
            <p>Best regards,<br>
            Wells Fargo Investigation Team</p>
        </div>
        <div class="footer">
            <p>&copy; {{ current_year }} Wells Fargo Bank, N.A. All rights reserved.</p>
            <p>This is an automated message. Please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
            """,
            
            "resolution_notification.html": """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Complaint Resolution - Wells Fargo</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #d71e2b; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .footer { background-color: #333; color: white; padding: 15px; text-align: center; font-size: 12px; }
        .resolution-box { background: white; padding: 15px; border-left: 4px solid #2e8b57; margin: 15px 0; }
        .button { background-color: #d71e2b; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Wells Fargo</h1>
            <h2>Complaint Resolution</h2>
        </div>
        <div class="content">
            <p>Dear {{ customer_name }},</p>
            
            <p>We are pleased to inform you that we have completed our investigation of your complaint and have taken action to resolve your concern.</p>
            
            <div class="resolution-box">
                <h3>Resolution Summary</h3>
                <p><strong>Complaint ID:</strong> {{ complaint_id }}</p>
                <p><strong>Resolution Date:</strong> {{ resolution_date }}</p>
                <p><strong>Total Resolution Time:</strong> {{ resolution_duration }} days</p>
            </div>
            
            <h3>What We Found</h3>
            <p>{{ investigation_findings }}</p>
            
            <h3>Actions Taken</h3>
            <p>{{ resolution_actions }}</p>
            
            {% if compensation %}
            <h3>Compensation</h3>
            <p>{{ compensation }}</p>
            {% endif %}
            
            <h3>Feedback</h3>
            <p>Your feedback is important to us. Please let us know about your experience with our complaint resolution process.</p>
            
            <a href="{{ feedback_link }}" class="button">Provide Feedback</a>
            
            <p>Thank you for bringing this matter to our attention and for giving us the opportunity to make things right.</p>
            
            <p>Best regards,<br>
            Wells Fargo Customer Resolution Team</p>
        </div>
        <div class="footer">
            <p>&copy; {{ current_year }} Wells Fargo Bank, N.A. All rights reserved.</p>
            <p>This is an automated message. Please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
            """
        }
        
        for filename, content in templates.items():
            template_path = template_dir / filename
            if not template_path.exists():
                template_path.write_text(content)

    async def send_confirmation_email(self, customer_email: str, complaint_id: str, theme: str, 
                                    customer_name: str = "Valued Customer", 
                                    estimated_resolution_time: str = "2-3 business days") -> bool:
        """Send complaint confirmation email to customer"""
        
        template_data = {
            "customer_name": customer_name,
            "complaint_id": complaint_id,
            "theme": theme,
            "submission_date": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            "estimated_resolution_time": estimated_resolution_time,
            "current_year": datetime.now().year
        }
        
        return await self._send_template_email(
            to_email=customer_email,
            subject=f"Complaint Confirmation - ID: {complaint_id}",
            template_name="complaint_confirmation.html",
            template_data=template_data
        )

    async def send_investigation_update(self, customer_email: str, complaint_id: str, 
                                      status: str, investigation_summary: str,
                                      customer_name: str = "Valued Customer",
                                      next_steps: Optional[List[str]] = None,
                                      estimated_completion: Optional[str] = None) -> bool:
        """Send investigation update email to customer"""
        
        template_data = {
            "customer_name": customer_name,
            "complaint_id": complaint_id,
            "status": status.title(),
            "update_date": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            "investigation_summary": investigation_summary,
            "next_steps": next_steps or [],
            "estimated_completion": estimated_completion,
            "current_year": datetime.now().year
        }
        
        return await self._send_template_email(
            to_email=customer_email,
            subject=f"Investigation Update - Complaint ID: {complaint_id}",
            template_name="investigation_update.html",
            template_data=template_data
        )

    async def send_resolution_notification(self, customer_email: str, complaint_id: str,
                                         investigation_findings: str, resolution_actions: str,
                                         customer_name: str = "Valued Customer",
                                         compensation: Optional[str] = None,
                                         resolution_duration: int = 0,
                                         feedback_link: str = "#") -> bool:
        """Send resolution notification email to customer"""
        
        template_data = {
            "customer_name": customer_name,
            "complaint_id": complaint_id,
            "resolution_date": datetime.now().strftime("%B %d, %Y"),
            "resolution_duration": resolution_duration,
            "investigation_findings": investigation_findings,
            "resolution_actions": resolution_actions,
            "compensation": compensation,
            "feedback_link": feedback_link,
            "current_year": datetime.now().year
        }
        
        return await self._send_template_email(
            to_email=customer_email,
            subject=f"Complaint Resolved - ID: {complaint_id}",
            template_name="resolution_notification.html",
            template_data=template_data
        )

    async def send_internal_notification(self, team_emails: List[str], complaint_id: str,
                                       investigation_report: Dict[str, Any]) -> bool:
        """Send internal notification to expert team"""
        
        subject = f"New Investigation Report - Complaint ID: {complaint_id}"
        
        # Create plain text content for internal team
        content = f"""
New Investigation Report Available

Complaint ID: {complaint_id}
Priority Level: {investigation_report.get('priority_level', 'Medium')}
Root Cause: {investigation_report.get('root_cause_analysis', 'Analysis pending')}

Recommended Actions:
{chr(10).join(['- ' + action for action in investigation_report.get('recommended_actions', [])])}

Estimated Resolution Time: {investigation_report.get('estimated_resolution_time', 'TBD')}

Please review the full report in the dashboard.

This is an automated notification from the Complaint Management System.
        """
        
        success = True
        for email in team_emails:
            email_sent = await self._send_plain_email(
                to_email=email,
                subject=subject,
                content=content
            )
            if not email_sent:
                success = False
        
        return success

    async def _send_template_email(self, to_email: str, subject: str, 
                                 template_name: str, template_data: Dict[str, Any]) -> bool:
        """Send email using HTML template"""
        try:
            # Render template
            template = self.template_env.get_template(template_name)
            html_content = template.render(**template_data)
            
            if self.use_sendgrid and self.sendgrid_client:
                return await self._send_sendgrid_email(to_email, subject, html_content)
            else:
                return await self._send_smtp_email(to_email, subject, html_content, is_html=True)
                
        except Exception as e:
            print(f"Error sending template email: {e}")
            return False

    async def _send_plain_email(self, to_email: str, subject: str, content: str) -> bool:
        """Send plain text email"""
        try:
            if self.use_sendgrid and self.sendgrid_client:
                return await self._send_sendgrid_email(to_email, subject, content, is_html=False)
            else:
                return await self._send_smtp_email(to_email, subject, content, is_html=False)
                
        except Exception as e:
            print(f"Error sending plain email: {e}")
            return False

    async def _send_sendgrid_email(self, to_email: str, subject: str, content: str, is_html: bool = True) -> bool:
        """Send email via SendGrid"""
        try:
            from_email = Email(self.from_email, self.from_name)
            to_email_obj = To(to_email)
            
            if is_html:
                content_obj = Content("text/html", content)
            else:
                content_obj = Content("text/plain", content)
            
            mail = Mail(from_email, to_email_obj, subject, content_obj)
            
            response = self.sendgrid_client.send(mail)
            
            if response.status_code in [200, 201, 202]:
                print(f"✅ Email sent successfully to {to_email}")
                return True
            else:
                print(f"❌ Failed to send email. Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"SendGrid error: {e}")
            return False

    async def _send_smtp_email(self, to_email: str, subject: str, content: str, 
                             is_html: bool = True, attachments: Optional[List[str]] = None) -> bool:
        """Send email via SMTP"""
        try:
            if not self.smtp_username or not self.smtp_password:
                print("SMTP credentials not configured")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add content
            if is_html:
                msg.attach(MIMEText(content, 'html'))
            else:
                msg.attach(MIMEText(content, 'plain'))
            
            # Add attachments if provided
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {os.path.basename(file_path)}'
                            )
                            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            print(f"SMTP error: {e}")
            return False

    def get_email_status(self, complaint_id: str) -> Dict[str, Any]:
        """Get email sending status for a complaint (placeholder for future tracking)"""
        # This could be expanded to track email delivery status
        return {
            "complaint_id": complaint_id,
            "emails_sent": [],
            "last_email_date": None,
            "pending_notifications": []
        }

    async def send_bulk_emails(self, recipients: List[Dict[str, str]], subject: str, 
                             template_name: str, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send bulk emails to multiple recipients"""
        results = {
            "total": len(recipients),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for recipient in recipients:
            try:
                # Personalize template data for each recipient
                personalized_data = template_data.copy()
                personalized_data.update(recipient)
                
                success = await self._send_template_email(
                    to_email=recipient['email'],
                    subject=subject,
                    template_name=template_name,
                    template_data=personalized_data
                )
                
                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to send to {recipient['email']}")
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Error sending to {recipient.get('email', 'unknown')}: {str(e)}")
        
        return results
    
