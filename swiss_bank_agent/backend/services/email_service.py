# backend/services/email_service.py
import smtplib
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
        # SMTP configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        
        # Default sender information
        self.from_email = os.getenv("FROM_EMAIL", "noreply@swissbank.com")
        self.from_name = os.getenv("FROM_NAME", "Swiss Bank Customer Service")
        
        # Initialize template engine
        self.template_env = self._setup_templates()

    def _setup_templates(self) -> jinja2.Environment:
        """Setup Jinja2 template environment"""
        template_dir = Path(__file__).parent.parent / "templates" / "emails"
        
        # Create directory if it doesn't exist
        template_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if template files exist, if not create them
        self._ensure_template_files_exist(template_dir)
        
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

    def _ensure_template_files_exist(self, template_dir: Path):
        """Ensure all required template files exist"""
        required_templates = [
            "base_template.html",
            "complaint_confirmation.html", 
            "investigation_update.html",
            "resolution_notification.html",
            "internal_notification.html"
        ]
        
        for template_name in required_templates:
            template_path = template_dir / template_name
            if not template_path.exists():
                print(f"⚠️  Template file missing: {template_name}")
                print(f"   Please create: {template_path}")
                print(f"   Refer to the template structure documentation.")

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
        """Send internal notification to expert team using HTML template"""
        
        template_data = {
            "complaint_id": complaint_id,
            "priority_level": investigation_report.get('priority_level', 'Medium'),
            "root_cause_analysis": investigation_report.get('root_cause_analysis', 'Analysis pending'),
            "recommended_actions": investigation_report.get('recommended_actions', []),
            "estimated_resolution_time": investigation_report.get('estimated_resolution_time', 'TBD'),
            "current_year": datetime.now().year
        }
        
        success = True
        for email in team_emails:
            email_sent = await self._send_template_email(
                to_email=email,
                subject=f"New Investigation Report - Complaint ID: {complaint_id}",
                template_name="internal_notification.html",
                template_data=template_data
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
            
            return await self._send_smtp_email(to_email, subject, html_content, is_html=True)
                
        except jinja2.TemplateNotFound as e:
            print(f"❌ Template not found: {e}")
            print(f"   Make sure {template_name} exists in templates/emails/ directory")
            return False
        except Exception as e:
            print(f"❌ Error sending template email: {e}")
            return False

    async def _send_plain_email(self, to_email: str, subject: str, content: str) -> bool:
        """Send plain text email"""
        try:
            return await self._send_smtp_email(to_email, subject, content, is_html=False)
        except Exception as e:
            print(f"❌ Error sending plain email: {e}")
            return False

    async def send_custom_email(self, to_email: str, subject: str, 
                              template_name: str, template_data: Dict[str, Any]) -> bool:
        """Send email with any custom template"""
        return await self._send_template_email(to_email, subject, template_name, template_data)

    async def _send_smtp_email(self, to_email: str, subject: str, content: str, 
                             is_html: bool = True, attachments: Optional[List[str]] = None) -> bool:
        """Send email via SMTP"""
        try:
            if not self.smtp_username or not self.smtp_password:
                print("❌ SMTP credentials not configured")
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
            print(f"❌ SMTP error: {e}")
            return False

    def get_email_status(self, complaint_id: str) -> Dict[str, Any]:
        """Get email sending status for a complaint"""
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

    def validate_templates(self) -> Dict[str, bool]:
        """Validate that all required templates exist and are readable"""
        template_dir = Path(__file__).parent.parent / "templates" / "emails"
        required_templates = [
            "base_template.html",
            "complaint_confirmation.html", 
            "investigation_update.html",
            "resolution_notification.html",
            "internal_notification.html"
        ]
        
        validation_results = {}
        
        for template_name in required_templates:
            template_path = template_dir / template_name
            try:
                if template_path.exists() and template_path.is_file():
                    # Try to load the template to check for syntax errors
                    self.template_env.get_template(template_name)
                    validation_results[template_name] = True
                    print(f"✅ Template validated: {template_name}")
                else:
                    validation_results[template_name] = False
                    print(f"❌ Template missing: {template_name}")
            except Exception as e:
                validation_results[template_name] = False
                print(f"❌ Template error in {template_name}: {e}")
        
        return validation_results
    



    