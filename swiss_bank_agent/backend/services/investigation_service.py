# backend/services/investigation_service.py
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

from services.database_service import DatabaseService
from services.llm_service import LLMService
from services.email_service import EmailService

class InvestigationService:
    def __init__(self):
        self.db_service = DatabaseService()
        self.llm_service = LLMService()
        self.email_service = EmailService()
        
        # Track ongoing investigations
        self.active_investigations = {}

    def start_investigation(self, complaint_id: str, complaint_data: Dict[str, Any]):
        """
        Start asynchronous investigation process
        """
        # Create investigation task
        task = asyncio.create_task(
            self.conduct_investigation(complaint_id, complaint_data)
        )
        
        self.active_investigations[complaint_id] = {
            "task": task,
            "started_at": datetime.now(),
            "status": "in_progress"
        }
        
        print(f"🔍 Investigation started for complaint {complaint_id}")

    async def conduct_investigation(self, complaint_id: str, complaint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct comprehensive investigation and RCA
        """
        try:
            print(f"🔍 Starting investigation for {complaint_id}")
            
            # Step 1: Gather customer history
            customer_history = await self.db_service.get_customer_complaint_history(
                complaint_data["customer_id"]
            )
            
            # Step 2: Find similar complaints
            similar_complaints = await self.db_service.get_similar_complaints(
                complaint_data["theme"], 
                complaint_data["customer_id"]
            )
            
            # Step 3: Generate investigation prompt
            investigation_prompt = await self.llm_service.generate_investigation_prompt(
                complaint_data, customer_history, similar_complaints
            )
            
            # Step 4: Conduct LLM-powered RCA
            if self.llm_service.use_anthropic:
                rca_response = await self.llm_service._call_anthropic(investigation_prompt)
            else:
                rca_response = await self.llm_service._call_openai(investigation_prompt)
            
            # Step 5: Parse investigation results
            investigation_report = self._parse_investigation_response(
                rca_response, complaint_id, complaint_data
            )
            
            # Step 6: Save investigation report
            investigation_id = await self.db_service.save_investigation_report(investigation_report)
            
            # Step 7: Update complaint status
            await self.db_service.update_complaint_status(
                complaint_id, 
                "investigating",
                f"Investigation completed. Report ID: {investigation_id}"
            )
            
            # Step 8: Send notifications
            await self._send_investigation_notifications(
                complaint_id, investigation_report, complaint_data
            )
            
            # Step 9: Clean up
            if complaint_id in self.active_investigations:
                self.active_investigations[complaint_id]["status"] = "completed"
                self.active_investigations[complaint_id]["completed_at"] = datetime.now()
            
            print(f"✅ Investigation completed for {complaint_id}")
            return investigation_report
            
        except Exception as e:
            print(f"❌ Investigation failed for {complaint_id}: {e}")
            
            # Update status to failed
            if complaint_id in self.active_investigations:
                self.active_investigations[complaint_id]["status"] = "failed"
                self.active_investigations[complaint_id]["error"] = str(e)
            
            # Create fallback report
            fallback_report = self._create_fallback_report(complaint_id, complaint_data)
            await self.db_service.save_investigation_report(fallback_report)
            
            return fallback_report

    def _parse_investigation_response(self, response: str, complaint_id: str, complaint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM investigation response
        """
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Validate and structure the report
            report = {
                "complaint_id": complaint_id,
                "root_cause_analysis": parsed_data.get("root_cause_analysis", "Unable to determine root cause"),
                "contributing_factors": parsed_data.get("contributing_factors", []),
                "pattern_analysis": parsed_data.get("pattern_analysis", "No patterns identified"),
                "customer_impact_assessment": parsed_data.get("customer_impact_assessment", "Medium impact"),
                "recommended_actions": parsed_data.get("recommended_actions", ["Follow up with customer"]),
                "preventive_measures": parsed_data.get("preventive_measures", []),
                "priority_level": parsed_data.get("priority_level", complaint_data.get("severity", "medium")),
                "estimated_resolution_time": parsed_data.get("estimated_resolution_time", "2-3 business days"),
                "escalation_recommended": parsed_data.get("escalation_recommended", False),
                "financial_impact_assessment": parsed_data.get("financial_impact_assessment", "No financial impact"),
                "follow_up_required": parsed_data.get("follow_up_required", True),
                "customer_compensation": parsed_data.get("customer_compensation", "None recommended"),
                "investigation_metadata": {
                    "investigation_method": "LLM-powered RCA",
                    "completion_time": datetime.now().isoformat(),
                    "data_sources": ["customer_history", "similar_complaints", "complaint_content"]
                }
            }
            
            return report
            
        except Exception as e:
            print(f"Error parsing investigation response: {e}")
            return self._create_fallback_report(complaint_id, complaint_data)

    def _create_fallback_report(self, complaint_id: str, complaint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fallback investigation report when LLM processing fails
        """
        theme = complaint_data.get("theme", "General")
        severity = complaint_data.get("severity", "medium")
        
        # Basic analysis based on complaint theme and severity
        basic_actions = {
            "Account Access": ["Reset account credentials", "Verify customer identity", "Check system logs"],
            "Transaction Dispute": ["Review transaction details", "Check for fraud indicators", "Contact merchant if needed"],
            "Service Quality": ["Review interaction logs", "Provide additional training", "Follow up with customer"],
            "Fees and Charges": ["Review fee structure", "Check for waiver eligibility", "Explain charges to customer"],
            "Technical Issues": ["Check system status", "Test functionality", "Escalate to IT if needed"],
            "Card Issues": ["Check card status", "Review recent activity", "Issue replacement if needed"]
        }
        
        return {
            "complaint_id": complaint_id,
            "root_cause_analysis": f"Standard {theme.lower()} issue requiring investigation. Analysis based on complaint category and severity level.",
            "contributing_factors": ["Customer experience gap", "Process improvement needed"],
            "pattern_analysis": "Requires manual analysis due to processing limitations",
            "customer_impact_assessment": f"{severity.capitalize()} impact on customer experience and satisfaction",
            "recommended_actions": basic_actions.get(theme, ["Manual review required", "Contact customer", "Follow standard procedures"]),
            "preventive_measures": ["Process review", "Training update", "System monitoring"],
            "priority_level": severity,
            "estimated_resolution_time": self._get_resolution_time_by_severity(severity),
            "escalation_recommended": severity in ["high", "critical"],
            "financial_impact_assessment": "Requires manual assessment",
            "follow_up_required": True,
            "customer_compensation": "To be determined based on manual review",
            "investigation_metadata": {
                "investigation_method": "Fallback processing",
                "completion_time": datetime.now().isoformat(),
                "requires_manual_review": True
            }
        }

    def _get_resolution_time_by_severity(self, severity: str) -> str:
        """Get expected resolution time based on severity"""
        resolution_times = {
            "low": "3-5 business days",
            "medium": "2-3 business days", 
            "high": "1-2 business days",
            "critical": "24 hours"
        }
        return resolution_times.get(severity, "2-3 business days")

    async def _send_investigation_notifications(self, complaint_id: str, report: Dict[str, Any], complaint_data: Dict[str, Any]):
        """
        Send investigation results to relevant parties
        """
        try:
            # Get customer info
            customer = await self.db_service.get_customer(complaint_data["customer_id"])
            
            # 1. Send email to customer
            await self.email_service.send_investigation_update_email(
                customer["email"],
                complaint_id,
                report,
                customer["name"]
            )
            
            # 2. Send to expert team (if escalation needed)
            if report.get("escalation_recommended", False):
                await self.email_service.send_escalation_email(
                    complaint_id,
                    report,
                    complaint_data
                )
            
            # 3. Update front desk dashboard (via database)
            await self.db_service.update_complaint_status(
                complaint_id,
                "investigation_complete",
                f"RCA completed. Priority: {report.get('priority_level', 'medium')}"
            )
            
            print(f"📧 Notifications sent for complaint {complaint_id}")
            
        except Exception as e:
            print(f"Error sending notifications: {e}")

    async def get_investigation_status(self, complaint_id: str) -> Dict[str, Any]:
        """
        Get current investigation status
        """
        if complaint_id in self.active_investigations:
            investigation = self.active_investigations[complaint_id]
            
            return {
                "complaint_id": complaint_id,
                "status": investigation["status"],
                "started_at": investigation["started_at"].isoformat(),
                "completed_at": investigation.get("completed_at", {}).isoformat() if investigation.get("completed_at") else None,
                "duration": (datetime.now() - investigation["started_at"]).total_seconds(),
                "error": investigation.get("error")
            }
        else:
            # Check database for completed investigations
            complaint = await self.db_service.get_complaint(complaint_id)
            if complaint and complaint.get("investigation_id"):
                return {
                    "complaint_id": complaint_id,
                    "status": "completed",
                    "investigation_id": complaint["investigation_id"]
                }
            else:
                return {
                    "complaint_id": complaint_id,
                    "status": "not_found"
                }

    async def get_investigation_report(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get investigation report by ID
        """
        # This would require a method in DatabaseService to fetch investigation reports
        # For now, return placeholder
        return await self.db_service.get_investigation_report(investigation_id)

    def get_active_investigations(self) -> List[Dict[str, Any]]:
        """
        Get list of currently active investigations
        """
        active = []
        for complaint_id, investigation in self.active_investigations.items():
            active.append({
                "complaint_id": complaint_id,
                "status": investigation["status"],
                "started_at": investigation["started_at"].isoformat(),
                "duration": (datetime.now() - investigation["started_at"]).total_seconds()
            })
        
        return active

    def cleanup_completed_investigations(self, max_age_hours: int = 24):
        """
        Clean up completed investigations older than specified hours
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for complaint_id, investigation in self.active_investigations.items():
            if (investigation["status"] in ["completed", "failed"] and 
                investigation.get("completed_at", investigation["started_at"]) < cutoff_time):
                to_remove.append(complaint_id)
        
        for complaint_id in to_remove:
            del self.active_investigations[complaint_id]
        
        print(f"🧹 Cleaned up {len(to_remove)} old investigations")


        