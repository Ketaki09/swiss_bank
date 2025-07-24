# backend/services/triage_agent_service.py - PROPER IMPLEMENTATION
import anthropic
import json
import uuid
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

@dataclass
class TriageResult:
    """Structured triage result with three sections"""
    original_complaint: Dict[str, Any]
    triage_analysis: Dict[str, Any] 
    routing_package: Dict[str, Any]
    is_new_theme: bool = False
    new_theme_alert: Optional[Dict[str, Any]] = None

class NewThemeDetector:
    """Detects if complaint represents a new theme requiring escalation"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.similarity_threshold = 0.6
        
    def detect_new_theme(self, complaint_text: str, classification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Determines if complaint represents a new theme"""
        
        # Check 1: Low confidence across all categories
        max_confidence = max(classification_results.get('confidence_scores', {}).values())
        
        if max_confidence < self.confidence_threshold:
            return self._flag_as_new_theme(
                reason="low_confidence",
                max_confidence=max_confidence,
                complaint_text=complaint_text
            )
        
        # Check 2: Novel terminology detection
        if self._contains_novel_terminology(complaint_text):
            return self._flag_as_new_theme(
                reason="novel_terminology", 
                complaint_text=complaint_text
            )
        
        # Check 3: Cross-category confusion
        if self._cross_category_confusion(classification_results):
            return self._flag_as_new_theme(
                reason="cross_category_confusion",
                classification_results=classification_results,
                complaint_text=complaint_text
            )
        
        # Not a new theme
        return {
            "is_new_theme": False,
            "classification": classification_results,
            "confidence": max_confidence
        }
    
    def _flag_as_new_theme(self, reason: str, **kwargs) -> Dict[str, Any]:
        """Flag complaint as new theme and prepare orchestrator alert"""
        
        alert_package = {
            "is_new_theme": True,
            "detection_reason": reason,
            "complaint_text": kwargs.get('complaint_text'),
            "timestamp": datetime.now().isoformat(),
            "requires_immediate_escalation": True,
            "suggested_actions": self._get_escalation_actions(reason),
            "metadata": {
                "confidence_scores": kwargs.get('classification_results', {}),
                "max_confidence": kwargs.get('max_confidence'),
                "detection_details": kwargs
            }
        }
        
        return alert_package
    
    def _contains_novel_terminology(self, complaint_text: str) -> bool:
        """Detect novel banking terminology"""
        novel_indicators = [
            'new app feature', 'latest update', 'cryptocurrency', 'NFT', 'blockchain',
            'digital wallet', 'biometric', 'voice banking', 'AI assistant',
            'quantum', 'metaverse', 'virtual reality', 'augmented reality',
            'social payment', 'peer-to-peer', 'decentralized', 'smart contract',
            'GDPR compliance', 'open banking', 'PSD2', 'real-time payments',
            'instant settlement', 'regulatory sandbox', 'fintech partnership'
        ]
        
        complaint_lower = complaint_text.lower()
        return any(term in complaint_lower for term in novel_indicators)
    
    def _cross_category_confusion(self, classification_results: Dict[str, Any]) -> bool:
        """Detect if AI is confused between multiple categories"""
        confidence_scores = classification_results.get('confidence_scores', {})
        
        # Count categories with high confidence (>0.5)
        high_confidence_categories = [
            category for category, score in confidence_scores.items() 
            if score > 0.5
        ]
        
        return len(high_confidence_categories) >= 3
    
    def _get_escalation_actions(self, reason: str) -> List[str]:
        """Generate specific actions for orchestrator"""
        actions_map = {
            "low_confidence": [
                "Route to human expert for manual classification",
                "Flag for training data enhancement",
                "Consider creating new category if pattern repeats"
            ],
            "novel_terminology": [
                "Alert product/compliance teams about new terminology", 
                "Update classification training data",
                "Investigate if new banking trend emerging"
            ],
            "cross_category_confusion": [
                "Human expert review required",
                "Potential multi-department coordination needed",
                "Consider if complaint spans multiple categories"
            ]
        }
        return actions_map.get(reason, ["Human expert review required"])

class TriageAgentService:
    """
    Swiss Bank Triage Agent - Intelligent complaint classification and routing
    """
    
    def __init__(self, database_service=None, eva_agent_service=None):
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("TRIAGE_API_KEY"))
        self.database_service = database_service
        self.eva_agent_service = eva_agent_service
        self.new_theme_detector = NewThemeDetector()
        
        # Swiss Bank complaint categories
        self.complaint_categories = [
            "fraudulent_activities_unauthorized_transactions",
            "account_freezes_holds_funds",
            "deposit_related_issues", 
            "dispute_resolution_issues",
            "bank_system_policy_failures",
            "atm_machine_issues",
            "check_related_issues",
            "poor_customer_service_communication",
            "delays_fund_availability",
            "overdraft_issues",
            "online_banking_technical_security_issues",
            "discrimination_unfair_practices",
            "mortgage_related_issues",
            "credit_card_issues",
            "ambiguity_unclear_unclassified",
            "debt_collection_harassment",
            "loan_issues_auto_personal_student",
            "insurance_claim_denials_delays"
        ]
        
        # Orchestrator alert queue (for future orchestrator integration)
        self.orchestrator_alerts = []
    
    # ==================== MAIN TRIAGE PROCESSING METHOD ===========================
    async def update_complaint_with_additional_context(self, context_data: Dict[str, Any]) -> bool:
        """
        NEW: Update complaint with additional context from Eva follow-up questions
        """
        try:
            conversation_id = context_data["conversation_id"]
            additional_info = context_data["additional_context"]
            
            # Store additional context for orchestrator
            additional_context_alert = {
                "alert_type": "ADDITIONAL_CONTEXT_COLLECTED",
                "alert_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,
                "additional_context": additional_info,
                "context_significance": "follow_up_details",
                "orchestrator_actions": [
                    "Review additional context for enhanced resolution",
                    "Update investigation priorities if needed",
                    "Ensure all details are included in case file"
                ]
            }
            
            # Store alert for orchestrator
            self.orchestrator_alerts.append(additional_context_alert)
            
            if self.database_service:
                await self.database_service.store_orchestrator_alert(additional_context_alert)
                
            print(f"✅ Additional context stored for conversation {conversation_id}")
            return True
        
        except Exception as e:
            print(f"❌ Error updating complaint with additional context: {e}")
            return False

    async def process_complaint(self, complaint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main triage processing with all functionalities:
        1. New vs Follow-up Detection
        2. Classification or Status Update  
        3. Three-section Output Structure
        4. New Theme Detection & Orchestrator Alerts
        """
        try:
            complaint_text = complaint_data["complaint_text"]
            customer_id = complaint_data["customer_id"]
            customer_context = complaint_data.get("customer_context", {})
            
            print(f"🎯 Processing complaint for customer {customer_id}")
            
            # STEP 1: Check if this is follow-up or new complaint
            followup_result = await self._detect_followup_complaint(
                complaint_text, customer_id, customer_context
            )
            
            if followup_result["is_followup"]:
                # Handle follow-up complaint
                return await self._handle_followup_complaint(
                    complaint_data, followup_result
                )
            else:
                # Handle new complaint with full classification
                return await self._handle_new_complaint(complaint_data)
                
        except Exception as e:
            print(f"❌ Error in triage processing: {e}")
            return self._create_error_response(str(e), complaint_data)
    
    # ===========================================
    # FOLLOW-UP DETECTION (Functionality 1)
    # ===========================================
    
    async def _detect_followup_complaint(self, complaint_text: str, customer_id: str, 
                                        customer_context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if complaint is follow-up or new"""
        
        # Get customer's open complaints
        open_complaints = await self._get_customer_open_complaints(customer_id)
        
        if not open_complaints:
            return {"is_followup": False, "reason": "no_open_complaints"}
        
        # Analyze complaint text for follow-up indicators
        followup_analysis = await self._analyze_complaint_similarity(
            complaint_text, open_complaints
        )
        
        return followup_analysis
    
    async def _get_customer_open_complaints(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get customer's open complaints"""
        if not self.database_service:
            return []
        
        try:
            # Get complaints with open status
            open_statuses = ["received", "investigating", "in_progress", "pending", "escalated"]
            open_complaints = await self.database_service.get_customer_open_complaints_by_status(
                customer_id, open_statuses
            )
            return open_complaints
        except Exception as e:
            print(f"Error getting open complaints: {e}")
            return []
    
    async def _analyze_complaint_similarity(self, new_complaint: str, 
                                          open_complaints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if new complaint is similar to existing ones"""
        
        for complaint in open_complaints:
            similarity_score = self._calculate_text_similarity(
                new_complaint, complaint.get("description", "")
            )
            
            if similarity_score > 0.7:  # High similarity threshold
                return {
                    "is_followup": True,
                    "related_complaint_id": complaint["complaint_id"],
                    "similarity_score": similarity_score,
                    "reason": "high_content_similarity"
                }
        
        return {"is_followup": False, "reason": "no_similar_complaints"}
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    # ===========================================
    # FOLLOW-UP COMPLAINT HANDLING
    # ===========================================
    
    async def _handle_followup_complaint(self, complaint_data: Dict[str, Any], 
                                        followup_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle follow-up complaint with status update"""
        
        related_complaint_id = followup_result["related_complaint_id"]
        
        # Get current status of related complaint
        current_status = await self._get_complaint_current_status(related_complaint_id)
        
        # Check if additional context is provided
        additional_context = await self._analyze_additional_context(
            complaint_data["complaint_text"], related_complaint_id
        )
        
        if additional_context["has_new_information"]:
            # Handle additional context (Functionality 2)
            return await self._handle_additional_context(
                complaint_data, related_complaint_id, additional_context
            )
        else:
            # Return status update
            return {
                "complaint_type": "followup",
                "related_complaint_id": related_complaint_id,
                "current_status": current_status,
                "eva_response": await self._generate_followup_response(current_status),
                "requires_orchestrator_action": False
            }
    
    async def _get_complaint_current_status(self, complaint_id: str) -> Dict[str, Any]:
        """Get detailed current status of complaint"""
        if not self.database_service:
            return {"status": "unknown", "message": "Database not available"}
        
        try:
            complaint = await self.database_service.get_complaint(complaint_id)
            if complaint:
                return {
                    "status": complaint.get("status", "unknown"),
                    "last_updated": complaint.get("updated_at", "unknown"),
                    "resolution_estimate": complaint.get("resolution_time_expected", "2-3 business days"),
                    "assigned_specialist": complaint.get("assigned_specialist", "Customer service team")
                }
            else:
                return {"status": "not_found", "message": "Complaint not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # ===========================================
    # ADDITIONAL CONTEXT HANDLING (Functionality 2)
    # ===========================================
    
    async def _analyze_additional_context(self, new_complaint: str, 
                                         related_complaint_id: str) -> Dict[str, Any]:
        """Analyze if new complaint provides additional context"""
        
        # Get original complaint
        if not self.database_service:
            return {"has_new_information": False}
        
        try:
            original_complaint = await self.database_service.get_complaint(related_complaint_id)
            if not original_complaint:
                return {"has_new_information": False}
            
            # Check for new financial amounts, dates, parties involved
            new_info_indicators = [
                "additional", "also", "furthermore", "another", "new evidence",
                "i forgot to mention", "update", "correction", "$", "amount",
                "happened again", "more information"
            ]
            
            new_complaint_lower = new_complaint.lower()
            has_new_info = any(indicator in new_complaint_lower for indicator in new_info_indicators)
            
            if has_new_info:
                significance = await self._assess_context_significance(new_complaint)
                return {
                    "has_new_information": True,
                    "significance_level": significance["level"],
                    "significance_reasons": significance["reasons"]
                }
            
            return {"has_new_information": False}
            
        except Exception as e:
            print(f"Error analyzing additional context: {e}")
            return {"has_new_information": False}
    
    async def _assess_context_significance(self, additional_context: str) -> Dict[str, Any]:
        """Assess significance of additional context"""
        
        context_lower = additional_context.lower()
        
        # High significance indicators
        high_significance = [
            "legal", "lawyer", "attorney", "lawsuit", "court", "police",
            "regulatory", "compliance", "fraud", "criminal", "investigation"
        ]
        
        # Medium significance indicators  
        medium_significance = [
            "additional amount", "more money", "other accounts", "different date",
            "correction", "mistake", "wrong information", "update"
        ]
        
        if any(indicator in context_lower for indicator in high_significance):
            return {
                "level": "high",
                "reasons": ["Legal or regulatory implications detected"]
            }
        elif any(indicator in context_lower for indicator in medium_significance):
            return {
                "level": "medium", 
                "reasons": ["Financial or factual corrections provided"]
            }
        else:
            return {
                "level": "low",
                "reasons": ["General additional information"]
            }
    
    async def _handle_additional_context(self, complaint_data: Dict[str, Any], 
                                        related_complaint_id: str,
                                        context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle additional context with dual notifications"""
        
        # Update database with additional context
        if self.database_service:
            try:
                await self.database_service.update_complaint_context(
                    related_complaint_id,
                    {
                        "additional_context": complaint_data["complaint_text"],
                        "context_significance": context_analysis["significance_level"],
                        "last_updated": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                print(f"Error updating complaint context: {e}")
        
        # Generate dual notifications
        orchestrator_notification = await self._generate_orchestrator_context_alert(
            related_complaint_id, context_analysis, complaint_data
        )
        
        eva_update = await self._generate_eva_context_response(
            context_analysis, related_complaint_id
        )
        
        return {
            "complaint_type": "additional_context",
            "related_complaint_id": related_complaint_id,
            "significance_level": context_analysis["significance_level"],
            "orchestrator_notification": orchestrator_notification,
            "eva_response": eva_update,
            "requires_orchestrator_action": context_analysis["significance_level"] in ["high", "medium"]
        }
    
    # ===========================================
    # NEW COMPLAINT HANDLING WITH THREE SECTIONS
    # ===========================================
    
    async def _handle_new_complaint(self, complaint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle new complaint with full three-section analysis"""
        
        complaint_text = complaint_data["complaint_text"]
        customer_context = complaint_data.get("customer_context", {})
        
        # SECTION 1: Original Complaint (preserve exactly as submitted)
        original_complaint = {
            "complaint_text": complaint_text,
            "customer_id": complaint_data["customer_id"],
            "submission_timestamp": complaint_data.get("submission_timestamp", datetime.now().isoformat()),
            "submission_method": complaint_data.get("submission_method", "web"),
            "customer_context": customer_context,
            "attachments": complaint_data.get("attachments", [])
        }
        
        # SECTION 2: Triage Analysis (AI-powered intelligence)
        triage_analysis = await self._perform_triage_analysis(complaint_text, customer_context)
        
        # Check for new theme BEFORE creating routing package
        new_theme_check = self.new_theme_detector.detect_new_theme(
            complaint_text, triage_analysis
        )
        
        if new_theme_check["is_new_theme"]:
            # Handle new theme with immediate orchestrator alert
            await self._alert_orchestrator_new_theme(new_theme_check)
            
            return {
                "complaint_type": "new_theme",
                "classification_result": "new_theme",
                "original_complaint": original_complaint,
                "triage_analysis": {
                    "classification": "new_theme",
                    "detection_reason": new_theme_check["detection_reason"],
                    "confidence_score": 0.0,
                    "requires_human_review": True
                },
                "new_theme_alert": new_theme_check,
                "requires_orchestrator_action": True,
                "orchestrator_alert_sent": True
            }
        
        # SECTION 3: Routing Package (actionable instructions)
        routing_package = await self._generate_routing_package(triage_analysis, customer_context)
        
        # Alert orchestrator about new complaint
        await self._alert_orchestrator_new_complaint(original_complaint, triage_analysis, routing_package)
        
        return {
            "complaint_type": "new_complaint",
            "original_complaint": original_complaint,
            "triage_analysis": triage_analysis,
            "routing_package": routing_package,
            "requires_orchestrator_action": True,
            "orchestrator_alert_sent": True
        }
    
    # ===========================================
    # TRIAGE ANALYSIS (SECTION 2)
    # ===========================================
    
    async def _perform_triage_analysis(self, complaint_text: str, 
                                      customer_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI-powered triage analysis"""
        
        try:
            # Build analysis prompt
            prompt = self._build_triage_analysis_prompt(complaint_text, customer_context)
            
            # Call Anthropic API
            response = await self._call_anthropic(prompt)
            
            # Parse response
            analysis = self._parse_triage_response(response)
            
            return analysis
            
        except Exception as e:
            print(f"Error in triage analysis: {e}")
            return self._fallback_triage_analysis(complaint_text)
    
    def _build_triage_analysis_prompt(self, complaint_text: str, 
                                     customer_context: Dict[str, Any]) -> str:
        """Build comprehensive triage analysis prompt"""
        
        customer_info = ""
        if customer_context:
            customer_info = f"""
CUSTOMER CONTEXT:
- Account Type: {customer_context.get('account_type', 'Standard')}
- Previous Complaints: {len(customer_context.get('recent_complaints', []))}
- Account Status: {customer_context.get('account_status', 'Active')}
- Customer Tenure: {customer_context.get('customer_since', 'Unknown')}
"""
        
        return f"""
You are a Swiss Bank complaint triage specialist. Analyze this complaint comprehensively:

COMPLAINT: {complaint_text}
{customer_info}

AVAILABLE CATEGORIES:
{', '.join(self.complaint_categories)}

Provide detailed analysis in JSON format:
{{
    "primary_category": "most_appropriate_category",
    "secondary_category": "secondary_category_or_null",
    "confidence_scores": {{
        "category1": 0.XX,
        "category2": 0.XX,
        "category3": 0.XX
    }},
    "urgency_level": "low|medium|high|critical",
    "emotional_state": "frustrated|angry|anxious|neutral|confused",
    "financial_impact": true_or_false,
    "estimated_financial_amount": "amount_if_applicable",
    "relationship_risk": "low|medium|high",
    "compliance_flags": ["list_of_compliance_concerns"],
    "key_entities": ["account_numbers", "transaction_ids", "dates"],
    "resolution_complexity": "simple|moderate|complex",
    "estimated_resolution_time": "time_estimate",
    "escalation_triggers": ["list_of_escalation_reasons"],
    "customer_expectations": "what_customer_wants",
    "previous_interaction_indicators": ["indicators_of_prior_contact"],
    "reasoning": "detailed_explanation_of_classification"
}}

Focus on INTENT and IMPACT, not just keywords. Consider the customer's emotional state and urgency.
"""
    
    def _parse_triage_response(self, response: str) -> Dict[str, Any]:
        """Parse AI triage response"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                # Add processing metadata
                analysis.update({
                    "processing_timestamp": datetime.now().isoformat(),
                    "processing_method": "anthropic_ai",
                    "triage_version": "v1.0"
                })
                
                return analysis
            else:
                return self._fallback_triage_analysis("")
        except Exception as e:
            print(f"Error parsing triage response: {e}")
            return self._fallback_triage_analysis("")
    
    def _fallback_triage_analysis(self, complaint_text: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails"""
        text_lower = complaint_text.lower()
        
        if any(word in text_lower for word in ["unauthorized", "fraud", "stolen"]):
            primary_category = "fraudulent_activities_unauthorized_transactions"
            urgency = "high"
        elif any(word in text_lower for word in ["dispute", "charge", "refund"]):
            primary_category = "dispute_resolution_issues"
            urgency = "medium"
        else:
            primary_category = "ambiguity_unclear_unclassified"
            urgency = "medium"
        
        return {
            "primary_category": primary_category,
            "secondary_category": None,
            "confidence_scores": {primary_category: 0.6},
            "urgency_level": urgency,
            "emotional_state": "neutral",
            "financial_impact": False,
            "relationship_risk": "medium",
            "compliance_flags": [],
            "resolution_complexity": "moderate",
            "estimated_resolution_time": "2-3 business days",
            "reasoning": "Fallback classification due to AI processing error"
        }
    
    # ===========================================
    # ROUTING PACKAGE GENERATION (SECTION 3)
    # ===========================================
    
    async def _generate_routing_package(self, triage_analysis: Dict[str, Any], 
                                       customer_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate routing package with orchestrator instructions"""
        
        primary_category = triage_analysis["primary_category"]
        urgency_level = triage_analysis["urgency_level"]
        
        # Determine specialist assignments
        specialist_assignment = self._determine_specialist_assignment(primary_category)
        
        # Generate SLA targets
        sla_targets = self._calculate_sla_targets(urgency_level, triage_analysis)
        
        # Generate orchestrator instructions
        orchestrator_instructions = self._generate_orchestrator_instructions(
            triage_analysis, specialist_assignment
        )
        
        # Generate Eva briefing
        eva_briefing = await self._generate_eva_briefing(triage_analysis, customer_context)
        
        return {
            "specialist_assignment": specialist_assignment,
            "coordination_requirements": self._assess_coordination_needs(triage_analysis),
            "sla_targets": sla_targets,
            "priority_level": self._calculate_priority_level(triage_analysis),
            "escalation_triggers": triage_analysis.get("escalation_triggers", []),
            "orchestrator_instructions": orchestrator_instructions,
            "eva_briefing": eva_briefing,
            "workflow_metadata": {
                "routing_timestamp": datetime.now().isoformat(),
                "routing_version": "v1.0",
                "auto_routing_enabled": True
            }
        }
    
    def _determine_specialist_assignment(self, primary_category: str) -> Dict[str, Any]:
        """Determine which specialist should handle the complaint"""
        
        specialist_mapping = {
            "fraudulent_activities_unauthorized_transactions": {
                "department": "fraud_investigation",
                "specialist_type": "senior_fraud_investigator",
                "required_skills": ["fraud_analysis", "transaction_investigation", "security_protocols"],
                "priority": "high"
            },
            "dispute_resolution_issues": {
                "department": "dispute_resolution",
                "specialist_type": "dispute_analyst",
                "required_skills": ["chargeback_processing", "merchant_communication", "documentation_review"],
                "priority": "medium"
            },
            "mortgage_related_issues": {
                "department": "mortgage_services",
                "specialist_type": "mortgage_specialist",
                "required_skills": ["loan_modification", "payment_assistance", "foreclosure_prevention"],
                "priority": "medium"
            },
            # Add more mappings as needed
        }
        
        return specialist_mapping.get(primary_category, {
            "department": "general_customer_service",
            "specialist_type": "customer_service_representative",
            "required_skills": ["general_inquiry_handling", "customer_communication"],
            "priority": "low"
        })
    
    def _calculate_sla_targets(self, urgency_level: str, triage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SLA targets based on urgency and complexity"""
        
        base_sla = {
            "critical": {"first_response": "1 hour", "resolution": "24 hours"},
            "high": {"first_response": "2 hours", "resolution": "48 hours"},
            "medium": {"first_response": "4 hours", "resolution": "72 hours"},
            "low": {"first_response": "8 hours", "resolution": "5 business days"}
        }
        
        sla = base_sla.get(urgency_level, base_sla["medium"])
        
        # Adjust based on complexity
        if triage_analysis.get("resolution_complexity") == "complex":
            sla["resolution"] = self._extend_sla_time(sla["resolution"])
        
        return sla
    
    def _extend_sla_time(self, original_time: str) -> str:
        """Extend SLA time for complex cases"""
        time_mapping = {
            "24 hours": "48 hours",
            "48 hours": "72 hours", 
            "72 hours": "5 business days",
            "5 business days": "7 business days"
        }
        return time_mapping.get(original_time, original_time)
    
    # ===========================================
    # ORCHESTRATOR ALERT SYSTEM (NEW FEATURE)
    # ===========================================
    
    async def _alert_orchestrator_new_complaint(self, original_complaint: Dict[str, Any],
                                               triage_analysis: Dict[str, Any],
                                               routing_package: Dict[str, Any]):
        """Alert orchestrator about new complaint (MAIN REQUIREMENT)"""
        
        orchestrator_alert = {
            "alert_type": "NEW_COMPLAINT",
            "alert_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "priority": routing_package["priority_level"],
            "complaint_summary": {
                "customer_id": original_complaint["customer_id"],
                "primary_category": triage_analysis["primary_category"],
                "urgency_level": triage_analysis["urgency_level"],
                "financial_impact": triage_analysis.get("financial_impact", False),
                "estimated_amount": triage_analysis.get("estimated_financial_amount")
            },
            "routing_instructions": {
                "target_department": routing_package["specialist_assignment"]["department"],
                "specialist_type": routing_package["specialist_assignment"]["specialist_type"],
                "required_skills": routing_package["specialist_assignment"]["required_skills"],
                "sla_targets": routing_package["sla_targets"],
                "coordination_needs": routing_package["coordination_requirements"]
            },
            "orchestrator_actions": [
                f"Route to {routing_package['specialist_assignment']['department']}",
                f"Assign {routing_package['specialist_assignment']['specialist_type']}",
                f"Set SLA: {routing_package['sla_targets']['resolution']}",
                "Monitor progress and escalate if needed"
            ],
            "background_information": {
                "customer_context": original_complaint["customer_context"],
                "related_transactions": triage_analysis.get("key_entities", []),
                "compliance_flags": triage_analysis.get("compliance_flags", []),
                "previous_interactions": triage_analysis.get("previous_interaction_indicators", [])
            }
        }
        
        # Store alert for orchestrator pickup
        self.orchestrator_alerts.append(orchestrator_alert)
        
        # If database available, store persistently
        if self.database_service:
            try:
                await self.database_service.store_orchestrator_alert(orchestrator_alert)
                print(f"✅ New complaint alert sent to orchestrator: {orchestrator_alert['alert_id']}")
            except Exception as e:
                print(f"⚠️ Failed to store orchestrator alert: {e}")
        
        print(f"🚨 ORCHESTRATOR ALERT: New {triage_analysis['urgency_level']} priority complaint - {triage_analysis['primary_category']}")
    
    async def _alert_orchestrator_new_theme(self, new_theme_data: Dict[str, Any]):
        """Alert orchestrator about new theme detection (CRITICAL REQUIREMENT)"""
        
        orchestrator_alert = {
            "alert_type": "NEW_THEME_DETECTED",
            "alert_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "priority": "CRITICAL",
            "new_theme_details": {
                "detection_reason": new_theme_data["detection_reason"],
                "complaint_text": new_theme_data["complaint_text"],
                "confidence_analysis": new_theme_data.get("metadata", {}),
                "requires_immediate_attention": True
            },
            "immediate_actions_required": new_theme_data["suggested_actions"],
            "orchestrator_actions": [
                "IMMEDIATE: Route to senior management",
                "CREATE: New category evaluation process",
                "ALERT: Product and compliance teams",
                "INVESTIGATE: If this represents emerging trend",
                "UPDATE: Classification training data"
            ],
            "escalation_level": "EXECUTIVE",
            "human_review_mandatory": True
        }
        
        # Store critical alert
        self.orchestrator_alerts.append(orchestrator_alert)
        
        if self.database_service:
            try:
                await self.database_service.store_orchestrator_alert(orchestrator_alert)
                print(f"🚨 CRITICAL: New theme alert sent to orchestrator: {orchestrator_alert['alert_id']}")
            except Exception as e:
                print(f"❌ FAILED to store critical new theme alert: {e}")
        
        print(f"🚨🚨🚨 CRITICAL ORCHESTRATOR ALERT: NEW THEME DETECTED - {new_theme_data['detection_reason']}")
    
    # ===========================================
    # RESPONSE GENERATION METHODS
    # ===========================================
    
    async def _generate_followup_response(self, current_status: Dict[str, Any]) -> str:
        """Generate response for follow-up complaints"""
        
        status = current_status.get("status", "unknown")
        
        status_responses = {
            "received": "Your complaint has been received and is currently being reviewed by our team.",
            "investigating": "We are actively investigating your complaint and gathering all necessary information.",
            "in_progress": "Your complaint is being processed and we're working on a resolution.",
            "pending": "Your complaint is pending additional information or approval.",
            "escalated": "Your complaint has been escalated to senior specialists for priority handling."
        }
        
        base_response = status_responses.get(status, "We're working on your complaint.")
        
        return f"""{base_response}

**Current Status:** {status.title()}
**Last Updated:** {current_status.get('last_updated', 'Recently')}
**Expected Resolution:** {current_status.get('resolution_estimate', '2-3 business days')}
**Handling Team:** {current_status.get('assigned_specialist', 'Customer service team')}

**What's happening next:**
• Your case is being actively monitored
• You'll receive updates as progress is made
• Our team will contact you if additional information is needed

Is there any additional information you'd like to provide about this case?"""
    
    async def _generate_orchestrator_context_alert(self, related_complaint_id: str, 
                                             context_analysis: Dict[str, Any],
                                             complaint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate orchestrator notification for additional context"""
        
        significance_level = context_analysis["significance_level"]
        
        return {
            "alert_type": "ADDITIONAL_CONTEXT",
            "alert_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "related_complaint_id": related_complaint_id,
            "priority": "HIGH" if significance_level == "high" else "MEDIUM",
            "context_update": {
                "additional_information": complaint_data["complaint_text"],
                "significance_level": significance_level,
                "significance_reasons": context_analysis.get("significance_reasons", []),
                "customer_id": complaint_data["customer_id"]
            },
            "orchestrator_actions": [
                f"Review additional context for complaint {related_complaint_id}",
                f"Assess if priority adjustment needed (current significance: {significance_level})",
                "Update assigned specialist with new information",
                "Consider if case complexity has changed"
            ],
            "requires_specialist_notification": significance_level in ["high", "medium"]
        }

    async def _generate_eva_context_response(self, context_analysis: Dict[str, Any], 
                                        related_complaint_id: str) -> str:
        """Generate Eva response for additional context"""
        
        significance_level = context_analysis["significance_level"]
        
        if significance_level == "high":
            return f"""Thank you for this important additional information about case {related_complaint_id}. 

    **This update is significant and I've immediately notified our specialist team.**

    **What's happening now:**
    - Your case priority has been elevated due to this new information
    - The assigned specialist will review this update within 2 hours
    - You may receive a call to discuss these new details
    - We'll provide an updated timeline once reviewed

    This additional context is very helpful for ensuring we address your concerns completely."""
        
        elif significance_level == "medium":
            return f"""Thank you for the additional details about case {related_complaint_id}.

    **What's happening now:**
    - I've updated your case file with this new information
    - Your assigned specialist will review this within 24 hours
    - This may help us resolve your case more effectively
    - You'll receive an update once this information has been reviewed

    **Your current case status:** Your case is actively being processed with the expected timeline of resolution."""
        
        else:
            return f"""Thank you for following up on case {related_complaint_id}.

    **What's happening now:**
    - I've noted this additional information in your case file
    - Your case continues to be processed according to the original timeline
    - This information will be available to your specialist for reference

    **Current status:** Your case is being handled and you'll receive updates as progress is made."""
        
    async def _generate_eva_briefing(self, triage_analysis: Dict[str, Any], 
                                    customer_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate briefing for Eva agent"""
        
        return {
            "customer_communication_guidelines": {
                "emotional_approach": self._get_emotional_approach(triage_analysis["emotional_state"]),
                "key_talking_points": self._generate_talking_points(triage_analysis),
                "specialist_to_mention": self._get_specialist_name(triage_analysis["primary_category"]),
                "timeline_to_communicate": triage_analysis["estimated_resolution_time"]
            },
            "classification_confirmation_needed": {
                "primary_category_friendly": self._translate_category_for_customer(triage_analysis["primary_category"]),
                "confidence_level": max(triage_analysis.get("confidence_scores", {}).values()),
                "requires_customer_confirmation": True
            },
            "next_steps_for_customer": self._generate_customer_next_steps(triage_analysis)
        }
    
    def _get_emotional_approach(self, emotional_state: str) -> str:
        """Get appropriate emotional approach for Eva"""
        
        emotional_approaches = {
            "frustrated": "Show understanding and patience. Acknowledge their frustration and focus on immediate action.",
            "angry": "Remain calm and professional. Validate their concerns and emphasize resolution commitment.",
            "anxious": "Provide reassurance and clear timelines. Emphasize security and careful handling.",
            "confused": "Use clear, simple language. Explain processes step-by-step.",
            "neutral": "Professional and friendly. Focus on efficient resolution."
        }
        
        return emotional_approaches.get(emotional_state, emotional_approaches["neutral"])
    
    def _translate_category_for_customer(self, category: str) -> str:
        """Translate technical category to customer-friendly language"""
        
        translations = {
            "fraudulent_activities_unauthorized_transactions": "Unauthorized transaction or fraud concern",
            "dispute_resolution_issues": "Transaction dispute or chargeback request",
            "mortgage_related_issues": "Mortgage or home loan concern",
            "poor_customer_service_communication": "Service quality concern",
            "online_banking_technical_security_issues": "Online banking technical issue",
            "ambiguity_unclear_unclassified": "General banking inquiry"
        }
        
        return translations.get(category, "Banking service inquiry")
    
    # ===========================================
    # UTILITY METHODS
    # ===========================================
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent classification
                messages=[{"role": "user", "content": prompt}]
            )
            
            content_text = ""
            for content_block in response.content:
                content_text += getattr(content_block, 'text', '') or str(getattr(content_block, 'content', ''))
            
            return content_text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            raise e
    
    def _assess_coordination_needs(self, triage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if multiple departments need coordination"""
        
        category = triage_analysis["primary_category"]
        
        # Categories that typically require coordination
        coordination_mapping = {
            "fraudulent_activities_unauthorized_transactions": {
                "requires_coordination": True,
                "departments": ["fraud_investigation", "security", "legal"],
                "coordination_type": "parallel_investigation"
            },
            "mortgage_related_issues": {
                "requires_coordination": True,
                "departments": ["mortgage_services", "legal", "collections"],
                "coordination_type": "sequential_review"
            },
            "dispute_resolution_issues": {
                "requires_coordination": False,
                "departments": ["dispute_resolution"],
                "coordination_type": "single_department"
            }
        }
        
        return coordination_mapping.get(category, {
            "requires_coordination": False,
            "departments": ["general_customer_service"],
            "coordination_type": "single_department"
        })
    
    def _calculate_priority_level(self, triage_analysis: Dict[str, Any]) -> str:
        """Calculate overall priority level"""
        
        urgency = triage_analysis["urgency_level"]
        financial_impact = triage_analysis.get("financial_impact", False)
        relationship_risk = triage_analysis.get("relationship_risk", "low")
        
        # Priority matrix
        if urgency == "critical" or (financial_impact and relationship_risk == "high"):
            return "P1_CRITICAL"
        elif urgency == "high" or (financial_impact and relationship_risk == "medium"):
            return "P2_HIGH"
        elif urgency == "medium":
            return "P3_MEDIUM"
        else:
            return "P4_LOW"
    
    def _generate_orchestrator_instructions(self, triage_analysis: Dict[str, Any], 
                                          specialist_assignment: Dict[str, Any]) -> List[str]:
        """Generate specific instructions for orchestrator"""
        
        instructions = [
            f"Route to {specialist_assignment['department']} department",
            f"Assign {specialist_assignment['specialist_type']} with skills: {', '.join(specialist_assignment['required_skills'])}"
        ]
        
        if triage_analysis.get("financial_impact"):
            instructions.append("Flag for financial impact review")
        
        if triage_analysis.get("compliance_flags"):
            instructions.append(f"Compliance review required: {', '.join(triage_analysis['compliance_flags'])}")
        
        if triage_analysis["urgency_level"] in ["high", "critical"]:
            instructions.append("Priority handling - expedite assignment")
        
        return instructions
    
    def _generate_customer_next_steps(self, triage_analysis: Dict[str, Any]) -> List[str]:
        """Generate next steps for customer communication"""
        
        category = triage_analysis["primary_category"]
        
        category_steps = {
            "fraudulent_activities_unauthorized_transactions": [
                "Freeze affected accounts immediately",
                "File police report if advised",
                "Monitor all account activity closely",
                "Await contact from fraud specialist within 2 hours"
            ],
            "dispute_resolution_issues": [
                "Gather supporting documentation",
                "Avoid using disputed payment method",
                "Respond to any requests for additional information",
                "Allow 3-5 business days for initial review"
            ]
        }
        
        return category_steps.get(category, [
            "Monitor your account for any updates",
            "Respond promptly to any requests for information",
            "Contact us immediately if situation worsens",
            "Expect resolution within estimated timeframe"
        ])
    
    def _get_specialist_name(self, category: str) -> str:
        """Get specialist name for customer communication"""
        
        # Use Eva's specialist names if available
        if self.eva_agent_service and hasattr(self.eva_agent_service, 'specialist_names'):
            specialists = self.eva_agent_service.specialist_names.get(category, [])
            if specialists:
                return specialists[0].get("name", "Customer Service Specialist")
        
        return "Customer Service Specialist"
    
    def _generate_talking_points(self, triage_analysis: Dict[str, Any]) -> List[str]:
        """Generate key talking points for Eva"""
        
        points = [
            f"We understand this is a {triage_analysis['emotional_state']} situation",
            f"This has been classified as {triage_analysis['urgency_level']} priority",
            f"Expected resolution timeframe: {triage_analysis['estimated_resolution_time']}"
        ]
        
        if triage_analysis.get("financial_impact"):
            points.append("We recognize the financial impact and will prioritize accordingly")
        
        return points
    
    def _create_error_response(self, error_message: str, complaint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create error response when processing fails"""
        
        return {
            "complaint_type": "processing_error",
            "error": error_message,
            "fallback_classification": "ambiguity_unclear_unclassified",
            "requires_manual_review": True,
            "original_complaint": {
                "complaint_text": complaint_data.get("complaint_text", ""),
                "customer_id": complaint_data.get("customer_id", "")
            },
            "orchestrator_alert_required": True
        }
    
    # ===========================================
    # SERVICE HEALTH AND STATUS METHODS
    # ===========================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for triage agent service"""
        
        try:
            # Test Anthropic API
            test_prompt = "Test prompt for health check"
            await self._call_anthropic("Respond with 'OK' for health check")
            anthropic_status = "healthy"
        except Exception:
            anthropic_status = "unhealthy"
        
        # Test database connection
        database_status = "healthy" if self.database_service else "not_configured"
        if self.database_service:
            try:
                await self.database_service.health_check()
            except Exception:
                database_status = "unhealthy"
        
        # Test Eva integration
        eva_status = "healthy" if self.eva_agent_service else "not_configured"
        
        warnings = []
        if anthropic_status != "healthy":
            warnings.append("Anthropic API unavailable - using fallback classification")
        if database_status != "healthy":
            warnings.append("Database unavailable - limited functionality")
        
        overall_status = "healthy" if not warnings else "degraded"
        
        return {
            "status": overall_status,
            "components": {
                "anthropic_api": anthropic_status,
                "database": database_status,
                "eva_integration": eva_status,
                "new_theme_detector": "healthy"
            },
            "warnings": warnings,
            "orchestrator_alerts_pending": len(self.orchestrator_alerts),
            "capabilities": {
                "followup_detection": True,
                "new_theme_detection": True,
                "three_section_analysis": True,
                "orchestrator_alerts": True
            }
        }
    
    async def get_pending_orchestrator_alerts(self) -> List[Dict[str, Any]]:
        """Get alerts pending for orchestrator"""
        return self.orchestrator_alerts.copy()
    
    async def clear_processed_alerts(self, alert_ids: List[str]) -> bool:
        """Clear alerts that have been processed by orchestrator"""
        try:
            self.orchestrator_alerts = [
                alert for alert in self.orchestrator_alerts 
                if alert["alert_id"] not in alert_ids
            ]
            return True
        except Exception:
            return False
        

