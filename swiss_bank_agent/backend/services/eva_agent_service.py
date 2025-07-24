# backend/services/eva_agent_service.py - FIXED VERSION with proper database integration
import anthropic
import json
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import re
import os
import asyncio
from dotenv import load_dotenv
import random

load_dotenv()

@dataclass
class ConversationContext:
    conversation_id: str
    customer_id: str
    customer_name: str
    messages: List[Dict[str, Any]]
    ongoing_issues: List[str]
    specialist_assignments: Dict[str, Any]
    emotional_state: str
    classification_pending: Optional[Dict[str, Any]] = None

@dataclass
class ClassificationFeedback:
    complaint_id: str
    original_classification: Dict[str, Any]
    customer_response: str
    feedback_type: str
    learning_weight: float
    timestamp: str

class EvaAgentService:
    """
    Eva - Personal Relationship Manager for Swiss Bank
    Implements all 5 requirements + reinforcement learning
    FIXED VERSION with proper database integration
    """
    
    def __init__(self, database_service=None, triage_service=None):
        # Initialize Claude client
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("EVA_API_KEY"))
        self.database_service = database_service
        self.triage_service = triage_service
        
        # FIXED: Check database availability during initialization
        self.database_available = False
        if self.database_service:
            try:
                # Test database connection
                self.database_available = self.database_service._check_connection()
                if self.database_available:
                    print("✅ Eva initialized with active database connection")
                else:
                    print("⚠️ Eva initialized but database not connected yet")
            except Exception as e:
                print(f"⚠️ Eva database test failed: {e}")
                self.database_available = False
        else:
            print("⚠️ Eva initialized without database service")
        
        # Conversation memory storage (Requirement 1) - now with database backing
        self.conversation_contexts = {}
        
        # NEW: Conversation flow state management for natural flow
        self.conversation_states = {}

        # Learning system storage
        self.classification_weights = {}
        self.feedback_history = []
        
        # Load learning weights from database if available
        if self.database_available:
            self._load_learning_weights_from_database()
        
        # Specialist name mappings (Requirement 5) - Enhanced with detailed credentials
        self.specialist_names = self._initialize_specialist_names()
        
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
    
        # NEW: Banking policy constraints
        self.banking_constraints = {
            "no_instant_refunds": True,
            "investigation_required": True,
            "regulatory_compliance": True,
            "documentation_protocols": True,
            "provisional_credit_conditions": True
        }

        # NEW: Realistic timelines by complaint category
        self.realistic_timelines = {
            "fraudulent_activities_unauthorized_transactions": {
                "security_action": "Immediate",
                "investigation_start": "2-4 Working hours",
                "provisional_credit_review": "1-3 business days",
                "final_resolution": "3-5 business days",
                "new_card_delivery": "24-48 hours"
            },
            "dispute_resolution_issues": {
                "case_creation": "Immediate",
                "investigation_start": "1-2 Working hours", 
                "provisional_credit_review": "1-2  business days",
                "final_resolution": "3-5 business days"
            },
            "account_freezes_holds_funds": {
                "security_review": "2-4 hours",
                "documentation_review": "4-24 hours",
                "access_restoration": "1-2 business days"
            },
            "online_banking_technical_security_issues": {
                "security_check": "Immediate",
                "technical_investigation": "2-4 hours",
                "resolution": "4-24 hours"
            },
            "default": {
                "initial_response": "2-4 hours",
                "investigation": "1-2 business days", 
                "resolution": "3-5 business days"
            }
        }
        
        print("✅ Eva initialized with natural flow and banking policy compliance")
    
    # ========================= Core Utility Methods ====================

    def _get_holidays_for_date(self, date_obj: datetime) -> List[str]:
        """Holiday detection for contextual greetings"""
        month_day = date_obj.strftime("%m-%d")
        
        fixed_holidays = {
            "01-01": "Happy New Year",
            "02-14": "Happy Valentine's Day",
            "07-04": "Happy Independence Day", 
            "10-31": "Happy Halloween",
            "12-24": "Happy Christmas Eve",
            "12-25": "Merry Christmas",
            "12-31": "Happy New Year's Eve"
        }
        
        holidays = []
        if month_day in fixed_holidays:
            holidays.append(fixed_holidays[month_day])
        
        return holidays
    
    def _check_special_occasions(self, date_obj: datetime) -> str:
        """Check for special occasions"""
        weekday = date_obj.weekday()  # 0=Monday, 6=Sunday
        
        if weekday in [5, 6]:  # Weekend
            return "I hope you're having a wonderful weekend!"
        elif date_obj.day == 1:  # First of month
            return f"Welcome to {date_obj.strftime('%B')}!"
        else:
            return ""
        
    def _detect_frustration(self, text: str) -> float:
        frustration_words = ["frustrated", "annoyed", "tired of", "sick of", "enough", "ridiculous", "unacceptable"]
        matches = sum(1 for word in frustration_words if word in text)
        return min(matches * 0.3, 1.0)
    
    def _detect_anxiety(self, text: str) -> float:
        anxiety_words = ["worried", "concerned", "scared", "nervous", "afraid", "anxious", "uncertain"]
        matches = sum(1 for word in anxiety_words if word in text)
        return min(matches * 0.35, 1.0)
    
    def _detect_anger(self, text: str) -> float:
        anger_words = ["angry", "furious", "outraged", "livid", "mad", "disgusted", "demanding"]
        matches = sum(1 for word in anger_words if word in text)
        return min(matches * 0.4, 1.0)
    
    def _detect_happiness(self, text: str) -> float:
        happy_words = ["thank you", "thanks", "amazing", "excellent", "wonderful", "great", "perfect", "helpful"]
        matches = sum(1 for word in happy_words if word in text)
        return min(matches * 0.4, 1.0)
    
    def _detect_confusion(self, text: str) -> float:
        confusion_words = ["confused", "don't understand", "unclear", "what does this mean", "explain"]
        matches = sum(1 for word in confusion_words if word in text)
        return min(matches * 0.3, 1.0)
    
    def _get_realistic_alternatives(self, violations: List[str]) -> List[str]:
        """NEW: Get realistic alternatives for unrealistic promises"""
        alternatives = {
            "instant refund": "expedited dispute processing for provisional credit review",
            "immediate refund": "priority investigation for fastest possible resolution",
            "money back now": "emergency dispute filing with urgent review",
            "credit your account immediately": "provisional credit consideration after initial investigation",
            "temporary credit back to your account within the next 2 hours": "expedited review for provisional credit eligibility within 1-3 business days"
        }
        
        return [alternatives.get(violation, "realistic timeline communication") for violation in violations]
    
    def _get_realistic_timeline(self, complaint_category: str) -> Dict[str, str]:
        """NEW: Get realistic timeline for complaint category"""
        return self.realistic_timelines.get(complaint_category, self.realistic_timelines["default"])
    
    def _calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate accuracy metrics from feedback history"""
        if not self.feedback_history:
            return {"overall_accuracy": 0.0, "total_feedback": 0}
        
        total = len(self.feedback_history)
        confirmed = sum(1 for f in self.feedback_history if f.feedback_type == "confirmed")
        partial = sum(1 for f in self.feedback_history if f.feedback_type == "partial_correction")
        
        accuracy = (confirmed + (partial * 0.5)) / total if total > 0 else 0.0
        
        return {
            "overall_accuracy": accuracy,
            "total_feedback": total,
            "confirmed_classifications": confirmed,
            "partial_corrections": partial
        }

    def _translate_category_for_customer(self, category: str) -> str:
        """NEW: Convert technical category to customer-friendly language"""
        translations = {
            "fraudulent_activities_unauthorized_transactions": "Fraudulent transaction / Card theft",
            "dispute_resolution_issues": "Transaction dispute",
            "account_freezes_holds_funds": "Account access issue",
            "online_banking_technical_security_issues": "Online banking technical issue",
            "poor_customer_service_communication": "Service quality concern",
            "mortgage_related_issues": "Mortgage or home loan concern",
            "credit_card_issues": "Credit card concern",
            "overdraft_issues": "Overdraft or fee concern",
            "ambiguity_unclear_unclassified": "General banking inquiry"
        }
        return translations.get(category, "Banking service inquiry")
    
    def _initialize_specialist_names(self) -> Dict[str, List[Dict[str, str]]]:
        """Initialize realistic specialist names with credentials (Requirement 5)"""
        return {
            "fraudulent_activities_unauthorized_transactions": [
                {"name": "Sarah Chen", "title": "Senior Fraud Investigator", "experience": "8 years", "specialty": "unauthorized transaction cases", "success_rate": "96%"},
                {"name": "Michael Rodriguez", "title": "Fraud Analysis Specialist", "experience": "6 years", "specialty": "identity theft investigations", "success_rate": "94%"},
                {"name": "Jennifer Williams", "title": "Security Specialist", "experience": "10 years", "specialty": "account compromise cases", "success_rate": "98%"}
            ],
            "dispute_resolution_issues": [
                {"name": "Jennifer Martinez", "title": "Dispute Resolution Specialist", "experience": "9 years", "specialty": "chargeback and dispute cases", "success_rate": "95%"},
                {"name": "Kevin Wu", "title": "Senior Dispute Analyst", "experience": "7 years", "specialty": "merchant transaction disputes", "success_rate": "93%"},
                {"name": "Amanda Foster", "title": "Dispute Resolution Manager", "experience": "11 years", "specialty": "complex dispute cases", "success_rate": "97%"}
            ],
            "mortgage_related_issues": [
                {"name": "David Rodriguez", "title": "Senior Mortgage Specialist", "experience": "12 years", "specialty": "loan modification and refinancing", "success_rate": "94%"},
                {"name": "Emily Zhang", "title": "Mortgage Resolution Specialist", "experience": "8 years", "specialty": "payment assistance programs", "success_rate": "92%"},
                {"name": "Christopher Lee", "title": "Home Loan Advisor", "experience": "10 years", "specialty": "foreclosure prevention", "success_rate": "96%"}
            ],
            "poor_customer_service_communication": [
                {"name": "Patricia Mitchell", "title": "Customer Experience Manager", "experience": "11 years", "specialty": "service recovery and escalations", "success_rate": "98%"},
                {"name": "Steven Garcia", "title": "Customer Relations Supervisor", "experience": "9 years", "specialty": "complaint resolution", "success_rate": "95%"},
                {"name": "Michelle Adams", "title": "Senior Customer Advocate", "experience": "8 years", "specialty": "relationship management", "success_rate": "97%"}
            ],
            "technical": [
                {"name": "Sarah Johnson", "title": "Technical Support Lead", "experience": "7 years", "specialty": "online banking systems", "success_rate": "94%"},
                {"name": "Mike Chen", "title": "IT Support Specialist", "experience": "5 years", "specialty": "mobile app issues", "success_rate": "92%"},
                {"name": "Emma Rodriguez", "title": "Systems Analyst", "experience": "6 years", "specialty": "platform integration", "success_rate": "95%"}
            ],
            "billing": [
                {"name": "David Park", "title": "Billing Specialist", "experience": "9 years", "specialty": "fee disputes and adjustments", "success_rate": "96%"},
                {"name": "Lisa Wang", "title": "Account Resolution Expert", "experience": "7 years", "specialty": "payment processing issues", "success_rate": "94%"},
                {"name": "James Miller", "title": "Financial Services Advisor", "experience": "10 years", "specialty": "account reconciliation", "success_rate": "97%"}
            ],
            "account": [
                {"name": "Rachel Green", "title": "Account Manager", "experience": "8 years", "specialty": "account access and security", "success_rate": "95%"},
                {"name": "Tom Wilson", "title": "Customer Account Specialist", "experience": "6 years", "specialty": "profile and settings management", "success_rate": "93%"},
                {"name": "Anna Smith", "title": "Banking Services Coordinator", "experience": "9 years", "specialty": "account setup and maintenance", "success_rate": "96%"}
            ],
            "general": [
                {"name": "Chris Taylor", "title": "Customer Service Representative", "experience": "5 years", "specialty": "general banking inquiries", "success_rate": "92%"},
                {"name": "Maria Garcia", "title": "Banking Advisor", "experience": "7 years", "specialty": "product information and guidance", "success_rate": "94%"},
                {"name": "Alex Brown", "title": "Customer Support Specialist", "experience": "6 years", "specialty": "multi-service assistance", "success_rate": "93%"}
            ]
        }

    def _load_learning_weights_from_database(self):
        """Load learning weights from database on startup"""
        try:
            if not self.database_available:
                return
            
        except Exception as e:
            print(f"⚠️ Failed to load learning weights: {e}")

    # ==================== EXTERNAL API & DATABASE METHODS ====================

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API with better error handling"""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,  # Allow longer responses for natural conversation
                temperature=0.7,  # Higher temperature for more natural responses
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Handle different content block types safely
            content_text = ""
            for content_block in response.content:
                content_text += getattr(content_block, 'text', '') or str(getattr(content_block, 'content', ''))
            
            return content_text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            raise e
        
    async def _load_learning_weights_async(self):
        """Load learning weights from database (async)"""
        try:
            if not self.database_available or not self.database_service:
                return
            
            weights_data = await self.database_service.get_eva_learning_weights()
            if weights_data:
                self.classification_weights = weights_data.get("classification_weights", {})
                print(f"✅ Loaded {len(self.classification_weights)} learning weights from database")
            else:
                print("📚 No existing learning weights found, starting fresh")
                
        except Exception as e:
            print(f"⚠️ Failed to load learning weights: {e}")
    
    async def _save_learning_weights_to_database(self):
        """Save learning weights to database"""
        try:
            if not self.database_available or not self.database_service:
                return False
            
            weights_data = {
                "classification_weights": self.classification_weights,
                "total_feedback_processed": len(self.feedback_history),
                "accuracy_metrics": self._calculate_accuracy_metrics(),
                "version_id": str(uuid.uuid4())
            }
            
            success = await self.database_service.store_eva_learning_weights(weights_data)
            if success:
                print("✅ Learning weights saved to database")
            return success
            
        except Exception as e:
            print(f"⚠️ Failed to save learning weights: {e}")
            return False
        
    # ==================== ANALYSIS & HELPER METHODS ====================
    
    async def _analyze_emotion(self, message: str) -> str:
        """Simple emotion analysis for message storage"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["angry", "furious", "mad"]):
            return "angry"
        elif any(word in message_lower for word in ["frustrated", "annoyed"]):
            return "frustrated" 
        elif any(word in message_lower for word in ["worried", "concerned", "scared"]):
            return "anxious"
        elif any(word in message_lower for word in ["thank", "thanks", "great", "excellent"]):
            return "happy"
        else:
            return "neutral"
        
    async def _analyze_customer_emotion(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Requirement 3: Advanced emotional intelligence"""
        
        message_lower = message.lower()
        
        # Emotion detection patterns
        emotions = {
            "frustrated": self._detect_frustration(message_lower),
            "anxious": self._detect_anxiety(message_lower), 
            "angry": self._detect_anger(message_lower),
            "happy": self._detect_happiness(message_lower),
            "confused": self._detect_confusion(message_lower)
        }
        
        # Determine primary emotion
        primary_emotion = max(emotions.items(), key=lambda x: x[1])
        
        return {
            "primary_emotion": primary_emotion[0] if primary_emotion[1] > 0 else "neutral",
            "emotion_intensity": "high" if primary_emotion[1] > 0.7 else "medium" if primary_emotion[1] > 0.3 else "low",
            "emotions_detected": emotions,
            "empathy_needed": primary_emotion[1] > 0.3
        }
    
    def _analyze_customer_confirmation(self, message: str) -> Dict[str, Any]:
        """NEW: Analyze customer's confirmation response"""
        message_lower = message.lower()
        
        confirmed_indicators = ["yes", "correct", "right", "accurate", "exactly", "that's it"]
        correction_indicators = ["no", "wrong", "not exactly", "but", "actually", "however"]
        
        if any(indicator in message_lower for indicator in confirmed_indicators):
            return {"confirmed": True, "needs_correction": False}
        elif any(indicator in message_lower for indicator in correction_indicators):
            return {"confirmed": False, "needs_correction": True}
        else:
            return {"confirmed": False, "needs_correction": False}
        
    async def _is_complaint(self, message: str) -> bool:
        """Detect if message contains a complaint"""
        complaint_indicators = [
            "problem", "issue", "complaint", "wrong", "error", "frustrated", 
            "unauthorized", "dispute", "denied", "refused", "terrible", 
            "awful", "unacceptable", "fraud", "stolen", "locked", "frozen",
            "furious", "charges", "never made", "angry"
        ]
        
        message_lower = message.lower()
        result = any(indicator in message_lower for indicator in complaint_indicators)
        print(f"🔍 COMPLAINT CHECK: '{message_lower[:50]}...' -> {result}")
        return result
    
    def _validate_response_promises(self, response_text: str) -> Dict[str, Any]:
        """
        NEW: Validate that response doesn't make unrealistic promises
        """
        unrealistic_phrases = [
            "instant refund", "immediate refund", "money back now",
            "credit your account immediately", "refund within hours",
            "instant credit", "immediate credit", "money available now",
            "temporary credit back to your account within the next 2 hours"
        ]
        
        violations = []
        for phrase in unrealistic_phrases:
            if phrase.lower() in response_text.lower():
                violations.append(phrase)
        
        return {
            "is_realistic": len(violations) == 0,
            "violations": violations,
            "suggestions": self._get_realistic_alternatives(violations) if violations else []
        }
    
    def _customer_wants_to_proceed(self, message: str) -> bool:
        """NEW: Check if customer wants to skip more questions"""
        proceed_indicators = [
            "let's proceed", "move on", "that's enough", "what's next",
            "fix this now", "take action", "resolve this"
        ]
        return any(indicator in message.lower() for indicator in proceed_indicators)

    # ========================= GENERATION METHODS ====================

    async def _generate_contextual_greeting(self, customer_context: Dict[str, Any]) -> str:
        """Requirement 4: Natural time-based greetings"""
        
        current_time = datetime.now()
        customer_name = customer_context.get("name", "valued customer")
        
        # Holiday detection
        holidays = self._get_holidays_for_date(current_time)
        
        # Time of day greeting
        hour = current_time.hour
        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 17:
            time_greeting = "Good afternoon"
        elif 17 <= hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Good evening"
        
        # Check for recent interactions or account status
        recent_complaints = customer_context.get("recent_complaints", [])
        account_status = customer_context.get("account_status", "active")
        
        if holidays:
            holiday_name = holidays[0]
            return f"Hi {customer_name}, {holiday_name}! I'm Eva, your personal relationship manager at Swiss Bank. How can I help you today?"
        elif recent_complaints:
            # Customer has recent issues
            return f"""{time_greeting}, {customer_name}! I see you've been in touch with us recently about some concerns. I'm Eva, your dedicated customer service assistant, and I'm here to make sure we resolve everything quickly and thoroughly.

How can I help you today? I have all your recent case details and I'm ready to jump right in."""
        elif account_status == "premium":
            # Premium customer
            return f"""{time_greeting}, {customer_name}! As one of our valued premium customers, I want to make sure you receive the exceptional service you deserve. I'm Eva, your personal customer service assistant.

What can I help you with today?"""
        else:
            # Check for special occasions
            special_occasion = self._check_special_occasions(current_time)
            if special_occasion:
                return f"{time_greeting}, {customer_name}! {special_occasion} I'm Eva, your personal relationship manager at Swiss Bank. How can I help you today?"
            else:
                return f"{time_greeting}, {customer_name}! I'm Eva, your customer service assistant at Swiss Bank. I'm here to help resolve any concerns you might have quickly and efficiently.\n\nWhat brings you to us today?"
    
    async def _generate_empathetic_acknowledgment(self, complaint_text: str, customer_name: str) -> str:
        """
        NEW: Generate natural empathetic acknowledgment based on complaint content
        """
        # Detect urgency and emotion
        urgency_indicators = ["rent money", "mortgage", "urgent", "emergency", "panic"]
        amount_match = re.search(r'\$([0-9,]+)', complaint_text)
        
        is_urgent = any(indicator in complaint_text.lower() for indicator in urgency_indicators)
        amount = amount_match.group(0) if amount_match else "significant amount"
        
        if is_urgent and "rent" in complaint_text.lower():
            return f"""
            {customer_name}, I can absolutely hear the panic in your message, and I completely understand - 
            when {amount} of your rent money is involved in unauthorized charges, this becomes an emergency situation. 
            
            You did exactly the right thing by contacting us immediately.
            """
        elif "furious" in complaint_text.lower() or "angry" in complaint_text.lower():
            return f"""
            {customer_name}, I can feel how frustrated and angry you are about these charges, and you have every right to be. 
            This is definitely not something you should have to deal with, and I'm going to make sure we resolve this quickly.
            """
        else:
            return f"""
            {customer_name}, I can see how concerned you are about these charges. 
            This is definitely not something you should have to deal with, and I'm here to help you get this resolved.
            """
    async def _generate_empathetic_fallback(self, emotional_analysis: Dict[str, Any], 
                                          context: ConversationContext) -> Dict[str, Any]:
        """Generate empathetic fallback response"""
        
        emotion = emotional_analysis.get("primary_emotion", "neutral")
        customer_name = context.customer_name
        
        empathy_responses = {
            "frustrated": f"I can absolutely understand your frustration, {customer_name}. Let me help resolve this for you right away.",
            "anxious": f"I can hear the concern in your message, {customer_name}. Let me ease your worries by addressing this immediately.",
            "angry": f"I can tell you're upset, {customer_name}, and I completely understand why. Let me focus on fixing this for you.",
            "confused": f"I can see this is confusing, {customer_name}. Let me clarify everything step by step for you."
        }
        
        response = empathy_responses.get(emotion, f"Thank you for reaching out, {customer_name}. I'm here to help you with whatever you need.")
        
        return {
            "content": response,
            "next_steps": [
                "I'm analyzing your request to provide the best assistance",
                "I'll connect you with the right specialist if needed", 
                "You'll receive a complete response shortly"
            ],
            "specialists_mentioned": []
        }
    
    async def _generate_fallback_response(self, customer_context: Dict[str, Any]) -> str:
        """
        Generate fallback response when errors occur in Eva processing
        This function should be placed after the _analyze_emotion method
        """
        customer_name = customer_context.get("name", "valued customer")
        
        # Different fallback responses based on context
        if customer_context.get("account_status") == "premium":
            return f"""
            I apologize, {customer_name}. I'm experiencing a brief technical issue that's preventing me from providing you with the premium service you deserve. 
            
            As one of our valued premium customers, I want to ensure you receive immediate assistance. Please call our dedicated premium support line at 1-800-SWISS-VIP, or I'll have a senior specialist call you within the next 30 minutes.
            
            Thank you for your patience, and I sincerely apologize for this inconvenience.
            """
        
        elif "recent_complaints" in customer_context and len(customer_context.get("recent_complaints", [])) > 0:
            return f"""
            I apologize, {customer_name}. I'm having a temporary technical issue, and I know this is especially frustrating given that you've already been in touch with us recently about concerns.
            
            Given your ongoing situation, I want to make sure you get immediate help. Please call our customer service line at 1-800-SWISS-BANK and mention that you were speaking with Eva about a follow-up issue. They'll prioritize your call.
            
            I'm truly sorry for this additional inconvenience.
            """
        
        else:
            return f"""
            I apologize, {customer_name}. I'm experiencing a brief technical issue that's preventing me from helping you properly right now.
            
            **Here are your immediate options:**
            • Call our customer service line at 1-800-SWISS-BANK for immediate assistance
            • Try chatting with me again in a few moments
            • Visit any Swiss Bank branch for in-person help
            
            I sincerely apologize for this inconvenience, and thank you for your patience.
            """

    async def _generate_next_followup_question(self, conversation_state: Dict[str, Any]) -> str:
        """NEW: Generate next follow-up question based on previous responses"""
        previous_responses = [info["response"] for info in conversation_state["gathered_info"]]
        
        prompt = f"""
        Based on previous customer responses: {' | '.join(previous_responses)}
        
        Generate the next logical follow-up question for this complaint investigation.
        Keep it conversational and focused on gathering helpful details.
        Don't repeat information already gathered.
        """
        
        return await self._call_anthropic(prompt)
    
    # ========================= CLASSIFICATION METHODS ====================

    def _fallback_classification(self, complaint_text: str) -> Dict[str, Any]:
        """Fallback classification when AI fails"""
        text_lower = complaint_text.lower()
        
        # Simple keyword-based fallback
        if any(word in text_lower for word in ["unauthorized", "fraud", "stolen"]):
            return {
                "primary_category": "fraudulent_activities_unauthorized_transactions",
                "sub_category": "unauthorized_transactions",
                "priority": "high",
                "sentiment": "negative",
                "theme": "Unauthorized Transaction",
                "confidence_score": 0.6,
                "estimated_resolution": "24-48 hours",
                "financial_impact": True,
                "urgency_indicators": ["unauthorized", "fraud"],
                "requires_callback": True,
                "requires_human_review": True,
                "compliance_flags": ["fraud_investigation"],
                "suggested_agent_skills": ["fraud_investigation"],
                "reasoning": "Keyword-based fallback classification for fraud indicators"
            }
        elif any(word in text_lower for word in ["dispute", "denied", "claim"]):
            return {
                "primary_category": "dispute_resolution_issues",
                "sub_category": "transaction_dispute",
                "priority": "medium",
                "sentiment": "negative",
                "theme": "Transaction Dispute",
                "confidence_score": 0.6,
                "estimated_resolution": "3-5 business days",
                "financial_impact": True,
                "urgency_indicators": [],
                "requires_callback": False,
                "requires_human_review": True,
                "compliance_flags": [],
                "suggested_agent_skills": ["dispute_resolution"],
                "reasoning": "Keyword-based fallback classification for dispute"
            }
        elif any(word in text_lower for word in ["payment", "transaction", "charge", "fee", "billing"]):
            return {
                "primary_category": "delays_fund_availability",
                "sub_category": "payment_processing",
                "priority": "medium",
                "sentiment": "neutral",
                "theme": "Payment Issue",
                "confidence_score": 0.5,
                "estimated_resolution": "2-3 business days",
                "financial_impact": False,
                "urgency_indicators": [],
                "requires_callback": False,
                "requires_human_review": False,
                "compliance_flags": [],
                "suggested_agent_skills": ["billing_support"],
                "reasoning": "Keyword-based fallback classification for payment"
            }
        else:
            return {
                "primary_category": "ambiguity_unclear_unclassified",
                "sub_category": None,
                "priority": "low",
                "sentiment": "neutral",
                "theme": "General Inquiry",
                "confidence_score": 0.4,
                "estimated_resolution": "2-3 business days",
                "financial_impact": False,
                "urgency_indicators": [],
                "requires_callback": False,
                "requires_human_review": True,
                "compliance_flags": [],
                "suggested_agent_skills": ["general_support"],
                "reasoning": "Unclear complaint - fallback classification"
            }
     
    def _apply_learning_weights(self, classification: Dict[str, Any], complaint_text: str) -> Dict[str, Any]:
        """Apply reinforcement learning weights to classification"""
        
        primary_category = classification["primary_category"]
        original_confidence = classification["confidence_score"]
        
        # Get learned weights for this category
        category_weights = self.classification_weights.get(primary_category, {
            "confidence_boost": 0.0,
            "confidence_penalty": 0.0,
            "successful_classifications": 0,
            "failed_classifications": 0
        })
        
        # Calculate success rate
        total_attempts = category_weights["successful_classifications"] + category_weights["failed_classifications"]
        success_rate = category_weights["successful_classifications"] / max(total_attempts, 1)
        
        # Adjust confidence based on learning
        confidence_adjustment = (category_weights["confidence_boost"] - category_weights["confidence_penalty"]) * 0.1
        success_rate_adjustment = (success_rate - 0.5) * 0.2
        
        adjusted_confidence = original_confidence + confidence_adjustment + success_rate_adjustment
        adjusted_confidence = max(0.1, min(0.99, adjusted_confidence))
        
        classification["confidence_score"] = adjusted_confidence
        classification["learning_applied"] = True
        classification["original_confidence"] = original_confidence
        
        return classification

    async def _contextual_complaint_classification(self, complaint_text: str, 
                                                 customer_context: Optional[Dict[str, Any]] = None,
                                                 attachments: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        FIXED: Intelligent complaint classification using Eva's AI capabilities
        Enhanced version for the complaint submission endpoint with better error handling
        """
        try:
            # Handle None values with safe defaults
            if customer_context is None:
                customer_context = {}
            if attachments is None:
                attachments = []
                
            # Enhanced prompt with customer context
            customer_info = ""
            if customer_context:
                customer_info = f"""
CUSTOMER CONTEXT:
- Customer Name: {customer_context.get('name', 'N/A')}
- Account Type: {customer_context.get('account_type', 'Standard')}
- Previous Issues: {len(customer_context.get('recent_complaints', []))} recent complaints
- Account Status: {customer_context.get('account_status', 'Active')}
"""
            
            attachment_info = ""
            if attachments:
                attachment_info = f"""
ATTACHMENTS PROVIDED: {len(attachments)} files
- File types: {', '.join([att.get('content_type', 'unknown') for att in attachments])}
"""
            
            prompt = f"""
Analyze this banking complaint and classify it contextually:

COMPLAINT: {complaint_text}
{customer_info}
{attachment_info}

CATEGORIES TO CHOOSE FROM:
{', '.join(self.complaint_categories)}

Analyze the CONTEXT and INTENT, not just keywords. Consider:
1. What is the customer's primary concern?
2. What department has the authority to resolve this?
3. What is the root cause vs symptoms?
4. What is the urgency level based on language and context?
5. Is there potential financial impact?

Respond with JSON:
{{
    "primary_category": "category_name",
    "sub_category": "specific_sub_issue_or_null",
    "priority": "low|medium|high",
    "sentiment": "positive|neutral|negative",
    "theme": "human_readable_theme",
    "confidence_score": 0.XX,
    "estimated_resolution": "time_estimate",
    "financial_impact": true_or_false,
    "urgency_indicators": ["list", "of", "urgent", "keywords"],
    "requires_callback": true_or_false,
    "requires_human_review": true_or_false,
    "compliance_flags": ["list_of_compliance_concerns"],
    "suggested_agent_skills": ["list", "of", "required", "skills"],
    "reasoning": "explanation of classification logic"
}}
"""
            
            response = await self._call_anthropic(prompt)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
                
                # Add processing metadata
                classification.update({
                    "processing_timestamp": datetime.now().isoformat(),
                    "processing_version": "eva_v2.0_fixed",
                    "customer_context_used": customer_context is not None,
                    "attachments_analyzed": len(attachments) if attachments else 0
                })
                
                return classification
            else:
                return self._fallback_classification(complaint_text)
                
        except Exception as e:
            print(f"Classification error: {e}")
            return self._fallback_classification(complaint_text)
    
    async def _classify_complaint_with_learning(self, message: str, 
                                              customer_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """FIXED: Classify complaint using contextual AI + reinforcement learning"""
        
        # Handle None customer_context
        if customer_context is None:
            customer_context = {}
        
        # Base classification using contextual analysis
        base_classification = await self._contextual_complaint_classification(message, customer_context)
        
        # Apply learning weights if available
        adjusted_classification = self._apply_learning_weights(base_classification, message)
        
        return adjusted_classification
    
    # ========================= CONTEXT & STORAGE METHODS ====================
    
    async def _store_conversation_context(self, context: ConversationContext):
        """FIXED: Store conversation context with proper database integration"""
        # Always cache in memory
        self.conversation_contexts[context.conversation_id] = context
        
        # Store in database if available
        if self.database_available and self.database_service:
            try:
                conversation_data = {
                    "conversation_id": context.conversation_id,
                    "customer_id": context.customer_id,
                    "customer_name": context.customer_name,
                    "messages": context.messages,
                    "ongoing_issues": context.ongoing_issues,
                    "specialist_assignments": context.specialist_assignments,
                    "emotional_state": context.emotional_state,
                    "classification_pending": context.classification_pending
                }
                
                success = await self.database_service.store_eva_conversation(conversation_data)
                if success:
                    print(f"✅ Conversation {context.conversation_id} saved to database")
                else:
                    print(f"⚠️ Failed to save conversation {context.conversation_id} to database")
                    
            except Exception as e:
                print(f"⚠️ Error storing conversation context: {e}")
                # Don't fail the entire operation if database storage fails
    
    async def _get_or_create_conversation_context(self, conversation_id: str, 
                                                 customer_context: Dict[str, Any]) -> ConversationContext:
        """Requirement 1: Get or create conversation context with database backing"""
        
        # First check in-memory cache
        if conversation_id in self.conversation_contexts:
            return self.conversation_contexts[conversation_id]
        
        # Then check database if available
        if self.database_available and self.database_service:
            try:
                stored_context = await self.database_service.get_eva_conversation(conversation_id)
                if stored_context:
                    # Reconstruct context from database
                    context = ConversationContext(
                        conversation_id=stored_context["conversation_id"],
                        customer_id=stored_context["customer_id"],
                        customer_name=stored_context["customer_name"],
                        messages=stored_context["messages"],
                        ongoing_issues=stored_context["ongoing_issues"],
                        specialist_assignments=stored_context["specialist_assignments"],
                        emotional_state=stored_context["emotional_state"],
                        classification_pending=stored_context.get("classification_pending")
                    )
                    # Cache in memory
                    self.conversation_contexts[conversation_id] = context
                    print(f"✅ Restored conversation {conversation_id} from database")
                    return context
            except Exception as e:
                print(f"⚠️ Failed to load conversation from database: {e}")
        
        # Create new context
        context = ConversationContext(
            conversation_id=conversation_id,
            customer_id=customer_context.get("customer_id", ""),
            customer_name=customer_context.get("name", "valued customer"),
            messages=[],
            ongoing_issues=[],
            specialist_assignments={},
            emotional_state="neutral"
        )
        
        # Add greeting if this is the first interaction
        if not context.messages:
            greeting = await self._generate_contextual_greeting(customer_context)
            greeting_message = {
                "role": "eva",
                "content": greeting,
                "timestamp": datetime.now().isoformat(),
                "is_greeting": True
            }
            context.messages.append(greeting_message)
        
        return context
    
    # ========================= RESPONSE GENERATION METHODS ====================

    async def _build_eva_prompt(self, message: str, context: ConversationContext,
                               emotional_analysis: Dict[str, Any], 
                               complaint_classification: Optional[Dict[str, Any]]) -> str:
        """Build comprehensive prompt for Eva"""
        
        # Get conversation history (last 5 exchanges)
        recent_messages = context.messages[-10:] if len(context.messages) > 10 else context.messages
        conversation_history = ""
        for msg in recent_messages:
            role = "Customer" if msg["role"] == "customer" else "Eva"
            conversation_history += f"{role}: {msg['content']}\n"
        
        # Emotional context
        emotion_context = ""
        if emotional_analysis["empathy_needed"]:
            emotion_context = f"\nCustomer appears {emotional_analysis['primary_emotion']} with {emotional_analysis['emotion_intensity']} intensity. Show appropriate empathy."
        
        # Complaint classification context
        classification_context = ""
        if complaint_classification:
            classification_context = f"""
            
COMPLAINT CLASSIFICATION DETECTED:
Primary Category: {complaint_classification.get('primary_category', 'Unknown')}
Secondary Category: {complaint_classification.get('sub_category', 'None')}
Priority: {complaint_classification.get('priority', 'medium')}
Confidence: {complaint_classification.get('confidence_score', 0):.2f}

You need to explain this classification to the customer and ask for confirmation.
"""
        
        return f"""
You are Eva, the personal relationship manager for Swiss Bank. You provide premium, personalized banking assistance with the warmth and expertise of a dedicated relationship banker.

CUSTOMER INFORMATION:
- Name: {context.customer_name}
- Customer ID: {context.customer_id}
- Conversation Context: This is an ongoing conversation

RECENT CONVERSATION:
{conversation_history}

CURRENT MESSAGE: {message}
{emotion_context}
{classification_context}

YOUR RESPONSE GUIDELINES:

1. CONVERSATION MEMORY: Always reference previous conversations and show complete understanding of ongoing issues.

2. NEXT STEPS FORMAT: Always structure your response with clear bullet points:
   **What I'm doing right now:**
   • [Immediate actions you're taking]
   
   **What happens next:**
   • [Timeline and next steps]
   
   **Your next actions:**
   • [What the customer should do]

3. EMOTIONAL INTELLIGENCE: 
   - If customer gives compliments: Accept gracefully with genuine appreciation
   - If customer is frustrated/angry: Show patience and understanding without being defensive
   - Always acknowledge their emotional state appropriately

4. HUMAN SPECIALISTS: When mentioning specialists, use real names with credentials:
   - Never say "our fraud team" - say "Sarah Chen, our Senior Fraud Investigator with 8 years of experience"
   - Include their specialty and success rate when relevant

5. NATURAL CONVERSATION: 
   - Be conversational and warm, not corporate or robotic
   - Use the customer's name naturally
   - Show personal investment in their success
   
If this is a complaint classification, explain the categories in customer-friendly language and ask for confirmation.

Respond as Eva with complete naturalness and professionalism.
"""
    
    async def _parse_eva_response(self, response: str, 
                                 complaint_classification: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse Eva's response and extract next steps (Requirement 2)"""
        
        # Extract specialist mentions for Requirement 5
        specialists_mentioned = []
        for category_specialists in self.specialist_names.values():
            for specialist in category_specialists:
                if specialist["name"] in response:
                    specialists_mentioned.append(specialist)
        
        # Extract next steps if they exist
        next_steps = []
        
        # Look for bullet point patterns
        bullet_patterns = [
            r"•\s*([^\n•]+)",
            r"-\s*([^\n-]+)", 
            r"\*\s*([^\n*]+)"
        ]
        
        for pattern in bullet_patterns:
            matches = re.findall(pattern, response)
            next_steps.extend([match.strip() for match in matches])
        
        return {
            "content": response,
            "next_steps": next_steps,
            "specialists_mentioned": specialists_mentioned,
            "requires_confirmation": complaint_classification is not None
        }
    
    async def _generate_eva_response(self, message: str, context: ConversationContext,
                                   emotional_analysis: Dict[str, Any], 
                                   complaint_classification: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate Eva's natural response with all requirements"""
        
        # Build context-aware prompt
        prompt = await self._build_eva_prompt(message, context, emotional_analysis, complaint_classification)
        
        try:
            response = await self._call_anthropic(prompt)
            
            # Parse response for next steps and specialist mentions
            parsed_response = await self._parse_eva_response(response, complaint_classification)
            
            return parsed_response
            
        except Exception as e:
            print(f"Error generating Eva response: {e}")
            return await self._generate_empathetic_fallback(emotional_analysis, context)
    
    # ========================= SPECIALIST & CONFIRMATION METHODS ====================
    
    def _get_specialist_for_category(self, primary_category: str, customer_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get specialist assignment for category with consistent assignment
        Place this method after _generate_triage_confirmation_response method
        """
        # Map category to specialist type
        if primary_category in self.specialist_names:
            specialists = self.specialist_names[primary_category]
        else:
            specialists = self.specialist_names.get("general", [
                {"name": "Chris Taylor", "title": "Customer Service Representative", "experience": "5 years"}
            ])
        
        # Use customer ID for consistent assignment if provided
        if customer_id and specialists:
            import hashlib
            index = int(hashlib.md5(customer_id.encode()).hexdigest(), 16) % len(specialists)
            return specialists[index]
        elif specialists:
            return specialists[0]
        else:
            # Ultimate fallback
            return {
                "name": "Customer Service Team",
                "title": "Customer Service Representative",
                "experience": "Available to assist"
            }

    def _assign_specialist_name(self, category: str, complaint_id: str) -> Dict[str, str]:
        """Assign consistent realistic specialist name (Requirement 5)"""
        
        if category not in self.specialist_names:
            # Fallback to general category
            category = "general"
        
        # Use complaint ID for consistent assignment
        specialists = self.specialist_names[category]
        index = int(hashlib.md5(complaint_id.encode()).hexdigest(), 16) % len(specialists)
        
        return specialists[index]
    
    async def _generate_triage_confirmation_response(self, triage_result: Dict[str, Any], 
                                                   customer_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate triage confirmation response for customer
        Place this method after _generate_fallback_response method
        """
        customer_name = customer_context.get("name", "valued customer")
        
        # Extract triage information
        if "triage_analysis" in triage_result:
            # New complaint from triage service
            analysis = triage_result["triage_analysis"]
            primary_category = analysis.get("primary_category", "general_inquiry")
            urgency_level = analysis.get("urgency_level", "medium")
            confidence = analysis.get("confidence_score", 0.8)
            estimated_resolution = analysis.get("estimated_resolution_time", "2-3 business days")
        else:
            # Fallback or Eva classification
            primary_category = triage_result.get("primary_category", "general_inquiry")
            urgency_level = triage_result.get("priority", "medium")
            confidence = triage_result.get("confidence_score", 0.8)
            estimated_resolution = triage_result.get("estimated_resolution_time", "2-3 business days")
        
        # Translate category to customer-friendly language
        friendly_category = self._translate_category_for_customer(primary_category)
        
        # Get specialist assignment
        specialist = self._get_specialist_for_category(primary_category, customer_context.get("customer_id"))
        
        # Generate confirmation response
        confirmation_response = f"""
        {customer_name}, I've just completed a thorough analysis of your situation with our specialist team.
        
        **Here's what our analysis shows:**
        
        **Issue Classification:** {friendly_category}
        **Priority Level:** {urgency_level.title()} priority
        **Confidence:** {confidence:.0%} match to our resolution protocols
        **Assigned Specialist:** {specialist['name']}, {specialist['title']} 
        **Expected Timeline:** {estimated_resolution}
        
        Based on your description, this looks like {friendly_category.lower()} that our {specialist['title'].lower()} handles regularly.
        
        **Does this assessment sound accurate to you?** 
        
        I want to make sure we're addressing exactly what you're experiencing before I outline our next steps.
        
        Also, is there anything important about this situation that I should know to help {specialist['name']} resolve this more effectively?
        """
        
        return {
            "response": confirmation_response,
            "triage_classification": {
                "category": primary_category,
                "urgency": urgency_level,
                "confidence": confidence,
                "specialist": specialist
            },
            "requires_confirmation": True,
            "awaiting_customer_response": True
        }
    
    # ========================= ACTION & FLOW METHODS ====================

    async def _request_clarification(self, context: ConversationContext, conversation_id: str) -> Dict[str, Any]:
        """NEW: Request clarification when customer response is unclear"""
        customer_name = context.customer_name
        
        clarification_response = f"""
        {customer_name}, I want to make sure I'm understanding your situation correctly. 
        
        Does the analysis I provided sound like it captures what you're experiencing? 
        Or would you like me to look at this differently?
        """
        
        return {
            "response": clarification_response,
            "conversation_id": conversation_id,
            "stage": "seeking_clarification",
            "awaiting_confirmation": True
        }
    
    async def _handle_classification_correction(self, message: str, context: ConversationContext, 
                                          conversation_id: str) -> Dict[str, Any]:
        """NEW: Handle when customer wants to correct classification"""
        customer_name = context.customer_name
        
        correction_response = f"""
        Thank you for that clarification, {customer_name}. I want to make sure I understand your situation exactly right.
        
        Could you help me understand what I might have missed or gotten wrong about your situation? 
        What would you say is the most important thing for me to focus on?
        """
        
        # Reset to follow-up questions stage
        self.conversation_states[conversation_id].update({
            "stage": "follow_up_questions",
            "questions_asked": 1,
            "max_questions": 2,
            "gathered_info": [{"question_number": 0, "response": message, "type": "correction"}]
        })
        
        return {
            "response": correction_response,
            "conversation_id": conversation_id,
            "stage": "classification_correction",
            "requires_clarification": True
        }
    
    async def _generate_realistic_action_response(self, action_type: str, 
                                                triage_results: Dict[str, Any],
                                                context: ConversationContext) -> str:
        """
        NEW: Generate realistic action response using banking policy constraints
        """
        # Extract complaint details
        if "triage_analysis" in triage_results:
            analysis = triage_results["triage_analysis"]
            complaint_type = analysis["primary_category"]
            urgency_level = analysis["urgency_level"]
            financial_impact = analysis.get("financial_impact", False)
            amount = analysis.get("estimated_financial_amount", "")
        else:
            complaint_type = triage_results.get("primary_category", "general_inquiry")
            urgency_level = triage_results.get("priority", "medium")
            financial_impact = triage_results.get("financial_impact", False)
            amount = triage_results.get("estimated_financial_amount", "")
        
        # Get realistic timelines for this complaint type
        timelines = self._get_realistic_timeline(complaint_type)
        
        # Build response prompt with banking constraints
        prompt = f"""
        Generate realistic banking response for "{action_type}" stage.
        
        Context:
        - Customer: {context.customer_name}
        - Complaint type: {complaint_type}
        - Urgency: {urgency_level}
        - Financial impact: {financial_impact}
        - Amount: {amount}
        - Realistic timelines: {timelines}
        
        BANKING CONSTRAINTS (CRITICAL):
        - NO instant refunds or money promises
        - Investigation required before any credits
        - Show process urgency, not money urgency
        - Use realistic timelines from provided data
        - Provisional credit is conditional, not guaranteed
        - Mention specific specialist names from our team
        
        Generate empathetic but realistic response for {action_type}.
        """
        
        return await self._call_anthropic(prompt)
    
    async def _regenerate_realistic_response(self, action_type: str, triage_results: Dict[str, Any],
                                       context: ConversationContext, violations: List[str]) -> str:
        """
        NEW: Regenerate response with stricter banking constraints
        """
        # Get realistic timelines
        if "triage_analysis" in triage_results:
            complaint_type = triage_results["triage_analysis"]["primary_category"]
        else:
            complaint_type = triage_results.get("primary_category", "general_inquiry")
        
        timelines = self._get_realistic_timeline(complaint_type)
        
        prompt = f"""
        Regenerate realistic banking response for "{action_type}".
        
        VIOLATIONS TO AVOID: {violations}
        
        Context: Customer {context.customer_name}
        Realistic timelines: {timelines}
        
        STRICT BANKING CONSTRAINTS:
        - NEVER promise instant money, refunds, or credits
        - Always mention investigation requirement
        - Use phrases like "review for provisional credit eligibility"
        - Show urgency in PROCESS, not MONEY
        - Be empathetic but honest about banking realities
        
        Generate response that shows immediate action on investigation/security but realistic expectations about money.
        """
        
        return await self._call_anthropic(prompt)
        
    async def _generate_action_step_2(self, conversation_id: str) -> Dict[str, Any]:
        """NEW: Generate second action message"""
        conversation_state = self.conversation_states[conversation_id]
        triage_results = conversation_state["triage_results"]
        context = self.conversation_contexts[conversation_id]
        
        action_2 = await self._generate_realistic_action_response(
            "what_happens_next", triage_results, context
        )
        
        conversation_state.update({
            "action_step": 2,
            "next_action_time": datetime.now() + timedelta(seconds=10)
        })
        
        return {
            "response": action_2,
            "conversation_id": conversation_id,
            "stage": "action_sequence_2",
            "next_message_in_seconds": 10
        }
    
    async def _generate_action_step_3(self, conversation_id: str) -> Dict[str, Any]:
        """NEW: Generate third action message"""
        conversation_state = self.conversation_states[conversation_id]
        triage_results = conversation_state["triage_results"]
        context = self.conversation_contexts[conversation_id]
        
        action_3 = await self._generate_realistic_action_response(
            "your_next_actions", triage_results, context
        )
        
        conversation_state.update({
            "action_step": 3,
            "stage": "action_complete"
        })
        
        return {
            "response": action_3,
            "conversation_id": conversation_id,
            "stage": "action_sequence_complete",
            "ready_for_questions": True
        }
    
    async def _complete_action_sequence(self, conversation_id: str) -> Dict[str, Any]:
        """NEW: Complete action sequence"""
        context = self.conversation_contexts[conversation_id]
        customer_name = context.customer_name
        
        completion_response = f"""
        {customer_name}, I've outlined everything we're doing to resolve your situation. 
        
        Your case is now in the hands of our specialist team, and you'll be hearing from them soon.
        
        Is there anything else about this situation that you'd like me to address or any other questions I can help you with right now?
        """
        
        # Reset conversation state for normal chat
        self.conversation_states[conversation_id] = {"stage": "normal_chat"}
        
        return {
            "response": completion_response,
            "conversation_id": conversation_id,
            "stage": "action_sequence_complete",
            "ready_for_normal_chat": True
        }
    
    async def _continue_action_sequence(self, conversation_id: str) -> Dict[str, Any]:
        """
        NEW: Continue with next part of action sequence
        """
        conversation_state = self.conversation_states[conversation_id]
        
        # Check if it's time for next message
        if datetime.now() < conversation_state.get("next_action_time", datetime.now()):
            return {
                "response": "Processing...",
                "conversation_id": conversation_id,
                "stage": "action_sequence_processing",
                "retry_in_seconds": 2
            }
        
        action_step = conversation_state["action_step"]
        
        if action_step == 1:
            # Generate "What happens next"
            return await self._generate_action_step_2(conversation_id)
        elif action_step == 2:
            # Generate "Your next actions"
            return await self._generate_action_step_3(conversation_id)
        else:
            # Action sequence complete
            return await self._complete_action_sequence(conversation_id)
    
    async def _initiate_realistic_action_sequence(self, conversation_id: str) -> Dict[str, Any]:
        """
        NEW: Initiate realistic action sequence with banking policy compliance
        """
        conversation_state = self.conversation_states[conversation_id]
        triage_results = conversation_state["triage_results"]
        context = self.conversation_contexts[conversation_id]
        
        # Generate first action response with banking policy compliance
        first_action = await self._generate_realistic_action_response(
            "what_doing_now", triage_results, context
        )
        
        # Validate response for realistic banking practices
        validation = self._validate_response_promises(first_action)
        
        if not validation["is_realistic"]:
            print(f"⚠️ Unrealistic response detected, regenerating...")
            # Regenerate with stricter constraints
            first_action = await self._regenerate_realistic_response(
                "what_doing_now", triage_results, context, validation["violations"]
            )
        
        # Update conversation state
        self.conversation_states[conversation_id].update({
            "stage": "action_sequence",
            "action_step": 1,
            "next_action_time": datetime.now() + timedelta(seconds=10)
        })
        
        return {
            "response": first_action,
            "conversation_id": conversation_id,
            "stage": "action_sequence_1",
            "next_message_in_seconds": 10,
            "banking_policy_validated": True
        }
    
    # ========================= FLOW CONTROL METHODS ===========================
    async def _initiate_follow_up_questions(self, context: ConversationContext, 
                                          conversation_id: str) -> Dict[str, Any]:
        """
        NEW: Start dynamic follow-up questions
        """
        conversation_state = self.conversation_states[conversation_id]
        triage_results = conversation_state["triage_results"]
        
        # Generate contextual follow-up questions using AI
        followup_prompt = f"""
        Customer {context.customer_name} confirmed the complaint classification.
        
        Generate 1-2 specific follow-up questions that would help investigate this case better.
        Base questions on the complaint type and customer's emotional state.
        
        Keep it conversational and empathetic, not interrogative.
        Focus on gathering helpful details for resolution.
        """
        
        followup_response = await self._call_anthropic(followup_prompt)
        
        # Update conversation state
        self.conversation_states[conversation_id].update({
            "stage": "follow_up_questions",
            "questions_asked": 1,
            "max_questions": 3,
            "gathered_info": []
        })
        
        return {
            "response": followup_response,
            "conversation_id": conversation_id,
            "stage": "follow_up_questions",
            "question_number": 1
        }
    
    async def _handle_follow_up_questions(self, message: str, context: ConversationContext, 
                                        conversation_id: str) -> Dict[str, Any]:
        """
        NEW: Handle follow-up question responses
        """
        conversation_state = self.conversation_states[conversation_id]
        
        # Store customer response
        conversation_state["gathered_info"].append({
            "question_number": conversation_state["questions_asked"],
            "response": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if more questions needed
        if (conversation_state["questions_asked"] < conversation_state["max_questions"] and 
            not self._customer_wants_to_proceed(message)):
            
            # Generate next question
            next_question = await self._generate_next_followup_question(conversation_state)
            conversation_state["questions_asked"] += 1
            
            return {
                "response": next_question,
                "conversation_id": conversation_id,
                "stage": "follow_up_questions",
                "question_number": conversation_state["questions_asked"]
            }
        else:
            # Move to action sequence
            return await self._initiate_realistic_action_sequence(conversation_id)
    
    async def _handle_triage_confirmation(self, message: str, context: ConversationContext, 
                                        conversation_id: str) -> Dict[str, Any]:
        """
        NEW: Handle customer's response to triage classification
        """
        confirmation_analysis = self._analyze_customer_confirmation(message)
        
        if confirmation_analysis["confirmed"]:
            # Customer confirms - proceed to follow-up questions
            return await self._initiate_follow_up_questions(context, conversation_id)
        elif confirmation_analysis["needs_correction"]:
            # Customer wants clarification or correction
            return await self._handle_classification_correction(message, context, conversation_id)
        else:
            # Need more clarification
            return await self._request_clarification(context, conversation_id)
    
    async def _present_triage_results(self, conversation_id: str) -> Dict[str, Any]:
        """
        NEW: Present triage results in natural, customer-friendly way
        """
        conversation_state = self.conversation_states[conversation_id]
        
        if conversation_state.get("stage") != "triage_results_ready":
            # Analysis still in progress
            return {
                "response": "I'm still analyzing your situation with our specialist team. This will just take another moment...",
                "conversation_id": conversation_id,
                "stage": "analysis_in_progress",
                "retry_in_seconds": 3
            }
        
        triage_results = conversation_state["triage_results"]
        context = self.conversation_contexts[conversation_id]
        customer_name = context.customer_name
        
        # Extract key information from triage results
        if "triage_analysis" in triage_results:
            # New complaint format from triage service
            analysis = triage_results["triage_analysis"]
            primary_category = analysis["primary_category"]
            urgency_level = analysis["urgency_level"]
            confidence = analysis.get("confidence_score", 0.8)
        else:
            # Eva classification format
            primary_category = triage_results.get("primary_category", "general_inquiry")
            urgency_level = triage_results.get("priority", "medium")
            confidence = triage_results.get("confidence_score", 0.8)
        
        # Generate natural presentation using AI
        presentation_prompt = f"""
        Present triage analysis results naturally to customer {customer_name}.
        
        Analysis Results:
        - Category: {self._translate_category_for_customer(primary_category)}
        - Urgency: {urgency_level}
        - Confidence: {confidence:.0%}
        
        Present this in customer-friendly language, ask for confirmation, and indicate what specialist team will handle this.
        
        Be conversational, not corporate. Show that analysis was thorough.
        Include a specialist name from our team.
        """
        
        presentation_response = await self._call_anthropic(presentation_prompt)
        
        # Update conversation state
        self.conversation_states[conversation_id].update({
            "stage": "triage_confirmation",
            "awaiting_customer_confirmation": True
        })
        
        return {
            "response": presentation_response,
            "conversation_id": conversation_id,
            "stage": "triage_confirmation_needed",
            "triage_classification": {
                "category": primary_category,
                "urgency": urgency_level,
                "confidence": confidence
            }
        }
    
    async def _run_background_triage_analysis(self, conversation_id: str, complaint_text: str, customer_id: str):
        """
        FIXED: Run triage analysis in background and properly update conversation state
        """
        try:
            print(f"🔍 Starting background triage analysis for conversation {conversation_id}")
            
            if not self.triage_service:
                print("⚠️ Triage service not available, using Eva classification")
                # Fall back to Eva's existing classification
                customer_context = await self.database_service.get_customer(customer_id) if self.database_service else {}
                triage_result = await self._classify_complaint_with_learning(complaint_text, customer_context)
            else:
                # Use proper triage service
                customer_context = await self.database_service.get_customer(customer_id) if self.database_service else {}
                complaint_data = {
                    "complaint_text": complaint_text,
                    "customer_id": customer_id,
                    "customer_context": customer_context,
                    "submission_timestamp": datetime.now().isoformat(),
                    "submission_method": "eva_chat"
                }
                
                print(f"🎯 Calling triage service for complaint: {complaint_text[:50]}...")
                triage_result = await self.triage_service.process_complaint(complaint_data)
                print(f"✅ Triage analysis complete: {triage_result.get('complaint_type', 'unknown')}")
            
            # 🔥 FIX: Ensure conversation_states exists and update it properly
            if conversation_id not in self.conversation_states:
                self.conversation_states[conversation_id] = {}
                
            # 🔥 CRITICAL FIX: Update state with results AND mark as ready
            self.conversation_states[conversation_id].update({
                "stage": "triage_results_ready",  # ✅ This is the key fix
                "triage_results": triage_result,
                "analysis_complete_time": datetime.now().isoformat(),
                "background_analysis_completed": True
            })
            
            print(f"✅ Background triage analysis complete for conversation {conversation_id}")
            print(f"🎯 State updated to: {self.conversation_states[conversation_id]['stage']}")
            print(f"🎯 Triage results keys: {list(triage_result.keys())}")
            
        except Exception as e:
            print(f"❌ Background triage analysis failed: {e}")
            # Set fallback state
            if conversation_id not in self.conversation_states:
                self.conversation_states[conversation_id] = {}
                
            self.conversation_states[conversation_id].update({
                "stage": "triage_analysis_failed",
                "error": str(e),
                "analysis_complete_time": datetime.now().isoformat()
            })      

    async def _handle_initial_complaint_with_triage(self, message: str, context: ConversationContext, 
                                                   conversation_id: str) -> Dict[str, Any]:
        """
        NEW: Handle initial complaint with background triage analysis
        """
        customer_name = context.customer_name
        
        # Step 1: Immediate empathetic response
        initial_response = await self._generate_empathetic_acknowledgment(message, customer_name)
        
        # Step 2: Show triage analysis starting
        analysis_message = f"""
        {initial_response}
        
        Let me immediately connect with our specialist analysis team to get a complete picture of what's happening. 
        I'm pulling up your account details and running this through our security and complaint classification protocols right now...
        
        """
        
        # Step 3: Update conversation state and start background triage
        self.conversation_states[conversation_id] = {
            "stage": "awaiting_triage_results",
            "complaint_text": message,
            "analysis_start_time": datetime.now(),
            "triage_initiated": True
        }
        
        # Store message in context using existing method
        context.messages.append({
            "role": "customer",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "complaint_detected": True
        })
        
        context.messages.append({
            "role": "eva",
            "content": analysis_message,
            "timestamp": datetime.now().isoformat(),
            "stage": "triage_analysis_initiated"
        })
        
        await self._store_conversation_context(context)
        
        # Trigger background triage analysis
        asyncio.create_task(self._run_background_triage_analysis(conversation_id, message, context.customer_id))
        
        return {
            "response": analysis_message,
            "conversation_id": conversation_id,
            "stage": "triage_analysis_initiated",
            "next_action": "await_triage_results",
            "background_processing": True
        }

    # ========================= LEARNING SYSTEM METHODS ====================
    
    async def _update_learning_weights(self, feedback: ClassificationFeedback):
        """FIXED: Update classification weights based on customer feedback"""
        
        primary_category = feedback.original_classification["primary_category"]
        reward_signal = self._analyze_customer_feedback(feedback.customer_response)["reward_signal"]
        
        if primary_category not in self.classification_weights:
            self.classification_weights[primary_category] = {
                "confidence_boost": 0.0,
                "confidence_penalty": 0.0,
                "successful_classifications": 0,
                "failed_classifications": 0
            }
        
        weights = self.classification_weights[primary_category]
        
        if reward_signal > 0:
            weights["confidence_boost"] += 0.1 * reward_signal
            weights["successful_classifications"] += 1
        else:
            weights["confidence_penalty"] += 0.1 * abs(reward_signal)
            weights["failed_classifications"] += 1
        
        # Save weights to database if available
        if self.database_available:
            await self._save_learning_weights_to_database()

    async def process_customer_feedback(self, complaint_id: str, customer_feedback: str,
                                      original_classification: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: Process customer confirmation/correction for reinforcement learning"""
        
        # Analyze feedback type
        feedback_analysis = self._analyze_customer_feedback(customer_feedback)
        
        # Create feedback record
        feedback_record = ClassificationFeedback(
            complaint_id=complaint_id,
            original_classification=original_classification,
            customer_response=customer_feedback,
            feedback_type=feedback_analysis["feedback_type"],
            learning_weight=feedback_analysis["learning_weight"],
            timestamp=datetime.now().isoformat()
        )
        
        # Update learning weights
        await self._update_learning_weights(feedback_record)
        
        # Store feedback for future training
        self.feedback_history.append(feedback_record)
        
        # Store in database if available
        if self.database_available and self.database_service:
            try:
                feedback_data = {
                    "complaint_id": complaint_id,
                    "customer_id": original_classification.get("customer_id"),
                    "original_classification": original_classification,
                    "customer_response": customer_feedback,
                    "feedback_type": feedback_analysis["feedback_type"],
                    "learning_weight": feedback_analysis["learning_weight"],
                    "confidence_adjustment": feedback_analysis.get("confidence_adjustment", 0)
                }
                
                await self.database_service.store_classification_feedback(feedback_data)
                print(f"✅ Feedback stored in database for complaint {complaint_id}")
                
            except Exception as e:
                print(f"⚠️ Failed to store feedback in database: {e}")
        
        return {
            "feedback_processed": True,
            "feedback_type": feedback_analysis["feedback_type"],
            "learning_applied": True,
            "confidence_adjustment": feedback_analysis.get("confidence_adjustment", 0)
        }

    # ========================= MAIN API METHODS ====================

    async def _format_response_with_bullets(self, response: str, context: ConversationContext) -> str:
        """Requirement 2: Ensure response follows bullet point format"""
        # If response doesn't already have bullets, add structure
        if "•" not in response and "**" not in response:
            lines = response.split('\n')
            if len(lines) > 1:
                # Format as bullet points
                formatted = lines[0] + "\n\n**Here's how I can help:**\n"
                for line in lines[1:]:
                    if line.strip():
                        formatted += f"• {line.strip()}\n"
                return formatted
        
        return response
    
    async def _generate_dynamic_followup_questions(self, complaint_classification: Dict, customer_context: Dict):
        # AI-generated questions based on complaint type and customer situation
        prompt = f"""
        Based on this fraud complaint classification:
        - Type: {complaint_classification['primary_category']}
        - Customer emotion: {complaint_classification['emotional_state']}
        - Amount: $2,500 (rent money - high urgency)
        
        Generate 1-2 specific follow-up questions to better understand:
        1. When/where the card was last used legitimately
        2. Any suspicious communications received
        3. Other potentially affected accounts
        
        Keep it conversational and empathetic, not interrogative.
        """   

    async def _generate_followup_response(self, customer_feedback: str, 
                                        feedback_result: Dict[str, Any], 
                                        customer_data: Dict[str, Any]) -> str:
        """Generate appropriate follow-up response based on feedback"""
        
        feedback_type = feedback_result["feedback_type"]
        customer_name = customer_data.get("name", "valued customer")
        
        if feedback_type == "confirmed":
            return f"""Perfect, {customer_name}! Thank you for confirming that understanding. I'm so glad we're on the same page about how to help you.

**Here's exactly what's happening next:**
• Your case is now being escalated to our specialist with high priority
• You'll receive a call within the next 2 hours with a detailed action plan
• I'm personally monitoring your case to ensure everything goes smoothly
• You'll get regular updates throughout the resolution process

Is there anything else you'd like me to clarify about the process?"""

        elif feedback_type == "major_correction":
            return f"""Thank you for that correction, {customer_name}. I really appreciate you taking the time to help me understand your situation better - that's exactly what helps us provide you with the right solution.

Let me ask a few quick questions so I can get this exactly right:

**What I need to understand better:**
• What would you say is your #1 priority for us to resolve?
• How would you describe the main problem in your own words?
• Is there anything important that I missed in my understanding?

Based on your clarification, I'll make sure we focus our efforts exactly where you need them most."""

        else:  # partial_correction or unclear
            return f"""Thank you for that feedback, {customer_name}. I want to make sure I get this completely right for you.

**Help me understand:**
• What's the most important thing for us to focus on first?
• What did I get right, and what needs adjustment?
• How would you prioritize the different issues you're facing?

I want to make sure our resolution plan addresses exactly what matters most to you, in the right order."""

    async def _get_next_steps_after_confirmation(self, feedback_type: str, 
                                               classification_data: Dict[str, Any]) -> List[str]:
        """Get next steps based on confirmation feedback"""
        
        if feedback_type == "confirmed":
            return [
                "Escalating to appropriate specialist team",
                "Creating priority case file with your details",
                "Specialist will contact you within 2 hours",
                "Regular progress updates will be provided"
            ]
        elif feedback_type == "major_correction":
            return [
                "Collecting additional clarification from you",
                "Re-analyzing your situation with corrected information", 
                "Routing to the correct specialist team",
                "Ensuring proper priority assignment"
            ]
        else:
            return [
                "Gathering more specific details about your priorities",
                "Refining our understanding of your situation",
                "Confirming the correct resolution approach",
                "Proceeding with accurate specialist assignment"
            ]
    
    async def eva_chat_response(self, message: str, customer_context: Dict[str, Any], 
                               conversation_id: str) -> Dict[str, Any]:
        """
        Main Eva chat response with all 5 requirements implemented
        FIXED VERSION with proper database integration
        """
        try:
            # Requirement 1: Conversation Memory Management (now with database backing)
            context = await self._get_or_create_conversation_context(
                conversation_id, customer_context
            )
            
            # Add customer message to context
            customer_message = {
                "role": "customer",
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "emotion": await self._analyze_emotion(message)
            }
            context.messages.append(customer_message)
            
            # Requirement 3: Emotional Intelligence
            emotional_analysis = await self._analyze_customer_emotion(message, context)
            
            # Check if this is a complaint that needs classification
            complaint_classification = None
            if await self._is_complaint(message):
                complaint_classification = await self._classify_complaint_with_learning(
                    message, customer_context
                )
            
            # Generate Eva's response
            eva_response = await self._generate_eva_response(
                message, context, emotional_analysis, complaint_classification
            )
            
            # Add Eva's response to context
            eva_message = {
                "role": "eva",
                "content": eva_response["content"],
                "timestamp": datetime.now().isoformat(),
                "next_steps": eva_response.get("next_steps", []),
                "specialists_mentioned": eva_response.get("specialists_mentioned", [])
            }
            context.messages.append(eva_message)
            
            # FIXED: Store updated context with proper database integration
            await self._store_conversation_context(context)
            
            return {
                "response": eva_response["content"],
                "conversation_id": conversation_id,
                "emotional_state": emotional_analysis["primary_emotion"],
                "classification_pending": complaint_classification,
                "requires_confirmation": complaint_classification is not None
            }
            
        except Exception as e:
            print(f"Error in eva_chat_response: {e}")
            return {
                "response": await self._generate_fallback_response(customer_context),
                "conversation_id": conversation_id,
                "error": str(e)
            }

    async def _generate_contextual_followup_question(self, gathered_info: List[Dict[str, Any]], 
                                               triage_results: Dict[str, Any]) -> str:
        """
        NEW: Generate contextual follow-up questions based on complaint type and previous responses
        """
        try:
            # Extract complaint details
            if "triage_analysis" in triage_results:
                analysis = triage_results["triage_analysis"]
                primary_category = analysis["primary_category"]
                emotional_state = analysis.get("emotional_state", "neutral")
                financial_impact = analysis.get("financial_impact", False)
            else:
                primary_category = triage_results.get("primary_category", "general")
                emotional_state = triage_results.get("emotional_state", "neutral")
                financial_impact = triage_results.get("financial_impact", False)
            
            # Get previous responses for context
            previous_responses = [info["response"] for info in gathered_info]
            questions_asked = len(gathered_info)
            
            # Generate category-specific questions
            if primary_category == "fraudulent_activities_unauthorized_transactions":
                return await self._generate_fraud_followup_question(previous_responses, questions_asked, emotional_state)
            elif primary_category == "dispute_resolution_issues":
                return await self._generate_dispute_followup_question(previous_responses, questions_asked)
            elif primary_category == "account_freezes_holds_funds":
                return await self._generate_account_access_followup_question(previous_responses, questions_asked)
            elif primary_category == "online_banking_technical_security_issues":
                return await self._generate_technical_followup_question(previous_responses, questions_asked)
            else:
                return await self._generate_general_followup_question(previous_responses, questions_asked, financial_impact)
                
        except Exception as e:
            print(f"Error generating contextual follow-up: {e}")
            return "Is there anything else about this situation that you think would be helpful for our investigation team to know?"

    async def _generate_fraud_followup_question(self, previous_responses: List[str], 
                                            questions_asked: int, emotional_state: str) -> str:
        """
        Generate fraud-specific follow-up questions
        """
        # Empathetic approach for fraud cases
        empathy_prefix = ""
        if emotional_state in ["anxious", "frustrated", "angry"]:
            empathy_prefix = "I understand this is very stressful. "
        
        fraud_questions = [
            f"{empathy_prefix}When did you last use your card legitimately, and do you still have it in your possession?",
            f"{empathy_prefix}Have you received any suspicious emails, texts, or phone calls recently asking for your banking information?",
            "Have you noticed any other unusual activity on any of your other accounts or cards?",
            "Did you make any online purchases or share your card information anywhere in the days leading up to this charge?",
            "Have you reported this to the police yet, or would you like guidance on whether that's necessary?"
        ]
        
        if questions_asked < len(fraud_questions):
            return fraud_questions[questions_asked]
        else:
            return "Is there anything else about this fraudulent activity that might help our investigation?"

    async def _generate_dispute_followup_question(self, previous_responses: List[str], 
                                                questions_asked: int) -> str:
        """
        Generate dispute-specific follow-up questions
        """
        dispute_questions = [
            "Do you have any receipts, confirmation emails, or other documentation related to this transaction?",
            "Did you attempt to resolve this directly with the merchant first? If so, what was their response?",
            "What outcome are you hoping for - a refund, exchange, or something else?",
            "How long ago did this transaction occur, and when did you first notice the issue?",
            "Have you disputed transactions with this merchant before?"
        ]
        
        if questions_asked < len(dispute_questions):
            return dispute_questions[questions_asked]
        else:
            return "Is there any other information about this dispute that would help us resolve it?"

    async def _generate_account_access_followup_question(self, previous_responses: List[str], 
                                                    questions_asked: int) -> str:
        """
        Generate account access follow-up questions
        """
        access_questions = [
            "When did you last successfully access your account, and what were you trying to do when it became inaccessible?",
            "Have you received any notifications from us about security concerns or required actions on your account?",
            "Are you able to access your account through other channels (mobile app, phone, ATM)?",
            "Have there been any recent changes to your contact information, address, or employment status?",
            "Do you have any upcoming payments or bills that depend on access to this account?"
        ]
        
        if questions_asked < len(access_questions):
            return access_questions[questions_asked]
        else:
            return "Is there anything else about your account access issue that we should be aware of?"

    async def _generate_technical_followup_question(self, previous_responses: List[str], 
                                                questions_asked: int) -> str:
        """
        Generate technical issue follow-up questions
        """
        technical_questions = [
            "What device and browser are you using, and when did this technical issue first occur?",
            "Are you getting any specific error messages? If so, what exactly do they say?",
            "Have you tried clearing your browser cache or using a different browser or device?",
            "Is this affecting all features of online banking or just specific functions?",
            "Are other people in your household able to access their banking without issues?"
        ]
        
        if questions_asked < len(technical_questions):
            return technical_questions[questions_asked]
        else:
            return "Are there any other technical details about this issue that might help our IT team resolve it?"

    async def _generate_general_followup_question(self, previous_responses: List[str], 
                                                questions_asked: int, financial_impact: bool) -> str:
        """
        Generate general follow-up questions
        """
        general_questions = [
            "Can you provide more details about when this issue started and what specific problems you're experiencing?",
            "Have you tried any steps to resolve this on your own, and if so, what happened?",
            "How is this issue affecting your day-to-day banking needs?",
            "Do you have any documentation or reference numbers related to this issue?",
            "What would be the ideal resolution for you?"
        ]
        
        # Add financial impact question if applicable
        if financial_impact and questions_asked == 2:
            return "Can you help me understand the financial impact this is having on you?"
        
        if questions_asked < len(general_questions):
            return general_questions[questions_asked]
        else:
            return "Is there anything else about your situation that you think would be important for us to know?"

    async def _handle_follow_up_questions_enhanced(self, message: str, context: ConversationContext, 
                                         conversation_id: str) -> Dict[str, Any]:
        """
        FIXED: Handle follow-up questions without duplication
        """
        conversation_state = self.conversation_states[conversation_id]
        customer_name = context.customer_name
        
        # ✅ NEW: Check if this is the first response after triage confirmation
        first_question_included = conversation_state.get("first_question_included_in_response", False)
        current_questions_asked = conversation_state.get("questions_asked", 0)
        
        # If first question was included in the triage confirmation response, this is actually question 1
        if first_question_included and current_questions_asked == 0:
            actual_question_number = 1
            # Clear the flag and update the count
            conversation_state["first_question_included_in_response"] = False
            conversation_state["questions_asked"] = 1
        else:
            actual_question_number = current_questions_asked

        # Store customer response to the current question
        if "gathered_additional_info" not in conversation_state:
            conversation_state["gathered_additional_info"] = []
        
        conversation_state["gathered_additional_info"].append({
            "question_number": actual_question_number,
            "response": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if customer indicates they're done
        completion_indicators = [
            "nothing else", "that's all", "no more", "that's everything", 
            "i think that covers it", "nothing additional", "that's it",
            "that covers everything", "nothing more"
        ]
        
        customer_done = any(indicator in message.lower() for indicator in completion_indicators)
        max_questions = conversation_state.get("max_questions", 3)
        
        # Check if we should continue or wrap up
        if customer_done or actual_question_number >= max_questions:
            # Complete follow-up phase
            completion_message = f"""Thank you, {customer_name}. I believe we have everything needed for this complaint. Our investigation team has access to your account transactions, documents, and all the details you've provided.

    If we need any further clarification during the investigation, we'll contact you directly.

    Let me now outline exactly what we're doing to resolve your situation..."""

            # Update state to move to action sequence
            self.conversation_states[conversation_id].update({
                "stage": "follow_up_complete",
                "ready_for_action_sequence": True
            })
            
            return {
                "response": completion_message,
                "conversation_id": conversation_id,
                "stage": "follow_up_complete"
            }
        else:
            # Ask next question
            next_question_number = actual_question_number + 1
            next_question = self._get_question_by_number(
                next_question_number, 
                conversation_state.get("current_question_category", "general"),
                conversation_state["gathered_additional_info"]
            )
            
            # Update questions asked count
            conversation_state["questions_asked"] = next_question_number
            
            # ✅ CRITICAL FIX: Keep the stage as "follow_up_questions_active" to prevent re-triggering
            return {
                "response": next_question,
                "conversation_id": conversation_id,
                "stage": "follow_up_questions_active",  # ✅ Changed from "follow_up_questions" to "follow_up_questions_active"
                "question_number": next_question_number
            }
        
    def _get_question_by_number(self, question_number: int, category: str, previous_responses: List[Dict]) -> str:
        """
        Get specific follow-up question by number and category
        """
        if category == "fraudulent_activities_unauthorized_transactions":
            fraud_questions = {
                2: "Have you received any suspicious emails, texts, or phone calls recently asking for your banking information?",
                3: "Have you noticed any other unusual activity on any of your other accounts or cards?"
            }
            return fraud_questions.get(question_number, "Is there anything else about this fraudulent activity that might help our investigation?")
        
        elif category == "dispute_resolution_issues":
            dispute_questions = {
                2: "Do you have any receipts, confirmation emails, or other documentation related to this transaction?",
                3: "Did you attempt to resolve this directly with the merchant first? If so, what was their response?"
            }
            return dispute_questions.get(question_number, "Is there any other information about this dispute that would help us resolve it?")
        
        else:
            general_questions = {
                2: "Have you tried any steps to resolve this on your own, and if so, what happened?",
                3: "How is this issue affecting your day-to-day banking needs?"
            }
            return general_questions.get(question_number, "Is there anything else about your situation that you think would be important for us to know?")

    async def _generate_contextual_followup_question_by_number(self, question_number: int,
                                                            gathered_info: List[Dict[str, Any]], 
                                                            triage_results: Dict[str, Any]) -> str:
        """
        Generate follow-up questions based on question number and category
        """
        # Extract complaint details
        if "triage_analysis" in triage_results:
            analysis = triage_results["triage_analysis"]
            primary_category = analysis["primary_category"]
            emotional_state = analysis.get("emotional_state", "neutral")
        else:
            primary_category = triage_results.get("primary_category", "general")
            emotional_state = triage_results.get("emotional_state", "neutral")
        
        # Get previous responses for context
        previous_responses = [info["response"] for info in gathered_info]
        
        # Generate category-specific questions
        if primary_category == "fraudulent_activities_unauthorized_transactions":
            return await self._generate_fraud_followup_question_by_number(
                question_number, previous_responses, emotional_state
            )
        elif primary_category == "dispute_resolution_issues":
            return await self._generate_dispute_followup_question_by_number(
                question_number, previous_responses
            )
        else:
            return await self._generate_general_followup_question_by_number(
                question_number, previous_responses
            )

    async def _generate_fraud_followup_question_by_number(self, question_number: int, 
                                                        previous_responses: List[str], 
                                                        emotional_state: str) -> str:
        """
        Generate fraud-specific follow-up questions by number
        """
        empathy_prefix = ""
        if emotional_state in ["anxious", "frustrated", "angry"]:
            empathy_prefix = "I understand this is very stressful. "
        
        fraud_questions = {
            2: f"{empathy_prefix}Have you received any suspicious emails, texts, or phone calls recently asking for your banking information?",
            3: "Have you noticed any other unusual activity on any of your other accounts or cards?",
        }
        
        return fraud_questions.get(question_number, "Is there anything else about this fraudulent activity that might help our investigation?")

    async def _generate_dispute_followup_question_by_number(self, question_number: int, 
                                                        previous_responses: List[str]) -> str:
        """
        Generate dispute-specific follow-up questions by number
        """
        dispute_questions = {
            2: "Do you have any receipts, confirmation emails, or other documentation related to this transaction?",
            3: "Did you attempt to resolve this directly with the merchant first? If so, what was their response?",
        }
        
        return dispute_questions.get(question_number, "Is there any other information about this dispute that would help us resolve it?")

    async def _generate_general_followup_question_by_number(self, question_number: int, 
                                                        previous_responses: List[str]) -> str:
        """
        Generate general follow-up questions by number
        """
        general_questions = {
            2: "Have you tried any steps to resolve this on your own, and if so, what happened?",
            3: "How is this issue affecting your day-to-day banking needs?",
        }
        
        return general_questions.get(question_number, "Is there anything else about your situation that you think would be important for us to know?")

    async def _check_sufficient_complaint_data(self, conversation_id: str) -> bool:
        """
        NEW: Check if we have sufficient data for the complaint
        """
        # For MVP, assume we have transaction and document access
        # In future, this will check actual data availability
        conversation_state = self.conversation_states[conversation_id]
        questions_asked = conversation_state.get("questions_asked", 0)
        
        # Consider sufficient if we've asked at least 2 questions
        return questions_asked >= 2

    async def _pass_additional_context_to_triage(self, conversation_id: str):
        """
        NEW: Pass collected additional info to triage agent
        """
        conversation_state = self.conversation_states[conversation_id]
        additional_info = conversation_state.get("gathered_additional_info", [])
        
        if self.triage_service and additional_info:
            try:
                context_data = {
                    "conversation_id": conversation_id,
                    "additional_context": additional_info,
                    "context_type": "follow_up_details",
                    "timestamp": datetime.now().isoformat()
                }
                
                # This method needs to be added to triage service
                await self.triage_service.update_complaint_with_additional_context(context_data)
                print(f"✅ Additional context passed to triage for conversation {conversation_id}")
                
            except Exception as e:
                print(f"⚠️ Failed to pass additional context to triage: {e}")

    async def _initiate_sequential_action_messages(self, conversation_id: str) -> Dict[str, Any]:
        """
        Start sequential action messages after follow-up is complete
        """
        conversation_state = self.conversation_states[conversation_id]
        context = self.conversation_contexts[conversation_id]
        triage_results = conversation_state["triage_results"]
        
        # Generate first action message using the realistic response generator
        first_action = await self._generate_realistic_action_response(
            "what_doing_now", 
            triage_results, 
            context
        )
        
        # Update state for sequential action messages
        self.conversation_states[conversation_id].update({
            "stage": "action_sequence",
            "action_step": 1,
            "next_action_time": datetime.now() + timedelta(seconds=10)
        })
        
        return {
            "response": first_action,
            "conversation_id": conversation_id,
            "stage": "action_sequence_1",
            "next_message_in_seconds": 10,
            "sequential_messages_active": True
        }

    async def _continue_action_sequence_with_timing(self, conversation_id: str) -> Dict[str, Any]:
        """
        UPDATED: Continue action sequence with voice-aware timing
        """
        conversation_state = self.conversation_states[conversation_id]
        
        # Check if it's time for next message (10-second delay for voice-disabled users)
        if datetime.now() < conversation_state.get("next_action_time", datetime.now()):
            return {
                "response": "Processing next steps...",
                "conversation_id": conversation_id,
                "stage": "action_sequence_processing",
                "retry_in_seconds": 2
            }
        
        action_step = conversation_state["action_step"]
        context = self.conversation_contexts[conversation_id]
        triage_results = conversation_state["triage_results"]
        
        if action_step == 1:
            # Generate "What happens next"
            action_2 = await self._generate_realistic_action_response(
                "what_happens_next", triage_results, context
            )
            
            conversation_state.update({
                "action_step": 2,
                "next_action_time": datetime.now() + timedelta(seconds=10)
            })
            
            return {
                "response": action_2,
                "conversation_id": conversation_id,
                "stage": "action_sequence_2",
                "next_message_in_seconds": 10
            }
            
        elif action_step == 2:
            # Generate "Your next actions"
            action_3 = await self._generate_realistic_action_response(
                "your_next_actions", triage_results, context
            )
            
            conversation_state.update({
                "action_step": 3,
                "stage": "action_complete"
            })
            
            return {
                "response": action_3,
                "conversation_id": conversation_id,
                "stage": "action_sequence_complete",
                "ready_for_normal_chat": True
            }
        else:
            # Sequence complete
            return await self._complete_action_sequence(conversation_id)

    async def eva_chat_response_with_natural_flow(self, message: str, customer_context: Dict[str, Any], 
                                         conversation_id: str) -> Dict[str, Any]:
        """
        FIXED: Enhanced Eva chat with proper triage confirmation flow - no duplicates
        """
        try:
            print(f"🎯 NATURAL FLOW METHOD CALLED: {message[:30]}...")
            context = await self._get_or_create_conversation_context(conversation_id, customer_context)
            conversation_state = self.conversation_states.get(conversation_id, {"stage": "initial"})
            
            print(f"🎯 CONVERSATION STATE: {conversation_state}") 
            print(f"🔍 IS COMPLAINT: {await self._is_complaint(message)}")

            # Route based on current stage
            if conversation_state["stage"] == "initial" and await self._is_complaint(message):
                print("🎯 COMPLAINT DETECTED - STARTING TRIAGE")
                return await self._handle_initial_complaint_with_triage(message, context, conversation_id)
            
            elif conversation_state["stage"] == "awaiting_triage_results":
                print("🎯 CHECKING TRIAGE RESULTS...")
                # Check if background analysis updated the state to ready
                current_state = self.conversation_states.get(conversation_id, {})
                if current_state.get("stage") == "triage_results_ready":
                    print("🎯 TRIAGE RESULTS READY - PRESENTING")
                    return await self._present_triage_for_confirmation(conversation_id)
                else:
                    print("🎯 TRIAGE STILL PROCESSING...")
                    return {
                        "response": "I'm still analyzing your situation with our specialist team. This will just take another moment...",
                        "conversation_id": conversation_id,
                        "stage": "analysis_in_progress",
                        "retry_in_seconds": 3
                    }
            
            # Direct check for triage_results_ready stage
            elif conversation_state["stage"] == "triage_results_ready":
                print("🎯 DIRECT TRIAGE RESULTS READY - PRESENTING")
                return await self._present_triage_for_confirmation(conversation_id)
            
            # Triage confirmation stage
            elif conversation_state["stage"] == "triage_confirmation_pending":
                return await self._handle_triage_confirmation_response(message, context, conversation_id)
            
            # ✅ CRITICAL FIX: Only handle follow_up_questions_active stage to prevent double handling
            elif conversation_state["stage"] == "follow_up_questions_active":
                print("🎯 HANDLING FOLLOW-UP QUESTION RESPONSE")
                return await self._handle_follow_up_questions_enhanced(message, context, conversation_id)
            
            # Check if follow-up is complete
            elif conversation_state["stage"] == "follow_up_complete":
                print("🎯 FOLLOW-UP COMPLETE - STARTING ACTION SEQUENCE")
                return await self._initiate_sequential_action_messages(conversation_id)
            
            elif conversation_state["stage"] == "action_sequence":
                return await self._continue_action_sequence(conversation_id)
            
            else:
                print("🎯 HANDLING AS NORMAL CONVERSATION")
                return await self.eva_chat_response(message, customer_context, conversation_id)
                
        except Exception as e:
            print(f"❌ Error in enhanced Eva flow: {e}")
            fallback_response = await self._generate_fallback_response(customer_context)
            return {
                "response": fallback_response,
                "conversation_id": conversation_id,
                "error": "Eva processing error - fallback response provided"
            }
    
    async def _present_triage_for_confirmation(self, conversation_id: str) -> Dict[str, Any]:
        """
        FIXED: Present triage results with proper label and reasoning for customer confirmation
        """
        conversation_state = self.conversation_states[conversation_id]
        
        # 🔥 FIX: More flexible state checking
        current_stage = conversation_state.get("stage")
        print(f"🎯 _present_triage_for_confirmation called with stage: {current_stage}")
        
        if current_stage != "triage_results_ready":
            print(f"⚠️ Triage results not ready, current stage: {current_stage}")
            
            # If we don't have results yet, check if we have triage_results data anyway
            if "triage_results" not in conversation_state:
                return {
                    "response": "I'm still analyzing your situation with our triage team. Just a moment more...",
                    "conversation_id": conversation_id,
                    "stage": "analysis_in_progress",
                    "retry_in_seconds": 3
                }
            # If we have results but stage isn't right, proceed anyway
            print("🎯 Found triage_results data, proceeding with presentation...")
        
        triage_results = conversation_state["triage_results"]
        context = self.conversation_contexts[conversation_id]
        customer_name = context.customer_name
        
        print(f"🎯 Presenting triage results for customer {customer_name}")
        
        # Extract triage details based on result format
        if "triage_analysis" in triage_results:
            # New complaint from triage service
            analysis = triage_results["triage_analysis"]
            primary_category = analysis["primary_category"]
            secondary_category = analysis.get("secondary_category", "High-priority financial impact case")
            reasoning = analysis.get("reasoning", "Based on content analysis and urgency indicators")
            confidence = analysis.get("confidence_scores", {}).get(primary_category, 0.8)
        else:
            # Eva classification format
            primary_category = triage_results.get("primary_category", "general_inquiry")
            secondary_category = triage_results.get("secondary_category", "General banking concern")
            reasoning = triage_results.get("reasoning", "Based on content analysis")
            confidence = triage_results.get("confidence_score", 0.8)
        
        # Translate to customer-friendly language
        friendly_category = self._translate_category_for_customer(primary_category)
        
        # Generate confirmation message with proper format
        confirmation_message = f"""{customer_name}, I've completed my analysis with our triage team. Here's what we determined:

    **Complaint Classification:**
    - **Primary Category:** {friendly_category}
    - **Secondary Category:** {secondary_category}
    - **Confidence Level:** {confidence:.0%}

    **Why we labeled it this way:**
    {reasoning}

    **Does this assessment accurately capture your situation?** Please let me know if this sounds right or if I need to adjust my understanding before we proceed with the resolution steps."""
        
        # 🔥 FIX: Update conversation state to await confirmation
        self.conversation_states[conversation_id].update({
            "stage": "triage_confirmation_pending",
            "awaiting_customer_confirmation": True,
            "triage_presented_at": datetime.now().isoformat()
        })
        
        print(f"✅ Triage results presented to customer, awaiting confirmation")
        
        return {
            "response": confirmation_message,
            "conversation_id": conversation_id,
            "stage": "triage_confirmation_pending",
            "requires_confirmation": True
        }
    
    async def _pass_confirmed_triage_to_orchestrator(self, conversation_id: str):
        """
        NEW: Pass confirmed triage results to orchestrator system
        """
        try:
            conversation_state = self.conversation_states[conversation_id]
            triage_results = conversation_state["triage_results"]
            context = self.conversation_contexts[conversation_id]
            
            # Generate orchestrator alert for confirmed triage
            orchestrator_alert = {
                "alert_type": "TRIAGE_CONFIRMED_BY_CUSTOMER",
                "alert_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,
                "customer_id": context.customer_id,
                "priority": "HIGH",
                "triage_confirmation": {
                    "customer_confirmed": True,
                    "confirmation_timestamp": datetime.now().isoformat(),
                    "original_classification": triage_results.get("triage_analysis", triage_results),
                    "customer_name": context.customer_name
                },
                "routing_instructions": {
                    "immediate_action": "CREATE_INVESTIGATION_QUEUE_ENTRY",
                    "assign_tracking_id": True,
                    "priority_level": triage_results.get("triage_analysis", {}).get("urgency_level", "medium"),
                    "department": self._determine_department_from_category(
                        triage_results.get("triage_analysis", {}).get("primary_category", "general")
                    )
                },
                "orchestrator_actions": [
                    "Generate unique tracking ID for complaint",
                    "Create investigation queue entry",
                    "Route to appropriate department specialist",
                    "Set SLA timers based on urgency level",
                    "Monitor case progress and escalation triggers"
                ],
                "case_details": {
                    "complaint_text": conversation_state.get("complaint_text", ""),
                    "urgency_level": triage_results.get("triage_analysis", {}).get("urgency_level", "medium"),
                    "financial_impact": triage_results.get("triage_analysis", {}).get("financial_impact", False),
                    "estimated_amount": triage_results.get("triage_analysis", {}).get("estimated_financial_amount"),
                    "customer_emotional_state": triage_results.get("triage_analysis", {}).get("emotional_state", "neutral")
                }
            }
            
            # Use triage service to send alert if available
            if self.triage_service:
                self.triage_service.orchestrator_alerts.append(orchestrator_alert)
                
            # Store in database if available
            if self.database_available and self.database_service:
                try:
                    await self.database_service.store_orchestrator_alert(orchestrator_alert)
                    print(f"✅ Confirmed triage passed to orchestrator: {orchestrator_alert['alert_id']}")
                except Exception as e:
                    print(f"⚠️ Failed to store orchestrator alert: {e}")
            
            # Update conversation state with tracking info
            tracking_id = f"TRK_{conversation_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M')}"
            conversation_state.update({
                "orchestrator_tracking_id": tracking_id,
                "case_status": "ROUTED_TO_INVESTIGATION",
                "orchestrator_notified_at": datetime.now().isoformat()
            })
            
            print(f"🚨 ORCHESTRATOR ALERT: Customer confirmed triage - Case {tracking_id} created")
            
        except Exception as e:
            print(f"❌ Error passing confirmed triage to orchestrator: {e}")

    def _determine_department_from_category(self, category: str) -> str:
        """
        Helper: Determine department from complaint category
        """
        department_mapping = {
            "fraudulent_activities_unauthorized_transactions": "fraud_investigation",
            "dispute_resolution_issues": "dispute_resolution", 
            "mortgage_related_issues": "mortgage_services",
            "account_freezes_holds_funds": "account_security",
            "online_banking_technical_security_issues": "technical_support",
            "poor_customer_service_communication": "customer_relations",
            "credit_card_issues": "credit_services",
            "overdraft_issues": "account_services"
        }
    
        return department_mapping.get(category, "general_customer_service")

    async def _handle_triage_confirmation_response(self, message: str, context: ConversationContext, 
                                            conversation_id: str) -> Dict[str, Any]:
        """
        FIXED: Handle customer's response to triage confirmation - no duplicates
        """
        confirmation_analysis = self._analyze_customer_confirmation(message)
        customer_name = context.customer_name
        
        if confirmation_analysis["confirmed"]:
            # Customer confirms - pass to orchestrator and generate complete response
            await self._pass_confirmed_triage_to_orchestrator(conversation_id)
            
            # Get triage results for context
            conversation_state = self.conversation_states[conversation_id]
            triage_results = conversation_state["triage_results"]
            
            # Generate tracking ID
            tracking_id = f"TRK_{conversation_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M')}"
            
            # Extract triage details for appropriate question
            if "triage_analysis" in triage_results:
                analysis = triage_results["triage_analysis"]
                primary_category = analysis["primary_category"]
                emotional_state = analysis.get("emotional_state", "neutral")
            else:
                primary_category = triage_results.get("primary_category", "general")
                emotional_state = triage_results.get("emotional_state", "neutral")
            
            # Generate first follow-up question based on category
            first_question = self._get_first_question_by_category(primary_category, emotional_state)
            
            # Create complete response with status AND first question
            complete_response = f"""Perfect, {customer_name}! I've immediately escalated your case to our orchestrator system.

    **Current Status:** Your complaint has been routed and is now in the investigation queue with a tracking ID: {tracking_id}.

    Now, let me gather some additional details to help our investigation team resolve this more effectively.

    {first_question}"""

            # 🔥 CRITICAL FIX: Set questions_asked to 0, not 1, because the first question is included in this response
            self.conversation_states[conversation_id].update({
                "stage": "follow_up_questions_active",  # Different stage name to prevent re-triggering
                "questions_asked": 0,  # ✅ FIXED: Changed from 1 to 0 since first question is in the response above
                "max_questions": 3,
                "gathered_additional_info": [],
                "orchestrator_notified": True,
                "current_question_category": primary_category,
                "first_question_included_in_response": True  # Flag to track this
            })
            
            return {
                "response": complete_response,
                "conversation_id": conversation_id,
                "stage": "follow_up_questions_active",
                "question_number": 1,  # This is the question number being asked
                "first_question_included": True
            }
            
        elif confirmation_analysis["needs_correction"]:
            # Customer wants correction
            correction_message = f"""Thank you for that clarification, {customer_name}. Let me understand your situation better.

    Could you help me correct my understanding? What would you say is the main issue you're facing, and what's the most important thing for us to focus on?"""

            return {
                "response": correction_message,
                "conversation_id": conversation_id,
                "stage": "triage_correction_needed"
            }
        else:
            # Need more clarification
            clarification_message = f"""I want to make sure I understand your situation correctly, {customer_name}. 

    Could you confirm whether my assessment captures what you're experiencing, or would you like me to look at this differently?"""
            
            return {
                "response": clarification_message,
                "conversation_id": conversation_id,
                "stage": "triage_confirmation_pending"
            }

    def _get_first_question_by_category(self, primary_category: str, emotional_state: str) -> str:
        """
        Get the first follow-up question based on complaint category
        """
        # Add empathy prefix for stressed customers
        empathy_prefix = ""
        if emotional_state in ["anxious", "frustrated", "angry"]:
            empathy_prefix = "I understand this is very stressful. "
        
        # Category-specific first questions
        if primary_category == "fraudulent_activities_unauthorized_transactions":
            return f"{empathy_prefix}When did you last use your card legitimately, and do you still have it in your possession?"
        
        elif primary_category == "dispute_resolution_issues":
            return f"{empathy_prefix}Can you tell me more about when this transaction occurred and what you expected to happen instead?"
        
        elif primary_category == "account_freezes_holds_funds":
            return f"{empathy_prefix}When did you first notice you couldn't access your account, and what were you trying to do at the time?"
        
        elif primary_category == "online_banking_technical_security_issues":
            return f"{empathy_prefix}What device and browser are you using, and when did this technical issue first start occurring?"
        
        elif primary_category == "mortgage_related_issues":
            return f"{empathy_prefix}Can you tell me more about your current mortgage situation and what specific issue you're experiencing?"
        
        elif primary_category == "credit_card_issues":
            return f"{empathy_prefix}What specific problem are you experiencing with your credit card, and when did it start?"
        
        else:
            # Generic first question for other categories
            return f"{empathy_prefix}Can you provide more details about when this issue started and what specific problems you're experiencing?"


    async def _generate_status_with_first_question(self, customer_name: str, 
                                            triage_results: Dict[str, Any], 
                                            conversation_id: str) -> str:
        """
        FIXED: Generate status update WITH immediate first follow-up question
        """
        # Generate tracking ID
        tracking_id = f"TRK_{conversation_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M')}"
        
        # Extract triage details
        if "triage_analysis" in triage_results:
            analysis = triage_results["triage_analysis"]
            primary_category = analysis["primary_category"]
            emotional_state = analysis.get("emotional_state", "neutral")
        else:
            primary_category = triage_results.get("primary_category", "general")
            emotional_state = triage_results.get("emotional_state", "neutral")
        
        # Generate category-specific first question
        first_question = await self._generate_first_followup_question_by_category(
            primary_category, emotional_state, customer_name
        )
        
        # Combine status update with first question
        status_message = f"""Perfect, {customer_name}! I've immediately escalated your case to our orchestrator system.

    **Current Status:** Your complaint has been routed and is now in the investigation queue with a tracking ID: {tracking_id}.

    Now, let me gather some additional details to help our investigation team resolve this more effectively.

    {first_question}"""
        
        return status_message

    async def _generate_first_followup_question_by_category(self, primary_category: str, 
                                                        emotional_state: str, 
                                                        customer_name: str) -> str:
        """
        Generate the first follow-up question based on complaint category
        """
        # Add empathy prefix for stressed customers
        empathy_prefix = ""
        if emotional_state in ["anxious", "frustrated", "angry"]:
            empathy_prefix = "I understand this is very stressful. "
        
        # Category-specific first questions
        if primary_category == "fraudulent_activities_unauthorized_transactions":
            return f"{empathy_prefix}When did you last use your card legitimately, and do you still have it in your possession?"
        
        elif primary_category == "dispute_resolution_issues":
            return f"{empathy_prefix}Can you tell me more about when this transaction occurred and what you expected to happen instead?"
        
        elif primary_category == "account_freezes_holds_funds":
            return f"{empathy_prefix}When did you first notice you couldn't access your account, and what were you trying to do at the time?"
        
        elif primary_category == "online_banking_technical_security_issues":
            return f"{empathy_prefix}What device and browser are you using, and when did this technical issue first start occurring?"
        
        elif primary_category == "mortgage_related_issues":
            return f"{empathy_prefix}Can you tell me more about your current mortgage situation and what specific issue you're experiencing?"
        
        elif primary_category == "credit_card_issues":
            return f"{empathy_prefix}What specific problem are you experiencing with your credit card, and when did it start?"
        
        else:
            # Generic first question for other categories
            return f"{empathy_prefix}Can you provide more details about when this issue started and what specific problems you're experiencing?"

    async def eva_chat_response_with_triage_confirmation(self, message: str, 
                                                       customer_context: Dict[str, Any], 
                                                       conversation_id: str) -> Dict[str, Any]:
        """
        FIXED: Enhanced Eva flow with proper error handling
        Replace the existing broken method with this corrected version
        """
        try:
            # Step 1: Acknowledge and show empathy
            if await self._is_complaint(message):
                
                # Step 2: Check if triage service is available
                if self.triage_service is not None:
                    # Use triage service
                    triage_result = await self.triage_service.process_complaint({
                        "complaint_text": message,
                        "customer_id": customer_context.get("customer_id", ""),
                        "customer_context": customer_context,
                        "submission_timestamp": datetime.now().isoformat(),
                        "submission_method": "eva_chat"
                    })
                else:
                    # Fallback to Eva's own classification
                    print("⚠️ Triage service not available, using Eva classification")
                    triage_result = await self._classify_complaint_with_learning(
                        message, customer_context
                    )
                
                # Step 3: Present analysis for confirmation (now with proper method)
                confirmation_response = await self._generate_triage_confirmation_response(
                    triage_result, customer_context
                )
                
                return confirmation_response
            
            else:
                # Not a complaint, use regular Eva response
                return await self.eva_chat_response(message, customer_context, conversation_id)
                
        except Exception as e:
            print(f"❌ Error in triage confirmation flow: {e}")
            # Fallback to regular Eva response
            return await self.eva_chat_response(message, customer_context, conversation_id)

    # ======================= DATABASE & CLEANUP mETHODS ====================
    
    async def check_database_integration(self) -> Dict[str, Any]:
        """Check if Eva's database integration is working properly"""
        try:
            if not self.database_service:
                return {
                    "success": False,
                    "error": "No database service provided",
                    "status": "no_database"
                }
            
            # Test database connection
            db_health = await self.database_service.eva_health_check()
            
            if db_health["status"] == "healthy":
                self.database_available = True
                return {
                    "success": True,
                    "status": "healthy",
                    "collections": db_health.get("eva_collections", {}),
                    "features": {
                        "conversation_persistence": True,
                        "learning_weights_storage": True,
                        "feedback_storage": True,
                        "natural_flow_enabled": True
                    }
                }
            else:
                self.database_available = False
                return {
                    "success": False,
                    "error": db_health.get("error", "Database unhealthy"),
                    "status": "degraded"
                }
                
        except Exception as e:
            self.database_available = False
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    
    async def cleanup(self):
        """Cleanup Eva resources"""
        try:
            # Save learning weights before cleanup
            if self.database_available and self.database_service and self.classification_weights:
                await self._save_learning_weights_to_database()
            
            print("✅ Eva agent cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Eva cleanup error: {e}")
    
    # =================  BULLET POINT RESPONSES ====================
    
    def _analyze_customer_feedback(self, customer_response: str) -> Dict[str, Any]:
        """Analyze customer feedback to determine learning signals"""
        
        response_lower = customer_response.lower()
        
        if any(phrase in response_lower for phrase in ["exactly right", "perfect", "correct", "yes that's it"]):
            return {
                "feedback_type": "confirmed",
                "learning_weight": 1.0,
                "reward_signal": 1.0,
                "confidence_adjustment": 0.05
            }
        elif any(phrase in response_lower for phrase in ["completely wrong", "not right", "disagree"]):
            return {
                "feedback_type": "major_correction", 
                "learning_weight": 1.0,
                "reward_signal": -0.5,
                "confidence_adjustment": -0.1
            }
        elif any(phrase in response_lower for phrase in ["partially", "sort of", "close but"]):
            return {
                "feedback_type": "partial_correction",
                "learning_weight": 0.7,
                "reward_signal": 0.5,
                "confidence_adjustment": 0.02
            }
        else:
            return {
                "feedback_type": "unclear",
                "learning_weight": 0.3,
                "reward_signal": 0.0,
                "confidence_adjustment": 0.0
            }
    

    
    
    
    

    
        
    
    
    
    
    



