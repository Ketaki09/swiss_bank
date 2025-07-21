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
from dotenv import load_dotenv

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
    
    def __init__(self, database_service=None):
        # Initialize Claude client
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.database_service = database_service
        
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
    
    # ===========================================
    # DATABASE INTEGRATION METHODS (NEW)
    # ===========================================
    
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
                        "feedback_storage": True
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
    
    def _load_learning_weights_from_database(self):
        """Load learning weights from database on startup"""
        try:
            if not self.database_available:
                return
            
            # This will be called async in the startup
            print("📚 Learning weights will be loaded from database")
            
        except Exception as e:
            print(f"⚠️ Failed to load learning weights: {e}")
    
    async def _load_learning_weights_async(self):
        """Load learning weights from database (async)"""
        try:
            if not self.database_available:
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
            if not self.database_available:
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
    
    async def cleanup(self):
        """Cleanup Eva resources"""
        try:
            # Save learning weights before cleanup
            if self.database_available and self.classification_weights:
                await self._save_learning_weights_to_database()
            
            print("✅ Eva agent cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Eva cleanup error: {e}")
    
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
            # Additional categories for comprehensive coverage
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
    
    # ===========================================
    # MAIN EVA CHAT RESPONSE (ALL REQUIREMENTS)
    # ===========================================
    
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
    
    # ===========================================
    # REQUIREMENT 1: CONVERSATION MEMORY (FIXED)
    # ===========================================
    
    async def _get_or_create_conversation_context(self, conversation_id: str, 
                                                 customer_context: Dict[str, Any]) -> ConversationContext:
        """Requirement 1: Get or create conversation context with database backing"""
        
        # First check in-memory cache
        if conversation_id in self.conversation_contexts:
            return self.conversation_contexts[conversation_id]
        
        # Then check database if available
        if self.database_available:
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
    
    async def _store_conversation_context(self, context: ConversationContext):
        """FIXED: Store conversation context with proper database integration"""
        # Always cache in memory
        self.conversation_contexts[context.conversation_id] = context
        
        # Store in database if available
        if self.database_available:
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
    
    # ===========================================
    # REQUIREMENT 2: BULLET POINT RESPONSES
    # ===========================================
    
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
    
    # ===========================================
    # REQUIREMENT 3: EMOTIONAL INTELLIGENCE
    # ===========================================
    
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
    
    # ===========================================
    # REQUIREMENT 4: CONTEXTUAL GREETINGS
    # ===========================================
    
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
    
    # ===========================================
    # REQUIREMENT 5: HUMAN SPECIALIST NAMES
    # ===========================================
    
    def _assign_specialist_name(self, category: str, complaint_id: str) -> Dict[str, str]:
        """Assign consistent realistic specialist name (Requirement 5)"""
        
        if category not in self.specialist_names:
            # Fallback to general category
            category = "general"
        
        # Use complaint ID for consistent assignment
        specialists = self.specialist_names[category]
        index = int(hashlib.md5(complaint_id.encode()).hexdigest(), 16) % len(specialists)
        
        return specialists[index]
    
    # ===========================================
    # COMPLAINT CLASSIFICATION WITH LEARNING (FIXED)
    # ===========================================
    
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
    
    # ===========================================
    # REINFORCEMENT LEARNING SYSTEM (FIXED)
    # ===========================================
    
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
        if self.database_available:
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
    
    # ===========================================
    # HELPER METHODS FOR API ENDPOINTS
    # ===========================================
    
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
    
    # ===========================================
    # CORE EVA RESPONSE GENERATION
    # ===========================================
    
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
    
    # ===========================================
    # UTILITY METHODS
    # ===========================================
    
    async def _is_complaint(self, message: str) -> bool:
        """Detect if message contains a complaint"""
        complaint_indicators = [
            "problem", "issue", "complaint", "wrong", "error", "frustrated", 
            "unauthorized", "dispute", "denied", "refused", "terrible", 
            "awful", "unacceptable", "fraud", "stolen", "locked", "frozen"
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in complaint_indicators)
    
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
        """Generate fallback response when errors occur"""
        customer_name = customer_context.get("name", "valued customer")
        return f"I apologize, {customer_name}. I'm experiencing a brief technical issue. Please give me a moment to assist you properly, or feel free to call our customer service line at 1-800-SWISS-BANK for immediate assistance."
    
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



