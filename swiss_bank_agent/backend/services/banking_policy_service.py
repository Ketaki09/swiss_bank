# backend/services/banking_policy_service.py
"""
Banking Policy Service - Integrates with existing Eva and Triage services
Ensures all responses follow realistic banking policies
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from .eva_agent_service import ConversationContext

load_dotenv()

class BankingPolicyService:
    """
    Service that ensures all customer communications follow realistic banking policies
    Integrates with existing Eva and Triage services
    """
    
    def __init__(self):
        # Universal banking constraints
        self.banking_constraints = {
            "no_instant_refunds": True,
            "investigation_required": True, 
            "regulatory_compliance": True,
            "documentation_protocols": True,
            "provisional_credit_conditions": True
        }
        
        # Realistic timelines by complaint category
        self.realistic_timelines = {
            "fraudulent_activities_unauthorized_transactions": {
                "security_action": "Immediate",
                "investigation_start": "2-4 hours",
                "provisional_credit_review": "1-3 business days",
                "final_resolution": "7-10 business days",
                "new_card_delivery": "24-48 hours"
            },
            "dispute_resolution_issues": {
                "case_creation": "Immediate",
                "investigation_start": "1-2 business days", 
                "provisional_credit_review": "3-5 business days",
                "final_resolution": "10-14 business days"
            },
            "account_freezes_holds_funds": {
                "security_review": "2-4 hours",
                "documentation_review": "4-24 hours",
                "access_restoration": "1-3 business days"
            },
            "online_banking_technical_security_issues": {
                "security_check": "Immediate",
                "technical_investigation": "2-4 hours",
                "resolution": "4-24 hours"
            },
            # Add other categories as needed
            "default": {
                "initial_response": "2-4 hours",
                "investigation": "1-2 business days", 
                "resolution": "3-5 business days"
            }
        }
    
    def get_realistic_timeline(self, complaint_category: str) -> Dict[str, str]:
        """Get realistic timeline for complaint category"""
        return self.realistic_timelines.get(complaint_category, self.realistic_timelines["default"])
    
    def validate_response_promises(self, response_text: str) -> Dict[str, Any]:
        """
        Validate that response doesn't make unrealistic promises
        Returns validation result and suggestions
        """
        unrealistic_phrases = [
            "instant refund", "immediate refund", "money back now",
            "credit your account immediately", "refund within hours",
            "instant credit", "immediate credit", "money available now"
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
    
    def _get_realistic_alternatives(self, violations: List[str]) -> List[str]:
        """Get realistic alternatives for unrealistic promises"""
        alternatives = {
            "instant refund": "expedited dispute processing for provisional credit review",
            "immediate refund": "priority investigation for fastest possible resolution",
            "money back now": "emergency dispute filing with urgent review",
            "credit your account immediately": "provisional credit consideration after initial investigation"
        }
        
        return [alternatives.get(violation, "realistic timeline communication") for violation in violations]

# backend/services/eva_agent_service_enhanced.py
"""
Enhanced Eva Agent Service that integrates with existing services
"""

from .eva_agent_service import EvaAgentService
from .banking_policy_service import BankingPolicyService
import asyncio

class EvaAgentServiceEnhanced(EvaAgentService):
    """
    Enhanced Eva Agent that integrates with Banking Policy Service
    Extends existing EvaAgentService without breaking current functionality
    """
    
    def __init__(self, database_service=None, triage_service=None):
        # Initialize parent Eva service
        super().__init__(database_service)
        
        # Add new services
        self.triage_service = triage_service
        self.banking_policy_service = BankingPolicyService()
        
        # Conversation flow state management
        self.conversation_states = {}
        
        print("✅ Enhanced Eva initialized with Banking Policy Service")
    
    async def eva_chat_response_with_natural_flow(self, message: str, customer_context: Dict[str, Any], 
                                                 conversation_id: str) -> Dict[str, Any]:
        """
        Enhanced Eva chat with natural triage flow and banking policy compliance
        This method extends the existing eva_chat_response method
        """
        try:
            # Get conversation context using parent method
            context = await self._get_or_create_conversation_context(conversation_id, customer_context)
            
            # Check conversation stage
            conversation_state = self.conversation_states.get(conversation_id, {"stage": "initial"})
            
            # Route to appropriate handler based on stage
            if conversation_state["stage"] == "initial" and await self._is_complaint(message):
                return await self._handle_initial_complaint_with_triage(message, context, conversation_id)
            
            elif conversation_state["stage"] == "awaiting_triage_results":
                return await self._present_triage_results(conversation_id)
            
            elif conversation_state["stage"] == "triage_confirmation":
                return await self._handle_triage_confirmation(message, context, conversation_id)
            
            elif conversation_state["stage"] == "follow_up_questions":
                return await self._handle_follow_up_questions(message, context, conversation_id)
            
            elif conversation_state["stage"] == "action_sequence":
                return await self._continue_action_sequence(conversation_id)
            
            else:
                # Fall back to original Eva functionality
                return await super().eva_chat_response(message, customer_context, conversation_id)
                
        except Exception as e:
            print(f"❌ Error in enhanced Eva flow: {e}")
            fallback_response = await self._generate_fallback_response(customer_context)
            return {
                "response": fallback_response,
                "conversation_id": conversation_id,
                "error": "Enhanced Eva processing error - fallback response provided"
            }
        
    async def _handle_initial_complaint_with_triage(self, message: str, context: ConversationContext, 
                                                   conversation_id: str) -> Dict[str, Any]:
        """
        Handle initial complaint with background triage analysis
        """
        customer_name = context.customer_name
        
        # Step 1: Immediate empathetic response
        initial_response = await self._generate_empathetic_acknowledgment(message, customer_name)
        
        # Step 2: Show triage analysis starting
        analysis_message = f"""
        {initial_response}
        
        Let me immediately connect with our specialist analysis team to get a complete picture of what's happening. 
        I'm pulling up your account details and running this through our security and classification protocols right now...
        
        *[Analysis in progress - this will take just a moment]*
        """
        
        # Step 3: Update conversation state and start background triage
        self.conversation_states[conversation_id] = {
            "stage": "awaiting_triage_results",
            "complaint_text": message,
            "analysis_start_time": datetime.now(),
            "triage_initiated": True
        }
        
        # Store message in context using parent method
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
    
    async def _run_background_triage_analysis(self, conversation_id: str, complaint_text: str, customer_id: str):
        """
        Run triage analysis in background and update conversation state
        """
        try:
            if not self.triage_service:
                print("⚠️ Triage service not available, using Eva classification")
                # Fall back to Eva's classification
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
                
                triage_result = await self.triage_service.process_complaint(complaint_data)
            
            # Update conversation state with results
            self.conversation_states[conversation_id].update({
                "stage": "triage_results_ready",
                "triage_results": triage_result,
                "analysis_complete_time": datetime.now()
            })
            
            print(f"✅ Background triage analysis complete for conversation {conversation_id}")
            
        except Exception as e:
            print(f"❌ Background triage analysis failed: {e}")
            # Set fallback state
            self.conversation_states[conversation_id].update({
                "stage": "triage_analysis_failed",
                "error": str(e)
            })
    
    async def _present_triage_results(self, conversation_id: str) -> Dict[str, Any]:
        """
        Present triage results in natural, customer-friendly way
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
            # New complaint format
            analysis = triage_results["triage_analysis"]
            primary_category = analysis["primary_category"]
            urgency_level = analysis["urgency_level"]
            confidence = analysis.get("confidence_score", 0.8)
        else:
            # Eva classification format
            primary_category = triage_results.get("primary_category", "general_inquiry")
            urgency_level = triage_results.get("priority", "medium")
            confidence = triage_results.get("confidence_score", 0.8)
        
        # Generate natural presentation using AI with banking constraints
        presentation_prompt = f"""
        Present triage analysis results naturally to customer {customer_name}.
        
        Analysis Results:
        - Category: {primary_category}
        - Urgency: {urgency_level}
        - Confidence: {confidence:.0%}
        
        Present this in customer-friendly language, ask for confirmation, and indicate what specialist team will handle this.
        
        Be conversational, not corporate. Show that analysis was thorough.
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
    
    async def _handle_triage_confirmation(self, message: str, context: ConversationContext, 
                                        conversation_id: str) -> Dict[str, Any]:
        """
        Handle customer's response to triage classification
        """
        confirmation_analysis = self._analyze_customer_confirmation(message)
        
        if confirmation_analysis["confirmed"]:
            # Customer confirms - proceed to follow-up questions
            return await self._initiate_follow_up_questions(context, conversation_id)
        else:
            # Customer wants clarification or correction
            return await self._handle_classification_correction(message, context, conversation_id)
    
    async def _initiate_follow_up_questions(self, context: ConversationContext, 
                                          conversation_id: str) -> Dict[str, Any]:
        """
        Start dynamic follow-up questions
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
        Handle follow-up question responses
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
    
    async def _initiate_realistic_action_sequence(self, conversation_id: str) -> Dict[str, Any]:
        """
        Initiate realistic action sequence using banking policy service
        """
        conversation_state = self.conversation_states[conversation_id]
        triage_results = conversation_state["triage_results"]
        context = self.conversation_contexts[conversation_id]
        
        # Generate first action response with banking policy compliance
        first_action = await self._generate_realistic_action_response(
            "what_doing_now", triage_results, context
        )
        
        # Validate response for realistic banking practices
        validation = self.banking_policy_service.validate_response_promises(first_action)
        
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
    
    async def _generate_realistic_action_response(self, action_type: str, 
                                                triage_results: Dict[str, Any],
                                                context: ConversationContext) -> str:
        """
        Generate realistic action response using banking policy constraints
        """
        # Extract complaint details
        if "triage_analysis" in triage_results:
            analysis = triage_results["triage_analysis"]
            complaint_type = analysis["primary_category"]
            urgency_level = analysis["urgency_level"]
        else:
            complaint_type = triage_results.get("primary_category", "general_inquiry")
            urgency_level = triage_results.get("priority", "medium")
        
        # Get realistic timelines for this complaint type
        timelines = self.banking_policy_service.get_realistic_timeline(complaint_type)
        
        # Build response prompt with banking constraints
        prompt = f"""
        Generate realistic banking response for "{action_type}" stage.
        
        Context:
        - Customer: {context.customer_name}
        - Complaint type: {complaint_type}
        - Urgency: {urgency_level}
        - Realistic timelines: {timelines}
        
        BANKING CONSTRAINTS (CRITICAL):
        - NO instant refunds or money promises
        - Investigation required before any credits
        - Show process urgency, not money urgency
        - Use realistic timelines from provided data
        - Provisional credit is conditional, not guaranteed
        
        Generate empathetic but realistic response for {action_type}.
        """
        
        return await self._call_anthropic(prompt)
    
    # Helper methods for conversation flow
    
    def _analyze_customer_confirmation(self, message: str) -> Dict[str, Any]:
        """Analyze customer's confirmation response"""
        message_lower = message.lower()
        
        confirmed_indicators = ["yes", "correct", "right", "accurate", "exactly", "that's it"]
        correction_indicators = ["no", "wrong", "not exactly", "but", "actually", "however"]
        
        if any(indicator in message_lower for indicator in confirmed_indicators):
            return {"confirmed": True, "needs_correction": False}
        elif any(indicator in message_lower for indicator in correction_indicators):
            return {"confirmed": False, "needs_correction": True}
        else:
            return {"confirmed": False, "needs_correction": False}
    
    def _customer_wants_to_proceed(self, message: str) -> bool:
        """Check if customer wants to skip more questions"""
        proceed_indicators = [
            "let's proceed", "move on", "that's enough", "what's next",
            "fix this now", "take action", "resolve this"
        ]
        return any(indicator in message.lower() for indicator in proceed_indicators)
    
    async def _generate_next_followup_question(self, conversation_state: Dict[str, Any]) -> str:
        """Generate next follow-up question based on previous responses"""
        previous_responses = [info["response"] for info in conversation_state["gathered_info"]]
        
        prompt = f"""
        Based on previous customer responses: {' | '.join(previous_responses)}
        
        Generate the next logical follow-up question for this complaint investigation.
        Keep it conversational and focused on gathering helpful details.
        Don't repeat information already gathered.
        """
        
        return await self._call_anthropic(prompt)
    
    async def _continue_action_sequence(self, conversation_id: str) -> Dict[str, Any]:
        """Continue with next part of action sequence"""
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
    
    async def _generate_action_step_2(self, conversation_id: str) -> Dict[str, Any]:
        """Generate second action message"""
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
        """Generate third action message"""
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

# Integration with main.py
"""
Update main.py to use enhanced Eva service
"""

# In main.py lifespan function, replace Eva initialization:
"""
# STEP 4: Initialize Enhanced Eva with Banking Policy compliance
print("\n🤖 Initializing Enhanced Eva Agent with Banking Policy Service...")
services["eva"] = EvaAgentServiceEnhanced(
    database_service=services["db"],
    triage_service=None  # Will be set after triage initialization
)

# STEP 5: Initialize Triage Agent Service  
print("\n🎯 Initializing Triage Agent with Eva integration...")
services["triage"] = TriageAgentService(
    database_service=services["db"],
    eva_agent_service=services["eva"]
)

# STEP 6: Link Triage service to Enhanced Eva
services["eva"].triage_service = services["triage"]
print("✅ Enhanced Eva and Triage services fully integrated")
"""