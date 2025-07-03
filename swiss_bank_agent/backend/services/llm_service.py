# backend/services/llm_service.py
import anthropic
from typing import Dict, Any, List, Optional
import json
import re
from datetime import datetime
import os

class LLMService:
    def __init__(self):
        # Initialize Claude client
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Chat session memory
        self.chat_sessions = {}

    async def process_complaint(self, text: str, customer_context: Dict[str, Any], attachments: List[str] = None) -> Dict[str, Any]:
        """
        Process complaint text and extract structured information
        """
        prompt = self._build_complaint_processing_prompt(text, customer_context, attachments)
        
        try:
            response = await self._call_anthropic(prompt)
            
            # Parse the structured response
            processed_data = self._parse_complaint_response(response, text, customer_context)
            return processed_data
            
        except Exception as e:
            print(f"Error processing complaint: {e}")
            # Fallback to basic processing
            return self._fallback_complaint_processing(text, customer_context)

    async def chat_response(self, message: str, customer_context: Dict[str, Any], session_id: str) -> str:
        """
        Generate chat response for customer interaction
        """
        # Get or create chat session
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = []
        
        # Add customer message to session
        self.chat_sessions[session_id].append({"role": "user", "content": message})
        
        prompt = self._build_chat_prompt(message, customer_context, self.chat_sessions[session_id])
        
        try:
            response = await self._call_anthropic(prompt)
            
            # Add bot response to session
            self.chat_sessions[session_id].append({"role": "assistant", "content": response})
            
            # Keep only last 10 exchanges to manage memory
            if len(self.chat_sessions[session_id]) > 20:
                self.chat_sessions[session_id] = self.chat_sessions[session_id][-20:]
            
            return response
            
        except Exception as e:
            print(f"Error generating chat response: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team directly. support@swissbank.com"

    def _build_complaint_processing_prompt(self, text: str, customer_context: Dict[str, Any], attachments: List[str] = None) -> str:
        """Build prompt for complaint processing"""
        attachments_text = ""
        if attachments:
            attachments_text = f"\nAttachments provided: {', '.join(attachments)}"
        
        return f"""
You are an expert complaint analyst for Swiss bank. Analyze the following customer complaint and extract structured information.

Customer Information:
- Name: {customer_context.get('name', 'Unknown')}
- Account Type: {customer_context.get('account_type', 'Unknown')}
- Previous Complaints: {len(customer_context.get('previous_complaints', []))} complaints
- Customer Since: {customer_context.get('registration_date', 'Unknown')}
- Location: {customer_context.get('location', 'Unknown')}

Complaint Text:
{text}
{attachments_text}

Please analyze and provide a JSON response with the following structure:
{{
    "theme": "primary category (e.g., 'Account Access', 'Transaction Dispute', 'Service Quality', 'Fees and Charges')",
    "title": "concise title summarizing the issue",
    "severity": "low/medium/high/critical based on financial impact and urgency",
    "customer_sentiment": "positive/neutral/negative/angry based on tone",
    "urgency_keywords": ["list", "of", "urgent", "keywords", "found"],
    "resolution_time_expected": "estimated time based on complexity",
    "financial_impact": estimated monetary impact if applicable (number or null),
    "related_transactions": ["list of transaction IDs or amounts mentioned"],
    "summary": "brief summary of the core issue",
    "recommended_priority": "low/medium/high/critical",
    "escalation_needed": true/false,
    "department": "which department should handle this"
}}

Focus on:
1. Identifying the root issue beyond surface complaints
2. Assessing urgency based on customer language and situation
3. Considering customer history and context
4. Extracting actionable information
"""

    def _build_chat_prompt(self, message: str, customer_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Build prompt for chat interaction"""
        history_text = ""
        if chat_history:
            recent_history = chat_history[-10:]  # Last 5 exchanges
            history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])
        
        return f"""
You are Eva, a helpful and empathetic customer service assistant for Wells Fargo. You help customers with their banking needs and complaints.

Customer Information:
- Name: {customer_context.get('name', 'Valued Customer')}
- Account Type: {customer_context.get('account_type', 'Standard')}
- Customer Since: {customer_context.get('registration_date', 'Unknown')}

Recent Conversation:
{history_text}

Current Message: {message}

Guidelines:
1. Be empathetic and professional
2. Acknowledge customer concerns
3. Provide helpful information when possible
4. If you need to escalate or collect a formal complaint, guide them appropriately
5. Use the customer's name when appropriate
6. Keep responses concise but thorough
7. If technical banking questions arise, provide general guidance but recommend speaking with a specialist

Respond naturally and helpfully to the customer's message.
"""

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            raise e

    def _parse_complaint_response(self, response: str, original_text: str, customer_context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # If no JSON found, create from text analysis
                data = self._fallback_complaint_processing(original_text, customer_context)
            
            # Validate and set defaults
            required_fields = {
                "theme": "General Inquiry",
                "title": original_text[:100] + "..." if len(original_text) > 100 else original_text,
                "severity": "medium",
                "customer_sentiment": "neutral",
                "urgency_keywords": [],
                "resolution_time_expected": "2-3 business days",
                "financial_impact": None,
                "related_transactions": [],
                "summary": original_text[:200] + "..." if len(original_text) > 200 else original_text,
                "recommended_priority": "medium",
                "escalation_needed": False,
                "department": "Customer Service"
            }
            
            # Ensure all required fields are present
            for field, default in required_fields.items():
                if field not in data:
                    data[field] = default
            
            # Add customer context
            data["customer_id"] = customer_context["customer_id"]
            data["channel"] = "web"
            data["description"] = original_text
            
            return data
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._fallback_complaint_processing(original_text, customer_context)

    def _fallback_complaint_processing(self, text: str, customer_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing when LLM fails"""
        text_lower = text.lower()
        
        # Simple keyword-based analysis
        urgency_keywords = []
        severity = "medium"
        
        urgent_words = ["urgent", "emergency", "immediately", "asap", "critical", "stuck", "locked out"]
        high_impact_words = ["money", "fraud", "unauthorized", "dispute", "stolen", "hacked"]
        negative_sentiment_words = ["angry", "frustrated", "disappointed", "terrible", "awful", "worst"]
        
        # Check for urgency keywords
        for word in urgent_words:
            if word in text_lower:
                urgency_keywords.append(word)
                severity = "high"
        
        # Check for high impact issues
        for word in high_impact_words:
            if word in text_lower:
                urgency_keywords.append(word)
                if severity != "high":
                    severity = "medium"
        
        # Determine sentiment
        sentiment = "neutral"
        negative_count = sum(1 for word in negative_sentiment_words if word in text_lower)
        if negative_count >= 2:
            sentiment = "negative"
        elif negative_count >= 1:
            sentiment = "neutral"
        
        # Determine theme based on keywords
        theme_keywords = {
            "Account Access": ["login", "password", "access", "locked", "blocked", "sign in"],
            "Transaction Dispute": ["transaction", "charge", "payment", "transfer", "dispute", "unauthorized"],
            "Service Quality": ["service", "staff", "wait", "hold", "representative", "experience"],
            "Fees and Charges": ["fee", "charge", "cost", "expensive", "billing", "overdraft"],
            "Technical Issues": ["app", "website", "online", "mobile", "technical", "error"],
            "Card Issues": ["card", "debit", "credit", "atm", "declined", "stolen"]
        }
        
        theme = "General Inquiry"
        for category, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                theme = category
                break
        
        # Estimate resolution time based on complexity
        resolution_time = "2-3 business days"
        if severity == "critical":
            resolution_time = "24 hours"
        elif severity == "high":
            resolution_time = "1-2 business days"
        elif any(word in text_lower for word in ["simple", "quick", "information"]):
            resolution_time = "Same day"
        
        return {
            "customer_id": customer_context["customer_id"],
            "theme": theme,
            "title": f"{theme} - {customer_context.get('name', 'Customer')} Inquiry",
            "description": text,
            "channel": "web",
            "severity": severity,
            "customer_sentiment": sentiment,
            "urgency_keywords": urgency_keywords,
            "resolution_time_expected": resolution_time,
            "financial_impact": None,
            "related_transactions": [],
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "recommended_priority": severity,
            "escalation_needed": severity in ["high", "critical"],
            "department": self._determine_department(theme),
            "processed_content": {
                "fallback_processing": True,
                "processing_timestamp": datetime.now().isoformat()
            }
        }
    
    def _determine_department(self, theme: str) -> str:
        """Determine which department should handle the complaint"""
        department_mapping = {
            "Account Access": "Technical Support",
            "Transaction Dispute": "Fraud Investigation",
            "Service Quality": "Customer Relations",
            "Fees and Charges": "Billing Department",
            "Technical Issues": "IT Support",
            "Card Issues": "Card Services",
            "General Inquiry": "Customer Service"
        }
        return department_mapping.get(theme, "Customer Service")
    
    async def generate_investigation_prompt(self, complaint_data: Dict[str, Any], customer_history: List[Dict[str, Any]], similar_complaints: List[Dict[str, Any]]) -> str:
        """Generate prompt for investigation agent"""
        history_summary = ""
        if customer_history:
            history_summary = f"Customer has {len(customer_history)} previous complaints:\n"
            for complaint in customer_history[-3:]:  # Last 3 complaints
                history_summary += f"- {complaint.get('theme', 'Unknown')}: {complaint.get('title', 'No title')} ({complaint.get('status', 'Unknown')})\n"
        
        similar_summary = ""
        if similar_complaints:
            similar_summary = f"Similar complaints found ({len(similar_complaints)}):\n"
            for complaint in similar_complaints[:3]:  # Top 3 similar
                similar_summary += f"- {complaint.get('title', 'No title')} (Resolution: {complaint.get('status', 'Unknown')})\n"
        
        return f"""
You are a complaint investigation specialist for Wells Fargo. Conduct a thorough Root Cause Analysis (RCA) for the following complaint.

CURRENT COMPLAINT:
Theme: {complaint_data.get('theme', 'Unknown')}
Title: {complaint_data.get('title', 'No title')}
Description: {complaint_data.get('description', 'No description')}
Severity: {complaint_data.get('severity', 'Unknown')}
Customer Sentiment: {complaint_data.get('customer_sentiment', 'Unknown')}

CUSTOMER HISTORY:
{history_summary if history_summary else "No previous complaints on record."}

SIMILAR CASES:
{similar_summary if similar_summary else "No similar cases found."}

Please provide a comprehensive RCA in JSON format:
{{
    "root_cause_analysis": "detailed analysis of the underlying cause",
    "contributing_factors": ["factor1", "factor2", "factor3"],
    "pattern_analysis": "analysis of patterns from similar cases",
    "customer_impact_assessment": "assessment of impact on customer",
    "recommended_actions": ["action1", "action2", "action3"],
    "preventive_measures": ["measure1", "measure2"],
    "priority_level": "low/medium/high/critical",
    "estimated_resolution_time": "timeframe for resolution",
    "escalation_recommended": true/false,
    "financial_impact_assessment": "assessment of monetary impact",
    "follow_up_required": true/false,
    "customer_compensation": "recommended compensation if any"
}}

Focus on:
1. Identifying systemic issues vs isolated incidents
2. Customer experience impact
3. Operational improvements needed
4. Risk mitigation strategies
"""

    def clear_chat_session(self, session_id: str):
        """Clear chat session memory"""
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]
    
    def get_chat_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of chat session"""
        if session_id not in self.chat_sessions:
            return {"message_count": 0, "session_exists": False}
        
        messages = self.chat_sessions[session_id]
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        bot_messages = [msg for msg in messages if msg["role"] == "assistant"]
        
        return {
            "session_exists": True,
            "message_count": len(messages),
            "user_message_count": len(user_messages),
            "bot_message_count": len(bot_messages),
            "last_activity": datetime.now().isoformat()
        }
    
