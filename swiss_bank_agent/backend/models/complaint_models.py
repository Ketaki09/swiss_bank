# backend/models/complaint_models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ComplaintStatus(str, Enum):
    RECEIVED = "received"
    IN_PROGRESS = "in_progress"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class ComplaintSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplaintChannel(str, Enum):
    WEB = "web"
    PHONE = "phone"
    EMAIL = "email"
    CHAT = "chat"
    VOICE = "voice"

class ComplaintCreate(BaseModel):
    customer_id: str
    title: str
    description: str
    channel: ComplaintChannel = ComplaintChannel.WEB
    attachments: Optional[List[str]] = []
    
class ComplaintResponse(BaseModel):
    complaint_id: str
    status: ComplaintStatus
    message: str
    estimated_resolution_time: Optional[str] = None

class ProcessedComplaint(BaseModel):
    complaint_id: str
    customer_id: str
    theme: str
    title: str
    description: str
    channel: ComplaintChannel
    severity: ComplaintSeverity
    submission_date: datetime
    status: ComplaintStatus
    attachments: List[str] = []
    related_transactions: List[str] = []
    customer_sentiment: str
    urgency_keywords: List[str] = []
    resolution_time_expected: str
    financial_impact: Optional[float] = None
    processed_content: Optional[Dict[str, Any]] = None

class Customer(BaseModel):
    customer_id: str
    name: str
    email: str
    phone: str
    account_number: str
    account_type: str
    registration_date: datetime
    previous_complaints: List[str] = []
    credit_score: Optional[int] = None
    monthly_balance: Optional[float] = None
    location: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[str] = None

class ChatMessage(BaseModel):
    session_id: str
    customer_id: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    is_bot: bool = False

class InvestigationReport(BaseModel):
    complaint_id: str
    investigation_id: str
    root_cause_analysis: str
    similar_complaints: List[str] = []
    recommended_actions: List[str] = []
    priority_level: str
    estimated_resolution_time: str
    financial_impact_assessment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "pending"

class DashboardComplaint(BaseModel):
    complaint_id: str
    customer_name: str
    theme: str
    severity: ComplaintSeverity
    status: ComplaintStatus
    submission_date: datetime
    days_open: int
    assigned_agent: Optional[str] = None