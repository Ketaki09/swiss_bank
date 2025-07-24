# backend/models/complaint_models.py - UPDATED for Triage Integration
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
    WAITING = "waiting"
    ROUTING = "routing"  # For multi-department coordination
    COMPLETED = "completed"

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

# NEW: Triage-specific models
class TriageCategory(str, Enum):
    FRAUDULENT_ACTIVITIES = "fraudulent_activities_unauthorized_transactions"
    ACCOUNT_FREEZES = "account_freezes_holds_funds"
    DEPOSIT_ISSUES = "deposit_related_issues"
    DISPUTE_RESOLUTION = "dispute_resolution_issues"
    SYSTEM_FAILURES = "bank_system_policy_failures"
    ATM_ISSUES = "atm_machine_issues"
    CHECK_ISSUES = "check_related_issues"
    POOR_SERVICE = "poor_customer_service_communication"
    FUND_DELAYS = "delays_fund_availability"
    OVERDRAFT = "overdraft_issues"
    ONLINE_BANKING = "online_banking_technical_security_issues"
    DISCRIMINATION = "discrimination_unfair_practices"
    MORTGAGE = "mortgage_related_issues"
    CREDIT_CARD = "credit_card_issues"
    UNCLEAR = "ambiguity_unclear_unclassified"
    DEBT_COLLECTION = "debt_collection_harassment"
    LOANS = "loan_issues_auto_personal_student"
    INSURANCE = "insurance_claim_denials_delays"

class TriageUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TriageClassification(BaseModel):
    """Triage classification result"""
    primary_category: TriageCategory
    secondary_category: Optional[TriageCategory] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    classification_reasoning: str
    emotional_state: str
    urgency_level: TriageUrgency
    financial_impact: str
    relationship_risk: str
    root_cause: str
    key_facts_extracted: List[str] = []
    departments_affected: List[str] = []
    is_followup: bool = False
    related_complaint_id: Optional[str] = None

class SpecialistAssignment(BaseModel):
    """Specialist assignment details"""
    specialist_name: str
    specialist_title: str
    experience: str
    specialty: str
    success_rate: str
    contact_info: str
    assignment_reason: str

class TriageResult(BaseModel):
    """Complete triage result with 3-section structure"""
    complaint_id: str
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Section 1: Original Complaint
    original_complaint: Dict[str, Any]
    
    # Section 2: Triage Analysis
    triage_analysis: TriageClassification
    
    # Section 3: Routing Package
    routing_package: Dict[str, Any]
    
    # Processing metadata
    processing_metadata: Dict[str, Any]

class FollowupDetection(BaseModel):
    """Follow-up complaint detection result"""
    is_followup: bool
    related_complaint_id: Optional[str] = None
    similarity_score: Optional[float] = None
    existing_status: Optional[ComplaintStatus] = None
    requires_confirmation: bool = True

class AdditionalContext(BaseModel):
    """Additional context for existing complaints"""
    complaint_id: str
    additional_info: str
    context_analysis: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    requires_orchestrator_action: bool = True

# UPDATED: Enhanced ProcessedComplaint with triage integration
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
    
    # NEW: Triage integration fields
    triage_classification: Optional[TriageClassification] = None
    assigned_specialists: Optional[List[SpecialistAssignment]] = []
    is_followup_complaint: bool = False
    parent_complaint_id: Optional[str] = None
    additional_contexts: Optional[List[AdditionalContext]] = []
    orchestrator_routing: Optional[Dict[str, Any]] = None

class ComplaintCreate(BaseModel):
    customer_id: str
    title: str
    description: str
    channel: ComplaintChannel = ComplaintChannel.WEB
    attachments: Optional[List[str]] = []
    
    # NEW: Optional triage context
    customer_context: Optional[Dict[str, Any]] = None
    is_followup_attempt: bool = False

class ComplaintResponse(BaseModel):
    complaint_id: str
    status: ComplaintStatus
    message: str
    estimated_resolution_time: Optional[str] = None
    
    # NEW: Triage response fields
    triage_result: Optional[TriageResult] = None
    followup_detection: Optional[FollowupDetection] = None
    requires_confirmation: bool = False

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
    
    # NEW: Triage-relevant customer data
    complaint_history_summary: Optional[Dict[str, Any]] = None
    relationship_risk_factors: Optional[List[str]] = []
    preferred_communication_method: Optional[str] = None

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
    
    # NEW: Triage dashboard fields
    triage_confidence: Optional[float] = None
    primary_specialist: Optional[str] = None
    departments_involved: Optional[List[str]] = []

# NEW: Triage-specific models for API responses
class TriageProcessingResult(BaseModel):
    """Result of triage processing"""
    type: str  # "new_complaint_processed", "followup_detected", etc.
    processing_result: str
    complaint_id: Optional[str] = None
    
    # Conditional fields based on type
    triage_output: Optional[TriageResult] = None
    eva_status_update: Optional[Dict[str, Any]] = None
    orchestrator_notification: Optional[Dict[str, Any]] = None
    confirmation_brief: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    processing_metadata: Optional[Dict[str, Any]] = None

class ClassificationFeedback(BaseModel):
    """Customer feedback for reinforcement learning"""
    complaint_id: str
    customer_response: str
    feedback_type: str
    original_classification: Dict[str, Any]
    learning_weight: float
    timestamp: datetime = Field(default_factory=datetime.now)

# NEW: Database document models for MongoDB
class TriageResultDocument(BaseModel):
    """MongoDB document for triage results"""
    complaint_id: str
    triage_result: Dict[str, Any]
    processing_timestamp: datetime
    triage_agent_version: str
    status: str = "ready_for_orchestrator"
    expires_at: datetime

class AdditionalContextDocument(BaseModel):
    """MongoDB document for additional context"""
    complaint_id: str
    additional_context: str
    analysis: Dict[str, Any]
    timestamp: datetime
    processed_by: str = "triage_agent"
    requires_orchestrator_action: bool = True