# backend/services/database_service.py
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid
import os

from models.complaint_models import ProcessedComplaint, Customer, InvestigationReport

class DatabaseService:
    def __init__(self):
        self.client = None
        self.database = None
        self.complaints_collection = None
        self.customers_collection = None
        self.investigations_collection = None
        self.chat_sessions_collection = None
        
        # MongoDB connection string - update with your credentials
        self.connection_string = os.getenv(
            "MONGODB_URL", 
            "mongodb://localhost:27017/"
        )
        self.database_name = os.getenv("DB_NAME", "swiss_bank")

    async def connect(self):
        """Initialize database connection"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            self.database = self.client[self.database_name]
            
            # Initialize collections
            self.complaints_collection = self.database.complaints
            self.customers_collection = self.database.customers
            self.investigations_collection = self.database.investigations
            self.chat_sessions_collection = self.database.chat_sessions
            
            # Create indexes for better performance
            await self.create_indexes()
            
            print("✅ Database connection established")
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise e

    async def disconnect(self):
        """Close database connection"""
        if self.client:
            self.client.close()

    async def create_indexes(self):
        """Create database indexes for optimal performance"""
        # Complaints indexes
        await self.complaints_collection.create_index([("customer_id", ASCENDING)])
        await self.complaints_collection.create_index([("status", ASCENDING)])
        await self.complaints_collection.create_index([("submission_date", DESCENDING)])
        await self.complaints_collection.create_index([("theme", ASCENDING)])
        
        # Customers indexes
        await self.customers_collection.create_index([("customer_id", ASCENDING)], unique=True)
        await self.customers_collection.create_index([("email", ASCENDING)])
        
        # Investigations indexes
        await self.investigations_collection.create_index([("complaint_id", ASCENDING)])

    async def save_complaint(self, complaint_data: Dict[str, Any]) -> str:
        """Save a new complaint to database"""
        complaint_id = str(uuid.uuid4())
        
        complaint_doc = {
            "complaint_id": complaint_id,
            "customer_id": complaint_data["customer_id"],
            "theme": complaint_data["theme"],
            "title": complaint_data["title"],
            "description": complaint_data["description"],
            "channel": complaint_data["channel"],
            "severity": complaint_data["severity"],
            "submission_date": datetime.now(),
            "status": "received",
            "attachments": complaint_data.get("attachments", []),
            "related_transactions": complaint_data.get("related_transactions", []),
            "customer_sentiment": complaint_data.get("customer_sentiment", "neutral"),
            "urgency_keywords": complaint_data.get("urgency_keywords", []),
            "resolution_time_expected": complaint_data.get("resolution_time_expected", "2-3 business days"),
            "financial_impact": complaint_data.get("financial_impact"),
            "processed_content": complaint_data.get("processed_content", {}),
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        await self.complaints_collection.insert_one(complaint_doc)
        
        # Update customer's complaint history
        await self.customers_collection.update_one(
            {"customer_id": complaint_data["customer_id"]},
            {"$addToSet": {"previous_complaints": complaint_id}}
        )
        
        return complaint_id

    async def get_complaint(self, complaint_id: str) -> Optional[Dict[str, Any]]:
        """Get complaint by ID (using custom complaint_id, not MongoDB _id)"""
        complaint = await self.complaints_collection.find_one(
            {"complaint_id": complaint_id},
            {"_id": 0}  # Exclude MongoDB ObjectId from response
        )
        return complaint
    
    async def get_complaint_by_object_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get complaint by MongoDB ObjectId (if needed for internal operations)"""
        try:
            complaint = await self.complaints_collection.find_one(
                {"_id": ObjectId(object_id)},
                {"_id": 0}
            )
            return complaint
        except Exception as e:
            print(f"Invalid ObjectId format: {e}")
            return None

    async def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by ID (using custom customer_id, not MongoDB _id)"""
        customer = await self.customers_collection.find_one(
            {"customer_id": customer_id},
            {"_id": 0}  # Exclude MongoDB ObjectId from response
        )
        return customer
    
    async def get_customer_by_object_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by MongoDB ObjectId (if needed for internal operations)"""
        try:
            customer = await self.customers_collection.find_one(
                {"_id": ObjectId(object_id)},
                {"_id": 0}
            )
            return customer
        except Exception as e:
            print(f"Invalid ObjectId format: {e}")
            return None

    async def get_customer_complaint_history(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get customer's complaint history"""
        complaints = await self.complaints_collection.find(
            {"customer_id": customer_id},
            {"_id": 0}
        ).sort("submission_date", DESCENDING).limit(limit).to_list(length=limit)
        
        return complaints

    async def get_similar_complaints(self, theme: str, customer_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar complaints for RCA"""
        similar_complaints = await self.complaints_collection.find(
            {
                "theme": theme,
                "customer_id": {"$ne": customer_id},  # Exclude current customer
                "status": {"$in": ["resolved", "closed"]}
            },
            {"_id": 0}
        ).limit(limit).to_list(length=limit)
        
        return similar_complaints

    async def update_complaint_status(self, complaint_id: str, status: str, notes: Optional[str] = None) -> bool:
        """Update complaint status"""
        update_doc = {
            "status": status,
            "updated_at": datetime.now()
        }
        
        if notes:
            update_doc["agent_notes"] = notes
        
        result = await self.complaints_collection.update_one(
            {"complaint_id": complaint_id},
            {"$set": update_doc}
        )
        
        return result.modified_count > 0

    async def save_investigation_report(self, report: Dict[str, Any]) -> str:
        """Save investigation report"""
        investigation_id = str(uuid.uuid4())
        
        report_doc = {
            "investigation_id": investigation_id,
            "complaint_id": report["complaint_id"],
            "root_cause_analysis": report["root_cause_analysis"],
            "similar_complaints": report.get("similar_complaints", []),
            "recommended_actions": report.get("recommended_actions", []),
            "priority_level": report.get("priority_level", "medium"),
            "estimated_resolution_time": report.get("estimated_resolution_time", "2-3 business days"),
            "financial_impact_assessment": report.get("financial_impact_assessment"),
            "created_at": datetime.now(),
            "status": "pending"
        }
        
        await self.investigations_collection.insert_one(report_doc)
        
        # Update complaint with investigation ID
        await self.complaints_collection.update_one(
            {"complaint_id": report["complaint_id"]},
            {"$set": {"investigation_id": investigation_id, "status": "investigating"}}
        )
        
        return investigation_id

    async def get_complaints_for_dashboard(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get complaints for dashboard view"""
        query = {}
        if status:
            query["status"] = status
            
        complaints = await self.complaints_collection.find(
            query,
            {"_id": 0}
        ).sort("submission_date", DESCENDING).limit(limit).to_list(length=limit)
        
        # Add customer names and calculate days open
        for complaint in complaints:
            customer = await self.get_customer(complaint["customer_id"])
            complaint["customer_name"] = customer["name"] if customer else "Unknown"
            
            # Calculate days open
            submission_date = complaint["submission_date"]
            days_open = (datetime.now() - submission_date).days
            complaint["days_open"] = days_open
        
        return complaints

    async def save_chat_message(self, session_id: str, customer_id: str, message: str, is_bot: bool = False):
        """Save chat message to session"""
        message_doc = {
            "session_id": session_id,
            "customer_id": customer_id,
            "message": message,
            "is_bot": is_bot,
            "timestamp": datetime.now()
        }
        
        await self.chat_sessions_collection.insert_one(message_doc)

    async def get_chat_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        messages = await self.chat_sessions_collection.find(
            {"session_id": session_id},
            {"_id": 0}
        ).sort("timestamp", ASCENDING).limit(limit).to_list(length=limit)
        
        return messages

    async def get_complaint_statistics(self) -> Dict[str, Any]:
        """Get complaint statistics for dashboard"""
        total_complaints = await self.complaints_collection.count_documents({})
        
        # Count by status
        status_pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        status_counts = await self.complaints_collection.aggregate(status_pipeline).to_list(length=None)
        
        # Count by severity
        severity_pipeline = [
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}}
        ]
        severity_counts = await self.complaints_collection.aggregate(severity_pipeline).to_list(length=None)
        
        # Recent complaints (last 7 days)
        last_week = datetime.now() - timedelta(days=7)
        recent_complaints = await self.complaints_collection.count_documents(
            {"submission_date": {"$gte": last_week}}
        )
        
        return {
            "total_complaints": total_complaints,
            "status_breakdown": {item["_id"]: item["count"] for item in status_counts},
            "severity_breakdown": {item["_id"]: item["count"] for item in severity_counts},
            "recent_complaints": recent_complaints
        }