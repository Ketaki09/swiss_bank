# backend/services/database_service.py
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import uuid
import os
import logging

from models.complaint_models import ProcessedComplaint, Customer, InvestigationReport

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.complaints_collection: Optional[AsyncIOMotorCollection] = None
        self.customers_collection: Optional[AsyncIOMotorCollection] = None
        self.investigations_collection: Optional[AsyncIOMotorCollection] = None
        self.chat_sessions_collection: Optional[AsyncIOMotorCollection] = None
        self.temp_data_collection: Optional[AsyncIOMotorCollection] = None
        
        # MongoDB connection string - update with your credentials
        self.connection_string = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
        self.database_name = os.getenv("DB_NAME", "swiss_bank")

    def _check_connection(self) -> bool:
        """Check if database connection is established"""
        return (
            self.client is not None and 
            self.database is not None and
            self.complaints_collection is not None and
            self.customers_collection is not None and
            self.investigations_collection is not None and
            self.chat_sessions_collection is not None and
            self.temp_data_collection is not None
        )

    async def connect(self):
        """Initialize database connection"""
        try:
            # Create MongoDB client with proper configuration
            self.client = AsyncIOMotorClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test connection first before setting up collections
            await self.client.admin.command('ping')
            logger.info("MongoDB connection successful")
            
            # Now set up database and collections
            self.database = self.client[self.database_name]
            
            # Initialize collections
            self.complaints_collection = self.database.complaints
            self.customers_collection = self.database.customers
            self.investigations_collection = self.database.investigations
            self.chat_sessions_collection = self.database.chat_sessions
            self.temp_data_collection = self.database.temp_data
            
            # Create indexes for better performance
            await self.create_indexes()
            
            logger.info("✅ Database connection established")
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            # Clean up on failure
            if self.client:
                self.client.close()
                self.client = None
                self.database = None
                self.complaints_collection = None
                self.customers_collection = None
                self.investigations_collection = None
                self.chat_sessions_collection = None
                self.temp_data_collection = None
            raise e

    async def disconnect(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            # Reset all references
            self.client = None
            self.database = None
            self.complaints_collection = None
            self.customers_collection = None
            self.investigations_collection = None
            self.chat_sessions_collection = None
            self.temp_data_collection = None
            logger.info("Database connection closed")

    async def create_indexes(self):
        """Create database indexes for optimal performance"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            # Type assertion to help Pylance understand these are not None
            complaints_col = self.complaints_collection
            customers_col = self.customers_collection
            investigations_col = self.investigations_collection
            temp_data_col = self.temp_data_collection
            
            assert complaints_col is not None
            assert customers_col is not None
            assert investigations_col is not None
            assert temp_data_col is not None
            
            # Complaints indexes
            await complaints_col.create_index([("customer_id", ASCENDING)])
            await complaints_col.create_index([("status", ASCENDING)])
            await complaints_col.create_index([("submission_date", DESCENDING)])
            await complaints_col.create_index([("theme", ASCENDING)])
            
            # Customers indexes
            await customers_col.create_index([("customer_id", ASCENDING)], unique=True)
            await customers_col.create_index([("email", ASCENDING)])
            
            # Investigations indexes
            await investigations_col.create_index([("complaint_id", ASCENDING)])
            
            # Temp data indexes - TTL index for automatic expiration
            await temp_data_col.create_index([("expires_at", 1)], expireAfterSeconds=0)
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            # Don't raise here as indexes might already exist

    async def save_complaint(self, complaint_data: Dict[str, Any]) -> str:
        """Save a new complaint to database"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            complaints_col = self.complaints_collection
            customers_col = self.customers_collection
            assert complaints_col is not None
            assert customers_col is not None
            
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
            
            result = await complaints_col.insert_one(complaint_doc)
            
            # Update customer's complaint history
            await customers_col.update_one(
                {"customer_id": complaint_data["customer_id"]},
                {"$addToSet": {"previous_complaints": complaint_id}},
                upsert=False  # Don't create customer if doesn't exist
            )
            
            logger.info(f"Complaint saved with ID: {complaint_id}")
            return complaint_id
            
        except Exception as e:
            logger.error(f"Error saving complaint: {e}")
            raise e

    async def get_complaint(self, complaint_id: str) -> Optional[Dict[str, Any]]:
        """Get complaint by ID (using custom complaint_id, not MongoDB _id)"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            complaint = await complaints_col.find_one(
                {"complaint_id": complaint_id},
                {"_id": 0}  # Exclude MongoDB ObjectId from response
            )
            return complaint
        except Exception as e:
            logger.error(f"Error getting complaint {complaint_id}: {e}")
            raise e  # Re-raise to maintain backward compatibility
    
    async def get_complaint_by_object_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get complaint by MongoDB ObjectId (if needed for internal operations)"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            complaint = await complaints_col.find_one(
                {"_id": ObjectId(object_id)},
                {"_id": 0}
            )
            return complaint
        except Exception as e:
            logger.error(f"Error getting complaint by ObjectId {object_id}: {e}")
            return None

    async def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by ID (using custom customer_id, not MongoDB _id)"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            customers_col = self.customers_collection
            assert customers_col is not None
            
            customer = await customers_col.find_one(
                {"customer_id": customer_id},
                {"_id": 0}  # Exclude MongoDB ObjectId from response
            )
            return customer
        except Exception as e:
            logger.error(f"Error getting customer {customer_id}: {e}")
            return None
    
    async def get_customer_by_object_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by MongoDB ObjectId (if needed for internal operations)"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            customers_col = self.customers_collection
            assert customers_col is not None
            
            customer = await customers_col.find_one(
                {"_id": ObjectId(object_id)},
                {"_id": 0}
            )
            return customer
        except Exception as e:
            logger.error(f"Error getting customer by ObjectId {object_id}: {e}")
            return None

    async def get_customer_complaint_history(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get customer's complaint history"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            complaints = await complaints_col.find(
                {"customer_id": customer_id},
                {"_id": 0}
            ).sort("submission_date", DESCENDING).limit(limit).to_list(length=limit)
            
            return complaints
        except Exception as e:
            logger.error(f"Error getting complaint history for customer {customer_id}: {e}")
            return []

    async def get_similar_complaints(self, theme: str, customer_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar complaints for RCA"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            similar_complaints = await complaints_col.find(
                {
                    "theme": theme,
                    "customer_id": {"$ne": customer_id},  # Exclude current customer
                    "status": {"$in": ["resolved", "closed"]}
                },
                {"_id": 0}
            ).limit(limit).to_list(length=limit)
            
            return similar_complaints
        except Exception as e:
            logger.error(f"Error getting similar complaints: {e}")
            return []

    async def update_complaint_status(self, complaint_id: str, status: str, notes: Optional[str] = None) -> bool:
        """Update complaint status"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            update_doc = {
                "status": status,
                "updated_at": datetime.now()
            }
            
            if notes:
                update_doc["agent_notes"] = notes
            
            result = await complaints_col.update_one(
                {"complaint_id": complaint_id},
                {"$set": update_doc}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating complaint status: {e}")
            return False

    async def find_customer(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find customer by query parameters"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            customers_col = self.customers_collection
            assert customers_col is not None
            
            customer = await customers_col.find_one(
                query,
                {"_id": 0} 
            )
            return customer
        except Exception as e:
            logger.error(f"Error finding customer: {e}")
            raise e
    
    async def save_investigation_report(self, report: Dict[str, Any]) -> str:
        """Save investigation report"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            investigations_col = self.investigations_collection
            complaints_col = self.complaints_collection
            assert investigations_col is not None
            assert complaints_col is not None
            
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
            
            await investigations_col.insert_one(report_doc)
            
            # Update complaint with investigation ID
            await complaints_col.update_one(
                {"complaint_id": report["complaint_id"]},
                {"$set": {"investigation_id": investigation_id, "status": "investigating"}}
            )
            
            logger.info(f"Investigation report saved with ID: {investigation_id}")
            return investigation_id
        except Exception as e:
            logger.error(f"Error saving investigation report: {e}")
            raise e

    async def get_complaints_for_dashboard(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get complaints for dashboard view"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            query = {}
            if status:
                query["status"] = status
                
            complaints = await complaints_col.find(
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
        except Exception as e:
            logger.error(f"Error getting complaints for dashboard: {e}")
            return []

    async def save_chat_message(self, session_id: str, customer_id: str, message: str, is_bot: bool = False):
        """Save chat message to session"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            chat_sessions_col = self.chat_sessions_collection
            assert chat_sessions_col is not None
            
            message_doc = {
                "session_id": session_id,
                "customer_id": customer_id,
                "message": message,
                "is_bot": is_bot,
                "timestamp": datetime.now()
            }
            
            await chat_sessions_col.insert_one(message_doc)
            logger.info(f"Chat message saved for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")

    async def get_chat_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            chat_sessions_col = self.chat_sessions_collection
            assert chat_sessions_col is not None
            
            messages = await chat_sessions_col.find(
                {"session_id": session_id},
                {"_id": 0}
            ).sort("timestamp", ASCENDING).limit(limit).to_list(length=limit)
            
            return messages
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []

    async def get_complaint_statistics(self) -> Dict[str, Any]:
        """Get complaint statistics for dashboard"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            total_complaints = await complaints_col.count_documents({})
            
            # Count by status
            status_pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_counts = await complaints_col.aggregate(status_pipeline).to_list(length=None)
            
            # Count by severity
            severity_pipeline = [
                {"$group": {"_id": "$severity", "count": {"$sum": 1}}}
            ]
            severity_counts = await complaints_col.aggregate(severity_pipeline).to_list(length=None)
            
            # Recent complaints (last 7 days)
            last_week = datetime.now() - timedelta(days=7)
            recent_complaints = await complaints_col.count_documents(
                {"submission_date": {"$gte": last_week}}
            )
            
            return {
                "total_complaints": total_complaints,
                "status_breakdown": {item["_id"]: item["count"] for item in status_counts},
                "severity_breakdown": {item["_id"]: item["count"] for item in severity_counts},
                "recent_complaints": recent_complaints
            }
        except Exception as e:
            logger.error(f"Error getting complaint statistics: {e}")
            return {
                "total_complaints": 0,
                "status_breakdown": {},
                "severity_breakdown": {},
                "recent_complaints": 0
            }

    async def store_temp_data(self, data: Dict[str, Any]) -> bool:
        """Store temporary data with TTL (Time To Live)"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            temp_data_col = self.temp_data_collection
            assert temp_data_col is not None
            
            result = await temp_data_col.replace_one(
                {"_id": data["_id"]},
                data,
                upsert=True
            )
            
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error storing temporary data: {e}")
            return False

    async def get_temp_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve temporary data by key"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            temp_data_col = self.temp_data_collection
            assert temp_data_col is not None
            
            result = await temp_data_col.find_one({"_id": key})
            
            if result:
                # Remove MongoDB's _id field from the result
                result.pop('_id', None)
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving temporary data: {e}")
            return None

    async def delete_temp_data(self, key: str) -> bool:
        """Delete temporary data by key"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            temp_data_col = self.temp_data_collection
            assert temp_data_col is not None
            
            result = await temp_data_col.delete_one({"_id": key})
            
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting temporary data: {e}")
            return False

    async def cleanup_expired_temp_data(self):
        """Manually cleanup expired temporary data (MongoDB TTL should handle this automatically)"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
            
        try:
            temp_data_col = self.temp_data_collection
            assert temp_data_col is not None
            
            # Delete documents where expires_at < current time
            result = await temp_data_col.delete_many({
                "expires_at": {"$lt": datetime.now()}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} expired temporary data entries")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error during manual cleanup: {e}")
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # Check if client exists
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "Database client not initialized"
                }
            
            await self.client.admin.command('ping')
            
            # Check if database exists by trying to get its name
            # This is a safer way to check than using boolean evaluation
            if self.database is None:
                return {
                    "status": "unhealthy",
                    "error": "Database not initialized"
                }
            
            # Try to get database name to verify it's properly initialized
            db_name = self.database.name
            if not db_name:
                return {
                    "status": "unhealthy",
                    "error": "Database name not accessible"
                }
            
            # Get database stats
            stats = await self.database.command("dbstats")
            
            return {
                "status": "healthy",
                "database_name": db_name,
                "collections": stats.get("collections", 0),
                "dataSize": stats.get("dataSize", 0),
                "storageSize": stats.get("storageSize", 0)
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    # OTP-specific methods for better organization
    async def store_otp(self, phone_number: str, otp: str, expires_minutes: int = 3) -> bool:
        """Store OTP with expiration time"""
        try:
            expires_at = datetime.now() + timedelta(minutes=expires_minutes)
            
            otp_data = {
                "_id": f"otp_{phone_number}",
                "phone_number": phone_number,
                "otp": otp,
                "created_at": datetime.now(),
                "expires_at": expires_at,
                "verified": False
            }
            
            return await self.store_temp_data(otp_data)
            
        except Exception as e:
            logger.error(f"Error storing OTP for {phone_number}: {e}")
            return False

    async def verify_otp(self, phone_number: str, otp: str) -> bool:
        """Verify OTP and mark as used"""
        try:
            otp_data = await self.get_temp_data(f"otp_{phone_number}")
            
            if not otp_data:
                logger.warning(f"No OTP found for {phone_number}")
                return False
            
            if otp_data["verified"]:
                logger.warning(f"OTP already verified for {phone_number}")
                return False
            
            if otp_data["expires_at"] < datetime.now():
                logger.warning(f"OTP expired for {phone_number}")
                await self.delete_temp_data(f"otp_{phone_number}")
                return False
            
            if otp_data["otp"] != otp:
                logger.warning(f"Invalid OTP for {phone_number}")
                return False
            
            # Mark as verified
            otp_data["verified"] = True
            await self.store_temp_data(otp_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying OTP for {phone_number}: {e}")
            return False

    async def cleanup_used_otp(self, phone_number: str) -> bool:
        """Clean up used OTP"""
        try:
            return await self.delete_temp_data(f"otp_{phone_number}")
        except Exception as e:
            logger.error(f"Error cleaning up OTP for {phone_number}: {e}")
            return False


