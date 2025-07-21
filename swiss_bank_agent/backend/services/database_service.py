# backend/services/database_service.py - FIXED VERSION
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import uuid
import os

from models.complaint_models import ProcessedComplaint, Customer, InvestigationReport

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
        try:
            self.client = AsyncIOMotorClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000, 
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            # Test connection first before setting up collections
            await self.client.admin.command('ping')
            # Now set up database and collections
            self.database = self.client[self.database_name]
            self.complaints_collection = self.database.complaints
            self.customers_collection = self.database.customers
            self.investigations_collection = self.database.investigations
            self.chat_sessions_collection = self.database.chat_sessions
            self.temp_data_collection = self.database.temp_data
            # Create indexes for better performance
            await self.create_indexes()
            await self.create_eva_indexes()  # Add Eva indexes
        except Exception as e:
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
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            self.complaints_collection = None
            self.customers_collection = None
            self.investigations_collection = None
            self.chat_sessions_collection = None
            self.temp_data_collection = None

    async def create_indexes(self):
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            complaints_col = self.complaints_collection
            customers_col = self.customers_collection
            investigations_col = self.investigations_collection
            temp_data_col = self.temp_data_collection

            assert complaints_col is not None
            assert customers_col is not None
            assert investigations_col is not None
            assert temp_data_col is not None

            await complaints_col.create_index([("customer_id", ASCENDING)])
            await complaints_col.create_index([("status", ASCENDING)])
            await complaints_col.create_index([("submission_date", DESCENDING)])
            await complaints_col.create_index([("theme", ASCENDING)])
            await customers_col.create_index([("customer_id", ASCENDING)], unique=True)
            await customers_col.create_index([("email", ASCENDING)])
            await investigations_col.create_index([("complaint_id", ASCENDING)])
            await temp_data_col.create_index([("expires_at", 1)], expireAfterSeconds=0)
        except Exception:
            pass

    # ==================== EVA AGENT DATABASE METHODS (NOW PROPER CLASS METHODS) ====================

    async def store_eva_conversation(self, conversation_data: Dict[str, Any]) -> bool:
        """Store Eva conversation context with full message history"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            # Proper type-safe checking
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            conversations_col = self.database["eva_conversations"]
            
            conversation_doc = {
                "conversation_id": conversation_data["conversation_id"],
                "customer_id": conversation_data["customer_id"],
                "customer_name": conversation_data["customer_name"],
                "messages": conversation_data["messages"],
                "ongoing_issues": conversation_data.get("ongoing_issues", []),
                "specialist_assignments": conversation_data.get("specialist_assignments", {}),
                "emotional_state": conversation_data.get("emotional_state", "neutral"),
                "classification_pending": conversation_data.get("classification_pending"),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(days=30)  # Keep for 30 days
            }
            
            # Upsert conversation
            await conversations_col.replace_one(
                {"conversation_id": conversation_data["conversation_id"]},
                conversation_doc,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            print(f"Error storing Eva conversation: {e}")
            return False

    async def get_eva_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve Eva conversation context"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return None
            
            conversations_col = self.database["eva_conversations"]
            
            conversation = await conversations_col.find_one(
                {"conversation_id": conversation_id},
                {"_id": 0}
            )
            
            return conversation
            
        except Exception as e:
            print(f"Error retrieving Eva conversation: {e}")
            return None

    async def store_classification_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Store customer feedback for reinforcement learning"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            feedback_col = self.database["eva_feedback"]
            
            feedback_id = str(uuid.uuid4())
            feedback_doc = {
                "feedback_id": feedback_id,
                "complaint_id": feedback_data["complaint_id"],
                "customer_id": feedback_data.get("customer_id"),
                "original_classification": feedback_data["original_classification"],
                "customer_response": feedback_data["customer_response"],
                "feedback_type": feedback_data["feedback_type"],
                "learning_weight": feedback_data["learning_weight"],
                "confidence_adjustment": feedback_data.get("confidence_adjustment", 0),
                "created_at": datetime.now(),
                "processed_for_training": False
            }
            
            await feedback_col.insert_one(feedback_doc)
            return feedback_id
            
        except Exception as e:
            print(f"Error storing classification feedback: {e}")
            raise e

    async def get_classification_feedback_for_training(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get unprocessed feedback for model training"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return []
            
            feedback_col = self.database["eva_feedback"]
            
            feedback_items = await feedback_col.find(
                {"processed_for_training": False},
                {"_id": 0}
            ).sort("created_at", DESCENDING).limit(limit).to_list(length=limit)
            
            return feedback_items
            
        except Exception as e:
            print(f"Error retrieving feedback for training: {e}")
            return []

    async def mark_feedback_as_processed(self, feedback_ids: List[str]) -> bool:
        """Mark feedback as processed for training"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return False
            
            feedback_col = self.database["eva_feedback"]
            
            result = await feedback_col.update_many(
                {"feedback_id": {"$in": feedback_ids}},
                {"$set": {"processed_for_training": True, "processed_at": datetime.now()}}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            print(f"Error marking feedback as processed: {e}")
            return False

    async def store_eva_learning_weights(self, weights_data: Dict[str, Any]) -> bool:
        """Store Eva's learning weights for persistence"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            weights_col = self.database["eva_learning_weights"]
            
            weights_doc = {
                "version_id": weights_data.get("version_id", str(uuid.uuid4())),
                "classification_weights": weights_data["classification_weights"],
                "total_feedback_processed": weights_data.get("total_feedback_processed", 0),
                "accuracy_metrics": weights_data.get("accuracy_metrics", {}),
                "created_at": datetime.now(),
                "is_active": True
            }
            
            # Deactivate previous weights
            await weights_col.update_many(
                {"is_active": True},
                {"$set": {"is_active": False}}
            )
            
            # Insert new weights
            await weights_col.insert_one(weights_doc)
            
            return True
            
        except Exception as e:
            print(f"Error storing Eva learning weights: {e}")
            return False

    async def get_eva_learning_weights(self) -> Optional[Dict[str, Any]]:
        """Get current Eva learning weights"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return None
            
            weights_col = self.database["eva_learning_weights"]
            
            weights = await weights_col.find_one(
                {"is_active": True},
                {"_id": 0}
            )
            
            return weights
            
        except Exception as e:
            print(f"Error retrieving Eva learning weights: {e}")
            return None

    async def get_eva_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get Eva performance analytics"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return {
                    "period_days": days,
                    "total_conversations": 0,
                    "total_feedback": 0,
                    "accuracy_rate": 0.0,
                    "feedback_breakdown": {},
                    "learning_active": False,
                    "error": "Database not initialized"
                }
            
            feedback_col = self.database["eva_feedback"]
            conversations_col = self.database["eva_conversations"]
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Get feedback analytics
            feedback_pipeline = [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$feedback_type",
                    "count": {"$sum": 1},
                    "avg_learning_weight": {"$avg": "$learning_weight"}
                }}
            ]
            
            feedback_stats = await feedback_col.aggregate(feedback_pipeline).to_list(length=None)
            
            # Get conversation analytics
            conversation_count = await conversations_col.count_documents({
                "created_at": {"$gte": start_date}
            })
            
            # Calculate accuracy rate
            total_feedback = sum(stat["count"] for stat in feedback_stats)
            confirmed_feedback = next((stat["count"] for stat in feedback_stats if stat["_id"] == "confirmed"), 0)
            partial_feedback = next((stat["count"] for stat in feedback_stats if stat["_id"] == "partial_correction"), 0)
            
            accuracy_rate = (confirmed_feedback + partial_feedback) / max(total_feedback, 1)
            
            return {
                "period_days": days,
                "total_conversations": conversation_count,
                "total_feedback": total_feedback,
                "accuracy_rate": accuracy_rate,
                "feedback_breakdown": {stat["_id"]: stat["count"] for stat in feedback_stats},
                "learning_active": total_feedback > 0,
                "improvement_trend": "improving" if accuracy_rate > 0.8 else "needs_attention"
            }
            
        except Exception as e:
            print(f"Error getting Eva analytics: {e}")
            return {
                "period_days": days,
                "total_conversations": 0,
                "total_feedback": 0,
                "accuracy_rate": 0.0,
                "feedback_breakdown": {},
                "learning_active": False,
                "error": str(e)
            }

    async def cleanup_eva_data(self) -> Dict[str, Any]:
        """Cleanup expired Eva data"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return {
                    "expired_conversations_removed": 0,
                    "old_feedback_removed": 0,
                    "error": "Database not initialized"
                }
            
            conversations_col = self.database["eva_conversations"]
            feedback_col = self.database["eva_feedback"]
            
            expired_conversations_count = 0
            old_feedback_count = 0
            
            # Clean up expired conversations
            expired_conversations = await conversations_col.delete_many({
                "expires_at": {"$lt": datetime.now()}
            })
            expired_conversations_count = expired_conversations.deleted_count
            
            # Clean up old feedback (keep for 90 days)
            old_feedback_date = datetime.now() - timedelta(days=90)
            old_feedback = await feedback_col.delete_many({
                "created_at": {"$lt": old_feedback_date},
                "processed_for_training": True
            })
            old_feedback_count = old_feedback.deleted_count
            
            return {
                "expired_conversations_removed": expired_conversations_count,
                "old_feedback_removed": old_feedback_count
            }
            
        except Exception as e:
            print(f"Error cleaning up Eva data: {e}")
            return {
                "expired_conversations_removed": 0,
                "old_feedback_removed": 0,
                "error": str(e)
            }

    async def create_eva_indexes(self):
        """Create indexes for Eva collections for better performance"""
        if not self._check_connection():
            print("❌ Database connection not established - skipping Eva indexes")
            return
        
        try:
            if self.database is None:
                print("❌ Database not initialized - skipping Eva indexes")
                return
            
            # Eva conversations indexes
            conversations_col = self.database["eva_conversations"]
            await conversations_col.create_index([("conversation_id", ASCENDING)], unique=True)
            await conversations_col.create_index([("customer_id", ASCENDING)])
            await conversations_col.create_index([("created_at", DESCENDING)])
            await conversations_col.create_index([("expires_at", 1)], expireAfterSeconds=0)
            
            # Eva feedback indexes
            feedback_col = self.database["eva_feedback"]
            await feedback_col.create_index([("feedback_id", ASCENDING)], unique=True)
            await feedback_col.create_index([("complaint_id", ASCENDING)])
            await feedback_col.create_index([("customer_id", ASCENDING)])
            await feedback_col.create_index([("feedback_type", ASCENDING)])
            await feedback_col.create_index([("processed_for_training", ASCENDING)])
            await feedback_col.create_index([("created_at", DESCENDING)])
            
            # Eva learning weights indexes
            weights_col = self.database["eva_learning_weights"]
            await weights_col.create_index([("version_id", ASCENDING)], unique=True)
            await weights_col.create_index([("is_active", ASCENDING)])
            await weights_col.create_index([("created_at", DESCENDING)])
            
            print("✅ Eva database indexes created successfully")
            
        except Exception as e:
            print(f"❌ Error creating Eva indexes: {e}")

    async def eva_health_check(self) -> Dict[str, Any]:
        """Health check specific to Eva functionality"""
        if not self._check_connection():
            return {"status": "unhealthy", "error": "Database connection not established"}
        
        try:
            if self.database is None:
                return {"status": "unhealthy", "error": "Database not initialized"}
            
            # Check Eva collections exist and are accessible
            collections_check = {}
            eva_collections = ["eva_conversations", "eva_feedback", "eva_learning_weights"]
            
            for collection_name in eva_collections:
                try:
                    collection = self.database[collection_name]
                    count = await collection.count_documents({})
                    collections_check[collection_name] = {"status": "healthy", "document_count": count}
                except Exception as e:
                    collections_check[collection_name] = {"status": "error", "error": str(e)}
            
            # Overall status
            all_healthy = all(
                check["status"] == "healthy" 
                for check in collections_check.values()
            )
            
            return {
                "status": "healthy" if all_healthy else "degraded",
                "eva_collections": collections_check,
                "database_name": self.database.name
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    # ==================== ORIGINAL DATABASE METHODS (KEEP ALL EXISTING) ====================

    async def save_complaint(self, complaint_data: Dict[str, Any]) -> str:
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
            await complaints_col.insert_one(complaint_doc)
            await customers_col.update_one(
                {"customer_id": complaint_data["customer_id"]},
                {"$addToSet": {"previous_complaints": complaint_id}},
                upsert=False
            )
            return complaint_id
        except Exception as e:
            raise e

    async def get_complaint(self, complaint_id: str) -> Optional[Dict[str, Any]]:
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            complaint = await complaints_col.find_one(
                {"complaint_id": complaint_id},
                {"_id": 0}
            )
            return complaint
        except Exception as e:
            raise e

    async def get_complaint_by_object_id(self, object_id: str) -> Optional[Dict[str, Any]]:
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
        except Exception:
            return None

    async def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            customers_col = self.customers_collection
            assert customers_col is not None
            customer = await customers_col.find_one(
                {"customer_id": customer_id},
                {"_id": 0}
            )
            return customer
        except Exception:
            return None

    async def get_customer_by_object_id(self, object_id: str) -> Optional[Dict[str, Any]]:
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
        except Exception:
            return None

    async def get_customer_complaint_history(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
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
        except Exception:
            return []

    async def get_similar_complaints(self, theme: str, customer_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            similar_complaints = await complaints_col.find(
                {
                    "theme": theme,
                    "customer_id": {"$ne": customer_id},
                    "status": {"$in": ["resolved", "closed"]}
                },
                {"_id": 0}
            ).limit(limit).to_list(length=limit)
            return similar_complaints
        except Exception:
            return []

    async def update_complaint_status(self, complaint_id: str, status: str, notes: Optional[str] = None) -> bool:
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
        except Exception:
            return False

    async def find_customer(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
            raise e

    async def save_investigation_report(self, report: Dict[str, Any]) -> str:
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
            await complaints_col.update_one(
                {"complaint_id": report["complaint_id"]},
                {"$set": {"investigation_id": investigation_id, "status": "investigating"}}
            )
            return investigation_id
        except Exception as e:
            raise e

    async def get_complaints_for_dashboard(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
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
            for complaint in complaints:
                customer = await self.get_customer(complaint["customer_id"])
                complaint["customer_name"] = customer["name"] if customer else "Unknown"
                submission_date = complaint["submission_date"]
                days_open = (datetime.now() - submission_date).days
                complaint["days_open"] = days_open
            return complaints
        except Exception:
            return []

    async def save_chat_message(self, session_id: str, customer_id: str, message: str, is_bot: bool = False):
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
        except Exception:
            pass

    async def get_chat_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
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
        except Exception:
            return []

    async def get_complaint_statistics(self) -> Dict[str, Any]:
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            total_complaints = await complaints_col.count_documents({})
            status_pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_counts = await complaints_col.aggregate(status_pipeline).to_list(length=None)
            severity_pipeline = [
                {"$group": {"_id": "$severity", "count": {"$sum": 1}}}
            ]
            severity_counts = await complaints_col.aggregate(severity_pipeline).to_list(length=None)
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
        except Exception:
            return {
                "total_complaints": 0,
                "status_breakdown": {},
                "severity_breakdown": {},
                "recent_complaints": 0
            }

    async def store_temp_data(self, data: Dict[str, Any]) -> bool:
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
        except Exception:
            return False

    async def get_temp_data(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            temp_data_col = self.temp_data_collection
            assert temp_data_col is not None
            result = await temp_data_col.find_one({"_id": key})
            if result:
                result.pop('_id', None)
                return result
            return None
        except Exception:
            return None

    async def delete_temp_data(self, key: str) -> bool:
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            temp_data_col = self.temp_data_collection
            assert temp_data_col is not None
            result = await temp_data_col.delete_one({"_id": key})
            return result.deleted_count > 0
        except Exception:
            return False

    async def cleanup_expired_temp_data(self):
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            temp_data_col = self.temp_data_collection
            assert temp_data_col is not None
            result = await temp_data_col.delete_many({
                "expires_at": {"$lt": datetime.now()}
            })
            return result.deleted_count
        except Exception:
            return 0

    async def health_check(self) -> Dict[str, Any]:
        try:
            if not self.client:
                return {"status": "unhealthy", "error": "Database client not initialized"}
            await self.client.admin.command('ping')
            if self.database is None:
                return {"status": "unhealthy", "error": "Database not initialized"}
            db_name = self.database.name
            if not db_name:
                return {"status": "unhealthy", "error": "Database name not accessible"}
            stats = await self.database.command("dbstats")
            return {
                "status": "healthy",
                "database_name": db_name,
                "collections": stats.get("collections", 0),
                "dataSize": stats.get("dataSize", 0),
                "storageSize": stats.get("storageSize", 0)
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def store_otp(self, phone_number: str, otp: str, expires_minutes: int = 3) -> bool:
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
        except Exception:
            return False

    async def verify_otp(self, phone_number: str, otp: str) -> bool:
        try:
            otp_data = await self.get_temp_data(f"otp_{phone_number}")
            if not otp_data:
                return False
            if otp_data["verified"]:
                return False
            if otp_data["expires_at"] < datetime.now():
                await self.delete_temp_data(f"otp_{phone_number}")
                return False
            if otp_data["otp"] != otp:
                return False
            otp_data["verified"] = True
            await self.store_temp_data(otp_data)
            return True
        except Exception:
            return False

    async def cleanup_used_otp(self, phone_number: str) -> bool:
        try:
            return await self.delete_temp_data(f"otp_{phone_number}")
        except Exception:
            return False
        
        