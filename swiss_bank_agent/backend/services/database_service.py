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
            await self.create_eva_indexes()  
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
            await self.create_complaint_config_indexes()
        except Exception:
            pass

    # ==================== EVA AGENT DATABASE METHODS ====================

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
                "expires_at": datetime.now() + timedelta(days=30)  
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

            critical_errors_col = self.database["eva_critical_errors"]
            await critical_errors_col.create_index([("error_type", ASCENDING)])
            await critical_errors_col.create_index([("timestamp", DESCENDING)])
            await critical_errors_col.create_index([("customer_id", ASCENDING)])
            await critical_errors_col.create_index([("conversation_id", ASCENDING)])
            await critical_errors_col.create_index([("acknowledged", ASCENDING)])
            await critical_errors_col.create_index([("resolved", ASCENDING)])
            await critical_errors_col.create_index([("created_at", DESCENDING)])
            await critical_errors_col.create_index([("severity", ASCENDING)])
            
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
    
    async def get_realistic_timelines(self) -> Dict[str, Dict[str, str]]:
        """Get realistic timelines from database configuration"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            config_col = self.database["complaint_configuration"]
            
            config = await config_col.find_one(
                {"config_id": "realistic_timelines", "active": True},
                {"_id": 0}
            )
            
            if config and "timelines" in config:
                return config["timelines"]
            else:
                # Return fallback timelines if none found in database
                return self._get_fallback_timelines()
                
        except Exception as e:
            print(f"❌ Error getting realistic timelines: {e}")
            return self._get_fallback_timelines()

    def _get_fallback_timelines(self) -> Dict[str, Dict[str, str]]:
        """Fallback timelines if database is unavailable"""
        return {
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
                "provisional_credit_review": "1-2 business days",
                "final_resolution": "3-5 business days",
                "appeal_process": "5-10 business days"
            },
            "default": {
                "initial_response": "2-4 hours",
                "investigation": "1-2 business days", 
                "resolution": "3-5 business days"
            }
        }
    # ==================== COMPLAINT CONFIGURATION METHODS ====================

    async def get_complaint_categories(self) -> List[str]:
        """Get complaint categories from database configuration"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            config_col = self.database["complaint_configuration"]
            
            config = await config_col.find_one(
                {"config_id": "complaint_categories", "active": True},
                {"_id": 0}
            )
            
            if config and "categories" in config:
                return config["categories"]
            else:
                raise ValueError("No complaint categories found in database")
                
        except Exception as e:
            print(f"❌ Error getting complaint categories: {e}")
            raise e


    async def update_realistic_timelines_configuration(self, new_data: Dict[str, Any]) -> bool:
        """Update realistic timelines configuration in database"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            config_col = self.database["complaint_configuration"]
            
            # Deactivate current active configuration
            await config_col.update_one(
                {"config_id": "realistic_timelines", "active": True},
                {"$set": {"active": False, "deactivated_at": datetime.now()}}
            )
            
            # Insert new configuration
            new_config = {
                **new_data,
                "config_id": "realistic_timelines",
                "version": f"1.{int(datetime.now().timestamp())}",
                "created_at": datetime.now(),
                "active": True
            }
            
            await config_col.insert_one(new_config)
            print(f"✅ Updated realistic timelines configuration")
            return True
            
        except Exception as e:
            print(f"❌ Error updating realistic timelines configuration: {e}")
            return False

    async def get_realistic_timelines_status(self) -> Dict[str, Any]:
        """Get status of realistic timelines configuration system"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            config_col = self.database["complaint_configuration"]
            
            # Get active timelines configuration
            config = await config_col.find_one(
                {"config_id": "realistic_timelines", "active": True},
                {"_id": 0, "version": 1, "created_at": 1}
            )
            
            if config:
                return {
                    "status": "active",
                    "version": config["version"],
                    "created_at": config["created_at"],
                    "loaded": True
                }
            else:
                return {
                    "status": "fallback",
                    "version": "fallback_v1.0",
                    "created_at": None,
                    "loaded": False
                }
            
        except Exception as e:
            print(f"❌ Error getting timelines configuration status: {e}")
            raise e

    async def create_realistic_timelines_indexes(self):
        """Create indexes for realistic timelines configuration collection"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            # Realistic timelines configuration indexes
            config_col = self.database["complaint_configuration"]
            await config_col.create_index([("config_id", ASCENDING)])
            await config_col.create_index([("active", ASCENDING)])
            await config_col.create_index([("version", DESCENDING)])
            await config_col.create_index([("created_at", DESCENDING)])
            
            print("✅ Realistic timelines configuration indexes created successfully")
            
        except Exception as e:
            print(f"❌ Error creating timelines config indexes: {e}")
            raise e
        
    async def get_banking_constraints(self) -> Dict[str, Any]:
        """Get banking constraints from database configuration"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            config_col = self.database["complaint_configuration"]
            
            config = await config_col.find_one(
                {"config_id": "banking_constraints", "active": True},
                {"_id": 0}
            )
            
            if config and "constraints" in config:
                return config["constraints"]
            else:
                raise ValueError("No banking constraints found in database")
                
        except Exception as e:
            print(f"❌ Error getting banking constraints: {e}")
            raise e

    async def update_complaint_configuration(self, config_id: str, new_data: Dict[str, Any]) -> bool:
        """Update complaint configuration in database"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            config_col = self.database["complaint_configuration"]
            
            # Deactivate current active configuration
            await config_col.update_one(
                {"config_id": config_id, "active": True},
                {"$set": {"active": False, "deactivated_at": datetime.now()}}
            )
            
            # Insert new configuration
            new_config = {
                **new_data,
                "config_id": config_id,
                "version": f"1.{int(datetime.now().timestamp())}",
                "created_at": datetime.now(),
                "active": True
            }
            
            await config_col.insert_one(new_config)
            print(f"✅ Updated complaint configuration: {config_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating complaint configuration: {e}")
            return False

    async def get_complaint_configuration_status(self) -> Dict[str, Any]:
        """Get status of complaint configuration system"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            config_col = self.database["complaint_configuration"]
            
            # Get all active configurations
            configs = await config_col.find(
                {"active": True},
                {"_id": 0, "config_id": 1, "version": 1, "created_at": 1}
            ).to_list(length=10)
            
            config_status = {}
            for config in configs:
                config_status[config["config_id"]] = {
                    "version": config["version"],
                    "created_at": config["created_at"],
                    "loaded": True
                }
            
            return {
                "status": "active",
                "configurations": config_status,
                "total_configs": len(configs)
            }
            
        except Exception as e:
            print(f"❌ Error getting configuration status: {e}")
            raise e

    # ==================== DATABASE INDEX CREATION ====================
    
    async def create_complaint_config_indexes(self):
        """Create indexes for complaint configuration collections"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            # Complaint configuration indexes
            config_col = self.database["complaint_configuration"]
            await config_col.create_index([("config_id", ASCENDING)])
            await config_col.create_index([("active", ASCENDING)])
            await config_col.create_index([("version", DESCENDING)])
            await config_col.create_index([("created_at", DESCENDING)])
            
            print("✅ Complaint configuration indexes created successfully")
            
        except Exception as e:
            print(f"❌ Error creating config indexes: {e}")
            raise e
    # ==================== TRIAGE AGENT DATABASE METHODS ====================

    async def update_complaint_followup(self, complaint_id: str, followup_data: Dict[str, Any]) -> bool:
        """Update complaint with follow-up information"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            result = await complaints_col.update_one(
                {"complaint_id": complaint_id},
                {
                    "$set": {
                        "followup_interactions": followup_data.get("followup_interactions"),
                        "last_customer_contact": followup_data.get("last_customer_contact"),
                        "customer_engagement_level": followup_data.get("customer_engagement_level", "active"),
                        "updated_at": datetime.now()
                    }
                }
            )
            return result.modified_count > 0
            
        except Exception as e:
            print(f"❌ Error updating complaint followup: {e}")
            return False

    async def update_complaint_context(self, complaint_id: str, context_data: Dict[str, Any]) -> bool:
        """Update complaint with additional context"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            result = await complaints_col.update_one(
                {"complaint_id": complaint_id},
                {
                    "$set": {
                        "additional_context": context_data.get("additional_context"),
                        "last_updated": context_data.get("last_updated"),
                        "updated_at": datetime.now()
                    }
                }
            )
            return result.modified_count > 0
            
        except Exception as e:
            print(f"❌ Error updating complaint context: {e}")
            return False

    async def get_customer_open_complaints_by_status(self, customer_id: str, 
                                                statuses: List[str]) -> List[Dict[str, Any]]:
        """Get customer complaints by specific statuses"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            complaints_col = self.complaints_collection
            assert complaints_col is not None
            
            complaints = await complaints_col.find(
                {
                    "customer_id": customer_id,
                    "status": {"$in": statuses}
                },
                {"_id": 0}
            ).sort("submission_date", DESCENDING).to_list(length=50)
            
            return complaints
            
        except Exception as e:
            print(f"❌ Error getting complaints by status: {e}")
            return []
        
    # ==================== ORIGINAL DATABASE METHODS  ====================

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



    # ============================== ORCHESTRATOR ALERT METHODS ===================================

    async def store_orchestrator_alert(self, alert_data: Dict[str, Any]) -> str:
        """Store orchestrator alert for processing"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            alerts_col = self.database["orchestrator_alerts"]
            
            alert_doc = {
                "alert_id": alert_data["alert_id"],
                "alert_type": alert_data["alert_type"],
                "priority": alert_data["priority"],
                "timestamp": alert_data["timestamp"],
                "complaint_summary": alert_data.get("complaint_summary", {}),
                "routing_instructions": alert_data.get("routing_instructions", {}),
                "orchestrator_actions": alert_data.get("orchestrator_actions", []),
                "background_information": alert_data.get("background_information", {}),
                "new_theme_details": alert_data.get("new_theme_details", {}),
                "immediate_actions_required": alert_data.get("immediate_actions_required", []),
                "escalation_level": alert_data.get("escalation_level", "STANDARD"),
                "human_review_mandatory": alert_data.get("human_review_mandatory", False),
                "processed": False,
                "processed_at": None,
                "processed_by": None,
                "created_at": datetime.now()
            }
            
            await alerts_col.insert_one(alert_doc)
            return alert_data["alert_id"]
            
        except Exception as e:
            print(f"❌ Error storing orchestrator alert: {e}")
            raise e

    async def get_pending_orchestrator_alerts(self, alert_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get unprocessed orchestrator alerts"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return []
            
            alerts_col = self.database["orchestrator_alerts"]
            
            query: Dict[str, Any] = {"processed": False}
            if alert_type:
                query["alert_type"] = alert_type
            
            alerts = await alerts_col.find(
                query,
                {"_id": 0}
            ).sort("timestamp", DESCENDING).to_list(length=100)
            
            return alerts
            
        except Exception as e:
            print(f"❌ Error getting orchestrator alerts: {e}")
            return []

    async def store_critical_error(self, error_details: Dict[str, Any]) -> bool:
        """
        Store critical Eva agent errors for monitoring and alerting
        """
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        
        try:
            if self.database is None:
                raise ConnectionError("Database not properly initialized")
            
            # Store in a dedicated 'eva_critical_errors' collection
            critical_errors_col = self.database["eva_critical_errors"]
            result = await critical_errors_col.insert_one({
                **error_details,
                "created_at": datetime.now(),
                "acknowledged": False,
                "resolved": False
            })
            
            # Trigger immediate alert if error count exceeds threshold
            recent_errors = await critical_errors_col.count_documents({
                "timestamp": {"$gte": (datetime.now() - timedelta(minutes=10)).isoformat()},
                "error_type": "EVA_AGENT_CORE_FAILURE"
            })
            
            if recent_errors >= 3:
                # Trigger high-priority alert
                await self._trigger_eva_failure_alert(error_details, recent_errors)
            return True
            
        except Exception as e:
            print(f"❌ Failed to store critical error: {e}")
            return False

    async def _trigger_eva_failure_alert(self, error_details: Dict[str, Any], error_count: int):
        """
        Trigger immediate alert for Eva agent failures
        """
        alert_message = f"""
        🚨 EVA AGENT CRITICAL FAILURE DETECTED
        
        Error Count: {error_count} in last 10 minutes
        Last Error: {error_details['error_message']}
        Customer Impact: {error_details['impact']}
        
        IMMEDIATE ACTION REQUIRED
        """

    async def mark_orchestrator_alerts_processed(self, alert_ids: List[str], processed_by: str) -> bool:
        """Mark orchestrator alerts as processed"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return False
            
            alerts_col = self.database["orchestrator_alerts"]
            
            result = await alerts_col.update_many(
                {"alert_id": {"$in": alert_ids}},
                {
                    "$set": {
                        "processed": True,
                        "processed_at": datetime.now(),
                        "processed_by": processed_by
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            print(f"❌ Error marking alerts as processed: {e}")
            return False

    async def get_orchestrator_alert_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get orchestrator alert statistics"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return {
                    "total_alerts": 0,
                    "pending_alerts": 0,
                    "new_theme_alerts": 0,
                    "alert_breakdown": {},
                    "error": "Database not initialized"
                }
            
            alerts_col = self.database["orchestrator_alerts"]
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Get alert type breakdown
            pipeline = [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$alert_type",
                    "total": {"$sum": 1},
                    "pending": {"$sum": {"$cond": [{"$eq": ["$processed", False]}, 1, 0]}},
                    "critical": {"$sum": {"$cond": [{"$eq": ["$priority", "CRITICAL"]}, 1, 0]}}
                }}
            ]
            
            breakdown_stats = await alerts_col.aggregate(pipeline).to_list(length=None)
            
            # Total counts
            total_alerts = await alerts_col.count_documents({
                "created_at": {"$gte": start_date}
            })
            
            pending_alerts = await alerts_col.count_documents({
                "created_at": {"$gte": start_date},
                "processed": False
            })
            
            new_theme_alerts = await alerts_col.count_documents({
                "created_at": {"$gte": start_date},
                "alert_type": "NEW_THEME_DETECTED"
            })
            
            return {
                "total_alerts": total_alerts,
                "pending_alerts": pending_alerts,
                "new_theme_alerts": new_theme_alerts,
                "alert_breakdown": {stat["_id"]: stat for stat in breakdown_stats},
                "period_days": days
            }
            
        except Exception as e:
            print(f"❌ Error getting alert statistics: {e}")
            return {
                "total_alerts": 0,
                "pending_alerts": 0,
                "new_theme_alerts": 0,
                "alert_breakdown": {},
                "error": str(e)
            }

    # ===================== ENHANCED TRIAGE DATABASE METHODS =====================

    async def create_triage_indexes(self):
        """Create indexes for triage and orchestrator collections"""
        if not self._check_connection():
            print("❌ Database connection not established - skipping triage indexes")
            return
        
        try:
            if self.database is None:
                print("❌ Database not initialized - skipping triage indexes")
                return
            
            # Orchestrator alerts indexes
            alerts_col = self.database["orchestrator_alerts"]
            await alerts_col.create_index([("alert_id", ASCENDING)], unique=True)
            await alerts_col.create_index([("alert_type", ASCENDING)])
            await alerts_col.create_index([("priority", ASCENDING)])
            await alerts_col.create_index([("processed", ASCENDING)])
            await alerts_col.create_index([("timestamp", DESCENDING)])
            await alerts_col.create_index([("created_at", DESCENDING)])
            
            # Triage processing logs (optional)
            triage_logs_col = self.database["triage_processing_logs"]
            await triage_logs_col.create_index([("complaint_id", ASCENDING)])
            await triage_logs_col.create_index([("customer_id", ASCENDING)])
            await triage_logs_col.create_index([("processing_timestamp", DESCENDING)])
            await triage_logs_col.create_index([("complaint_type", ASCENDING)])
            
            print("✅ Triage database indexes created successfully")
            
        except Exception as e:
            print(f"❌ Error creating triage indexes: {e}")

    async def log_triage_processing(self, processing_data: Dict[str, Any]) -> str:
        """Log triage processing for analytics"""
        if not self._check_connection():
            return ""
        
        try:
            if self.database is None:
                return ""
            
            logs_col = self.database["triage_processing_logs"]
            
            log_id = str(uuid.uuid4())
            log_doc = {
                "log_id": log_id,
                "complaint_id": processing_data.get("complaint_id"),
                "customer_id": processing_data.get("customer_id"),
                "complaint_type": processing_data.get("complaint_type"),
                "processing_timestamp": datetime.now(),
                "classification_result": processing_data.get("classification_result"),
                "confidence_score": processing_data.get("confidence_score"),
                "processing_time_ms": processing_data.get("processing_time_ms"),
                "new_theme_detected": processing_data.get("new_theme_detected", False),
                "orchestrator_alert_sent": processing_data.get("orchestrator_alert_sent", False),
                "error_occurred": processing_data.get("error_occurred", False),
                "error_message": processing_data.get("error_message")
            }
            
            await logs_col.insert_one(log_doc)
            return log_id
            
        except Exception as e:
            print(f"❌ Error logging triage processing: {e}")
            return ""

    async def get_triage_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get triage processing analytics"""
        if not self._check_connection():
            raise ConnectionError("Database connection not established")
        try:
            if self.database is None:
                return {
                    "period_days": days,
                    "total_processed": 0,
                    "new_complaints": 0,
                    "followup_complaints": 0,
                    "new_themes_detected": 0,
                    "average_processing_time": 0,
                    "error_rate": 0.0,
                    "error": "Database not initialized"
                }
            
            logs_col = self.database["triage_processing_logs"]
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Basic statistics
            total_processed = await logs_col.count_documents({
                "processing_timestamp": {"$gte": start_date}
            })
            
            new_complaints = await logs_col.count_documents({
                "processing_timestamp": {"$gte": start_date},
                "complaint_type": "new_complaint"
            })
            
            followup_complaints = await logs_col.count_documents({
                "processing_timestamp": {"$gte": start_date},
                "complaint_type": "followup"
            })
            
            new_themes = await logs_col.count_documents({
                "processing_timestamp": {"$gte": start_date},
                "new_theme_detected": True
            })
            
            errors = await logs_col.count_documents({
                "processing_timestamp": {"$gte": start_date},
                "error_occurred": True
            })
            
            # Average processing time
            avg_time_pipeline = [
                {"$match": {
                    "processing_timestamp": {"$gte": start_date},
                    "processing_time_ms": {"$exists": True}
                }},
                {"$group": {
                    "_id": None,
                    "avg_time": {"$avg": "$processing_time_ms"}
                }}
            ]
            
            avg_time_result = await logs_col.aggregate(avg_time_pipeline).to_list(length=1)
            avg_processing_time = avg_time_result[0]["avg_time"] if avg_time_result else 0
            
            error_rate = (errors / max(total_processed, 1)) * 100
            
            return {
                "period_days": days,
                "total_processed": total_processed,
                "new_complaints": new_complaints,
                "followup_complaints": followup_complaints,
                "new_themes_detected": new_themes,
                "average_processing_time_ms": round(avg_processing_time, 2),
                "error_count": errors,
                "error_rate_percent": round(error_rate, 2),
                "processing_efficiency": "high" if error_rate < 5 else "needs_improvement"
            }
            
        except Exception as e:
            print(f"❌ Error getting triage analytics: {e}")
            return {
                "period_days": days,
                "total_processed": 0,
                "new_complaints": 0,
                "followup_complaints": 0,
                "new_themes_detected": 0,
                "average_processing_time_ms": 0,
                "error_count": 0,
                "error_rate_percent": 0.0,
                "error": str(e)
            }    
        
        
            