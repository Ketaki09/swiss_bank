# backend/services/database_service.py
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
