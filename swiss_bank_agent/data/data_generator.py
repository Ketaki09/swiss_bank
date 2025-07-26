# data/data_generator.py
"""
Swiss Bank Complaint Synthetic Data Generator - MongoDB Only
"""
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pymongo
from dataclasses import dataclass, asdict
from enum import Enum
import faker

# Initialize Faker for realistic data generation
fake = faker.Faker()


# =============================================================================
# PHONE NUMBER GENERATOR FUNCTION
# =============================================================================

def generate_phone_number() -> str:
    """
    Generate phone number in strict format: +19966118088
    """
    # Generate exactly 10 digits for the phone number
    phone_digits = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    return f"+1{phone_digits}"


# =============================================================================
# DATA MODELS AND ENUMS
# =============================================================================

class ComplaintTheme(Enum):
    """Types of banking complaints"""
    FRAUDULENT_ACTIVITIES = "fraudulent_activities"
    ACCOUNT_FREEZES = "account_freezes"
    DEPOSIT_ISSUES = "deposit_issues"
    DISPUTE_RESOLUTION = "dispute_resolution"
    ATM_ISSUES = "atm_issues"


class ComplaintChannel(Enum):
    """Communication channels for complaints"""
    PHONE = "phone"
    EMAIL = "email"
    CHAT = "chat"
    BRANCH = "branch"
    MOBILE_APP = "mobile_app"


class SeverityLevel(Enum):
    """Complaint severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CustomerProfile:
    """Customer profile data structure"""
    customer_id: str
    name: str
    email: str
    phone: str
    account_number: str
    account_type: str
    registration_date: datetime
    previous_complaints: int
    credit_score: int
    monthly_balance: float
    location: str
    age: int
    occupation: str


@dataclass
class ComplaintData:
    """Complaint data structure"""
    complaint_id: str
    customer_id: str
    theme: ComplaintTheme
    title: str
    description: str
    channel: ComplaintChannel
    severity: SeverityLevel
    submission_date: datetime
    status: str
    attachments: List[str]
    related_transactions: List[Dict]
    customer_sentiment: str
    urgency_keywords: List[str]
    resolution_time_expected: int  # hours
    financial_impact: float


# =============================================================================
# MAIN SYNTHETIC DATA GENERATOR CLASS
# =============================================================================

class ComplaintConfigurationGenerator:
    """
    Generator for complaint configuration data including categories and realistic timelines.
    This replaces hardcoded values in eva_agent_service.py
    """
    
    def __init__(self, mongo_connection_string: str = "mongodb://localhost:27017/"):
        self.mongo_client = pymongo.MongoClient(mongo_connection_string)
        self.mongo_db = self.mongo_client['swiss_bank']
        self.config_collection = self.mongo_db['complaint_configuration']
    
    
    def get_realistic_timelines_config(self) -> Dict[str, Any]:
        """Generate realistic timelines configuration - COMPLETE VERSION with all categories"""
        return {
            "config_id": "realistic_timelines",
            "version": "1.0", 
            "created_at": datetime.now().isoformat(),
            "timelines": {
                "fraudulent_activities_unauthorized_transactions": {
                    "security_action": "Immediate",
                    "investigation_start": "2-4 Working hours",
                    "provisional_credit_review": "1-3 business days",
                    "final_resolution": "3-5 business days",
                    "new_card_delivery": "24-48 hours"
                },
                "account_freezes_holds_funds": {
                    "security_review": "2-4 hours",
                    "documentation_review": "4-24 hours",
                    "access_restoration": "1-2 business days",
                    "compliance_review": "1-3 business days"
                },
                "deposit_related_issues": {
                    "deposit_verification": "4-24 hours",
                    "investigation": "1-3 business days",
                    "credit_correction": "2-4 business days",
                    "reconciliation": "1-2 business days"
                },
                "dispute_resolution_issues": {
                    "case_creation": "Immediate",
                    "investigation_start": "1-2 Working hours", 
                    "provisional_credit_review": "1-2 business days",
                    "final_resolution": "3-5 business days",
                    "appeal_process": "5-10 business days"
                },
                "bank_system_policy_failures": {
                    "system_diagnosis": "2-6 hours",
                    "policy_review": "1-2 business days",
                    "correction_implementation": "2-5 business days",
                    "customer_compensation": "3-5 business days"
                },
                "atm_machine_issues": {
                    "immediate_response": "2-4 hours",
                    "transaction_verification": "4-24 hours", 
                    "credit_adjustment": "1-2 business days",
                    "equipment_investigation": "1-3 business days"
                },
                "check_related_issues": {
                    "check_verification": "1-2 business days",
                    "bank_verification": "2-3 business days",
                    "investigation": "3-5 business days",
                    "resolution": "5-7 business days"
                },
                "delays_fund_availability": {
                    "hold_review": "4-24 hours",
                    "risk_assessment": "1-2 business days",
                    "hold_release": "1-3 business days",
                    "policy_exception": "2-4 business days"
                },
                "overdraft_issues": {
                    "account_review": "2-4 hours",
                    "fee_analysis": "1-2 business days",
                    "fee_reversal_review": "2-3 business days",
                    "resolution": "2-3 business days"
                },
                "online_banking_technical_security_issues": {
                    "security_check": "Immediate",
                    "technical_investigation": "2-4 hours",
                    "system_fix": "4-24 hours",
                    "resolution": "4-24 hours"
                },
                "discrimination_unfair_practices": {
                    "complaint_escalation": "Immediate",
                    "management_review": "1-2 business days",
                    "investigation": "5-10 business days",
                    "resolution": "10-15 business days",
                    "compliance_review": "3-5 business days"
                },
                "mortgage_related_issues": {
                    "initial_review": "1-2 business days",
                    "documentation_gathering": "3-5 business days",
                    "specialist_review": "5-10 business days",
                    "underwriting_review": "7-14 business days",
                    "resolution": "10-15 business days"
                },
                "credit_card_issues": {
                    "account_review": "2-4 hours",
                    "investigation": "1-3 business days",
                    "credit_bureau_verification": "3-5 business days",
                    "resolution": "3-5 business days"
                },
                "ambiguity_unclear_unclassified": {
                    "triage_review": "4-24 hours",
                    "specialist_assignment": "1-2 business days",
                    "investigation": "2-5 business days",
                    "resolution": "3-7 business days"
                },
                "debt_collection_harassment": {
                    "immediate_review": "2-4 hours",
                    "compliance_investigation": "1-3 business days",
                    "corrective_action": "2-5 business days",
                    "resolution": "3-7 business days"
                },
                "loan_issues_auto_personal_student": {
                    "application_review": "2-5 business days",
                    "documentation_review": "5-10 business days",
                    "underwriting_decision": "7-14 business days",
                    "decision_processing": "10-15 business days"
                },
                "insurance_claim_denials_delays": {
                    "claim_review": "1-3 business days",
                    "documentation_verification": "3-5 business days",
                    "insurance_coordination": "5-10 business days",
                    "resolution": "7-14 business days"
                },
                "default": {
                    "initial_response": "2-4 hours",
                    "investigation": "1-2 business days", 
                    "resolution": "3-5 business days"
                }
            },
            "active": True
        }
    
    
    def generate_and_save_realistic_timelines_only(self) -> Optional[Dict[str, Any]]:
        """Generate and save all complaint configuration to database"""
        print("🚀 Generating complaint configuration data...")
        
        try:
            # Check if configurations already exist
            existing_config = self.config_collection.find_one({
                "config_id": "realistic_timelines", 
                "active": True
            })

            if existing_config:
                print(f"⚠️  Found existing realistic timelines configuration. Skipping generation.")
                return existing_config
            
            # Generate realistic timelines configuration
            timelines_config = self.get_realistic_timelines_config()
            result = self.config_collection.insert_one(timelines_config)
            print(f"✅ Saved realistic timelines configuration to database")
            print(f"   - {len(timelines_config['timelines'])} timeline configurations")
            print(f"   - Document ID: {result.inserted_id}")
            
            return timelines_config
            
        except Exception as e:
            print(f"❌ Error saving complaint configuration: {e}")
            return None  # Return None instead of empty list
    
    def update_timelines_configuration(self, new_timelines_data: Dict[str, Any]) -> bool:
        """Update existing realistic timelines configuration"""
        try:
            # Deactivate old version
            self.config_collection.update_one(
                {"config_id": "realistic_timelines", "active": True},
                {"$set": {"active": False, "deactivated_at": datetime.now().isoformat()}}
            )
            
            # Insert new version
            new_config = {
                "config_id": "realistic_timelines",
                "version": f"1.{int(datetime.now().timestamp())}",
                "created_at": datetime.now().isoformat(),
                "timelines": new_timelines_data,
                "active": True
            }
            
            self.config_collection.insert_one(new_config)
            print(f"✅ Updated realistic timelines configuration")
            return True
            
        except Exception as e:
            print(f"❌ Error updating realistic timelines configuration: {e}")
            return False

class SyntheticDataGenerator:
    """
    Main class for generating synthetic banking complaint data.
    """
    
    def __init__(self, mongo_connection_string: str = "mongodb://localhost:27017/"):
        """
        Initialize the synthetic data generator with database connection
        
        """
        # MongoDB setup for storing complaint data and embeddings
        self.mongo_client = pymongo.MongoClient(mongo_connection_string)
        self.mongo_db = self.mongo_client['swiss_bank']  
        self.complaints_collection = self.mongo_db['complaints']
        self.customers_collection = self.mongo_db['customers']
        
        # Initialize complaint templates for each theme
        self.complaint_templates = self._initialize_complaint_templates()
        
    # =========================================================================
    # DATABASE VALIDATION METHODS
    # =========================================================================
    
    def _customer_exists_mongodb(self, customer_id: str) -> bool:
        """Check if customer already exists in MongoDB"""
        try:
            return self.customers_collection.count_documents({"customer_id": customer_id}) > 0
        except Exception:
            return False
    
    def _complaint_exists_mongodb(self, complaint_id: str) -> bool:
        """Check if complaint already exists in MongoDB"""
        try:
            return self.complaints_collection.count_documents({"complaint_id": complaint_id}) > 0
        except Exception:
            return False
        
    # =========================================================================
    # COMPLAINT TEMPLATE INITIALIZATION
    # =========================================================================
    
    def _initialize_complaint_templates(self) -> Dict:
        """
        Initialize complaint templates for each theme.
        
        Each theme contains:
        - titles: Common complaint titles
        - descriptions: Template descriptions with placeholders
        - urgency_keywords: Keywords indicating urgency
        - severity_distribution: Probability distribution for severity levels
        
        Returns:
            Dictionary mapping complaint themes to their templates
        """
        return {
            ComplaintTheme.FRAUDULENT_ACTIVITIES: {
                "titles": [
                    "Unauthorized withdrawal from my account",
                    "Fraudulent charges on my debit card",
                    "Someone opened an account using my identity",
                    "Suspicious transactions I didn't authorize",
                    "Credit card fraud - charges I never made"
                ],
                "descriptions": [
                    "I noticed ${amount} was withdrawn from my account on {date} without my authorization. I was not at the ATM location in {location} at that time. I need immediate investigation and refund.",
                    "There are multiple charges on my card totaling ${amount} that I did not make. The transactions occurred at {merchant_name} which I have never visited. Please reverse these charges immediately.",
                    "I received notifications about a new account opened in my name, but I never applied for it. Someone has stolen my identity and I need this investigated urgently.",
                    "My account shows transactions for ${amount} that I didn't authorize. This includes purchases at {merchant_name} on {date}. I was {alibi} at that time.",
                    "I found fraudulent charges of ${amount} on my statement. The merchant {merchant_name} is unknown to me and I never authorized these transactions."
                ],
                "urgency_keywords": ["urgent", "fraudulent", "unauthorized", "stolen", "identity theft", "immediate"],
                "severity_distribution": {"critical": 0.4, "high": 0.4, "medium": 0.2, "low": 0.0}
            },
            
            ComplaintTheme.ACCOUNT_FREEZES: {
                "titles": [
                    "My account has been frozen without explanation",
                    "Cannot access my funds due to account hold",
                    "Account suspended - need immediate resolution",
                    "Funds held for no apparent reason",
                    "Account freeze causing financial hardship"
                ],
                "descriptions": [
                    "My account was frozen {days_ago} days ago without any prior notice. I have bills to pay and cannot access my ${balance}. No one can explain why this happened.",
                    "I tried to withdraw money but found my account is on hold. I deposited a ${amount} check {days_ago} days ago and now I can't access any of my funds including my salary.",
                    "Your system flagged my account for suspicious activity but it's just my regular salary deposit of ${amount}. I need access to my money immediately as I have mortgage payments due.",
                    "I cannot understand why my account is frozen. I've been a customer for {years} years and suddenly you're holding ${amount} of my money without explanation.",
                    "The account freeze is causing severe financial distress. I have {dependents} dependents and cannot pay for groceries or utilities. This needs immediate resolution."
                ],
                "urgency_keywords": ["frozen", "hold", "suspended", "cannot access", "financial hardship", "immediate"],
                "severity_distribution": {"critical": 0.3, "high": 0.5, "medium": 0.2, "low": 0.0}
            },
            
            ComplaintTheme.DEPOSIT_ISSUES: {
                "titles": [
                    "My deposit was not credited to my account",
                    "Check deposit showing wrong amount",
                    "Mobile deposit failed but check was processed",
                    "Delayed fund availability on deposit",
                    "Deposit disappeared from my account"
                ],
                "descriptions": [
                    "I deposited a check for ${amount} on {date} but it's not showing in my account. The check was from {payer} and should have cleared by now.",
                    "I made a cash deposit of ${amount} at branch {branch_name} but only ${credited_amount} shows in my account. I have the receipt showing the correct amount.",
                    "My mobile deposit of ${amount} failed with an error message, but the check was somehow processed and I can't deposit it again. Now I'm missing my money.",
                    "I deposited ${amount} on {date} but the funds are still on hold after {days} days. Your policy says funds should be available in {expected_days} business days.",
                    "A deposit of ${amount} was credited to my account and then mysteriously disappeared. I need to know where my money went and get it back immediately."
                ],
                "urgency_keywords": ["not credited", "wrong amount", "failed", "delayed", "disappeared", "missing money"],
                "severity_distribution": {"critical": 0.2, "high": 0.4, "medium": 0.3, "low": 0.1}
            },
            
            ComplaintTheme.DISPUTE_RESOLUTION: {
                "titles": [
                    "My fraud dispute was denied unfairly",
                    "No response to my dispute claim",
                    "Dispute investigation was inadequate",
                    "Refused to investigate obvious fraud",
                    "Dispute resolution taking too long"
                ],
                "descriptions": [
                    "I filed a dispute for ${amount} in fraudulent charges {days_ago} days ago but it was denied without proper investigation. The evidence clearly shows I was not at {location} when the transaction occurred.",
                    "I submitted a dispute claim {weeks_ago} weeks ago regarding unauthorized transactions totaling ${amount} but have received no updates or communication from your team.",
                    "Your investigation into my dispute was clearly inadequate. You only took {investigation_days} days to investigate ${amount} in fraudulent charges, which is not enough time for proper review.",
                    "I provided clear evidence that the ${amount} charge was fraudulent including alibis and receipts, but you still refuse to investigate properly. This is unacceptable customer service.",
                    "My dispute has been pending for {months} months now. The ${amount} in fraudulent charges is affecting my credit score and I need this resolved immediately."
                ],
                "urgency_keywords": ["denied", "no response", "inadequate", "refused", "too long", "unacceptable"],
                "severity_distribution": {"critical": 0.1, "high": 0.4, "medium": 0.4, "low": 0.1}
            },
            
            ComplaintTheme.ATM_ISSUES: {
                "titles": [
                    "ATM ate my card and dispensed no cash",
                    "ATM charged me but didn't dispense money",
                    "Deposit not credited after ATM transaction",
                    "ATM out of order after taking my transaction",
                    "Double charged by malfunctioning ATM"
                ],
                "descriptions": [
                    "The ATM at {location} took my card and charged my account ${amount} but dispensed no cash. The machine showed an error message and my card is stuck inside.",
                    "I used the ATM at {location} to withdraw ${amount} and was charged, but no money came out. The screen went blank and I couldn't get my card back immediately.",
                    "I made a ${amount} deposit at the ATM on {date} but the money never appeared in my account. I have the transaction receipt showing the deposit was accepted.",
                    "The ATM processed my withdrawal of ${amount} and then displayed 'out of order'. I was charged but received no cash and had to wait {minutes} minutes to get my card back.",
                    "I was charged twice for the same ${amount} withdrawal at the ATM on {date}. The machine malfunctioned during my transaction but processed the charge multiple times."
                ],
                "urgency_keywords": ["ate my card", "no cash", "not credited", "out of order", "malfunctioned", "charged twice"],
                "severity_distribution": {"critical": 0.2, "high": 0.3, "medium": 0.4, "low": 0.1}
            }
        }
    
    # =========================================================================
    # DATA GENERATION METHODS
    # =========================================================================
    
    def generate_customer_profile(self) -> CustomerProfile:
        """
        Generate a realistic customer profile with banking details.
        
        Returns:
            CustomerProfile object with randomized but realistic data
        """
        return CustomerProfile(
            customer_id=str(uuid.uuid4()),
            name=fake.name(),
            email=fake.email(),
            phone=generate_phone_number(),  # Using our custom phone generator
            account_number=fake.bban(),
            account_type=random.choice(["checking", "savings", "business", "premium"]),
            registration_date=datetime.combine(fake.date_between(start_date='-10y', end_date='-1y'), datetime.min.time()),
            previous_complaints=random.randint(0, 5),
            credit_score=random.randint(300, 850),
            monthly_balance=round(random.uniform(100, 50000), 2),
            location=fake.city(),
            age=random.randint(18, 80),
            occupation=fake.job()
        )
    
    def generate_related_transactions(self, theme: ComplaintTheme, amount: float) -> List[Dict]:
        """
        Generate contextual transaction history related to the complaint.
        
        Args:
            theme: The complaint theme to generate relevant transactions for
            amount: Base amount for generating related transaction amounts
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(random.randint(3, 8)):
            transaction_date = base_date + timedelta(days=random.randint(0, 30))
            
            if theme == ComplaintTheme.FRAUDULENT_ACTIVITIES:
                transactions.append({
                    "transaction_id": str(uuid.uuid4()),
                    "date": transaction_date.isoformat(),
                    "amount": round(random.uniform(10, amount * 1.5), 2),
                    "merchant": fake.company(),
                    "location": fake.city(),
                    "transaction_type": random.choice(["purchase", "withdrawal", "transfer"]),
                    "suspicious": random.choice([True, False])
                })
            elif theme == ComplaintTheme.ACCOUNT_FREEZES:
                transactions.append({
                    "transaction_id": str(uuid.uuid4()),
                    "date": transaction_date.isoformat(),
                    "amount": round(random.uniform(1000, amount * 2), 2),
                    "description": random.choice(["Salary Deposit", "Large Check Deposit", "Wire Transfer"]),
                    "status": random.choice(["pending", "cleared", "flagged"]),
                    "transaction_type": "deposit"
                })
            else:
                transactions.append({
                    "transaction_id": str(uuid.uuid4()),
                    "date": transaction_date.isoformat(),
                    "amount": round(random.uniform(10, 500), 2),
                    "description": fake.sentence(nb_words=4),
                    "transaction_type": random.choice(["purchase", "withdrawal", "deposit", "transfer"])
                })
        
        return transactions
    
    def generate_complaint(self, theme: ComplaintTheme, customer_profile: CustomerProfile) -> ComplaintData:
        """
        Generate a single complaint based on theme and customer profile.
        
        Args:
            theme: The type of complaint to generate
            customer_profile: Customer profile to associate with the complaint
            
        Returns:
            ComplaintData object with realistic complaint details
        """
        template = self.complaint_templates[theme]
        
        # Select random title and description template
        title = random.choice(template["titles"])
        description_template = random.choice(template["descriptions"])
        
        # Generate complaint-specific data
        amount = round(random.uniform(50, 5000), 2)
        date = fake.date_between(start_date='-30d', end_date='now')
        
        # Fill in the template with realistic data
        description = description_template.format(
            amount=amount,
            date=date.strftime('%m/%d/%Y'),
            location=fake.city(),
            merchant_name=fake.company(),
            alibi=random.choice(["at work", "out of town", "at home", "in the hospital"]),
            days_ago=random.randint(1, 30),
            balance=customer_profile.monthly_balance,
            years=datetime.now().year - customer_profile.registration_date.year,
            dependents=random.randint(1, 4),
            payer=fake.company(),
            credited_amount=round(amount * 0.8, 2),
            branch_name=f"Swiss Bank {fake.city()} Branch",
            expected_days=random.randint(1, 5),
            days=random.randint(1, 10),
            weeks_ago=random.randint(1, 8),
            investigation_days=random.randint(1, 3),
            months=random.randint(1, 6),
            minutes=random.randint(5, 30)
        )
        
        # Determine severity based on theme distribution
        severity_dist = template["severity_distribution"]
        severity = random.choices(
            list(SeverityLevel),
            weights=[severity_dist.get(s.value, 0) for s in SeverityLevel]
        )[0]
        
        # Generate attachments based on complaint type
        attachments = []
        if theme in [ComplaintTheme.FRAUDULENT_ACTIVITIES, ComplaintTheme.DISPUTE_RESOLUTION]:
            attachments = random.choices(
                ["bank_statement.pdf", "receipt.jpg", "police_report.pdf", "photo_evidence.jpg"],
                k=random.randint(1, 3)
            )
        
        return ComplaintData(
            complaint_id=str(uuid.uuid4()),
            customer_id=customer_profile.customer_id,
            theme=theme,
            title=title,
            description=description,
            channel=random.choice(list(ComplaintChannel)),
            severity=severity,
            submission_date=datetime.now() - timedelta(days=random.randint(0, 7)),
            status=random.choice(["new", "in_progress", "pending_review", "resolved", "escalated"]),
            attachments=attachments,
            related_transactions=self.generate_related_transactions(theme, amount),
            customer_sentiment=random.choice(["angry", "frustrated", "concerned", "neutral", "disappointed"]),
            urgency_keywords=template["urgency_keywords"],
            resolution_time_expected=random.randint(24, 100),  # 1-4 days in hours
            financial_impact=amount
        )
    
    def generate_synthetic_dataset(self, total_complaints: int = 100) -> Dict[str, List]:
        """
        Generate a complete synthetic dataset with customers and complaints.
        
        Args:
            total_complaints: Number of complaints to generate
            
        Returns:
            Dictionary containing lists of customers and complaints
        """
        customers = []
        complaints = []
        
        # Generate customers first (some customers may have multiple complaints)
        num_customers = max(total_complaints // 3, 20)
        for _ in range(num_customers):
            customers.append(self.generate_customer_profile())
        
        # Define theme distribution (realistic proportions)
        theme_distribution = {
            ComplaintTheme.FRAUDULENT_ACTIVITIES: 0.3,  # Most common - fraud concerns
            ComplaintTheme.ACCOUNT_FREEZES: 0.2,        # Account access issues
            ComplaintTheme.DEPOSIT_ISSUES: 0.2,         # Transaction problems
            ComplaintTheme.DISPUTE_RESOLUTION: 0.15,    # Follow-up complaints
            ComplaintTheme.ATM_ISSUES: 0.15             # Technical issues
        }
        
        # Generate complaints
        for _ in range(total_complaints):
            # Select theme based on realistic distribution
            theme = random.choices(
                list(ComplaintTheme),
                weights=[theme_distribution[t] for t in ComplaintTheme]
            )[0]
            
            # Select customer (allowing for multiple complaints per customer)
            customer = random.choice(customers)
            
            # Generate complaint
            complaint = self.generate_complaint(theme, customer)
            complaints.append(complaint)
        
        return {
            "customers": customers,
            "complaints": complaints
        }
    
    # =========================================================================
    # DATABASE PERSISTENCE METHODS
    # =========================================================================
    
    def save_to_mongodb(self, dataset: Dict[str, List]):
        """
        Save generated data to MongoDB (NoSQL storage for complex complaint data).
        Prevents duplicate records by checking existing data.
        
        Args:
            dataset: Generated dataset containing customers and complaints
        """
        # Prepare customers and filter out duplicates
        new_customers = []
        duplicate_customers = 0
        for customer in dataset["customers"]:
            if not self._customer_exists_mongodb(customer.customer_id):
                customer_doc = asdict(customer)
                customer_doc["registration_date"] = customer_doc["registration_date"].isoformat()
                new_customers.append(customer_doc)
            else:
                duplicate_customers += 1
        
        # Insert new customers
        if new_customers:
            self.customers_collection.insert_many(new_customers)
        
        # Prepare complaints and filter out duplicates
        new_complaints = []
        duplicate_complaints = 0
        for complaint in dataset["complaints"]:
            if not self._complaint_exists_mongodb(complaint.complaint_id):
                doc = asdict(complaint)
                # Convert enums to string values
                doc["theme"] = doc["theme"].value
                doc["channel"] = doc["channel"].value
                doc["severity"] = doc["severity"].value
                doc["submission_date"] = doc["submission_date"].isoformat()
                new_complaints.append(doc)
            else:
                duplicate_complaints += 1
        
        # Insert new complaints
        if new_complaints:
            self.complaints_collection.insert_many(new_complaints)
        
        print(f"✓ MongoDB: Saved {len(new_customers)} new customers, {len(new_complaints)} new complaints")
        if duplicate_customers > 0 or duplicate_complaints > 0:
            print(f"  (Skipped {duplicate_customers} duplicate customers, {duplicate_complaints} duplicate complaints)")
    
    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================
    
    def generate_and_save_dataset(self, total_complaints: int = 100) -> Dict[str, List]:
        """
        Main method to generate and save complete synthetic dataset.
        
        Args:
            total_complaints: Number of complaints to generate
            
        Returns:
            Generated dataset dictionary
        """
        print(f"🚀 Generating synthetic dataset with {total_complaints} complaints...")
        
        # Generate the dataset
        dataset = self.generate_synthetic_dataset(total_complaints)
        
        # Save to MongoDB
        try:
            self.save_to_mongodb(dataset)
        except Exception as e:
            print(f"❌ MongoDB save failed: {e}")
        
        print("✨ Dataset generation complete!")
        return dataset


# =============================================================================
# USAGE EXAMPLE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Initialize generator with MongoDB connection
    generator = SyntheticDataGenerator(mongo_connection_string="mongodb://localhost:27017/")
    
    # Generate and save dataset
    dataset = generator.generate_and_save_dataset(total_complaints=200)
    
    # Display sample data for verification
    print("\n" + "="*50)
    print("SAMPLE GENERATED DATA")
    print("="*50)
    
    print("\n📋 Sample Customer Profile:")
    sample_customer = asdict(dataset["customers"][0])
    print(json.dumps(sample_customer, indent=2, default=str))
    
    print("\n🎫 Sample Complaint:")
    sample_complaint = asdict(dataset["complaints"][0])
    # Convert enums to strings for JSON serialization
    sample_complaint["theme"] = sample_complaint["theme"].value
    sample_complaint["channel"] = sample_complaint["channel"].value
    sample_complaint["severity"] = sample_complaint["severity"].value
    print(json.dumps(sample_complaint, indent=2, default=str))
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   • Total Customers: {len(dataset['customers'])}")
    print(f"   • Total Complaints: {len(dataset['complaints'])}")

    # Generate configuration with proper error handling
    config_generator = ComplaintConfigurationGenerator(mongo_connection_string="mongodb://localhost:27017/")
    
    # Generate and save configuration
    timelines_config = config_generator.generate_and_save_realistic_timelines_only()
    
    # Fixed conditional check - now properly handles None case
    if timelines_config and isinstance(timelines_config, dict):
        print(f"\n📋 Generated Realistic Timelines Configuration:")
        print(f"   • Configuration ID: {timelines_config.get('config_id', 'N/A')}")
        print(f"   • Version: {timelines_config.get('version', 'N/A')}")
        print(f"   • Timeline categories: {len(timelines_config.get('timelines', {}))}")
        print(f"   • Active: {timelines_config.get('active', False)}")
    else:
        print(f"\n❌ Failed to generate or retrieve timelines configuration")