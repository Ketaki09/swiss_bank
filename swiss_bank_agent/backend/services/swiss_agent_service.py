# Path: Swiss_bank_agent/backend/services/swiss_agent_service.py

import logging
import os
import threading
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

from collections import deque
from dataclasses import dataclass, field

# AI and Vector Database
import anthropic
from anthropic.types import TextBlock

# Configuration
from pathlib import Path
from dotenv import load_dotenv
from .rag_service import AnthropicContextualRAGService, RetrievalStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationEntity:
    """AI-extracted entity with semantic understanding"""
    text: str
    category: str 
    confidence: float
    first_mentioned: datetime = field(default_factory=datetime.now)
    last_mentioned: datetime = field(default_factory=datetime.now)
    mention_count: int = 1
    context_snippets: List[str] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)

@dataclass 
class ConversationTopic:
    """AI-identified conversation topic"""
    topic_name: str
    semantic_description: str
    confidence: float
    entities_involved: Set[str] = field(default_factory=set)
    last_mentioned: datetime = field(default_factory=datetime.now)
    message_count: int = 0

class AISemanticMemory:
    """
    Generalized AI-powered conversation memory using Claude for semantic understanding
    
    This approach mirrors Anthropic's methodology:
    - No hardcoded domain patterns
    - AI-driven entity extraction
    - Semantic topic identification  
    - Context-aware reference resolution
    - Dynamic domain adaptation
    """
    
    def __init__(self, claude_client,rate_limiter_func=None, max_history: int = 100, context_window: int = 20):
        self.claude_client = claude_client
        self.rate_limiter_func = rate_limiter_func 
        self.max_history = max_history
        self.context_window = context_window
        
        # AI-powered semantic components
        self.entities: Dict[str, ConversationEntity] = {}
        self.topics: Dict[str, ConversationTopic] = {}
        self.message_history: deque = deque(maxlen=max_history)
        
        # Semantic understanding cache
        self.semantic_cache: Dict[str, Dict[str, Any]] = {}
        self.reference_resolution_cache: Dict[str, str] = {}
        
        # Current conversation state
        self.current_topic: Optional[str] = None
        self.conversation_domain: Optional[str] = None
    
    def add_message(self, message: Dict[str, Any]):
        """Add message and perform AI-powered semantic analysis"""
        self.message_history.append(message)
        
        if message.get("role") == "user":
            content = message.get("content", "")
            
            # AI-powered semantic analysis
            self._extract_entities_with_ai(content)
            self._identify_topics_with_ai(content)

    
    def _extract_entities_with_ai(self, content: str):
        """Use AI to extract entities from any domain - fully generalized"""
        if not self.claude_client or len(content.strip()) < 10:
            return
        
        # Check cache first
        content_hash = str(hash(content))
        if content_hash in self.semantic_cache:
            cached_entities = self.semantic_cache[content_hash].get("entities", [])
            self._process_extracted_entities(cached_entities, content)
            return
        
        prompt = f"""Analyze this text and extract all meaningful entities with their semantic categories.

Text: "{content}"

Extract entities that could be important for conversation continuity in a financial services context. Include:
- Names (people, companies, products, projects, systems)
- Financial instruments (ETFs, funds, accounts, etc.)
- Processes (procedures, workflows, operations)
- Concepts (strategies, methodologies, frameworks)
- Any other domain-specific entities

Return a JSON array of entities with this format:
[
    {{
        "text": "entity name",
        "category": "semantic category (e.g., person, project, financial_instrument, process, concept, etc.)",
        "confidence": 0.95
    }}
]

Return only the JSON array, no other text."""

        try:
            def make_claude_call():
                if self.claude_client is None:
                    raise ValueError("Claude client not initialized")
                return self.claude_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            # CHANGE: Use rate limiter if available, otherwise direct call
            if self.rate_limiter_func:
                response = self.rate_limiter_func(make_claude_call, "entity_extraction")
            else:
                response = make_claude_call()
            
            response_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    response_text += block.text
                elif isinstance(block, str):
                    response_text += block
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    entities = json.loads(json_str)
                    
                    # Cache the result
                    self.semantic_cache[content_hash] = {"entities": entities}
                    
                    # Process extracted entities
                    self._process_extracted_entities(entities, content)
            
            except json.JSONDecodeError:
                # Fallback: simple entity extraction
                self._fallback_entity_extraction(content)
                
        except Exception as e:
            logger.warning(f"⚠️ AI entity extraction failed: {e}")
            self._fallback_entity_extraction(content)
    
    def _process_extracted_entities(self, entities: List[Dict[str, Any]], content: str):
        """Process AI-extracted entities into conversation memory"""
        for entity_data in entities:
            if not isinstance(entity_data, dict):
                continue
                
            entity_text = entity_data.get("text", "").strip()
            category = entity_data.get("category", "unknown")
            confidence = entity_data.get("confidence", 0.5)
            
            if not entity_text or confidence < 0.3:
                continue
            
            entity_key = entity_text.lower()
            
            if entity_key in self.entities:
                # Update existing entity
                self.entities[entity_key].last_mentioned = datetime.now()
                self.entities[entity_key].mention_count += 1
                
                # Add context snippet if not too many
                if len(self.entities[entity_key].context_snippets) < 3:
                    snippet = content[:150] + "..." if len(content) > 150 else content
                    self.entities[entity_key].context_snippets.append(snippet)
            else:
                # Create new entity
                snippet = content[:150] + "..." if len(content) > 150 else content
                self.entities[entity_key] = ConversationEntity(
                    text=entity_text,
                    category=category,
                    confidence=confidence,
                    context_snippets=[snippet]
                )
    
        
    def _identify_topics_with_ai(self, content: str):
        """Use AI to identify conversation topics - fully generalized"""
        if not self.claude_client or len(content.strip()) < 15:
            return
        
        # Get recent conversation context for topic identification
        recent_messages = list(self.message_history)[-5:]
        context_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}"
            for msg in recent_messages
        ])
        
        prompt = f"""Analyze the conversation context and current message to identify the main topic being discussed.

Recent conversation:
{context_text}

Current message: "{content}"

Identify the primary topic of discussion. Consider:
- What domain/area is being discussed (banking, investments, operations, etc.)
- What specific aspect within that domain
- The semantic theme of the conversation

Return a JSON object with this format:
{{
    "topic_name": "concise topic name",
    "semantic_description": "brief description of what this topic covers",
    "confidence": 0.95,
    "domain": "general domain area"
}}

Return only the JSON object, no other text."""

        try:
            def make_claude_call():
                if self.claude_client is None:
                    raise ValueError("Claude client not initialized")
                return self.claude_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=300,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            # CHANGE: Use rate limiter if available, otherwise direct call
            if self.rate_limiter_func:
                response = self.rate_limiter_func(make_claude_call, "topic_identification")
            else:
                response = make_claude_call()
            
            response_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    response_text += block.text
                elif isinstance(block, str):
                    response_text += block
            
            # Parse JSON response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    topic_data = json.loads(json_str)
                    
                    topic_name = topic_data.get("topic_name", "general_discussion")
                    semantic_desc = topic_data.get("semantic_description", "")
                    confidence = topic_data.get("confidence", 0.5)
                    domain = topic_data.get("domain", "unknown")
                    
                    if confidence > 0.4:
                        # Update or create topic
                        if topic_name in self.topics:
                            self.topics[topic_name].last_mentioned = datetime.now()
                            self.topics[topic_name].message_count += 1
                        else:
                            self.topics[topic_name] = ConversationTopic(
                                topic_name=topic_name,
                                semantic_description=semantic_desc,
                                confidence=confidence
                            )
                        
                        self.current_topic = topic_name
                        if not self.conversation_domain:
                            self.conversation_domain = domain
            
            except json.JSONDecodeError:
                pass  # Continue without topic identification
                
        except Exception as e:
            logger.warning(f"⚠️ AI topic identification failed: {e}")
    
    
    def _fallback_entity_extraction(self, content: str):
        """Simple fallback when AI extraction fails"""
        # Capitalize words that might be entities
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', content)
        
        for word_phrase in capitalized_words:
            if len(word_phrase.strip()) > 2 and word_phrase not in ['The', 'This', 'That', 'What', 'Where', 'When', 'How']:
                entity_key = word_phrase.lower().strip()
                
                if entity_key not in self.entities:
                    snippet = content[:150] + "..." if len(content) > 150 else content
                    self.entities[entity_key] = ConversationEntity(
                        text=word_phrase,
                        category="unknown",
                        confidence=0.3,
                        context_snippets=[snippet]
                    )
    
    def resolve_references_with_ai(self, message: str) -> str:
        """AI-powered reference resolution - works across all domains"""
        if not self.claude_client:
            return message
        
        # Check if message likely contains references
        reference_indicators = ["it", "that", "this", "they", "them", "one", "ones", "which"]
        message_lower = message.lower()
        
        if not any(indicator in message_lower for indicator in reference_indicators):
            return message
        
        # Check cache
        cache_key = f"{message}|{len(self.entities)}"
        if cache_key in self.reference_resolution_cache:
            return self.reference_resolution_cache[cache_key]
        
        # Get recent entities for context
        recent_entities = []
        if self.entities:
            # Sort by last mentioned time, take top 5
            sorted_entities = sorted(
                self.entities.values(),
                key=lambda x: x.last_mentioned,
                reverse=True
            )[:5]
            
            for entity in sorted_entities:
                recent_entities.append({
                    "text": entity.text,
                    "category": entity.category,
                    "last_mentioned": entity.last_mentioned.isoformat(),
                    "context": entity.context_snippets[-1] if entity.context_snippets else ""
                })
        
        if not recent_entities:
            return message
        
        prompt = f"""Resolve references in this message using conversation context.

Recent entities mentioned:
{json.dumps(recent_entities, indent=2)}

Current topic: {self.current_topic or "unknown"}

Message with references: "{message}"

Replace pronouns and references (it, that, this, they, the first one, etc.) with the appropriate entity names based on conversation context.

Return only the resolved message, no other text."""

        try:
            def make_claude_call():
                if self.claude_client is None:
                    raise ValueError("Claude client not initialized")
                return self.claude_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=200,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            # CHANGE: Use rate limiter if available, otherwise direct call
            if self.rate_limiter_func:
                resolved_response = self.rate_limiter_func(make_claude_call, "reference_resolution")
            else:
                resolved_response = make_claude_call()
            
            resolved_message = ""
            for block in resolved_response.content:
                if hasattr(block, 'text'):
                    resolved_message += block.text
                elif isinstance(block, str):
                    resolved_message += block
            
            resolved_message = resolved_message.strip()
            
            # Cache the result
            if resolved_message and resolved_message != message:
                self.reference_resolution_cache[cache_key] = resolved_message
                return resolved_message
            
        except Exception as e:
            logger.warning(f"⚠️ AI reference resolution failed: {e}")
        
        return message
    
    def get_relevant_context(self, current_message: str) -> Dict[str, Any]:
        """Get AI-curated relevant context for current message"""
        current_message_lower = current_message.lower()
        
        # Find entities mentioned or related to current message
        relevant_entities = []
        for entity_key, entity_info in self.entities.items():
            if (entity_key in current_message_lower or 
                entity_info.text.lower() in current_message_lower):
                relevant_entities.append(entity_info)
        
        # Get current topic info
        current_topic_info = None
        if self.current_topic and self.current_topic in self.topics:
            current_topic_info = self.topics[self.current_topic]
        
        # Get recent high-quality messages
        recent_messages = list(self.message_history)[-self.context_window:]
        
        return {
            "relevant_entities": relevant_entities,
            "current_topic": current_topic_info,
            "conversation_domain": self.conversation_domain,
            "recent_messages": recent_messages,
            "entity_count": len(self.entities),
            "topic_count": len(self.topics),
            "semantic_memory_active": True
        }

class SwissAgentService:
    """
    Generalized Swiss Agent Service following Anthropic's AI-first approach
    
    Key principles:
    - AI-powered semantic understanding (no hardcoded patterns)
    - Domain-agnostic conversation memory
    - Adaptive to any financial services domain
    - Claude-driven entity extraction and topic identification
    - Semantic reference resolution
    """
    
    def __init__(self, 
                 rag_service_instance=None,
                 chroma_db_path: str = "./chroma_db",
                 collection_name: str = "contextual_documents",
                 embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
                 quiet_queries: bool = True):
        """Initialize Generalized Swiss Agent with AI-powered semantic understanding"""
        
        self.quiet_queries = quiet_queries
        self._setup_consistent_logging()

        # API rate limiting coordination
        self.api_call_delay = 1.0
        self.last_api_call_time = 0
        self.api_call_lock = threading.Lock()

        # Agent-specific cost tracking
        self.agent_api_calls_count = 0
        self.agent_tokens_used = 0
        self.agent_cost_estimate = 0.0

        # Cost tracking per agent operation type
        self.agent_operation_costs = {
            "query_enhancement": {"calls": 0, "tokens": 0, "cost": 0.0},
            "response_generation": {"calls": 0, "tokens": 0, "cost": 0.0},
            "entity_extraction": {"calls": 0, "tokens": 0, "cost": 0.0},
            "topic_identification": {"calls": 0, "tokens": 0, "cost": 0.0},
            "reference_resolution": {"calls": 0, "tokens": 0, "cost": 0.0}
        }
        
        # Initialize Claude client
        self.claude_client = None
        self._initialize_claude()
        
        # Initialize AI-powered semantic memory
        self.semantic_memory = AISemanticMemory(
            claude_client=self.claude_client,
            max_history=100,
            context_window=20
        )

        # Initialize RAG service
        if rag_service_instance:
            self.rag_service = rag_service_instance
            self.rag_available = True
        else:
            self._connect_to_rag_service(chroma_db_path, collection_name, embedding_model)
        
        self.max_retrieval_results = 5
        
        if not self.quiet_queries:
            logger.info(f"✅ Swiss Agent initialized with {'existing' if rag_service_instance else 'new'} RAG connection")

    def _connect_to_rag_service(self, chroma_db_path: str, collection_name: str, embedding_model: str):
        """Create lightweight connection to existing RAG service"""
        try:
            # CHANGE: Create RAG service in query-only mode (no processing capabilities)
            self.rag_service = AnthropicContextualRAGService(
                chroma_db_path=chroma_db_path,
                collection_name=collection_name,
                retrieval_strategy=RetrievalStrategy.CONTEXTUAL_HYBRID,
                embedding_model=embedding_model,
                quiet_mode=True 
            )
            
            # Test connection
            health = self.rag_service.health_check()
            self.rag_available = health["status"] == "healthy"
            
            if not self.rag_available:
                logger.warning("⚠️ RAG service connection unhealthy - agent will use Claude-only mode")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to RAG service: {e}")
            self.rag_service = None
            self.rag_available = False

    def _setup_consistent_logging(self):
        """Setup consistent logging across services"""
        if self.quiet_queries:
            loggers_to_quiet = [
                'httpx', 'chromadb', 'sentence_transformers', 
                'chromadb.telemetry', 'urllib3', '__main__', 'numexpr.utils'
            ]
            for logger_name in loggers_to_quiet:
                logging.getLogger(logger_name).setLevel(logging.WARNING)

    def _rate_limited_api_call(self, api_call_func, operation_type="general", *args, **kwargs):
        """API rate limiting with cost tracking for agent operations"""
        with self.api_call_lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call_time
            
            if time_since_last_call < self.api_call_delay:
                sleep_time = self.api_call_delay - time_since_last_call
                time.sleep(sleep_time)
            
            try:
                self.agent_api_calls_count += 1
                response = api_call_func(*args, **kwargs)
                self.last_api_call_time = time.time()
                
                # ADD: Track cost for agent operations
                if response and hasattr(response, 'usage'):
                    input_tokens = getattr(response.usage, 'input_tokens', 0)
                    output_tokens = getattr(response.usage, 'output_tokens', 0)
                    total_tokens = input_tokens + output_tokens
                    
                    # Estimate cost (Claude 3.5 Haiku pricing)
                    cost = (input_tokens * 0.25 + output_tokens * 1.25) / 1_000_000
                    
                    self.agent_tokens_used += total_tokens
                    self.agent_cost_estimate += cost
                    
                    # Track by operation type
                    if operation_type in self.agent_operation_costs:
                        self.agent_operation_costs[operation_type]["calls"] += 1
                        self.agent_operation_costs[operation_type]["tokens"] += total_tokens
                        self.agent_operation_costs[operation_type]["cost"] += cost
                    
                    if not self.quiet_queries:
                        logger.info(f"💰 Agent API call ({operation_type}): {total_tokens} tokens, ~${cost:.4f}")
                
                return response
                
            except Exception as e:
                if not self.quiet_queries:
                    logger.warning(f"⚠️ Agent API call failed: {e}")
                self.last_api_call_time = time.time()
                return None
    
    def _initialize_claude(self):
        """Initialize Claude API client for agent operations with dedicated API key"""
        try:
            backend_dir = Path(__file__).parent.parent
            env_path = backend_dir / '.env'
            load_dotenv(env_path)
            
            # CHANGE: Use Agent-specific API key with fallback
            api_key = os.getenv("AGENT_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("⚠️ AGENT_API_KEY environment variable not set")
                logger.info("💡 Set AGENT_API_KEY in .env file for agent operations")
                return
            
            self.claude_client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"❌ Failed to initialize Claude API for agent: {e}")
            self.claude_client = None
    
    def _enhance_query_with_ai_semantic_context(self, message: str, user_id: Optional[str] = None) -> str:
        """Enhanced query enhancement with better error handling"""
        if not self.claude_client:
            return self.semantic_memory.resolve_references_with_ai(message)
        
        try:
            # Get semantic context
            context_info = self.semantic_memory.get_relevant_context(message)
            
            # Resolve references first
            resolved_message = self.semantic_memory.resolve_references_with_ai(message)
            
            # Check if enhancement is needed
            if not any([
                context_info["relevant_entities"],
                context_info["current_topic"],
                context_info["conversation_domain"]
            ]):
                return resolved_message
            
            # Build context for enhancement
            context_parts = []
            
            if context_info["relevant_entities"]:
                entity_context = [f"- {e.text} ({e.category})" for e in context_info["relevant_entities"][:3]]
                context_parts.extend(["Relevant entities:"] + entity_context)
            
            if context_info["current_topic"]:
                topic = context_info["current_topic"]
                context_parts.append(f"Topic: {topic.topic_name} - {topic.semantic_description}")
            
            if context_info["conversation_domain"]:
                context_parts.append(f"Domain: {context_info['conversation_domain']}")
            
            if not context_parts:
                return resolved_message
            
            # Enhanced prompt for query improvement
            context_text = "\n".join(context_parts)
            
            prompt = f"""Enhance this financial services query using conversation context.

    Context:
    {context_text}

    Query: "{resolved_message}"

    Instructions:
    1. If the query references conversation context, make it standalone
    2. Add relevant entity names and context
    3. Preserve original intent while adding necessary context
    4. Keep it concise and search-friendly
    5. Return unchanged if already clear

    Enhanced query:"""

            def make_claude_call():
                if not self.claude_client:
                    return resolved_message
                return self.claude_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=200,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            response = self._rate_limited_api_call(make_claude_call, "query_enhancement")
            
            if response:
                enhanced_query = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        enhanced_query += block.text
                    elif isinstance(block, str):
                        enhanced_query += block
                
                enhanced_query = enhanced_query.strip()
                
                if enhanced_query and enhanced_query != resolved_message:
                    if not self.quiet_queries:
                        logger.info(f"🧠 Query enhanced: {message[:50]}... -> {enhanced_query[:50]}...")
                    return enhanced_query
            
            return resolved_message
            
        except Exception as e:
            logger.warning(f"⚠️ Query enhancement failed: {e}")
            return message
        
    def _generate_ai_enhanced_response(self, query: str, search_results: Optional[Dict[str, Any]] = None) -> str:
        """Generate AI-enhanced response with improved direct answer handling"""
        if not self.claude_client:
            return "AI processing is not available. Please contact the relevant department for assistance."
        
        try:
            # Extract search results with better error handling
            context_documents = []
            direct_answer = None
            search_metadata = {}
            
            if search_results and search_results.get("success"):
                context_documents = search_results.get("documents", [])[:3]
                direct_answer = search_results.get("direct_answer")
                search_metadata = {
                    "retrieval_method": search_results.get("retrieval_method"),
                    "total_results": search_results.get("total_results", 0),
                    "enhanced_query": search_results.get("enhanced_query"),
                    "contextual_enhancement": search_results.get("contextual_enhancement", False)
                }
                
                # Validate direct answer quality
                if direct_answer and isinstance(direct_answer, str):
                    if len(direct_answer.strip()) < 20 or "Information not available" in direct_answer:
                        direct_answer = None

            # Get conversation context
            context_info = self.semantic_memory.get_relevant_context(query)
            
            # Build improved prompt with structured sections
            prompt_sections = [
                "You are an intelligent financial services assistant.",
                "Provide comprehensive, actionable responses using all available information.",
                ""
            ]
            
            # Handle direct answer with validation
            if direct_answer:
                prompt_sections.extend([
                    f"PRIMARY ANSWER (from document analysis): {direct_answer}",
                    "",
                    "Instructions for enhancing this answer:",
                    "1. Verify the information makes sense and is complete",
                    "2. Add relevant context and explanations",
                    "3. Provide actionable next steps or recommendations",
                    "4. Structure the response clearly with specific details",
                    ""
                ])
            
            # Add supporting documents with relevance scoring
            if context_documents:
                prompt_sections.append("SUPPORTING DOCUMENTATION:")
                for i, doc in enumerate(context_documents, 1):
                    source = doc.get("source_file", "Unknown")
                    content = doc.get("content", "")[:400]
                    similarity = doc.get("similarity_score", 0)
                    fusion_score = doc.get("fusion_score", 0)
                    
                    prompt_sections.append(
                        f"[Source {i}] {source} (similarity: {similarity:.3f}, fusion: {fusion_score:.3f}):\n{content}"
                    )
                prompt_sections.append("")
            
            # Add conversation context with entity awareness
            if context_info.get("relevant_entities"):
                entities = [f"{e.text} ({e.category})" for e in context_info["relevant_entities"][:3]]
                prompt_sections.append(f"Conversation Context - Entities: {', '.join(entities)}")
            
            if context_info.get("current_topic"):
                topic = context_info["current_topic"]
                prompt_sections.append(f"Current Topic: {topic.topic_name} - {topic.semantic_description}")
            
            prompt_sections.extend([
                f"User Question: {query}",
                "",
                "Response Requirements:",
                "1. Provide a clear, professional answer",
                "2. Use specific information from documents when available",
                "3. Include relevant procedures, contacts, or next steps",
                "4. Maintain conversation continuity",
                "5. If information is limited, acknowledge this clearly",
                "",
                "Response:"
            ])
            
            prompt = "\n".join(prompt_sections)
            
            # Generate response with rate limiting
            def make_claude_call():
                if not self.claude_client:
                    return "AI processing is not available. Please contact the relevant department for assistance."
                
                return self.claude_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1200,  # Increased for more comprehensive responses
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            message = self._rate_limited_api_call(make_claude_call, "response_generation")
            
            if message is None:
                return "I'm experiencing technical difficulties. Please try again or contact support."
            
            # Extract and process response
            response_text = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    response_text += block.text
                elif isinstance(block, str):
                    response_text += block
            
            # Add structured source attribution
            if context_documents:
                sources = []
                for doc in context_documents:
                    source = doc.get("source_file", "Unknown")
                    similarity = doc.get("similarity_score", 0)
                    sources.append(f"{source} (relevance: {similarity:.3f})")
                
                response_text += f"\n\n**Sources:** {', '.join(sources)}"
            
            # Add search metadata if available
            if search_metadata.get("retrieval_method"):
                response_text += f"\n*Search method: {search_metadata['retrieval_method']}*"
            
            return response_text.strip() if response_text else "I was unable to generate a response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"❌ AI response generation failed: {e}")
            return "I encountered an error while generating a response. Please try your question again."

    async def process_message(self, message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process message optimized for production use with direct RAG service calls"""
        try:
            if not self.quiet_queries:
                logger.info(f"🔍 Processing: {message[:100]}...")
            
            # Store user message in semantic memory
            user_message = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
            self.semantic_memory.add_message(user_message)
            
            # AI-powered semantic query enhancement
            enhanced_query = self._enhance_query_with_ai_semantic_context(message, user_id)
            
            # Streamlined response structure for production
            response_data = {
                "success": True,
                "response": "",
                "timestamp": datetime.now().isoformat(),
                "message_id": f"msg_{int(datetime.now().timestamp() * 1000)}",
                "query_enhanced": enhanced_query != message
            }
            
            # CHANGE: Call RAG service directly instead of using query_documents method
            if self.rag_available and self.rag_service:
                try:
                    # Direct call to RAG service's query_documents method
                    search_results = self.rag_service.query_documents(
                        query=enhanced_query,
                        top_k=self.max_retrieval_results,
                        retrieval_strategy=RetrievalStrategy.CONTEXTUAL_HYBRID
                    )
                    
                    if search_results.get("success") and search_results.get("documents"):
                        response_text = self._generate_ai_enhanced_response(message, search_results)
                        
                        # Simplified metadata for production
                        response_data.update({
                            "response": response_text,
                            "source": "rag_enhanced",
                            "documents_found": len(search_results.get("documents", [])),
                            "direct_answer": search_results.get("direct_answer"),
                            "sources": [
                                doc.get("source_file", "Unknown") 
                                for doc in search_results.get("documents", [])[:3]
                            ],
                            "search_method": search_results.get("retrieval_method", "contextual_hybrid")
                        })
                    else:
                        # Search returned no documents - use Claude only
                        response_text = self._generate_ai_enhanced_response(message, None)
                        response_data.update({
                            "response": response_text,
                            "source": "claude_only_no_docs",
                            "documents_found": 0,
                            "search_status": "no_documents_found"
                        })
                        
                except Exception as e:
                    logger.error(f"❌ RAG service call failed: {e}")
                    response_text = self._generate_ai_enhanced_response(message, None)
                    response_data.update({
                        "response": response_text,
                        "source": "claude_only_rag_error",
                        "error": str(e),
                        "fallback_reason": "rag_service_exception"
                    })
            else:
                # RAG service not available - Claude only
                response_text = self._generate_ai_enhanced_response(message, None)
                response_data.update({
                    "response": response_text,
                    "source": "claude_only",
                    "rag_status": "unavailable",
                    "fallback_reason": "rag_service_unavailable"
                })
            
            # Store assistant response with operation type for cost tracking
            assistant_message = {
                "role": "assistant",
                "content": response_data["response"],
                "timestamp": response_data["timestamp"],
                "user_id": user_id,
                "source": response_data.get("source"),
                "documents_used": response_data.get("documents_found", 0)
            }
            self.semantic_memory.add_message(assistant_message)
            
            return response_data
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            return {
                "success": False,
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "source": "error"
            }
        
    
    def health_check(self) -> Dict[str, Any]:
        """Production-focused health check"""
        try:
            return {
                "status": "healthy" if self.claude_client else "degraded",
                "service": "swiss_agent_production",
                "components": {
                    "claude_api": "available" if self.claude_client else "unavailable",
                    "rag_connection": "available" if self.rag_available else "unavailable",
                    "semantic_memory": "active",
                    "ai_capabilities": "active"
                },
                "performance": {
                    "total_messages": len(self.semantic_memory.message_history),
                    "entities_tracked": len(self.semantic_memory.entities),
                    "topics_identified": len(self.semantic_memory.topics),
                    "current_domain": self.semantic_memory.conversation_domain
                },
                "api_usage": {
                    "rate_limit_delay": self.api_call_delay,
                    "last_call": datetime.fromtimestamp(self.last_api_call_time).isoformat() if self.last_api_call_time > 0 else "never"
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    def get_agent_cost_statistics(self) -> Dict[str, Any]:
        """Get detailed cost statistics for agent operations"""
        return {
            "service": "swiss_agent",
            "api_key_used": "AGENT_API_KEY",
            "total_api_calls": self.agent_api_calls_count,
            "total_tokens_used": self.agent_tokens_used,
            "estimated_total_cost": round(self.agent_cost_estimate, 4),
            "cost_breakdown": {
                operation: {
                    "calls": data["calls"],
                    "tokens": data["tokens"],
                    "cost": round(data["cost"], 4),
                    "avg_cost_per_call": round(data["cost"] / max(data["calls"], 1), 4),
                    "avg_tokens_per_call": round(data["tokens"] / max(data["calls"], 1), 1)
                }
                for operation, data in self.agent_operation_costs.items()
                if data["calls"] > 0  # Only show operations that were used
            },
            "conversation_stats": {
                "total_messages": len(self.semantic_memory.message_history),
                "entities_extracted": len(self.semantic_memory.entities),
                "topics_identified": len(self.semantic_memory.topics),
                "current_domain": self.semantic_memory.conversation_domain
            },
            "efficiency_metrics": {
                "cost_per_message": round(self.agent_cost_estimate / max(len(self.semantic_memory.message_history), 1), 4),
                "tokens_per_message": round(self.agent_tokens_used / max(len(self.semantic_memory.message_history), 1), 1)
            },
            "timestamp": datetime.now().isoformat()
        }

    def get_combined_cost_report(self) -> Dict[str, Any]:
        """Get combined cost report including RAG service if available"""
        agent_costs = self.get_agent_cost_statistics()
        
        combined_report = {
            "agent_service": agent_costs,
            "total_agent_cost": agent_costs["estimated_total_cost"]
        }
        
        # Add RAG costs if service is available
        if self.rag_service and hasattr(self.rag_service, 'get_rag_cost_statistics'):
            rag_costs = self.rag_service.get_rag_cost_statistics()
            combined_report["rag_service"] = rag_costs
            combined_report["total_rag_cost"] = rag_costs["estimated_total_cost"]
            combined_report["total_combined_cost"] = round(
                agent_costs["estimated_total_cost"] + rag_costs["estimated_total_cost"], 4
            )
        
        combined_report["timestamp"] = datetime.now().isoformat()
        return combined_report

    def clear_ai_semantic_memory(self, user_id: Optional[str] = None):
        """Clear AI-powered semantic memory for fresh start"""
        if user_id:
            # Clear specific user's messages
            self.semantic_memory.message_history = deque([
                msg for msg in self.semantic_memory.message_history 
                if msg.get("user_id") != user_id
            ], maxlen=self.semantic_memory.max_history)
        else:
            # Clear all AI semantic memory
            self.semantic_memory = AISemanticMemory(
                claude_client=self.claude_client,
                max_history=self.semantic_memory.max_history,
                context_window=self.semantic_memory.context_window
            )
        
        logger.info(f"🧹 AI semantic memory cleared for {'user ' + user_id if user_id else 'all users'}")
    
    def get_ai_semantic_context(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive AI-powered semantic context and memory state"""
        context_info = self.semantic_memory.get_relevant_context("")
        
        return {
            "ai_semantic_memory": {
                "total_messages": len(self.semantic_memory.message_history),
                "ai_extracted_entities": len(self.semantic_memory.entities),
                "ai_identified_topics": len(self.semantic_memory.topics),
                "current_topic": self.semantic_memory.current_topic,
                "conversation_domain": self.semantic_memory.conversation_domain,
                "approach": "generalized_ai_powered"
            },
            "ai_extracted_entities": {
                name: {
                    "text": entity.text,
                    "ai_category": entity.category,
                    "confidence": entity.confidence,
                    "mentions": entity.mention_count,
                    "last_mentioned": entity.last_mentioned.isoformat(),
                    "context_snippets": entity.context_snippets[-2:] if entity.context_snippets else []
                }
                for name, entity in self.semantic_memory.entities.items()
            },
            "ai_identified_topics": {
                topic_id: {
                    "topic_name": topic.topic_name,
                    "semantic_description": topic.semantic_description,
                    "confidence": topic.confidence,
                    "message_count": topic.message_count,
                    "last_mentioned": topic.last_mentioned.isoformat(),
                    "entities_involved": list(topic.entities_involved)
                }
                for topic_id, topic in self.semantic_memory.topics.items()
            },
            "capabilities": {
                "domain_agnostic": True,
                "ai_powered_entity_extraction": True,
                "ai_powered_topic_identification": True,
                "ai_powered_reference_resolution": True,
                "semantic_query_enhancement": True,
                "adaptive_to_any_financial_domain": True,
                "no_hardcoded_patterns": True
            }
        }
    
    def get_conversation_history(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history with AI semantic metadata"""
        try:
            if user_id:
                history = [
                    msg for msg in self.semantic_memory.message_history 
                    if msg.get("user_id") == user_id
                ]
            else:
                history = list(self.semantic_memory.message_history)
            
            # Apply limit
            limited_history = history[-limit:] if limit else history
            
            # Add AI semantic metadata
            for i, msg in enumerate(limited_history):
                if msg.get("role") == "assistant":
                    msg["ai_semantic_metadata"] = {
                        "entities_at_time": len(self.semantic_memory.entities),
                        "topics_at_time": len(self.semantic_memory.topics),
                        "message_index": i,
                        "ai_powered": True,
                        "domain_adaptive": True
                    }
            
            return limited_history
            
        except Exception as e:
            logger.error(f"❌ Error getting conversation history: {e}")
            return []

    
# Convenience functions for creating generalized AI-powered Swiss Agent
def create_generalized_swiss_agent(
    chroma_db_path: str = "./chroma_db",
    collection_name: str = "contextual_documents", 
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
    quiet_queries: bool = True
) -> SwissAgentService:
    """
    Create a generalized AI-powered Swiss Agent Service following Anthropic's methodology
    
    Key Features:
    - AI-powered semantic understanding (no hardcoded patterns)
    - Domain-agnostic conversation memory
    - Adaptive to any financial services domain (private banking, corporate banking, ETFs, etc.)
    - Claude-driven entity extraction and topic identification
    - Semantic reference resolution
    - Generalized for MNC fintech with diverse domains
    """
    return SwissAgentService(
        chroma_db_path=chroma_db_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        quiet_queries=quiet_queries
    )

# Backward compatibility functions
def create_enhanced_swiss_agent(
    chroma_db_path: str = "./chroma_db",
    collection_name: str = "contextual_documents",
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"  
) -> SwissAgentService:
    """Create an instance of SwissAgentService with enhanced RAG capabilities"""
    return SwissAgentService(
        chroma_db_path=chroma_db_path,
        collection_name=collection_name,
        embedding_model=embedding_model
    )

def create_swiss_agent(
    chroma_db_path: str = "./chroma_db",
    collection_name: str = "contextual_documents",
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
) -> SwissAgentService:
    """Backward compatibility function with generalized AI-powered approach"""
    return create_generalized_swiss_agent(chroma_db_path, collection_name, embedding_model)
