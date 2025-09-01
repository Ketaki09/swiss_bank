"""
Internal Swiss Agent Package
High-performance banking assistant with domain guardrails
"""

from .optimized_swiss_agent import create_optimized_swiss_agent, initialize_claude_client
from .rag_service import create_anthropic_rag_service, AnthropicContextualRAGService, RetrievalStrategy

__all__ = [
    'create_optimized_swiss_agent',
    'initialize_claude_client',
    'create_anthropic_rag_service', 
    'AnthropicContextualRAGService',
    'RetrievalStrategy'
]