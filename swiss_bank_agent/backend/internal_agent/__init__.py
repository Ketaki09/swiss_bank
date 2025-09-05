"""
Internal Swiss Agent Package
High-performance banking assistant with domain guardrails
"""

from .optimized_swiss_agent import (
    create_optimized_swiss_agent, 
    initialize_claude_client,
    OptimizedSwissAgent,
    BankingDomainGuardrails,
    EnhancedConversationMemory,
    IntentType,
    IntentClassificationResult,
    PerformanceBenchmark,
    create_swiss_agent_with_sessions
)

from .rag_service import (
    create_anthropic_rag_service, 
    AnthropicContextualRAGService, 
    RetrievalStrategy,
    DuplicateHandlingMode,
    ChunkingStrategy
)

__all__ = [
    # Factory functions
    'create_optimized_swiss_agent',
    'create_swiss_agent_with_sessions',
    'initialize_claude_client',
    'create_anthropic_rag_service',
    
    # Core classes
    'OptimizedSwissAgent',
    'AnthropicContextualRAGService',
    'BankingDomainGuardrails',
    'EnhancedConversationMemory',
    'PerformanceBenchmark',
    
    # Enums and types
    'IntentType',
    'IntentClassificationResult',
    'RetrievalStrategy',
    'DuplicateHandlingMode',
    'ChunkingStrategy'
]