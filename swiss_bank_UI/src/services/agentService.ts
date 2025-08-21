// Path: swiss_bank_UI/src/services/agentService.ts

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

interface ChatResponse {
  message: string;
  success: boolean;
  error?: string;
  message_id?: string;
  timestamp?: string;
}

interface SwissAgentResponse {
  success: boolean;
  response: string;
  timestamp: string;
  message_id: string;
  query_enhanced?: boolean;
  source?: string;
  documents_found?: number;
  error?: string;
}

interface SwissAgentHistoryResponse {
  success: boolean;
  messages: Array<{
    role: string;
    content: string;
    timestamp: string;
    user_id?: string;
    source?: string;
    documents_used?: number;
  }>;
  count: number;
  ai_powered: boolean;
  semantic_understanding_active: boolean;
  framework_type: string;
}

// Added proper interfaces for performance and components
interface PerformanceMetrics {
  response_time_ms?: number;
  cpu_usage_percent?: number;
  memory_usage_mb?: number;
  cache_hit_rate?: number;
  active_sessions?: number;
  requests_per_minute?: number;
}

interface SystemComponents {
  database?: {
    status: string;
    connection_pool_size?: number;
    query_cache_enabled?: boolean;
  };
  ai_service?: {
    status: string;
    model_version?: string;
    context_window_size?: number;
  };
  semantic_engine?: {
    status: string;
    embedding_model?: string;
    vector_store_size?: number;
  };
}

// Added proper interface for semantic context
interface SemanticContext {
  user_id: string;
  conversation_context: {
    topics: string[];
    entities: string[];
    sentiment?: string;
    intent?: string;
  };
  semantic_memory: {
    relevant_documents?: number;
    context_relevance_score?: number;
    memory_window_size?: number;
  };
  query_enhancements: {
    expanded_terms?: string[];
    synonyms_used?: string[];
    context_applied?: boolean;
  };
}

interface SwissAgentStatusResponse {
  status: string;
  service: string;
  version: string;
  timestamp: string;
  ai_capabilities: {
    claude_api: boolean;
    semantic_memory: boolean;
    entity_extraction: boolean;
    topic_identification: boolean;
    reference_resolution: boolean;
    domain_agnostic: boolean;
    anthropic_methodology: boolean;
  };
  capabilities: {
    ai_powered_entity_extraction: boolean;
    ai_powered_topic_identification: boolean;
    semantic_query_enhancement: boolean;
    reference_resolution: boolean;
    domain_adaptive: boolean;
    no_hardcoded_patterns: boolean;
    generalized_for_mnc_fintech: boolean;
  };
  performance?: PerformanceMetrics;
  components?: SystemComponents;
}

class AgentService {
  private baseURL: string;
  private currentUserId: string;

  constructor() {
    // Use your backend URL - update this to match your backend port
    this.baseURL = 'http://localhost:8001'; // Updated to match your main.py port
    
    // Generate a unique user ID for this session (for internal employee)
    this.currentUserId = this.generateEmployeeUserId();
  }

  private generateEmployeeUserId(): string {
    // Generate a unique employee session ID
    const timestamp = Date.now();
    const random = Math.random().toString(36).substr(2, 9);
    return `employee_${timestamp}_${random}`;
  }

  async sendMessage(message: string): Promise<ChatResponse> {
    try {
      // Prepare the request body to match SwissAgentMessage model
      const requestBody = {
        message: message.trim(),
        timestamp: new Date().toISOString(),
        user_id: this.currentUserId
      };

      console.log('Sending message to Swiss Agent:', requestBody);

      const response = await fetch(`${this.baseURL}/api/swiss-agent/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('HTTP error response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
      }

      const data: SwissAgentResponse = await response.json();
      console.log('Swiss Agent response:', data);

      if (!data.success) {
        throw new Error(data.error || 'Swiss Agent processing failed');
      }

      return {
        message: data.response,
        success: true,
        message_id: data.message_id,
        timestamp: data.timestamp,
      };

    } catch (error) {
      console.error('Error sending message to Swiss Agent:', error);
      
      // Return user-friendly error message
      let errorMessage = 'Sorry, I encountered an error while processing your request. Please try again.';
      
      if (error instanceof Error) {
        // Check for specific error types
        if (error.message.includes('Failed to fetch')) {
          errorMessage = 'Unable to connect to Swiss Agent service. Please check if the backend is running.';
        } else if (error.message.includes('503')) {
          errorMessage = 'Swiss Agent service is temporarily unavailable. Please try again in a moment.';
        } else if (error.message.includes('500')) {
          errorMessage = 'Swiss Agent encountered an internal error. Please try rephrasing your question.';
        }
      }

      return {
        message: errorMessage,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async getChatHistory(): Promise<Message[]> {
    try {
      const url = `${this.baseURL}/api/swiss-agent/history?user_id=${encodeURIComponent(this.currentUserId)}&limit=50`;
      console.log('Fetching chat history from:', url);

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SwissAgentHistoryResponse = await response.json();
      console.log('Chat history response:', data);

      if (!data.success) {
        console.warn('Failed to fetch chat history');
        return [];
      }

      // Convert backend message format to frontend Message interface
      const messages: Message[] = data.messages.map((msg, index) => ({
        id: this.generateMessageId(),
        content: msg.content,
        role: msg.role === 'user' ? 'user' : 'assistant',
        timestamp: new Date(msg.timestamp),
      }));

      return messages;

    } catch (error) {
      console.error('Error fetching chat history:', error);
      return [];
    }
  }

  async clearChatHistory(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/api/swiss-agent/history?user_id=${encodeURIComponent(this.currentUserId)}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Clear history response:', data);

      return data.success || false;

    } catch (error) {
      console.error('Error clearing chat history:', error);
      return false;
    }
  }

  async getServiceStatus(): Promise<SwissAgentStatusResponse | null> {
    try {
      const response = await fetch(`${this.baseURL}/api/swiss-agent/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SwissAgentStatusResponse = await response.json();
      console.log('Swiss Agent status:', data);

      return data;

    } catch (error) {
      console.error('Error fetching Swiss Agent status:', error);
      return null;
    }
  }

  async getSemanticContext(): Promise<SemanticContext | null> {
    try {
      const response = await fetch(`${this.baseURL}/api/swiss-agent/semantic-context?user_id=${encodeURIComponent(this.currentUserId)}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SemanticContext = await response.json();
      console.log('Semantic context:', data);

      return data;

    } catch (error) {
      console.error('Error fetching semantic context:', error);
      return null;
    }
  }

  generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Get current user ID for this session
  getCurrentUserId(): string {
    return this.currentUserId;
  }

  // Utility method to check if backend is available
  async testConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      return response.ok;
    } catch (error) {
      console.error('Backend connection test failed:', error);
      return false;
    }
  }
}

export default new AgentService();
export type { 
  Message, 
  ChatResponse, 
  SwissAgentResponse, 
  SwissAgentStatusResponse, 
  SemanticContext, 
  PerformanceMetrics, 
  SystemComponents 
};