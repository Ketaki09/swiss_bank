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
}

class AgentService {
  private baseURL: string;

  constructor() {
    // Use default backend URL directly
    this.baseURL = 'http://localhost:8000';
  }

  async sendMessage(message: string): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseURL}/api/swiss-agent/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          timestamp: new Date().toISOString(),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return {
        message: data.response,
        success: true,
      };
    } catch (error) {
      console.error('Error sending message to Agent:', error);
      return {
        message: 'Sorry, I encountered an error while processing your request. Please try again.',
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async getChatHistory(): Promise<Message[]> {
    try {
      const response = await fetch(`${this.baseURL}/api/swiss-agent/history`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.messages || [];
    } catch (error) {
      console.error('Error fetching chat history:', error);
      return [];
    }
  }

  generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

export default new AgentService();
export type { Message, ChatResponse };