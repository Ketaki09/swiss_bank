// Path: swiss_bank_UI/src/components/agent/ChatHistory.tsx

import React, { useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import { Message } from '../../services/agentService';
import { Loader2 } from 'lucide-react';

interface ChatHistoryProps {
  messages: Message[];
  isLoading?: boolean;
  isTyping?: boolean;
}

const ChatHistory: React.FC<ChatHistoryProps> = ({ 
  messages, 
  isLoading = false,
  isTyping = false 
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // If no messages, show welcome screen without scroll container
  if (messages.length === 0 && !isLoading) {
    return (
      <div className="flex items-center justify-center h-full bg-black">
        <div className="text-center max-w-md px-4">
          <div className="mb-6">
            <div className="w-16 h-16 bg-yellow-500/20 border-2 border-yellow-400 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse-border">
              <svg className="w-8 h-8 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <p className="text-white leading-relaxed font-serif">
              How may I help you today?
            </p>
          </div>
        </div>
      </div>
    );
  }

  // If there are messages, show scrollable container
  return (
    <div className="flex-1 overflow-y-auto bg-black">
      {/* Messages */}
      {messages.map((message) => (
        <ChatMessage key={message.id} message={message} />
      ))}

      {/* Typing Indicator */}
      {isTyping && (
        <div className="flex gap-4 p-6 bg-gray-900/50 border-b border-gray-700">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-yellow-500/20 border border-yellow-400 text-yellow-400 flex items-center justify-center">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="font-medium text-sm text-white font-serif">Swiss Agent</span>
              <span className="text-xs text-yellow-400">typing...</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-yellow-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-yellow-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-yellow-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Scroll anchor */}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatHistory;