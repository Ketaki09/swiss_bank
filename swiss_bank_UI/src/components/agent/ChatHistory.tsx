// Path: swiss_bank_UI/src/components/agent/ChatHistory.tsx

import React, { useEffect, useRef, useState } from 'react';
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
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentTypingText, setCurrentTypingText] = useState('Standby...');

  // Typing text rotation
  useEffect(() => {
    if (!isTyping) return;

    const typingTexts = [
      'Standby...',
      'Ruminating...',
      'Analyzing...',
      'Processing...',
      'Thinking...'
    ];
    
    let index = 0;
    const interval = setInterval(() => {
      index = (index + 1) % typingTexts.length;
      setCurrentTypingText(typingTexts[index]);
    }, 5000); // Change every 5 seconds

    return () => clearInterval(interval);
  }, [isTyping]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // If no messages, show welcome screen
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

  // Main chat container with proper scrolling
  return (
    <div 
      ref={containerRef}
      className="h-full overflow-y-auto overflow-x-hidden bg-black scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800"
      style={{
        scrollBehavior: 'smooth'
      }}
    >
      {/* Messages Container */}
      <div className="min-h-full flex flex-col">
        {/* Spacer to push messages to bottom initially */}
        <div className="flex-1 min-h-0"></div>
        
        {/* Messages */}
        <div className="flex-shrink-0">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {/* Enhanced Typing Indicator */}
          {isTyping && (
            <div className="flex gap-4 p-6 bg-gray-900/20 border-b border-gray-700/30">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-yellow-500/20 border border-yellow-400/60 text-yellow-400 flex items-center justify-center animate-pulse-slow">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-3">
                  <span className="font-medium text-sm text-white font-serif">Swiss Agent</span>
                  <div className="flex items-center gap-2">
                    {/* Status dot with pulse */}
                    <div className="w-2 h-2 bg-yellow-400 rounded-full animate-ping"></div>
                    {/* Typing text with fade animation */}
                    <span 
                      key={currentTypingText}
                      className="text-xs text-yellow-400/80 font-serif animate-fade-in-out"
                    >
                      {currentTypingText}
                    </span>
                  </div>
                </div>
                
                {/* Sophisticated thinking animation */}
                <div className="flex items-center gap-3">
                  {/* Neural network style animation */}
                  <div className="flex items-center gap-1">
                    <div className="w-1.5 h-1.5 bg-yellow-400/60 rounded-full animate-pulse-delay-0"></div>
                    <div className="w-0.5 h-4 bg-gradient-to-t from-yellow-400/40 to-transparent animate-pulse-delay-1"></div>
                    <div className="w-1.5 h-1.5 bg-yellow-400/60 rounded-full animate-pulse-delay-2"></div>
                    <div className="w-0.5 h-4 bg-gradient-to-t from-yellow-400/40 to-transparent animate-pulse-delay-3"></div>
                    <div className="w-1.5 h-1.5 bg-yellow-400/60 rounded-full animate-pulse-delay-4"></div>
                  </div>
                  
                  {/* Brain wave animation */}
                  <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-transparent via-yellow-400/50 to-transparent w-20 animate-brain-wave"></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} className="h-1" />
        </div>
      </div>
    </div>
  );
};

export default ChatHistory;