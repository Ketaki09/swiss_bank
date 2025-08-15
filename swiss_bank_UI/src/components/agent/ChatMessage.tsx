// Path: swiss_bank_UI/src/components/agent/ChatMessage.tsx

import React from 'react';
import { User, Bot } from 'lucide-react';
import { Message } from '../../services/agentService';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-4 p-6 ${
      isUser ? 'bg-black' : 'bg-gray-900/30'
    } border-b border-gray-700/50`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center border-2 ${
        isUser 
          ? 'bg-gray-800 border-gray-600 text-gray-300' 
          : 'bg-yellow-500/20 border-yellow-400 text-yellow-400'
      }`}>
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2">
          <span className="font-medium text-sm text-white font-serif">
            {isUser ? 'You' : 'Swiss Agent'}
          </span>
          <span className="text-xs text-yellow-400">
            {new Date(message.timestamp).toLocaleTimeString([], { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </span>
        </div>
        
        <div className="prose prose-sm max-w-none">
          <div className={`whitespace-pre-wrap leading-relaxed font-serif ${
            isUser ? 'text-white font-normal' : 'text-gray-100 font-normal'
          }`}>
            {message.content}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;