// Path: swiss_bank_UI/src/components/agent/ChatInput.tsx

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
  placeholder?: string;
}

const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  isLoading = false,
  placeholder = "Message Swiss Agent..."
}) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedMessage = message.trim();
    if (trimmedMessage && !isLoading) {
      onSendMessage(trimmedMessage);
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  return (
    <div className="border-t border-black bg-black p-4">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto w-full">
        <div className="relative flex items-center w-full bg-gray-800 border border-black rounded-xl focus-within:ring-2 focus-within:ring-yellow-400 focus-within:border-yellow-400 transition-all duration-200">
          {/* Message Input */}
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="flex-1 px-4 py-3 pr-12 bg-transparent text-white placeholder:text-gray-400 resize-none focus:outline-none min-h-[52px] font-sans text-base leading-relaxed"
            style={{
              height: 'auto',
              overflow: 'hidden',
              fontFamily:
                "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
              fontWeight: 400,
            }}
            disabled={isLoading}
            rows={1}
          />

          {/* Send Button Inside Input */}
          <button
            type="submit"
            disabled={!message.trim() || isLoading}
            className={`absolute right-2 bottom-2 p-2 rounded-lg transition-all duration-200 ${
              message.trim() && !isLoading
                ? 'bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-400 hover:to-yellow-500 text-black shadow-md hover:shadow-lg hover:shadow-yellow-400/25'
                : 'bg-gray-700 text-gray-400 cursor-not-allowed border border-black'
            }`}
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInput;