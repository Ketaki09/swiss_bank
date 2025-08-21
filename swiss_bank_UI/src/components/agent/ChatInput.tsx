// Path: swiss_bank_UI/src/components/agent/ChatInput.tsx

import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, Paperclip } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  isLoading = false, 
  placeholder = "What do you want to know?",
  disabled = false 
}) => {
  const [message, setMessage] = useState('');
  const [isListening, setIsListening] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 120)}px`;
    }
  }, [message]);

  // Initialize speech recognition
  useEffect(() => {
    if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setMessage(prev => prev + transcript);
        setIsListening(false);
      };

      recognitionRef.current.onerror = () => {
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (disabled || isLoading || !message.trim()) {
      return;
    }

    onSendMessage(message.trim());
    setMessage('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const toggleVoiceInput = () => {
    if (disabled || isLoading) return;

    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      if (recognitionRef.current) {
        recognitionRef.current.start();
        setIsListening(true);
      }
    }
  };

  const handleAttachment = () => {
    // Handle file attachment logic here
    console.log('Attachment clicked');
  };

  // Combined button logic: voice when empty, send when has text
  const hasText = message.trim().length > 0;
  const canSend = !disabled && !isLoading && hasText;
  const canUseVoice = !disabled && !isLoading && recognitionRef.current && !hasText;

  const handleVoiceSendButton = () => {
  if (hasText) {
    // Send message with a mock React.FormEvent
    handleSubmit({
      preventDefault: () => {},
    } as React.FormEvent<HTMLFormElement>);
  } else {
    // Toggle voice
    toggleVoiceInput();
  }
};


  return (
    <div className="flex flex-col items-center w-full gap-1 p-2 font-serif">
      <form onSubmit={handleSubmit} className="w-full text-base flex flex-col gap-2 items-center justify-center relative z-10 mt-2">
        <div className="flex flex-col gap-0 justify-center w-full relative items-center max-w-4xl">
          {/* Main Input Container */}
          <div className="query-bar group z-10 bg-black ring-yellow-400 hover:ring-yellow-500 focus-within:ring-yellow-500 relative w-full overflow-hidden shadow shadow-black/5 max-w-4xl ring-1 ring-inset focus-within:ring-1 pb-12 px-2 sm:px-3 rounded-3xl transition-all duration-100">
            
            {/* Textarea Container */}
            <div className="relative z-10">
              {/* Placeholder */}
              {!message && (
                <span className="absolute px-2 sm:px-3 py-5 text-gray-400 pointer-events-none select-none font-serif">
                  {placeholder}
                </span>
              )}
              
              {/* Textarea */}
              <textarea
                ref={textareaRef}
                dir="auto"
                aria-label="Ask Swiss Agent anything"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={disabled || isLoading}
                className="w-full px-2 sm:px-3 bg-transparent focus:outline-none text-white align-bottom min-h-14 pt-5 my-0 mb-5 resize-none font-serif"
                style={{ height: '44px' }}
              />
            </div>

            {/* Bottom Controls */}
            <div className="flex gap-1.5 absolute inset-x-0 bottom-0 border-2 border-transparent p-2 max-w-full">
              
              {/* Attachment Button */}
              <button
                type="button"
                onClick={handleAttachment}
                disabled={disabled || isLoading}
                className="inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium leading-normal cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-yellow-400 disabled:opacity-60 disabled:cursor-not-allowed transition-colors duration-100 select-none border border-gray-600 text-yellow-400 hover:bg-gray-800 disabled:hover:bg-transparent h-10 w-10 rounded-full"
                aria-label="Attach"
              >
                <Paperclip className="w-4 h-4" />
              </button>

              {/* Spacer */}
              <div className="flex grow gap-1.5 max-w-full">
                <div className="grow flex gap-1.5 max-w-full">
                  {/* Empty space - no middle button */}
                </div>
              </div>

              {/* Combined Voice/Send Button */}
              <div className="ml-auto flex flex-row items-end gap-1">
                <button
                  type={hasText ? "submit" : "button"}
                  onClick={hasText ? undefined : handleVoiceSendButton}
                  disabled={disabled || isLoading || (!hasText && !canUseVoice)}
                  className={`group flex flex-col justify-center rounded-full focus:outline-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-yellow-400 transition-all duration-200 ${
                    (hasText && canSend) || (!hasText && canUseVoice)
                      ? hasText
                        ? 'bg-yellow-500 hover:bg-yellow-600 text-black hover:scale-105 active:scale-95'
                        : isListening
                          ? 'bg-red-600 hover:bg-red-700 text-white'
                          : 'bg-yellow-500 hover:bg-yellow-600 text-black hover:scale-105 active:scale-95'
                      : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  }`}
                  aria-label={hasText ? "Send message" : isListening ? "Stop recording" : "Start voice input"}
                >
                  <div className="h-10 w-10 relative aspect-square flex items-center justify-center rounded-full">
                    {isLoading ? (
                      <div className="w-4 h-4 animate-spin rounded-full border-2 border-gray-400 border-t-current"></div>
                    ) : hasText ? (
                      <Send className="w-4 h-4" />
                    ) : isListening ? (
                      <MicOff className="w-4 h-4" />
                    ) : (
                      <div className="flex items-center justify-center gap-0.5">
                        {/* Voice Wave Animation */}
                        <div className="w-0.5 rounded-full bg-current" style={{height: '0.4rem'}}></div>
                        <div className="w-0.5 rounded-full bg-current" style={{height: '0.8rem'}}></div>
                        <div className="w-0.5 rounded-full bg-current" style={{height: '1.2rem'}}></div>
                        <div className="w-0.5 rounded-full bg-current" style={{height: '0.7rem'}}></div>
                        <div className="w-0.5 rounded-full bg-current" style={{height: '1rem'}}></div>
                        <div className="w-0.5 rounded-full bg-current" style={{height: '0.4rem'}}></div>
                      </div>
                    )}
                  </div>
                </button>
              </div>
            </div>
          </div>
        </div>
      </form>

      {/* Status Messages */}
      <div className="w-full max-w-4xl">
        {isListening && (
          <div className="text-center">
            <span className="text-sm text-red-400 font-serif animate-pulse">
              🎤 Listening... Speak now
            </span>
          </div>
        )}
        
        {disabled && (
          <div className="text-center">
            <span className="text-sm text-gray-500 font-serif">
              Service unavailable - please check connection
            </span>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="w-full max-w-4xl mt-3">
        <p className="text-xs text-yellow-100 text-center font-serif">
          Swiss Agent can make mistakes. Please double-check with official bank policies.
        </p>
      </div>
    </div>
  );
};

export default ChatInput;