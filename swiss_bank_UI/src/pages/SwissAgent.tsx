// Path: swiss_bank_UI/src/pages/SwissAgent.tsx

import React, { useState, useCallback } from 'react';
import { ArrowLeft, Plus, MessageSquare } from 'lucide-react';
import ChatHistory from '../components/agent/ChatHistory';
import ChatInput from '../components/agent/ChatInput';
import agentService, { Message } from '../services/agentService';

const SwissAgent: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleSendMessage = useCallback(async (content: string) => {
    // Create user message
    const userMessage: Message = {
      id: agentService.generateMessageId(),
      content,
      role: 'user',
      timestamp: new Date(),
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setIsTyping(true);

    try {
      // Send message to backend
      const response = await agentService.sendMessage(content);
      
      // Simulate typing delay for better UX
      setTimeout(() => {
        setIsTyping(false);
        
        // Create assistant message
        const assistantMessage: Message = {
          id: agentService.generateMessageId(),
          content: response.message,
          role: 'assistant',
          timestamp: new Date(),
        };

        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);
      }, 1000);

    } catch (error) {
      setIsTyping(false);
      setIsLoading(false);
      
      // Add error message
      const errorMessage: Message = {
        id: agentService.generateMessageId(),
        content: 'Sorry, I encountered an error. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  }, []);

  const handleGoBack = () => {
    // Navigate back to home page
    window.location.href = '/';
  };

  const handleNewChat = () => {
    setMessages([]);
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="flex flex-col h-screen bg-black font-serif">
      {/* Full Width Navigation Bar */}
      <header className="flex items-center justify-between p-4 border-b border-black bg-black w-full">
        <div className="flex items-center gap-3">
          <button
            onClick={handleGoBack}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors duration-200"
            title="Back to Home"
          >
            <ArrowLeft className="w-5 h-5 text-yellow-400" />
          </button>
          
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-black rounded-full flex items-center justify-center overflow-hidden">
              <img 
                src="/Images_upload/bank_logo.png" 
                alt="Swiss Bank Logo" 
                className="w-8 h-8 object-contain"
              />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-white font-serif">Swiss Agent</h1>
              <p className="text-sm text-yellow-400 font-serif">Internal Banking Assistant</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Layout Below Navigation */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Below Navigation Bar */}
        <div className={`flex-shrink-0 border-r border-black bg-black transition-all duration-200 ${sidebarOpen ? 'w-72' : 'w-14'}`}>
          <nav className="h-full flex flex-col gap-3 pb-2 px-0 transition duration-200">
            {/* Sidebar Header with Toggle Button */}
            <div className="flex w-full items-center gap-px p-2">
              <button
                onClick={toggleSidebar}
                className="inline-flex items-center justify-center relative shrink-0 select-none disabled:pointer-events-none disabled:opacity-50 text-yellow-400 border-transparent transition font-serif tracking-tight duration-300 ease-custom-ease hover:bg-gray-800 aria-pressed:bg-gray-700 aria-checked:bg-gray-700 aria-expanded:bg-gray-700 hover:text-yellow-300 aria-pressed:text-yellow-300 aria-checked:text-yellow-300 aria-expanded:text-yellow-300 h-8 w-8 rounded-md active:scale-95 group aria-expanded:bg-transparent aria-expanded:hover:bg-gray-800"
                type="button"
                data-testid="pin-sidebar-toggle"
                aria-label="Sidebar"
                aria-expanded={sidebarOpen}
              >
                <div className="relative w-4 h-4">
                  <div className={`flex items-center justify-center transition-all duration-300 text-yellow-400 ${sidebarOpen ? 'opacity-100 scale-100 group-hover:opacity-0 group-hover:scale-80' : 'opacity-0 scale-50 group-hover:opacity-100 group-hover:scale-100'}`} style={{width: '16px', height: '16px'}}>
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg" className="shrink-0">
                      <path d="M16.5 4C17.3284 4 18 4.67157 18 5.5V14.5C18 15.3284 17.3284 16 16.5 16H3.5C2.67157 16 2 15.3284 2 14.5V5.5C2 4.67157 2.67157 4 3.5 4H16.5ZM7 15H16.5C16.7761 15 17 14.7761 17 14.5V5.5C17 5.22386 16.7761 5 16.5 5H7V15ZM3.5 5C3.22386 5 3 5.22386 3 5.5V14.5C3 14.7761 3.22386 15 3.5 15H6V5H3.5Z"></path>
                    </svg>
                  </div>
                  <div className={`flex items-center justify-center absolute inset-0 transition-all duration-300 text-yellow-300 ${sidebarOpen ? 'opacity-0 scale-50 group-hover:scale-100 group-hover:opacity-100' : 'opacity-100 scale-100 group-hover:opacity-0 group-hover:scale-80'}`} style={{width: '16px', height: '16px'}}>
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg" className="shrink-0">
                      <path d="M3.5 3C3.77614 3 4 3.22386 4 3.5V16.5L3.99023 16.6006C3.94371 16.8286 3.74171 17 3.5 17C3.25829 17 3.05629 16.8286 3.00977 16.6006L3 16.5V3.5C3 3.22386 3.22386 3 3.5 3ZM11.2471 5.06836C11.4476 4.95058 11.7104 4.98547 11.8721 5.16504C12.0338 5.34471 12.0407 5.60979 11.9023 5.79688L11.835 5.87207L7.80371 9.5H16.5C16.7761 9.5 17 9.72386 17 10C17 10.2761 16.7761 10.5 16.5 10.5H7.80371L11.835 14.1279C12.0402 14.3127 12.0568 14.6297 11.8721 14.835C11.6873 15.0402 11.3703 15.0568 11.165 14.8721L6.16504 10.3721L6.09473 10.2939C6.03333 10.2093 6 10.1063 6 10C6 9.85828 6.05972 9.72275 6.16504 9.62793L11.165 5.12793L11.2471 5.06836Z"></path>
                    </svg>
                  </div>
                </div>
              </button>
              
              {/* Swiss Agent Logo - Only show when expanded */}
              {sidebarOpen && (
                <div className="flex flex-col justify-start items-top ml-1 transition-all duration-200">
                </div>
              )}
            </div>

            {/* New Chat Button */}
            <div className="flex flex-col px-2 pt-1 gap-px mb-6">
              <div className="mb-1">
                <button
                  onClick={handleNewChat}
                  className={`inline-flex items-center justify-center relative shrink-0 select-none disabled:pointer-events-none disabled:opacity-50 h-9 rounded-lg min-w-[5rem] active:scale-[0.985] whitespace-nowrap group transition ease-in-out active:!scale-100 hover:bg-transparent hover:!bg-yellow-500/[0.08] active:!bg-yellow-500/15 font-serif ${sidebarOpen ? 'px-4 py-2 flex !justify-start !min-w-0 w-full' : 'w-8 !p-0 !min-w-0'}`}
                  aria-label="New chat"
                  title={!sidebarOpen ? "New chat" : undefined}
                >
                  {sidebarOpen ? (
                    <div className="-mx-3 flex flex-row items-center gap-2">
                      <div className="w-6 h-6 flex items-center justify-center group-active:!scale-[0.98] group-active:!shadow-none group-active:bg-yellow-600 group-hover:-rotate-2 group-hover:scale-105 group-active:rotate-3 rounded-full transition-all ease-in-out bg-yellow-500 group-hover:shadow-md">
                        <div className="flex items-center justify-center group-hover:scale-105 transition text-black" style={{width: '12px', height: '12px'}}>
                          <Plus className="w-3 h-3" />
                        </div>
                      </div>
                      <div className="transition-all duration-200 text-yellow-400 font-medium text-sm tracking-tight">New chat</div>
                    </div>
                  ) : (
                    <div className="w-4 h-4 flex items-center justify-center rounded-full transition-all ease-in-out bg-yellow-500 group-hover:bg-yellow-600">
                      <Plus className="w-3 h-3 text-black" />
                    </div>
                  )}
                </button>
              </div>
            </div>

            {/* Chat History Section */}
            <div className="flex flex-grow flex-col overflow-y-auto overflow-x-hidden relative px-2 mb-2">
              <div className="transition-all duration-200">
                <div className="flex flex-col">
                  {sidebarOpen && (
                    <h3 className="text-yellow-400 pb-2 mt-1 text-xs select-none pl-2 sticky top-0 z-10 bg-gradient-to-b from-black from-50% to-black/40 font-serif">Chat History</h3>
                  )}
                  <ul className="flex flex-col gap-px">
                    <li style={{opacity: 1, height: 'auto'}}>
                      <div className="relative group">
                        <div className={`inline-flex items-center justify-center relative shrink-0 select-none disabled:pointer-events-none disabled:opacity-50 text-yellow-400 border-transparent transition font-serif tracking-tight duration-300 ease-custom-ease hover:bg-gray-800 h-8 rounded-md active:scale-[0.985] whitespace-nowrap !text-xs overflow-hidden !min-w-0 group active:bg-gray-700 active:scale-[1.0] ${sidebarOpen ? 'px-3 min-w-[4rem] w-full px-4' : 'w-8 !px-0 !min-w-0'}`}
                          title={!sidebarOpen ? "No previous chats" : undefined}
                        >
                          {sidebarOpen ? (
                            <div className="-translate-x-2 w-full flex flex-row items-center justify-start gap-3">
                              <MessageSquare className="w-4 h-4 text-yellow-400 opacity-60" />
                              <span className="truncate text-sm whitespace-nowrap w-full text-yellow-400 opacity-60 font-serif">No previous chats</span>
                            </div>
                          ) : (
                            <MessageSquare className="w-4 h-4 text-yellow-400 opacity-60" />
                          )}
                        </div>
                      </div>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Gradient fade at bottom - only when expanded */}
            {sidebarOpen && (
              <div className="bg-gradient-to-t from-black to-transparent h-4"></div>
            )}
          </nav>
        </div>

        {/* Centered Chat Area */}
        <div className="flex flex-col flex-1 min-w-0 bg-black">
          {/* Chat Messages - Centered Content */}
          <div className="flex-1 overflow-hidden flex items-center justify-center">
            <div className="w-full max-w-4xl mx-auto h-full flex flex-col">
              <div className="flex-1 overflow-hidden">
                <ChatHistory 
                  messages={messages} 
                  isLoading={isLoading}
                  isTyping={isTyping}
                />
              </div>
            </div>
          </div>

          {/* Chat Input and Footer - Fixed at bottom, also centered */}
          <div className="flex-shrink-0">
            <div className="w-full max-w-4xl mx-auto">
              <ChatInput 
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                placeholder="Ask Swiss Agent about banking policies, procedures, or any internal questions..."
              />

              {/* Footer */}
              <div className="border-t border-black bg-black px-4 py-2">
                <p className="text-xs text-yellow-400 text-center font-serif">
                  Swiss Agent can make mistakes. Please double-check with official bank policies.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SwissAgent;