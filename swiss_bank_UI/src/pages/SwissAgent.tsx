// Path: swiss_bank_UI/src/pages/SwissAgent.tsx

import React, { useState, useCallback, useEffect } from 'react';
import { ArrowLeft, Plus, MessageSquare, Wifi, WifiOff, AlertCircle, Search, ChevronRight, MoreVertical, ChevronsLeft, ChevronsRight } from 'lucide-react';
import ChatHistory from '../components/agent/ChatHistory';
import ChatInput from '../components/agent/ChatInput';
import WelcomeScreen from '../components/agent/WelcomeScreen';
import agentService, { Message, SwissAgentStatusResponse } from '../services/agentService';

const SwissAgent: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [historyExpanded, setHistoryExpanded] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [serviceStatus, setServiceStatus] = useState<SwissAgentStatusResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Mock chat history for demonstration - replace with actual chat sessions
  const mockChatSessions = [
    { id: '2', title: 'Banking Policy Questions', date: 'Today', active: false },
    { id: '3', title: 'Account Management Procedures', date: 'June', active: false },
    { id: '4', title: 'Compliance Guidelines', date: 'June', active: false },
    { id: '5', title: 'Customer Service Protocols', date: 'May', active: false },
    { id: '6', title: 'Risk Assessment Framework', date: 'May', active: false },
  ];

  // Create current conversation entry if there are active messages
  const getCurrentConversationTitle = () => {
    if (messages.length === 0) return 'New Conversation';
    
    // Get the first user message to create a title
    const firstUserMessage = messages.find(msg => msg.role === 'user');
    if (firstUserMessage) {
      // Take first few words of the message as title
      const words = firstUserMessage.content.trim().split(' ');
      const title = words.slice(0, 4).join(' ');
      return title.length > 30 ? title.substring(0, 27) + '...' : title;
    }
    return 'Current Conversation';
  };

  // Combine current conversation with mock sessions
  const allSessions = messages.length > 0 
    ? [
        { 
          id: 'current', 
          title: getCurrentConversationTitle(), 
          date: 'Today', 
          active: true 
        },
        ...mockChatSessions
      ]
    : mockChatSessions;

  // Group chat sessions by date
  const groupedSessions = allSessions.reduce((acc, session) => {
    if (!acc[session.date]) {
      acc[session.date] = [];
    }
    acc[session.date].push(session);
    return acc;
  }, {} as Record<string, typeof allSessions>);

  // Check backend connection and service status on component mount
  useEffect(() => {
    const checkConnection = async () => {
      setConnectionStatus('checking');
      
      const isConnected = await agentService.testConnection();
      if (isConnected) {
        setConnectionStatus('connected');
        
        // Get service status
        const status = await agentService.getServiceStatus();
        setServiceStatus(status);
        
        // Load chat history
        const history = await agentService.getChatHistory();
        setMessages(history);
        
        setErrorMessage(null);
      } else {
        setConnectionStatus('disconnected');
        setErrorMessage('Unable to connect to Swiss Agent service. Please ensure the backend is running on localhost:8001.');
      }
    };

    checkConnection();
  }, []);

  const handleSendMessage = useCallback(async (content: string) => {
    // Check connection before sending
    if (connectionStatus === 'disconnected') {
      setErrorMessage('Cannot send message. Backend service is not available.');
      return;
    }

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
    setErrorMessage(null);

    try {
      // Send message to backend
      const response = await agentService.sendMessage(content);
      
      if (response.success) {
        // Simulate typing delay for better UX
        setTimeout(() => {
          setIsTyping(false);
          
          // Create assistant message
          const assistantMessage: Message = {
            id: response.message_id || agentService.generateMessageId(),
            content: response.message,
            role: 'assistant',
            timestamp: new Date(response.timestamp || new Date()),
          };

          setMessages(prev => [...prev, assistantMessage]);
          setIsLoading(false);
        }, 1000);
      } else {
        throw new Error(response.error || 'Unknown error occurred');
      }

    } catch (error) {
      setIsTyping(false);
      setIsLoading(false);
      
      const errorMsg = error instanceof Error ? error.message : 'An unexpected error occurred';
      setErrorMessage(errorMsg);
      
      // Add error message to chat
      const errorMessage: Message = {
        id: agentService.generateMessageId(),
        content: `Sorry, I encountered an error: ${errorMsg}. Please try again.`,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  }, [connectionStatus]);

  const handleGoBack = () => {
    window.location.href = '/';
  };

  const handleNewChat = async () => {
    if (connectionStatus === 'connected') {
      const success = await agentService.clearChatHistory();
      if (success) {
        setMessages([]);
        setErrorMessage(null);
      } else {
        setErrorMessage('Failed to clear chat history. Please try again.');
      }
    } else {
      setMessages([]);
      setErrorMessage(null);
    }
  };

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Wifi className="w-4 h-4 text-green-400" />;
      case 'disconnected':
        return <WifiOff className="w-4 h-4 text-red-400" />;
      case 'checking':
        return <Wifi className="w-4 h-4 text-yellow-400 animate-pulse" />;
      default:
        return <WifiOff className="w-4 h-4 text-gray-400" />;
    }
  };

  const getConnectionText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'disconnected':
        return 'Disconnected';
      case 'checking':
        return 'Connecting...';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="flex h-screen bg-black font-serif">
      {/* Grok-Style Sidebar */}
      <div className={`flex-shrink-0 border-r border-gray-800 bg-black transition-all duration-200 ${sidebarOpen ? 'w-72' : 'w-16'}`}>
        <div className="flex h-full w-full flex-col">
          
          {/* Header - Logo Section */}
          <div className="h-16 flex flex-row justify-between items-center gap-0 shrink-0 px-2">
            {sidebarOpen ? (
              <div className="flex items-center gap-3 px-1">
                <div className="w-10 h-10 bg-black rounded-full flex items-center justify-center overflow-hidden">
                  <img 
                    src="/Images_upload/bank_logo.png" 
                    alt="Swiss Bank Logo" 
                    className="w-8 h-8 object-contain"
                  />
                </div>
              </div>
            ) : (
              <div className="w-full flex justify-center">
                <div className="w-10 h-10 bg-black rounded-full flex items-center justify-center overflow-hidden">
                  <img 
                    src="/Images_upload/bank_logo.png" 
                    alt="Swiss Bank Logo" 
                    className="w-6 h-6 object-contain"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Content */}
          <div className="flex min-h-0 flex-col overflow-auto grow relative overflow-x-hidden" style={{maskImage: 'linear-gradient(black 85%, transparent 100%)', maskComposite: 'add'}}>
            
            {/* Search Bar */}
            {sidebarOpen && (
              <div className="relative w-full min-w-0 flex-col px-3 shrink-0 transition-[width,transform,opacity] duration-200 h-12 flex justify-center py-1">
                <button className="flex items-center gap-2 overflow-hidden text-left outline-none ring-yellow-400 transition-[width,height,padding] focus-visible:ring-1 hover:text-yellow-400 text-sm hover:bg-gray-800 flex-1 px-3 rounded-full border border-gray-700 bg-gray-900 justify-start text-yellow-400 h-10 mx-1">
                  <div className="flex items-center justify-center w-6 h-6 shrink-0">
                    <Search className="w-4 h-4" />
                  </div>
                  <span className="space-x-1 align-baseline">
                    <span>Search</span>
                    <span className="text-xs text-gray-400">Ctrl+K</span>
                  </span>
                </button>
              </div>
            )}

            {/* New Chat Section */}
            <div className="relative flex w-full min-w-0 flex-col px-3 py-1 shrink-0 transition-[width,transform,opacity] duration-200">
              <ul className="flex w-full min-w-0 flex-col cursor-default gap-px">
                <li className="group/menu-item whitespace-nowrap font-semibold mx-1 relative">
                  <button
                    onClick={handleNewChat}
                    disabled={connectionStatus === 'disconnected'}
                    className={`flex items-center gap-2 overflow-hidden rounded-xl text-left outline-none ring-yellow-400 transition-[width,height,padding] focus-visible:ring-1 hover:text-yellow-400 text-sm h-9 border-transparent hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed w-full flex-row group/sidebar-item transition-colors p-2 text-yellow-400 border-transparent ${
                      sidebarOpen ? 'justify-start' : 'justify-center'
                    }`}
                    title={!sidebarOpen ? "New chat" : undefined}
                  >
                    {sidebarOpen ? (
                      <div className="flex flex-row items-center gap-2">
                        <div className="w-6 h-6 flex items-center justify-center group-active:scale-95 group-hover:scale-105 group-hover:bg-yellow-600 rounded-full transition-all ease-in-out bg-yellow-500 group-hover:shadow-md">
                          <Plus className="w-3 h-3 text-black" />
                        </div>
                        <div className="transition-all duration-200 text-yellow-400 font-medium text-sm tracking-tight">
                          New chat
                        </div>
                      </div>
                    ) : (
                      <div className="w-4 h-4 flex items-center justify-center rounded-full transition-all ease-in-out bg-yellow-500 group-hover:bg-yellow-600">
                        <Plus className="w-3 h-3 text-black" />
                      </div>
                    )}
                  </button>
                </li>
              </ul>
            </div>

            {/* History Section */}
            <div className="relative flex w-full min-w-0 flex-col px-3 py-1 shrink-0 transition-[width,transform,opacity] duration-200">
              <ul className="flex w-full min-w-0 flex-col cursor-default gap-px">
                <li className="group/menu-item whitespace-nowrap font-semibold mx-1 relative">
                  <div
                    onClick={() => setHistoryExpanded(!historyExpanded)}
                    className={`flex items-center gap-2 overflow-hidden rounded-xl text-left outline-none ring-yellow-400 transition-[width,height,padding] focus-visible:ring-1 hover:text-yellow-400 text-sm h-9 border-transparent hover:bg-gray-800 w-full flex-row justify-start bg-background text-yellow-400 rounded-xl group/sidebar-item transition-colors p-2 border-transparent cursor-pointer ${
                      !sidebarOpen ? 'justify-center' : ''
                    }`}
                    title={!sidebarOpen ? "History" : undefined}
                  >
                    <div className="w-6 h-6 flex items-center justify-center shrink-0 transition-transform">
                      <button className="flex items-center justify-center h-6 w-6 rounded-lg">
                        <ChevronRight 
                          className={`w-4 h-4 transition-transform duration-200 ${historyExpanded ? 'rotate-90' : ''}`} 
                        />
                      </button>
                    </div>
                    {sidebarOpen && (
                      <span className="transition-all duration-200">History</span>
                    )}
                  </div>
                </li>

                {/* History Items */}
                {historyExpanded && sidebarOpen && (
                  <div className="overflow-hidden">
                    <div className="flex flex-row gap-px mx-1">
                      <div className="cursor-pointer ms-2 me-1 py-1">
                        <div className="border-l border-gray-700 h-full ms-3 me-1"></div>
                      </div>
                      <div className="flex flex-col gap-px w-full min-w-0">
                        {Object.entries(groupedSessions).map(([date, sessions]) => (
                          <div key={date}>
                            <div className="py-1 pl-3 text-xs text-yellow-400 sticky top-0 z-20 text-nowrap font-semibold">
                              {date}
                            </div>
                            {sessions.map((session) => (
                              <div key={session.id} style={{opacity: 1}}>
                                <a 
                                  href={session.id === 'current' ? '#' : `/chat/${session.id}`}
                                  className={`flex items-center gap-2 overflow-hidden rounded-xl text-left outline-none ring-yellow-400 transition-[width,height,padding] focus-visible:ring-1 hover:border-yellow-400 text-sm h-9 border border-transparent hover:bg-transparent group/sidebar-menu-item pl-3 pr-1.5 h-8 text-sm w-full flex-row items-center gap-2 text-white focus:outline-none ${
                                    session.active ? 'bg-gray-800 border-yellow-400' : ''
                                  }`}
                                  onClick={(e) => {
                                    if (session.id === 'current') {
                                      e.preventDefault();
                                      // Current conversation is already active, no navigation needed
                                    }
                                  }}
                                >
                                  <span 
                                    className="flex-1 select-none text-nowrap max-w-full overflow-hidden inline-block"
                                    style={{maskImage: 'linear-gradient(to right, black 85%, transparent 100%)'}}
                                  >
                                    {session.title}
                                  </span>
                                  <button 
                                    className="items-center justify-center gap-2 whitespace-nowrap text-sm font-medium leading-normal cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-yellow-400 disabled:opacity-60 disabled:cursor-not-allowed transition-colors duration-100 select-none text-gray-400 hover:text-white hover:bg-gray-700 disabled:hover:text-gray-400 disabled:hover:bg-transparent border border-transparent h-6 w-6 hidden group-hover/sidebar-menu-item:flex rounded-lg"
                                    onClick={(e) => {
                                      e.preventDefault();
                                      e.stopPropagation();
                                      // Handle options menu
                                    }}
                                    title="Options"
                                  >
                                    <MoreVertical className="w-3 h-3" />
                                  </button>
                                </a>
                              </div>
                            ))}
                          </div>
                        ))}
                        <button className="inline-flex items-center gap-2 whitespace-nowrap cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-yellow-400 disabled:opacity-60 disabled:cursor-not-allowed transition-colors duration-100 select-none text-gray-400 bg-transparent hover:text-white disabled:hover:text-gray-400 w-full justify-start px-3 text-xs font-semibold no-wrap pb-2 mt-1">
                          See all
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </ul>
            </div>
          </div>

          {/* Footer */}
          <div className="flex flex-col gap-2 mt-auto relative shrink-0 h-14">
            {/* Profile/DP Button - Dynamic positioning */}
            <div className={`absolute bottom-3 transition-all duration-300 ${sidebarOpen ? 'start-2' : 'opacity-0 pointer-events-none'}`}>
              <button className="inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium leading-normal cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-yellow-400 disabled:opacity-60 disabled:cursor-not-allowed transition-colors duration-100 select-none text-yellow-400 hover:bg-gray-800 disabled:hover:bg-transparent border border-transparent h-10 w-10 p-1 rounded-full">
                <span className="relative flex shrink-0 overflow-hidden rounded-full border border-gray-600 hover:opacity-75 transition-opacity duration-150 w-8 h-8">
                  <div className="aspect-square h-full w-full bg-yellow-500 flex items-center justify-center text-black font-bold text-xs">
                    SA
                  </div>
                </span>
              </button>
            </div>
            
            {/* Toggle Button - Dynamic positioning */}
            <div className="cursor-w-resize grow">
              <button 
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className={`inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium leading-normal cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-yellow-400 disabled:opacity-60 disabled:cursor-not-allowed transition-all duration-300 select-none text-gray-400 hover:text-yellow-400 hover:bg-gray-800 disabled:hover:text-gray-400 disabled:hover:bg-transparent h-10 w-10 rounded-full absolute bottom-3 ${
                  sidebarOpen ? 'end-2' : 'start-1/2 transform -translate-x-1/2'
                }`}
                title="Toggle Sidebar"
              >
                {/* Grok-style double chevron icons */}
                {sidebarOpen ? (
                  <ChevronsLeft className="w-4 h-4 transition-transform duration-200" />
                ) : (
                  <ChevronsRight className="w-4 h-4 transition-transform duration-200" />
                )}
                <span className="sr-only">Toggle Sidebar</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex flex-col flex-1 min-w-0 bg-black">

        {/* Error Message Banner */}
        {errorMessage && (
          <div className="bg-red-900/20 border-l-4 border-red-500 p-3 mx-4 mt-2 rounded">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <p className="text-sm text-red-300 font-serif">{errorMessage}</p>
              <button 
                onClick={() => setErrorMessage(null)}
                className="ml-auto text-red-400 hover:text-red-300"
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Chat Area - Centered Content */}
        <div className="flex-1 overflow-hidden">
          <div className="w-full h-full flex flex-col">
            {messages.length === 0 ? (
              <WelcomeScreen onSendMessage={handleSendMessage} />
            ) : (
              <div className="flex-1 overflow-hidden">
                <ChatHistory 
                  messages={messages} 
                  isLoading={isLoading}
                  isTyping={isTyping}
                />
              </div>
            )}
          </div>
        </div>

        {/* Chat Input - Fixed at bottom */}
        <div className="flex-shrink-0">
          <ChatInput 
            onSendMessage={handleSendMessage}
            isLoading={isLoading || connectionStatus !== 'connected'}
            placeholder={
              connectionStatus === 'connected' 
                ? "What do you want to know?"
                : "Connecting to Swiss Agent service..."
            }
            disabled={connectionStatus !== 'connected'}
          />
        </div>
      </div>
    </div>
  );
};

export default SwissAgent;