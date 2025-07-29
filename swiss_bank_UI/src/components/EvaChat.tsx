import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { X, Send, Mic, MicOff, Image, FileText, Plus, AlertCircle, CheckCircle, Volume2, VolumeX, Clock, RefreshCw, Brain, Sparkles, Users, Copy } from "lucide-react";
import { toast } from "sonner";
import { getAuthService } from "../services/authService";
import { VoiceService } from "../services/VoiceService";
import { config } from "../lib/config";
import type { AuthResponse } from "../services/authService";
import { useOTPStatus } from '../hooks/useOTPStatus';
import { CheckCircle2, ArrowRight, Shield, AlertTriangle, CreditCard, Eye, TrendingUp, Users2, MessageSquare, Target, Zap} from "lucide-react";
import React from 'react';

interface ClassificationData {
  complaint_id?: string;
  theme: string;
  category?: string;
  priority?: string;
  confidence_score?: number;
  suggested_actions?: string[];
  department?: string;
}

interface TriageResults {
  triage_analysis?: {
    primary_category?: string;
    confidence_scores?: Record<string, number>;
    secondary_category?: string;
    urgency_level?: string;
  };
  routing_package?: {
    specialist_assignment?: {
      department?: string;
      specialist_type?: string;
    };
  };
}

interface Message {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
  messageType?: 'text' | 'voice' | 'image' | 'document';
  emotional_state?: string;
  next_steps?: string[];
  specialists_mentioned?: Array<{
    name: string;
    title: string;
    experience: string;
    specialty: string;
    success_rate: string;
  }>;
  classification_pending?: ClassificationData;
  requires_confirmation?: boolean;
  processed?: boolean;
}

interface CustomerData {
  customer_id: string;
  name: string;
  email: string;
  phone?: string;
  account_type?: string;
  monthly_balance?: number;
  credit_score?: number;
  previous_complaints?: number;
}

interface EvaResponse {
  response: string;
  conversation_id: string;
  emotional_state?: string;
  classification_pending?: ClassificationData;
  requires_confirmation?: boolean;
  eva_version?: string;
  sequential_messages_active?: boolean;  
  next_message_in_seconds?: number;      
  stage?: string;
  background_processing?: boolean;  
  next_action?: string;            
  retry_in_seconds?: number;
  needs_first_question?: boolean;
  question_number?: number;
  ready_for_normal_chat?: boolean; 
}

interface ParsedMessage {
  intro?: string;
  sections: {
    title: string;
    items: string[];
    icon: string;
    emoji?: string;
  }[];
  conclusion?: string;
}

interface ClassificationConfirmationData {
  feedback_processed: boolean;
  feedback_type: string;
  learning_applied: boolean;
  followup_response: string;
  next_steps: string[];
}


interface ComplexEvaResponse {
  messageStructure: 'simple' | 'complex';
  greeting?: string;
  classification?: {
    primary: string;
    secondary: string;
    confidence: string;
  };
  reasoning?: string;
  question?: string;
  actionButtons?: string[];
}

// Triage Analysis Animation Component
const TriageAnalysisAnimation = () => (
  <div className="flex justify-start">
    <div className="bg-yellow-500 text-black border-2 border-yellow-400 p-3 rounded-lg animate-pulse-border">
      <div className="flex items-center space-x-2">
        <Brain className="w-4 h-4 animate-pulse" />
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-black rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-black rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
          <div className="w-2 h-2 bg-black rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
        </div>
        <span className="text-xs">Triage analysis in progress...</span>
      </div>
    </div>
  </div>
);

// CollapsibleCard Component with persistent state management
const CollapsibleCard = React.memo(({ 
  title, 
  children, 
  defaultExpanded = false,
  className = "",
  cardId // Add unique ID to maintain state
}: {
  title: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  className?: string;
  cardId?: string;
}) => {
  // Create a truly stable key that won't change across re-renders
  const stableKey = useMemo(() => 
    `collapsible_${cardId || title.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}`, 
    [cardId, title]
  );
  
  // Use a ref to maintain state across re-renders and prevent auto-closing
  const [isExpanded, setIsExpanded] = useState(() => {
    // Try to get saved state from sessionStorage as fallback, but default to defaultExpanded
    try {
      const saved = sessionStorage.getItem(stableKey);
      return saved !== null ? JSON.parse(saved) : defaultExpanded;
    } catch {
      return defaultExpanded;
    }
  });

  // Persist state to prevent auto-closing during re-renders
  const handleToggle = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    setIsExpanded(prev => {
      const newState = !prev;
      // Save state to sessionStorage to persist across re-renders
      try {
        sessionStorage.setItem(stableKey, JSON.stringify(newState));
      } catch {
        // Ignore sessionStorage errors
      }
      return newState;
    });
  }, [stableKey]);

  // Prevent the component from losing state during parent re-renders
  const memoizedContent = useMemo(() => children, [children]);
  
  return (
    <div className={`border border-black/20 rounded-lg transition-all duration-200 ${className}`}>
      <button
        onClick={handleToggle}
        className="w-full flex items-center justify-between p-3 text-left hover:bg-black/5 transition-colors focus:outline-none focus:ring-2 focus:ring-black/20"
        type="button"
        aria-expanded={isExpanded}
        aria-controls={`${stableKey}-content`}
      >
        <span className="font-semibold text-sm text-black">{title}</span>
        <div className={`transform transition-transform duration-300 ease-in-out ${isExpanded ? 'rotate-180' : 'rotate-0'}`}>
          <svg 
            className="w-4 h-4 text-black/70" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M19 9l-7 7-7-7" 
            />
          </svg>
        </div>
      </button>
      
      {isExpanded && (
        <div 
          id={`${stableKey}-content`}
          className="px-3 pb-3 space-y-2 animate-in slide-in-from-top-2 duration-200"
        >
          {memoizedContent}
        </div>
      )}
    </div>
  );
});


// simple text message component with better spacing for follow-up questions
const SimpleTextMessage = ({ content, messageId }: { content: string; messageId: string }) => {
  // Detect if this is a follow-up question message
  const isFollowUpQuestion = content.trim().endsWith('?') && 
                            (content.toLowerCase().includes('can you tell me') ||
                             content.toLowerCase().includes('approximately when') ||
                             content.toLowerCase().includes('how many') ||
                             content.toLowerCase().includes('what') ||
                             content.length < 200); // Simple question pattern

  return (
    <div className={`text-sm text-black ${isFollowUpQuestion ? 'space-y-3 p-6' : 'space-y-2 p-4'}`}>
      {content.split('\n').map((line, lineIndex) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={lineIndex} className="h-2" />; // Spacing
        
        // Clean the line by removing all markdown asterisks and bullet points
        const cleanedLine = trimmed.replace(/\*+/g, '').replace(/^•\s*/, '').trim();
        
        // Detect different line types
        const isMainGreeting = lineIndex === 0 && !cleanedLine.includes(':');
        
        // Detect section headings that need to be bold
        const isBoldHeading = (
          cleanedLine.toLowerCase().startsWith('current status:') ||
          cleanedLine.toLowerCase().startsWith('what i\'m doing right now:') ||
          cleanedLine.toLowerCase().startsWith('what happens next:') ||
          cleanedLine.toLowerCase().startsWith('your next actions:')
        );
        
        // Detect the specific investigation text that needs to be bold
        const isInvestigationText = cleanedLine.toLowerCase().includes('to ensure the fastest resolution') && 
                                  cleanedLine.toLowerCase().includes('additional details');

        // Detect bullet point content under specific headings
        const isInBulletSection = () => {
          const allLines = content.split('\n').map(l => l.trim().replace(/\*+/g, '').replace(/^•\s*/, '').trim());
          let foundBulletSectionStart = false;
          
          for (let i = 0; i <= lineIndex; i++) {
            const currentLine = allLines[i].toLowerCase();
            
            // Check if we hit a bullet section heading
            if (currentLine.startsWith('what i\'m doing right now:') || 
                currentLine.startsWith('what happens next:') || 
                currentLine.startsWith('your next actions:')) {
              foundBulletSectionStart = true;
              continue;
            }

            // If we found a bullet section and hit another major heading, stop
            if (foundBulletSectionStart && 
                (currentLine.startsWith('current status:') || 
                currentLine.includes('to ensure the fastest') ||
                currentLine.includes('?'))) {
              foundBulletSectionStart = false;
            }
          }
          
          return foundBulletSectionStart;
        };

        const isBulletContent = isInBulletSection() && !isBoldHeading && !isInvestigationText && !cleanedLine.includes('?');

        // Restore missing variables
        const isStatusLine = (
          cleanedLine.toLowerCase().includes('tracking id:') || 
          cleanedLine.toLowerCase().includes('has been routed') ||
          cleanedLine.toLowerCase().includes('escalated your case')
        ) && !cleanedLine.toLowerCase().startsWith('current status:');

        const isQuestionLine = cleanedLine.includes('?');

        // Apply appropriate styling with enhanced spacing for follow-up questions
        if (isMainGreeting) {
          return (
            <div key={lineIndex} className={`leading-relaxed font-medium ${isFollowUpQuestion ? 'mb-4' : 'mb-3'}`}>
              {cleanedLine}
            </div>
          );
        }
        
        // Task 1: Bold headings
        if (isBoldHeading) {
          // Split heading from content
          const parts = cleanedLine.split(':');
          const heading = parts[0] + ':';
          const content = parts.slice(1).join(':').trim();
          
          return (
            <div key={lineIndex} className={`leading-relaxed ${isFollowUpQuestion ? 'mb-3' : 'mb-2'}`}>
              <span className="font-bold">{heading}</span>
              {content && <span className="font-normal ml-1">{content}</span>}
            </div>
          );
        }
                
        // Task 3: Bold investigation text
        if (isInvestigationText) {
          return (
            <div key={lineIndex} className={`leading-relaxed font-bold ${isFollowUpQuestion ? 'mb-3' : 'mb-2'}`}>
              {cleanedLine}
            </div>
          );
        }
        
        // Task 2: Bullet point content
        if (isBulletContent) {
          return (
            <div key={lineIndex} className={`leading-relaxed ml-4 flex items-start ${isFollowUpQuestion ? 'mb-3' : 'mb-2'}`}>
              <span className="mr-2 mt-1 text-black/70">•</span>
              <span>{cleanedLine}</span>
            </div>
          );
        }
        
        if (isStatusLine) {
          // Check if this line contains tracking ID
          if (cleanedLine.toLowerCase().includes('tracking id:')) {
            const parts = cleanedLine.split(':');
            const beforeColon = parts[0] + ':';
            const afterColon = parts.slice(1).join(':').trim();
            
            return (
              <div key={lineIndex} className={`leading-relaxed bg-black/5 p-2 rounded ${isFollowUpQuestion ? 'mb-3' : 'mb-2'}`}>
                {beforeColon} <span className="font-bold">{afterColon}</span>
              </div>
            );
          }
          
          return (
            <div key={lineIndex} className={`leading-relaxed bg-black/5 p-2 rounded ${isFollowUpQuestion ? 'mb-3' : 'mb-2'}`}>
              {cleanedLine}
            </div>
          );
        }
        
        if (isQuestionLine) {
          // Enhanced spacing and styling for follow-up questions
          return (
            <div key={lineIndex} className={`leading-relaxed font-medium bg-blue-50 rounded border-l-4 border-blue-400 italic ${
              isFollowUpQuestion ? 'mb-4 p-6 text-base' : 'mb-3 p-4'
            }`}>
              {cleanedLine}
            </div>
          );
        }
                
        // Default line with enhanced spacing for follow-up questions
        return (
          <div key={lineIndex} className={`leading-relaxed ${isFollowUpQuestion ? 'mb-3' : 'mb-2'}`}>
            {cleanedLine}
          </div>
        );
      })}
    </div>
  );
};

const VoiceToggleButton = ({ 
  message, 
  currentlyPlayingMessageId, 
  onToggleVoice,
  onCopyMessage 
}: {
  message: Message;
  currentlyPlayingMessageId: string | null;
  onToggleVoice: (content: string, messageId: string) => void;
  onCopyMessage: (content: string) => void;
}) => {
  const isPlaying = currentlyPlayingMessageId === message.id;
  
  return (
    <div className="flex items-center space-x-3 mt-2 justify-end animate-fade-in">
      {/* Fixed Voice Toggle Button - Direct icon with yellow hover, no tooltips */}
      <button
        onClick={() => onToggleVoice(message.content, message.id)}
        className={`flex items-center justify-center transition-all duration-300 hover:scale-110 transform ${
          isPlaying
            ? 'text-yellow-400 animate-pulse'
            : 'text-gray-400 hover:text-yellow-400'
        }`}
      >
        {/* Icon with smooth transition - no circular background */}
        {isPlaying ? (
          <VolumeX className="w-3 h-3 transition-all duration-200" />
        ) : (
          <Volume2 className="w-3 h-3 transition-all duration-200" />
        )}
      </button>
      
      {/* Fixed Copy Button - Direct icon with yellow hover, no tooltips */}
      <button
        onClick={() => onCopyMessage(message.content)}
        className="flex items-center justify-center text-gray-400 hover:text-yellow-400 transition-all duration-300 hover:scale-110 transform"
      >
        <Copy className="w-3 h-3 transition-all duration-200" />
      </button>
    </div>
  );
};

// Greeting message style
const MicrosoftCopilotCard = ({ content, messageId }: { content: string; messageId: string }) => {
  // Parse dynamic greeting and message parts
  const parseGreetingMessage = (text: string) => {
    const lines = text.split('\n').filter(line => line.trim());
    
    // Detect greeting pattern (Good morning/afternoon/evening + name)
    const greetingRegex = /^(Good\s+(morning|afternoon|evening),?\s+([^!]+)!?)/i;
    const greetingMatch = text.match(greetingRegex);
    
    let greeting = '';
    let restOfMessage = text;
    
    if (greetingMatch) {
      greeting = greetingMatch[0];
      restOfMessage = text.replace(greetingMatch[0], '').trim();
    }
    
    // Split remaining content into introduction and question
    const questionRegex = /(What\s+[^?]*\?|How\s+[^?]*\?)/i;
    const questionMatch = restOfMessage.match(questionRegex);
    
    let introduction = restOfMessage;
    let question = '';
    
    if (questionMatch) {
      question = questionMatch[0];
      introduction = restOfMessage.replace(questionMatch[0], '').trim();
    }
    
    return { greeting, introduction, question };
  };

  const { greeting, introduction, question } = parseGreetingMessage(content);
  
  return (
    <>
      {/* Header with greeting */}
      {greeting && (
        <div className="border-b border-yellow-400/40 px-4 py-3 bg-yellow-600/10">
          <div className="flex items-center space-x-2">
            <span className="text-lg">🌅</span>
            <span className="font-semibold text-black text-sm">{greeting}</span>
          </div>
        </div>
      )}
      
      {/* Main content */}
      <div className="px-4 py-3 space-y-3">
        {/* Introduction section */}
        {introduction && (
          <div className="text-sm text-black leading-relaxed">
            {introduction}
          </div>
        )}
      </div>
      
      {/* Question section */}
      {question && (
        <div className="border-t border-yellow-400/20 px-4 py-3 bg-yellow-500/5">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-black">{question}</span>
          </div>
        </div>
      )}
    </>
  );
};

const AuthenticatedEvaChat = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [customerData, setCustomerData] = useState<CustomerData | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [authService] = useState(() => getAuthService(config.backendUrl));
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showAttachments, setShowAttachments] = useState(false);
  const [authStep, setAuthStep] = useState<'session' | 'contact' | 'otp' | 'authenticated'>('session');
  const [email, setEmail] = useState('');
  const [otp, setOtp] = useState('');
  const [preferredMethod, setPreferredMethod] = useState<'email' | 'sms'>('email');
  const [authLoading, setAuthLoading] = useState(false);
  const [inputHeight, setInputHeight] = useState(40);
  
  // Eva-specific states
  const [pendingClassification, setPendingClassification] = useState<ClassificationData | null>(null);
  const [awaitingFeedback, setAwaitingFeedback] = useState(false);
  const [evaStatus, setEvaStatus] = useState<{
    status: string;
    capabilities?: Record<string, boolean>;
    learning_stats?: Record<string, number>;
  } | null>(null);
  
  const [showAnalysisAnimation, setShowAnalysisAnimation] = useState(false);
  
  // OTP Status using the hook
  const { 
    otpStatus, 
    error: otpStatusError,
    refreshStatus: refreshOTPStatus,
    formatTimeRemaining,
    getUrgencyLevel 
  } = useOTPStatus(sessionId);
  
  // Voice-aware timing states
  const [isVoiceDisabled, setIsVoiceDisabled] = useState(false);
  const [pendingSequentialMessage, setPendingSequentialMessage] = useState<{
    conversationId: string;
    nextMessageTime: number;
  } | null>(null);

  // Voice-related state
  const [isListening, setIsListening] = useState(false);
  const [showRepeatInfo, setShowRepeatInfo] = useState(false);
  
  // Enhanced voice state management
  const [currentlyPlayingMessageId, setCurrentlyPlayingMessageId] = useState<string | null>(null);
  const [voiceQueue, setVoiceQueue] = useState<string[]>([]);
  
  // NEW: Global voice control states - DEFAULT MUTED
  const [isVoiceMuted, setIsVoiceMuted] = useState(true);
  const [autoPlayEnabled, setAutoPlayEnabled] = useState(false);
  
  // Voice service
  const [voiceService] = useState(() => new VoiceService());

  // Update voice service with customer data when available
  useEffect(() => {
    if (customerData && voiceService) {
      voiceService.setCustomerData(customerData);
    }
  }, [customerData, voiceService]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add this function inside AuthenticatedEvaChat component
  const handleSetInputTextWithFocus = useCallback((text: string) => {
    setInputText(text);
    
    // Focus the input and position cursor at the end
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus();
        inputRef.current.setSelectionRange(text.length, text.length);
      }
    }, 100);
  }, []);

  // Enhanced cleanup effect for proper voice management
  useEffect(() => {
    // Cleanup function to stop voice when component unmounts or chat closes
    const cleanup = () => {
      try {
        if (speechSynthesis.speaking) {
          speechSynthesis.cancel();
          console.log('🧹 Voice cleanup - synthesis cancelled');
        }
        setCurrentlyPlayingMessageId(null);
      } catch (error) {
        console.error('Error during voice cleanup:', error);
      }
    };

    return cleanup;
  }, []);

  // Stop voice when chat is closed
  useEffect(() => {
    if (!isOpen && currentlyPlayingMessageId) {
      try {
        speechSynthesis.cancel();
        setCurrentlyPlayingMessageId(null);
        console.log('🔇 Voice stopped - chat closed');
      } catch (error) {
        console.error('Error stopping voice on chat close:', error);
      }
    }
  }, [isOpen, currentlyPlayingMessageId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Function to play voice for bot messages with queue management
  const playVoiceForBotMessage = useCallback(async (content: string, messageId: string) => {
    // NEVER auto-play - only play when explicitly requested
    console.log('🔇 Auto-play disabled by default');
    return;
  }, []); 

  // Function to play specific message (for individual play buttons)
  const playSpecificMessage = useCallback(async (content: string, messageId: string) => {
    try {
      // Stop any currently playing voice first
      speechSynthesis.cancel();
      
      console.log('🔊 Playing specific message:', messageId);
      
      // Set as currently playing
      setCurrentlyPlayingMessageId(messageId);
      
      // Play the message
      await voiceService.textToSpeech(content);
      
      // Clear currently playing when done (only if this message is still playing)
      setCurrentlyPlayingMessageId(prev => prev === messageId ? null : prev);
      
      toast.success('🔊 Message played');
    } catch (error) {
      console.error('Error playing specific message:', error);
      setCurrentlyPlayingMessageId(null);
    }
  }, [voiceService]);
  // Function to stop current voice playback
  const stopCurrentVoice = useCallback(() => {
  try {
    speechSynthesis.cancel(); 
    setCurrentlyPlayingMessageId(null);
    toast.info('🔇 Voice playback stopped');
    console.log('🛑 All voice playback stopped');
  } catch (error) {
    console.error('Error stopping voice playback:', error);
    setCurrentlyPlayingMessageId(null);
  }
}, []);


  // Simplified and robust toggle voice for specific message
  const toggleMessageVoice = useCallback(async (content: string, messageId: string) => {
  console.log('🎯 Toggle voice called for message:', messageId, 'Currently playing:', currentlyPlayingMessageId);
  
  try {
    if (currentlyPlayingMessageId === messageId) {
      // STOP: This message is currently playing
      console.log('🛑 Stopping voice for message:', messageId);
      
      // Force stop all speech synthesis
      speechSynthesis.cancel();
      
      // Clear the playing state immediately
      setCurrentlyPlayingMessageId(null);
      console.log('✅ Voice successfully stopped for message:', messageId);
      
    } else {
      // PLAY: Start playing this message
      console.log('🔊 Starting voice for message:', messageId);
      
      // Stop any other currently playing voice first
      if (currentlyPlayingMessageId || speechSynthesis.speaking) {
        console.log('🛑 Stopping previous voice before starting new one');
        speechSynthesis.cancel();
        
        // Wait a moment for the cancellation to take effect
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      // Set this message as currently playing BEFORE starting
      setCurrentlyPlayingMessageId(messageId);
      console.log('🎯 Set message as currently playing:', messageId);
      
      // Start playing the message with proper error handling
      try {
        // Create and configure speech synthesis
        const utterance = new SpeechSynthesisUtterance(content);
        
        // Configure voice settings
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        // Set up event handlers
        utterance.onend = () => {
          console.log('✅ Speech ended for message:', messageId);
          setCurrentlyPlayingMessageId(prev => {
            if (prev === messageId) {
              console.log('✅ Clearing playing state for completed message:', messageId);
              return null;
            }
            console.log('🔄 Another message is playing, keeping state');
            return prev;
          });
        };
        
        utterance.onerror = (event) => {
          console.error('❌ Speech synthesis error:', event.error);
          setCurrentlyPlayingMessageId(null);
        };
        
        utterance.onstart = () => {
          console.log('🎵 Speech started for message:', messageId);
        };
        
        // Start speaking
        speechSynthesis.speak(utterance);
        
      } catch (speechError) {
        console.error('❌ Speech synthesis error:', speechError);
        setCurrentlyPlayingMessageId(null);
      }
    }
  } catch (error) {
    console.error('❌ Error in toggleMessageVoice:', error);
    setCurrentlyPlayingMessageId(null);
    toast.error('Voice operation failed');
  }
}, [currentlyPlayingMessageId]);

  // Copy message function
  const copyMessage = (content: string) => {
    // Clean the content from any markdown or special characters for copying
    const cleanContent = content
      .replace(/\*\*/g, '') 
      .replace(/\*/g, '')   
      .replace(/\n\s*\n/g, '\n') 
      .trim();
      
    navigator.clipboard.writeText(cleanContent).then(() => {
      toast.success('📋 Message copied to clipboard');
    }).catch(err => {
      console.error('Failed to copy message:', err);

      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = cleanContent;
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        document.execCommand('copy');
        toast.success('📋 Message copied to clipboard');
      } catch (fallbackErr) {
        toast.error('Failed to copy message');
      }
      document.body.removeChild(textArea);
    });
  };

  const parseComplexEvaResponse = (content: string): ComplexEvaResponse => {
    console.log('🔍 Parsing complex Eva response:', content.substring(0, 100) + '...');
    
    // Check if this is a complex triage confirmation response
    const hasClassification = content.includes('Classification:') || 
                             content.includes('Primary Category:') ||
                             content.includes('Complaint Classification:');
    
    const hasReasoning = content.includes('Why we labeled it this way:') ||
                        content.includes('Why we categorized') ||
                        content.includes('This complaint clearly falls under');
    
    const hasQuestion = content.includes('Does this assessment') ||
                       content.includes('accurately capture') ||
                       content.includes('sound right');
    
    if (!hasClassification && !hasReasoning) {
      return { messageStructure: 'simple' };
    }
    
    // Extract greeting (everything before classification)
    const greetingMatch = content.match(/^([^:\n]*?)(?=\n.*Classification:|Classification:|Primary Category:)/s);
    const greeting = greetingMatch?.[1]?.trim() || '';
    
    // Extract classification details
    let classification;
    const classificationSection = content.match(/(?:Complaint Classification:|Classification:)(.*?)(?=Why we labeled|Does this assessment|$)/s);
    if (classificationSection) {
      const classText = classificationSection[1];
      const primaryMatch = classText.match(/Primary Category[:\s]*([^\n]+)/i);
      const secondaryMatch = classText.match(/Secondary Category[:\s]*([^\n]+)/i);
      const confidenceMatch = classText.match(/Confidence Level?[:\s]*([^\n]+)/i);
      
      classification = {
        primary: primaryMatch?.[1]?.trim() || 'Not specified',
        secondary: secondaryMatch?.[1]?.trim() || 'None',
        confidence: confidenceMatch?.[1]?.trim() || 'Not specified'
      };
    }
    
    // Extract reasoning (Why we labeled it this way section)
    const reasoningMatch = content.match(/Why we labeled it this way:(.*?)(?=Does this assessment|$)/s);
    const reasoning = reasoningMatch?.[1]?.trim() || '';
    
    // Extract final question
    const questionMatch = content.match(/(Does this assessment.*?resolution steps\.)/s);
    const question = questionMatch?.[1]?.trim() || '';
    
    return {
      messageStructure: 'complex',
      greeting: greeting || undefined,
      classification,
      reasoning: reasoning || undefined,
      question: question || undefined,
      actionButtons: question ? ['Yes, this is accurate', 'No, needs adjustment'] : undefined
    };
  };

  // Complex message component for triage confirmation responses
  const ComplexStructuredMessage = React.memo(({ 
    parsedResponse, 
    messageId,
    onSendMessage,
    onSetInputText 
  }: { 
    parsedResponse: ReturnType<typeof parseComplexEvaResponse>; 
    messageId: string;
    onSendMessage: (message: string) => void;
    onSetInputText: (text: string) => void;
  }) => {
    const [confirmationSent, setConfirmationSent] = useState(false);
    
    const handleConfirmation = useCallback((response: string) => {
      if (confirmationSent) return;
      setConfirmationSent(true);
      
      console.log('🎯 User confirmation:', response);
      
      if (response.toLowerCase().includes('yes')) {
        console.log('✅ Sending YES confirmation to bot');
        onSendMessage('Yes, this is accurate');
        
        // Reset after message is sent
        setTimeout(() => setConfirmationSent(false), 1000);
      } else if (response.toLowerCase().includes('no')) {
        // Pre-fill input with "No. Cause, " and focus for user to add details
        console.log('❌ Setting NO response in input for user to complete');
        onSetInputText('No. Cause, ');
        
        // Reset confirmation state immediately so user can type
        setTimeout(() => setConfirmationSent(false), 100);
      }
    }, [confirmationSent, onSendMessage, onSetInputText]);
    
    if (parsedResponse.messageStructure === 'simple') {
      return null; // Use regular parsing for simple messages
    }
    
    return (
      <div className="space-y-3">
        {/* Greeting */}
        {parsedResponse.greeting && (
          <div className="text-sm leading-relaxed text-black">
            {parsedResponse.greeting}
          </div>
        )}
        
        {/* Classification Summary (Always Visible) */}
        {parsedResponse.classification && (
          <div className="space-y-1 text-sm text-black">
            <div className="font-bold text-sm text-black mb-3 pb-1 border-b-2 border-black/30">
                Analysis Complete - Classification Results
            </div>
            <div className="space-y-1 text-sm text-black ml-4">
              <div><strong>Primary Category:</strong> {parsedResponse.classification.primary.replace(/\*+\s*/g, '')}</div>
              <div><strong>Secondary Category:</strong> {parsedResponse.classification.secondary.replace(/\*+\s*/g, '')}</div>
              <div><strong>Confidence Level:</strong> {parsedResponse.classification.confidence.replace(/\*+\s*/g, '')}</div>
            </div>
          </div>
        )}
        
        {/* Detailed Analysis (Collapsible) - Fixed with stable ID */}
        {parsedResponse.reasoning && (
          <CollapsibleCard 
            title="Why we labeled it this way:" 
            defaultExpanded={false}
            className="bg-black/5"
            cardId={`reasoning-${messageId}`} // Unique stable ID
          >
            <div className="text-sm leading-relaxed text-black space-y-3">
              {/* Break reasoning into logical paragraphs */}
              {parsedResponse.reasoning.split(/\.\s+(?=[A-Z])/).map((sentence, index) => {
                if (sentence.trim().length === 0) return null;
                
                const cleanSentence = sentence.trim();
                const finalSentence = cleanSentence.endsWith('.') ? cleanSentence : cleanSentence + '.';
                
                return (
                  <p key={index} className="text-black">
                    {finalSentence}
                  </p>
                );
              })}
            </div>
          </CollapsibleCard>
        )}
        
        {/* Confirmation Question */}
        {parsedResponse.question && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="text-sm text-black mb-3">
              {parsedResponse.question}
            </div>
            
            {parsedResponse.actionButtons && (
              <div className="flex gap-2">
                {parsedResponse.actionButtons.map((button, index) => (
                  <button
                    key={index}
                    onClick={() => handleConfirmation(button)}
                    disabled={confirmationSent && button.toLowerCase().includes('yes')}
                    className={`px-3 py-1 text-xs rounded transition-colors ${
                      button.toLowerCase().includes('yes') 
                        ? 'bg-green-600 text-white hover:bg-green-700 disabled:bg-green-400' 
                        : 'bg-gray-600 text-white hover:bg-gray-700 disabled:bg-gray-400'
                    }`}
                  >
                    {(confirmationSent && button.toLowerCase().includes('yes')) ? 'Sent...' : button}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    );
  });

  // Enhanced parsing function with follow-up question extraction
  const parseEvaMessage = (content: string): { parsedMessage: ParsedMessage | null, followUpQuestions: string[] } => {
    console.log('🔍 Parsing content:', content.substring(0, 100) + '...');
    
    // Extract follow-up questions (anything that ends with ?)
    const followUpQuestions: string[] = [];
    const questionMatches = content.match(/[^.!]*\?[^.!]*?(?=\s*$|\s*[A-Z])/g);
    if (questionMatches) {
      questionMatches.forEach(question => {
        const cleanQuestion = question.trim();
        if (cleanQuestion && !cleanQuestion.includes('**')) {
          followUpQuestions.push(cleanQuestion);
        }
      });
    }
    
    // Check for existing structured patterns
    const hasExistingStructure = content.includes('**What I\'m doing right now:**') || 
                                content.includes('**What happens next:**') || 
                                content.includes('**Your next actions:**') ||
                                content.includes('**What I\'m doing**') ||
                                content.includes('**What happens**') ||
                                content.includes('**Your actions**') ||
                                content.includes('**Next steps**') ||
                                content.includes('**Current status**');
    
    console.log('📋 Structure detected:', hasExistingStructure);
    
    // If existing structure found, use original parsing
    if (hasExistingStructure) {
      return parseOriginalStructure(content, followUpQuestions);
    }

    // NEW: For unstructured content, return null to use simple text rendering
    return { parsedMessage: null, followUpQuestions };
  };

  // Original parsing logic (unchanged)
  const parseOriginalStructure = (content: string, followUpQuestions: string[]): { parsedMessage: ParsedMessage | null, followUpQuestions: string[] } => {
    const sections: ParsedMessage['sections'] = [];
    let intro = '';
    let conclusion = '';

    // Remove follow-up questions from content for processing sections
    let contentWithoutQuestions = content;
    followUpQuestions.forEach(question => {
      contentWithoutQuestions = contentWithoutQuestions.replace(question, '').trim();
    });

    // Split by lines and process properly
    const lines = contentWithoutQuestions.split('\n').filter(line => line.trim());
    let currentSection: string | null = null;
    let currentItems: string[] = [];
    const introLines: string[] = [];
    const conclusionLines: string[] = [];
    let currentPhase: 'intro' | 'sections' | 'conclusion' = 'intro';

    for (const line of lines) {
      const trimmedLine = line.trim();
      
      // Check if this is a section header
      if (trimmedLine.startsWith('**') && trimmedLine.endsWith('**')) {
        // Save previous section if exists
        if (currentSection && currentItems.length > 0) {
          sections.push({
            title: currentSection,
            items: [...currentItems],
            ...getSectionStyling(currentSection)
          });
        }
        
        // Start new section
        currentSection = trimmedLine.replace(/\*\*/g, '').replace(':', '').trim();
        currentItems = [];
        currentPhase = 'sections';
        console.log('📝 New section found:', currentSection);
        
      } else if (trimmedLine.startsWith('•') && currentSection) {
        // This is a bullet point
        const item = trimmedLine.substring(1).trim();
        if (item) {
          currentItems.push(item);
          console.log('  • Added item:', item);
        }
        
      } else if (trimmedLine.length > 0) {
        // Regular content
        if (currentPhase === 'intro') {
          introLines.push(trimmedLine);
        } else if (currentPhase === 'sections' && currentSection && currentItems.length === 0) {
          // Content right after section header, add as first item
          currentItems.push(trimmedLine);
        } else if (currentPhase === 'sections') {
          // We're past sections, this is conclusion
          currentPhase = 'conclusion';
          conclusionLines.push(trimmedLine);
        } else if (currentPhase === 'conclusion') {
          conclusionLines.push(trimmedLine);
        }
      }
    }

    // Save the last section
    if (currentSection && currentItems.length > 0) {
      sections.push({
        title: currentSection,
        items: [...currentItems],
        ...getSectionStyling(currentSection)
      });
    }

    // Clean up content
    intro = introLines.join(' ').trim();
    conclusion = conclusionLines.join(' ').trim();

    console.log('✅ Original parsing complete:', {
      hasIntro: !!intro,
      sectionsCount: sections.length,
      hasConclusion: !!conclusion,
      followUpQuestionsCount: followUpQuestions.length
    });

    const parsedMessage = {
      intro: intro || undefined,
      sections,
      conclusion: conclusion || undefined
    };

    return { parsedMessage, followUpQuestions };
  };

  // NEW: Smart parsing for unstructured content
  const parseUnstructuredContent = (content: string, followUpQuestions: string[]): { parsedMessage: ParsedMessage | null, followUpQuestions: string[] } => {
    console.log('🧠 Using smart parsing for unstructured content');
    
    // Remove questions from content
    let contentWithoutQuestions = content;
    followUpQuestions.forEach(question => {
      contentWithoutQuestions = contentWithoutQuestions.replace(question, '').trim();
    });
    
    // Split into paragraphs
    const paragraphs = contentWithoutQuestions.split('\n\n').filter(p => p.trim());
    
    if (paragraphs.length <= 1) {
      // Single paragraph, no structure needed
      console.log('📝 Single paragraph detected, no structuring needed');
      return { parsedMessage: null, followUpQuestions };
    }
    
    const sections: ParsedMessage['sections'] = [];
    let intro = '';
    
    // First paragraph is usually the intro/greeting
    if (paragraphs.length > 0) {
      intro = paragraphs[0].trim();
      console.log('📝 Intro detected:', intro.substring(0, 50) + '...');
    }
    
    // Process remaining paragraphs as sections
    for (let i = 1; i < paragraphs.length; i++) {
      const paragraph = paragraphs[i].trim();
      
      // Smart section detection with cleaned content
      const sectionInfo = detectSectionTypeAndClean(paragraph);
      
      // Only add section if it has content after cleaning
      if (sectionInfo.items.length > 0 && sectionInfo.items[0].trim() !== '') {
        console.log('📝 Section detected:', sectionInfo.title);
        
        sections.push({
          title: sectionInfo.title,
          items: sectionInfo.items,
          icon: sectionInfo.icon,
          emoji: sectionInfo.emoji
        });
      }
    }
    
    const parsedMessage = {
      intro: intro || undefined,
      sections,
      conclusion: undefined
    };
    
    console.log('✅ Smart parsing complete:', {
      hasIntro: !!intro,
      sectionsCount: sections.length,
      followUpQuestionsCount: followUpQuestions.length
    });
    
    return { parsedMessage, followUpQuestions };
  };

  // NEW: Smart section detection with content cleaning
  const detectSectionTypeAndClean = (paragraph: string): { title: string, items: string[], icon: string, emoji: string } => {
    const text = paragraph.toLowerCase();
    
    // Status/tracking patterns
    if (text.includes('status') || text.includes('tracking') || text.includes('routed') || text.includes('queue') || text.includes('escalated')) {
      return {
        title: 'Current Status',
        items: cleanRedundantContent(paragraph, 'current status'),
        icon: 'message-square',
        emoji: '📊'
      };
    }
    
    // Next steps/what happens patterns
    if (text.includes('specialist team') || text.includes('will begin') || text.includes('updates') || text.includes('receive') || text.includes('investigating')) {
      return {
        title: 'What Happens Next',
        items: cleanRedundantContent(splitIntoItems(paragraph), 'what happens next'),
        icon: 'clock',
        emoji: '⏱️'
      };
    }
    
    // Information gathering patterns
    if (text.includes('gather') || text.includes('additional details') || text.includes('help our') || text.includes('investigation team') || text.includes('let me gather')) {
      return {
        title: 'Information Gathering',
        items: cleanRedundantContent([paragraph], 'information gathering'),
        icon: 'target',
        emoji: '🔍'
      };
    }
    
    // Default
    return {
      title: 'Additional Information',
      items: cleanRedundantContent([paragraph], 'additional information'),
      icon: 'message-square',
      emoji: '📋'
    };
  };

  // NEW: Clean redundant content from items
  const cleanRedundantContent = (items: string | string[], titleToRemove: string): string[] => {
    const itemsArray = Array.isArray(items) ? items : [items];
    
    return itemsArray.map(item => {
      let cleanedItem = item.trim();
      
      // Remove redundant title patterns
      const patterns = [
        new RegExp(`^\\*\\*${titleToRemove}\\*\\*:?\\s*`, 'i'),
        new RegExp(`^${titleToRemove}:?\\s*`, 'i'),
        /^\*\*[^*]+\*\*:?\s*/,  // Remove any **Title**: pattern
      ];
      
      patterns.forEach(pattern => {
        cleanedItem = cleanedItem.replace(pattern, '');
      });
      
      // Clean up extra spaces and formatting
      cleanedItem = cleanedItem.replace(/\s+/g, ' ').trim();
      
      return cleanedItem;
    }).filter(item => item.length > 0);
  };

  // Helper to split paragraph into logical items (unchanged)
  const splitIntoItems = (paragraph: string): string[] => {
    // Split by sentences that seem like separate items
    const sentences = paragraph.split(/(?<=[.!])\s+(?=[A-Z])/);
    
    if (sentences.length > 1) {
      return sentences.map(s => s.trim()).filter(s => s.length > 0);
    }
    
    return [paragraph];
};

  // Section styling with single icon only
  const getSectionStyling = (title: string) => {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('specialist') || titleLower.includes('team')) {
      return {
        icon: 'users2',
        emoji: '👥'
      };
    }
    
    // Default styling
    return {
      icon: 'message-square',
      emoji: '📋'
    };
  };


  // Helper function
  const getSectionIcon = (title: string) => {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('doing') || titleLower.includes('right now')) {
      return '⚡';
    } else if (titleLower.includes('happens next') || titleLower.includes('what happens')) {
      return '⏱️';
    } else if (titleLower.includes('your') || titleLower.includes('action')) {
      return '📋';
    } else if (titleLower.includes('status') || titleLower.includes('current')) {
      return '📊';
    }
    return '•';
  };

  const getSectionClass = (title: string) => {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('doing') || titleLower.includes('right now')) {
      return 'section-immediate';
    } else if (titleLower.includes('happens next')) {
      return 'section-upcoming';
    } else if (titleLower.includes('your') || titleLower.includes('action')) {
      return 'section-action';
    }
    return 'section-immediate';
  };
  
  // Helper function to get the icon component
  const getIconComponent = (iconName: string, className: string = "w-4 h-4") => {
    const icons: Record<string, JSX.Element> = {
      'zap': <Zap className={className} />,
      'clock': <Clock className={className} />,
      'target': <Target className={className} />,
      'eye': <Eye className={className} />,
      'users2': <Users2 className={className} />,
      'message-square': <MessageSquare className={className} />,
      'check': <CheckCircle2 className={className} />,
      'arrow': <ArrowRight className={className} />,
      'shield': <Shield className={className} />,
      'alert': <AlertTriangle className={className} />,
      'file': <FileText className={className} />,
      'card': <CreditCard className={className} />,
      'trend': <TrendingUp className={className} />
    };
    
    return icons[iconName] || icons['message-square'];
  };

  // Enhanced StructuredMessage component with black theme and single icon
  const StructuredMessage = ({ parsedMessage, messageId }: { 
    parsedMessage: ParsedMessage; 
    messageId: string; 
  }) => {
    return (
      <div className="space-y-3">
        {/* Intro with enhanced styling */}
        {parsedMessage.intro && (
          <div className="rounded-lg border border-black/20 bg-black/5 p-3 structured-section transition-all duration-200 hover:shadow-md">
            <div className="text-sm leading-relaxed text-black">
              {parsedMessage.intro}
            </div>
          </div>
        )}

        {/* Enhanced Structured Sections */}
        {parsedMessage.sections.map((section, index) => {
          const sectionClass = getSectionClass(section.title);
          const sectionIcon = getSectionIcon(section.title);
          
          return (
            <div 
              key={`${messageId}-section-${index}`}
              className={`rounded-lg border p-3 structured-section transition-all duration-200 hover:shadow-md ${sectionClass}`}
            >
              {/* Section Header with icon */}
              <div className="flex items-center space-x-2 mb-3">
                <span className="text-lg">{sectionIcon}</span>
                <h4 className="font-semibold text-sm text-black">
                  {section.title}
                </h4>
              </div>
              
              {/* Section Items with enhanced styling */}
              <div className="space-y-2 ml-6">
                {section.items.map((item, itemIndex) => (
                  <div 
                    key={`${messageId}-item-${index}-${itemIndex}`}
                    className="flex items-start space-x-3 text-sm group"
                  >
                    <div className="mt-2 w-1.5 h-1.5 bg-black rounded-full flex-shrink-0 opacity-70 group-hover:opacity-100 transition-opacity bullet-point" />
                    <span className="leading-relaxed text-black">{item}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}

        {/* Conclusion with enhanced styling */}
        {parsedMessage.conclusion && (
          <div className="text-sm leading-relaxed text-black italic pl-4 border-l-2 border-black/20">
            {parsedMessage.conclusion}
          </div>
        )}
      </div>
    );
  };


  // Fetch Eva system status on component mount
  useEffect(() => {
    const fetchEvaStatus = async () => {
      try {
        const response = await fetch(`${config.backendUrl}/api/eva/status`);
        if (response.ok) {
          const status = await response.json();
          setEvaStatus(status);
          console.log('🤖 Eva Status:', status);
        }
      } catch (error) {
        console.warn('Could not fetch Eva status:', error);
      }
    };
    
    fetchEvaStatus();
  }, []);

  // Close attachments when clicking anywhere in the chat area
  const handleChatAreaClick = useCallback(() => {
    if (showAttachments) {
      setShowAttachments(false);
    }
  }, [showAttachments]);

  // Adjust input height based on content with smooth scrolling
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
    
    // Reset height to auto to get the correct scrollHeight
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      const scrollHeight = inputRef.current.scrollHeight;
      
      // Calculate new height (min 40px, max 120px)
      const newHeight = Math.min(Math.max(scrollHeight, 40), 120);
      setInputHeight(newHeight);
      inputRef.current.style.height = `${newHeight}px`;
    }
  };

  const resetInputHeight = () => {
    setInputHeight(40);
    if (inputRef.current) {
      inputRef.current.style.height = '40px';
    }
  };

  const addSystemMessage = useCallback((content: string) => {
    const systemMessage: Message = {
      id: Date.now().toString(),
      type: 'system',
      content,
      timestamp: new Date(),
      messageType: 'text'
    };
    setMessages(prev => [...prev, systemMessage]);
  }, []);

  // Fixed function with proper typing
  const generateTriageConfirmationMessage = (triageResults: TriageResults): string => {
    const category = triageResults.triage_analysis?.primary_category || 'general_inquiry';
    
    // Fix the confidence calculation with proper type checking
    let confidence = 0.8; 
    if (triageResults.triage_analysis?.confidence_scores) {
      const scores = Object.values(triageResults.triage_analysis.confidence_scores);
      if (scores.length > 0 && scores.every(score => typeof score === 'number')) {
        confidence = Math.max(...scores);
      }
    }
    
    // Convert category to human-readable format
    const friendlyCategory = category
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
    
    return `I've completed my analysis with our triage team. Here's what we determined:

  **Complaint Classification:**
  - **Primary Category:** ${friendlyCategory}
  - **Confidence Level:** ${Math.round(confidence * 100)}%

  **Does this assessment accurately capture your situation?** Please let me know if this sounds right or if I need to adjust my understanding before we proceed with the resolution steps.`;
  };

  // Add follow-up questions as separate messages
  const addFollowUpQuestions = useCallback((questions: string[]) => {
    questions.forEach((question, index) => {
      const questionMessage: Message = {
        id: `followup_${Date.now()}_${index}`,
        type: 'bot',
        content: question,
        timestamp: new Date(),
        messageType: 'text'
      };
      
      setTimeout(() => {
        setMessages(prev => [...prev, questionMessage]);
        // Play voice for follow-up questions ONLY if user explicitly requests
      }, 1000 + (index * 1500)); 
    });
  }, []);

  const startAuthentication = useCallback(async () => {
    setAuthLoading(true);
    try {
      const response = await authService.createSession();
      if (response.success) {
        setSessionId(response.session_id || authService.getSessionId());
        setAuthStep('contact');
      } else {
        toast.error('Failed to start authentication. Please try again.');
      }
    } catch (error) {
      console.error('Authentication start failed:', error);
      toast.error('Authentication failed. Please try again.');
    } finally {
      setAuthLoading(false);
    }
  }, [authService]);

  const validateSession = useCallback(async () => {
    try {
      const status = await authService.getSessionStatus();
      if (status.success && status.data) {
        if (status.data.authenticated) {
          setIsAuthenticated(true);
          setAuthStep('authenticated');
          setCustomerData(status.data.customer_data);
        } else if (status.data.state === 'otp_verification') {
          setAuthStep('otp');
          setCustomerData(status.data.customer_data);
        } else if (status.data.contact_verified) {
          setAuthStep('otp');
          setCustomerData(status.data.customer_data);
        } else {
          setAuthStep('contact');
        }
      } else {
        // Session invalid, start fresh
        startAuthentication();
      }
    } catch (error) {
      console.error('Session validation failed:', error);
      startAuthentication();
    }
  }, [authService, startAuthentication]);

  // Initialize authentication on component mount
  useEffect(() => {
    const existingSessionId = authService.getSessionId();
    if (existingSessionId) {
      setSessionId(existingSessionId);
      validateSession();
    } else {
      // No existing session, prepare for fresh auth
      setAuthStep('contact');
    }
  }, [authService, validateSession]);

  // Enhanced Eva greeting when authenticated
  useEffect(() => {
    if (isAuthenticated && customerData && messages.length === 0) {
      // Test Eva's greeting capability first
      const testEvaGreeting = async () => {
        try {
          const response = await fetch(`${config.backendUrl}/api/eva/test-greeting`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${sessionId}`
            }
          });
          
          if (response.ok) {
            const data = await response.json();
            const greetingMessage: Message = {
              id: `eva_greeting_${Date.now()}`,
              type: 'bot',
              content: data.greeting,
              timestamp: new Date(),
              messageType: 'text',
              emotional_state: 'neutral'
            };
            setMessages([greetingMessage]);
            
            // Play greeting voice message ONLY if user explicitly enables it
            // No auto-play by default
          } else {
            // Fallback greeting
            const fallbackGreeting: Message = {
              id: `fallback_greeting_${Date.now()}`,
              type: 'bot',
              content: `Hello ${customerData.name}! I'm Eva, your AI banking assistant. How can I help you today?`,
              timestamp: new Date(),
              messageType: 'text'
            };
            setMessages([fallbackGreeting]);
          }
        } catch (error) {
          console.error('Error testing Eva greeting:', error);
          // Fallback greeting
          const fallbackGreeting: Message = {
            id: `error_greeting_${Date.now()}`,
            type: 'bot',
            content: `Hello ${customerData.name}! I'm Eva, your AI banking assistant. How can I help you today?`,
            timestamp: new Date(),
            messageType: 'text'
          };
          setMessages([fallbackGreeting]);
        }
      };
      
      testEvaGreeting();
      
      // Show repeat info popup for first-time users
      setTimeout(() => {
        setShowRepeatInfo(true);
        setTimeout(() => setShowRepeatInfo(false), 5000);
      }, 2000);
    }
  }, [isAuthenticated, customerData, messages.length, sessionId, evaStatus, addSystemMessage]);

  // Sequential message handling with voice-aware timing
  useEffect(() => {
    if (pendingSequentialMessage) {
      const delay = isVoiceMuted ? 10000 : 5000; 
      
      const timer = setTimeout(async () => {
        try {
          const formData = new FormData();
          formData.append('session_id', pendingSequentialMessage.conversationId);
          
          const response = await fetch(`${config.backendUrl}/api/eva/continue-action-sequence`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${sessionId}`
            },
            body: formData
          });
          
          if (response.ok) {
            const data = await response.json();
            
            const botMessage: Message = {
              id: (Date.now() + 1).toString(),
              type: 'bot',
              content: data.response,
              timestamp: new Date(),
              messageType: 'text'
            };
            
            setMessages(prev => [...prev, botMessage]);
            
            // Check if more messages are coming
            if (data.next_message_in_seconds) {
              setPendingSequentialMessage({
                conversationId: pendingSequentialMessage.conversationId,
                nextMessageTime: Date.now() + (data.next_message_in_seconds * 1000)
              });
            } else {
              setPendingSequentialMessage(null);
            }
            
            // Play voice only if user explicitly requests it
            // No auto-play by default
          }
        } catch (error) {
          console.error('Error getting sequential message:', error);
          setPendingSequentialMessage(null);
        }
      }, delay);
      
      return () => clearTimeout(timer);
    }
  }, [pendingSequentialMessage, isVoiceMuted, currentlyPlayingMessageId, sessionId]);

  // Enhanced event listeners
  useEffect(() => {
    const handleAuthSuccess = (data: AuthResponse) => {
      setIsAuthenticated(true);
      setAuthStep('authenticated');
      setCustomerData(data.customer_data);
    };

    const handleContactVerified = (data: AuthResponse) => {
      setAuthStep('otp');
      setCustomerData(data.customer_data);
    };

    const handleSessionCreated = (data: AuthResponse) => {
      setSessionId(data.session_id || authService.getSessionId());
      setAuthStep('contact');
    };

    authService.on('authenticationSuccess', handleAuthSuccess);
    authService.on('contactVerified', handleContactVerified);
    authService.on('sessionCreated', handleSessionCreated);

    return () => {
      authService.off('authenticationSuccess', handleAuthSuccess);
      authService.off('contactVerified', handleContactVerified);
      authService.off('sessionCreated', handleSessionCreated);
    };
  }, [authService]);

  const handleContactVerification = async () => {
    if (!email) {
      toast.error('Please enter your email address');
      return;
    }

    setAuthLoading(true);
    try {
      // Step 1: Verify contact details
      const response = await authService.verifyContact(email, undefined, preferredMethod);
      if (response.success) {
        toast.success(`Contact verified for ${response.customer_name || 'customer'}! Sending OTP...`);
        
        // Step 2: Initiate OTP after successful contact verification
        try {
          console.log('🔄 Initiating OTP for session:', sessionId);
          
          const formData = new FormData();
          formData.append('session_id', sessionId!);
          
          const otpResponse = await fetch(`${config.backendUrl}/api/auth/initiate-otp-enhanced`, {
            method: 'POST',
            body: formData
          });
          
          if (otpResponse.ok) {
            const otpData = await otpResponse.json();
            if (otpData.success) {
              toast.success(`✅ OTP sent successfully! Check your ${otpData.otp_method || 'email'}.`); 
            } else {
              toast.error(`Failed to send OTP: ${otpData.message}`);
              addSystemMessage('❌ Failed to send OTP. You can try the Resend button.');
            }
          } else {
            toast.error('Failed to send OTP. You can try the Resend button.');
            addSystemMessage('❌ Failed to send OTP. You can try the Resend button.');
          }
        } catch (otpError) {
          console.error('OTP initiation error:', otpError);
          toast.error('Contact verified but failed to send OTP. You can try the Resend button.');
          addSystemMessage('❌ Failed to send OTP. You can try the Resend button.');
        }
      } else {
        toast.error(authService.getUserFriendlyError(response));
      }
    } catch (error: unknown) {
      toast.error(authService.getUserFriendlyError(error as AuthResponse));
    } finally {
      setAuthLoading(false);
    }
  };

  const handleOTPVerification = async () => {
    if (!otp || otp.length !== 6) {
      toast.error('Please enter a valid 6-digit OTP');
      return;
    }

    // Check if OTP is expired using the hook data
    if (otpStatus && !otpStatus.otp_active) {
      toast.error('OTP has expired. Please click Resend to get a new code.');
      return;
    }

    setAuthLoading(true);
    try {
      const response = await authService.verifyOTP(otp);
      if (response.success) {
        // Authentication successful - OTP status will be cleared automatically
      } else {
        toast.error(authService.getUserFriendlyError(response));
      }
    } catch (error: unknown) {
      toast.error(authService.getUserFriendlyError(error as AuthResponse));
    } finally {
      setAuthLoading(false);
    }
  };

  const handleResendOTP = async () => {
    setAuthLoading(true);
    try {
      const response = await authService.resendOTP();
      if (response.success) {
        toast.success('New verification code sent!');
        // The OTP status will be automatically updated by the hook
        await refreshOTPStatus(); 
      } else {
        toast.error(authService.getUserFriendlyError(response));
      }
    } catch (error: unknown) {
      toast.error(authService.getUserFriendlyError(error as AuthResponse));
    } finally {
      setAuthLoading(false);
    }
  };
  
  const getOTPStatusStyling = (urgency: string) => {
    const styles = {
      normal: {
        borderColor: 'border-green-500/50',
        bgColor: 'bg-green-900/20',
        textColor: 'text-green-400',
        progressColor: 'bg-green-400'
      },
      warning: {
        borderColor: 'border-yellow-500/50',
        bgColor: 'bg-yellow-900/20',
        textColor: 'text-yellow-400',
        progressColor: 'bg-yellow-400'
      },
      urgent: {
        borderColor: 'border-orange-500/50',
        bgColor: 'bg-orange-900/20',
        textColor: 'text-orange-400',
        progressColor: 'bg-orange-400'
      },
      critical: {
        borderColor: 'border-red-500/50',
        bgColor: 'bg-red-900/20',
        textColor: 'text-red-400',
        progressColor: 'bg-red-400'
      },
      expired: {
        borderColor: 'border-red-500/50',
        bgColor: 'bg-red-900/20',
        textColor: 'text-red-400',
        progressColor: 'bg-red-400'
      }
    };
    
    return styles[urgency as keyof typeof styles] || styles.normal;
  };

  // Handle classification confirmation
  const handleClassificationConfirmation = useCallback(async (feedback: string) => {
    if (!pendingClassification || !awaitingFeedback) {
      console.log('⚠️ No pending classification or not awaiting feedback');
      return;
    }
    
    console.log('🎯 Handling classification confirmation:', feedback);
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('complaint_id', pendingClassification.complaint_id || `temp_${Date.now()}`);
      formData.append('customer_feedback', feedback);
      formData.append('original_classification', JSON.stringify(pendingClassification));
      formData.append('session_id', sessionId!);
      
      const response = await fetch(`${config.backendUrl}/api/eva/confirm-classification`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${sessionId}`
        },
        body: formData
      });
      
      if (response.ok) {
        const data: ClassificationConfirmationData = await response.json();
        
        const followupMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: data.followup_response,
          timestamp: new Date(),
          messageType: 'text',
          next_steps: data.next_steps
        };
        
        setMessages(prev => [...prev, followupMessage]);
        
 
        
        // Show learning confirmation
        if (data.learning_applied) {
          addSystemMessage(`🎯 Thank you! I've learned from your feedback (${data.feedback_type}). This helps me improve for future interactions.`);
        }
        
        // Reset confirmation state
        setPendingClassification(null);
        setAwaitingFeedback(false);
        
        console.log('✅ Classification confirmation processed successfully');
      } else {
        console.error('❌ Classification confirmation failed:', response.status);
        addSystemMessage('❌ Failed to process your feedback, but I appreciate the input!');
      }
    } catch (error) {
      console.error('❌ Classification confirmation error:', error);
      addSystemMessage('❌ Failed to process your feedback, but I appreciate the input!');
    } finally {
      setIsProcessing(false);
    }
  }, [pendingClassification, awaitingFeedback, sessionId, addSystemMessage]);

  // Handle sending messages
  const handleSendMessage = async (content: string, type: 'text' | 'voice' | 'image' | 'document' = 'text') => {
  if (!content.trim() && type === 'text') return;
  if (!isAuthenticated) {
    toast.error('Please authenticate first');
    return;
  }

  const userMessage: Message = {
    id: Date.now().toString(),
    type: 'user',
    content,
    timestamp: new Date(),
    messageType: type
  };

  setMessages(prev => [...prev, userMessage]);
  setInputText('');
  resetInputHeight();
  setIsProcessing(true);

  try {
    const formData = new FormData();
    formData.append('message', content);
    formData.append('session_id', sessionId!);

    console.log('🤖 Sending message to Eva Agent:', content);

    const response = await fetch(`${config.backendUrl}/api/eva/chat-natural`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${sessionId}`
      },
      body: formData
    });

    console.log('📡 Response status:', response.status);

    if (response.ok) {
      const data: EvaResponse = await response.json();
      console.log('🤖 Eva Response received:', data);
      console.log('🎯 Eva Stage:', data.stage);
      
      // Clean response
      const cleanedResponse = data.response
        .replace(/\*\[Analysis in progress[^\]]*\]\*/g, '')
        .replace(/\.\.\.\s*$/g, '')
        .trim();
      
      const { parsedMessage, followUpQuestions } = parseEvaMessage(cleanedResponse);
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: cleanedResponse,
        timestamp: new Date(),
        messageType: 'text',
        emotional_state: data.emotional_state,
      };
      
      setMessages(prev => [...prev, botMessage]);

      // Handle separated question flow
      if (data.stage === "ready_for_first_question" && data.needs_first_question) {
        console.log('🎯 Triggering first follow-up question after delay');
        
        // Automatically trigger the first question after a short delay
        setTimeout(async () => {
          try {
            const formData = new FormData();
            formData.append('message', 'continue_first_question'); // Special trigger
            formData.append('session_id', sessionId!);
            
            const questionResponse = await fetch(`${config.backendUrl}/api/eva/chat-natural`, {
              method: 'POST',
              headers: { 'Authorization': `Bearer ${sessionId}` },
              body: formData
            });
            
            if (questionResponse.ok) {
              const questionData = await questionResponse.json();
              
              const questionMessage: Message = {
                id: (Date.now() + 3).toString(),
                type: 'bot',
                content: questionData.response,
                timestamp: new Date(),
                messageType: 'text'
              };
              
              setMessages(prev => [...prev, questionMessage]);
              console.log('✅ First follow-up question added');
            }
          } catch (error) {
            console.error('❌ Error getting first question:', error);
          }
        }, 1500); 
      }

      // Check if needs background processing
      const needsBackgroundProcessing = (
        data.stage === 'triage_analysis_initiated' || 
        data.background_processing === true ||
        data.response.includes('analyzing') ||
        data.response.includes('analysis in progress') ||
        data.stage === 'awaiting_triage_results'
      );
      
      console.log('🎯 Needs background processing:', needsBackgroundProcessing, 'Stage:', data.stage);
      
      if (needsBackgroundProcessing) {
        console.log('🎯 Starting analysis animation WITH AUTO-POLLING');
        setShowAnalysisAnimation(true);
        
        // Start polling for triage results after 3 seconds
        setTimeout(async () => {
          try {
            // First check if results are ready without sending new message
            const statusResponse = await fetch(`${config.backendUrl}/api/eva/triage-status/${sessionId}`, {
              method: 'GET',
              headers: { 'Authorization': `Bearer ${sessionId}` }
            });
            
            if (statusResponse.ok) {
              const statusData = await statusResponse.json();
              
              if (statusData.triage_results_ready) {
                // Results are ready, get them by sending a continue message
                const formData = new FormData();
                formData.append('message', 'continue_triage'); // Special trigger
                formData.append('session_id', sessionId!);
                
                const response = await fetch(`${config.backendUrl}/api/eva/chat-natural`, {
                  method: 'POST',
                  headers: { 'Authorization': `Bearer ${sessionId}` },
                  body: formData
                });
                
                if (response.ok) {
                  const resultData = await response.json();
                  
                  setShowAnalysisAnimation(false);
                  
                  const triageMessage: Message = {
                    id: (Date.now() + 2).toString(),
                    type: 'bot',
                    content: resultData.response,
                    timestamp: new Date(),
                    messageType: 'text'
                  };
                  
                  setMessages(prev => [...prev, triageMessage]);                   
                }
              } else {
                // Results not ready yet, continue showing animation
                setShowAnalysisAnimation(false);
                console.log('🎯 Triage results not ready yet');
              }
            }
          } catch (error) {
            console.error('Auto-polling error:', error);
            setShowAnalysisAnimation(false);
          }
        }, 3000); // 3 seconds like in test case
      }
      
      // Handle sequential messages
      if (data.sequential_messages_active && data.next_message_in_seconds) {
        setPendingSequentialMessage({
          conversationId: sessionId!,
          nextMessageTime: Date.now() + (data.next_message_in_seconds * 1000)
        });
      }

      // Handle follow-up questions
      if (followUpQuestions.length > 0) {
        addFollowUpQuestions(followUpQuestions);
      }

    } else {
      const errorText = await response.text();
      console.error('❌ Eva API Error:', {
        status: response.status,
        statusText: response.statusText,
        error: errorText
      });
      throw new Error(`Eva API failed: ${response.status} - ${errorText}`);
    }
  } catch (error) {
    console.error('❌ Eva chat error:', error);
    
    const errorMessage = (error as Error).message.toLowerCase();
    let fallbackMessage = `I apologize, ${customerData?.name || 'valued customer'}. `;
    
    if (errorMessage.includes('triage')) {
      fallbackMessage += `I'm having trouble analyzing your request right now. Our team has been notified and will review your message manually.`;
    } else {
      fallbackMessage += `I'm experiencing a technical issue with my AI systems. Let me try to help you in a different way. What specific banking service do you need assistance with today?`;
    }
    
    const botMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'bot',
      content: fallbackMessage,
      timestamp: new Date(),
      messageType: 'text'
    };
    
    setMessages(prev => [...prev, botMessage]);
    addSystemMessage(`⚠️ ${errorMessage.includes('triage') ? 'Triage system' : 'Eva AI'} temporarily offline. Error: ${(error as Error).message}`);
  } finally {
    setIsProcessing(false);
  }
};

  // Voice input with timeout and animations
  const handleVoiceInput = async () => {
    if (!isAuthenticated) {
      toast.error('Please authenticate first');
      return;
    }

    if (!voiceService.isSupported()) {
      toast.error('Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.');
      return;
    }

    if (isListening) {
      voiceService.stopListening();
      setIsListening(false);
      return;
    }

    try {
      setIsListening(true);
      
      const transcript = await voiceService.startListening();
      
      if (transcript && transcript.trim()) {
        handleSendMessage(transcript, 'voice');
      }
    } catch (error) {
      console.error('Voice input error:', error);
      if ((error as Error).message !== 'Speech recognition timeout') {
        toast.error('Voice input failed. Please check your microphone permissions and try again.');
      }
    } finally {
      setIsListening(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!isAuthenticated) {
      toast.error('Please authenticate first');
      return;
    }

    const file = event.target.files?.[0];
    if (!file) return;

    const fileType = file.type.includes('image') ? 'image' : 'document';
    const fileName = file.name;
    
    toast.success(`${fileType === 'image' ? 'Image' : 'Document'} uploaded: ${fileName}`);
    
    handleSendMessage(`I've uploaded a ${fileType}: ${fileName}. Can you help me with this?`, fileType);
    setShowAttachments(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(inputText);
    }
  };

  if (!isOpen) {
    return (
      <div className="fixed bottom-6 right-6 z-30">
        <div className="absolute inset-0 rounded-full bg-yellow-400/30 blur-lg animate-pulse"></div>
        <div className="absolute inset-0 rounded-full bg-yellow-400/20 blur-xl"></div>
        
        <button
          onClick={() => {
            setIsOpen(true);
            if (!sessionId && !isAuthenticated) {
              startAuthentication();
            }
          }}
          className="relative h-16 w-16 rounded-full hover:scale-110 transition-transform duration-300 focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:ring-offset-2"
        >
          <img 
            src="/Images_upload/chat-bot-3d-icon_235528-2179.jpeg" 
            alt="Eva - AI Assistant" 
            className="w-16 h-16 rounded-full object-cover shadow-2xl hover:shadow-3xl transition-shadow duration-300"
          />
        </button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-30">
      <Card className="w-96 h-[500px] bg-black border-gray-700 shadow-2xl">
        <CardHeader className="!flex !flex-row !items-center !justify-between !p-2 !space-y-0 bg-black text-white border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <img 
              src="/Images_upload/chat-bot-3d-icon_235528-2179.jpeg"
              alt="Eva - AI Assistant" 
              className="w-6 h-6 rounded-full object-cover"
            />
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-semibold font-serif text-sm">Eva - AI Assistant</span>
                {evaStatus?.status === 'active' && (
                  <div className="flex items-center space-x-1">
                    {evaStatus.capabilities?.reinforcement_learning && (
                      <Sparkles className="w-2 h-2 text-yellow-400 animate-pulse" />
                    )}
                  </div>
                )}
              </div>
              <div className="flex items-center space-x-2">
                <span className="font-semibold font-serif text-xs text-yellow-400">Bank of Swiss</span>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsOpen(false)}
              className="!h-6 !w-6 !p-0 hover:bg-gray-800 text-gray-400 hover:text-white"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        </CardHeader>

        <CardContent className="!p-0 flex flex-col h-[calc(100%-50px)]">

          {/* Authentication Flow or Messages */}
          <div 
            className="flex-1 overflow-y-auto p-4 space-y-4 bg-black"
            onClick={handleChatAreaClick}
          >
            
            {!isAuthenticated ? (
              <div className="space-y-4">
                {/* Authentication Steps */}
                {authStep === 'contact' && (
                  <div className="space-y-3">
                    <div className="text-left">
                      <AlertCircle className="w-8 h-8 text-yellow-400 mb-2" />
                      <h3 className="font-semibold text-white">Verify Your Identity</h3>
                      <p className="text-sm text-gray-300">Please enter your email to continue</p>
                    </div>
                    
                    <div className="space-y-2">
                      <Input
                        type="email"
                        placeholder="your.email@example.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="bg-gray-800 border-gray-600 text-white"
                        disabled={authLoading}
                      />
                      
                      <div className="flex gap-2">
                        <Button
                          variant={preferredMethod === 'email' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setPreferredMethod('email')}
                          className={preferredMethod === 'email' ? 'bg-yellow-400 text-black' : ''}
                        >
                          Email
                        </Button>
                        <Button
                          variant={preferredMethod === 'sms' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setPreferredMethod('sms')}
                          className={preferredMethod === 'sms' ? 'bg-yellow-400 text-black' : ''}
                        >
                          SMS
                        </Button>
                      </div>
                      
                      <Button 
                        onClick={handleContactVerification}
                        disabled={authLoading || !email}
                        className="w-full bg-yellow-400 text-black hover:bg-yellow-500"
                      >
                        {authLoading ? 'Verifying...' : 'Verify Contact'}
                      </Button>
                    </div>
                  </div>
                )}

                {authStep === 'otp' && (
                  <div className="space-y-3">
                    <div className="text-left">
                      <CheckCircle className="w-8 h-8 text-green-400 mb-2" />
                      <h3 className="font-semibold text-white">Enter Verification Code</h3>
                      <p className="text-sm text-gray-300">
                        We sent a 6-digit code to your {preferredMethod}
                      </p>
                      {customerData && (
                        <p className="text-xs text-yellow-400 mt-1">
                          Welcome, {customerData.name}!
                        </p>
                      )}
                    </div>

                    {/* Enhanced OTP Status and Countdown */}
                    {otpStatus && otpStatus.otp_initiated && (
                      <div className={`border rounded-lg p-3 space-y-2 transition-all duration-300 ${
                        getOTPStatusStyling(getUrgencyLevel(otpStatus.remaining_seconds)).borderColor
                      } ${getOTPStatusStyling(getUrgencyLevel(otpStatus.remaining_seconds)).bgColor}`}>
                        
                        {/* Header row */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <Clock className={`w-4 h-4 ${
                              otpStatus.otp_active ? 'text-blue-400' : 'text-gray-400'
                            }`} />
                            <span className="text-xs text-gray-300">
                              Code sent to {otpStatus.masked_contact}
                            </span>
                          </div>
                          <Badge variant="outline" className="bg-gray-700 border-gray-500">
                            {otpStatus.method.toUpperCase()}
                          </Badge>
                        </div>

                        {/* Countdown row */}
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-gray-400">
                            {otpStatus.otp_active ? 'Time remaining:' : 'Status:'}
                          </span>
                          <div className="flex items-center space-x-2">
                            {otpStatus.otp_active ? (
                              <>
                                <span className={`text-sm font-mono font-semibold ${
                                  getOTPStatusStyling(getUrgencyLevel(otpStatus.remaining_seconds)).textColor
                                }`}>
                                  {formatTimeRemaining(otpStatus.remaining_seconds)}
                                </span>
                                {otpStatus.remaining_seconds <= 60 && (
                                  <span className="text-xs text-red-400 animate-pulse">
                                    ⚠️
                                  </span>
                                )}
                              </>
                            ) : (
                              <span className="text-sm font-semibold text-red-400">
                                Expired
                              </span>
                            )}
                          </div>
                        </div>
                        
                        {/* Progress bar */}
                        <div className="w-full bg-gray-700 rounded-full h-1.5">
                          <div 
                            className={`h-1.5 rounded-full transition-all duration-1000 ${
                              getOTPStatusStyling(getUrgencyLevel(otpStatus.remaining_seconds)).progressColor
                            }`}
                            style={{ 
                              width: `${otpStatus.progress_percentage}%` 
                            }}
                          />
                        </div>
                        
                        {/* Attempts info */}
                        {otpStatus.attempts_used > 0 && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-gray-400">Attempts:</span>
                            <span className={`${
                              otpStatus.remaining_attempts <= 1 ? 'text-red-400' : 'text-gray-300'
                            }`}>
                              {otpStatus.remaining_attempts} remaining
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                    
                    <div className="space-y-2">
                      <Input
                        type="text"
                        placeholder="123456"
                        value={otp}
                        onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                        className="bg-gray-800 border-gray-600 text-white text-center text-lg tracking-widest"
                        maxLength={6}
                        disabled={authLoading || (otpStatus && !otpStatus.otp_active)}
                      />
                      
                      <div className="flex gap-2">
                        <Button 
                          onClick={handleOTPVerification}
                          disabled={authLoading || otp.length !== 6 || (otpStatus && !otpStatus.otp_active)}
                          className="flex-1 bg-yellow-400 text-black hover:bg-yellow-500"
                        >
                          {authLoading ? 'Verifying...' : 'Verify'}
                        </Button>
                        <Button 
                          variant="outline"
                          onClick={handleResendOTP}
                          disabled={authLoading}
                          className="border-gray-600 text-white hover:bg-gray-800 flex items-center space-x-1"
                        >
                          <RefreshCw className={`w-3 h-3 ${authLoading ? 'animate-spin' : ''}`} />
                          <span>Resend</span>
                        </Button>
                      </div>
                      
                      {/* Dynamic status messages */}
                      {otpStatus && (
                        <>
                          {!otpStatus.otp_active && otpStatus.otp_initiated && (
                            <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-2">
                              <p className="text-xs text-red-400 text-center">
                                ⏰ Verification code expired. Click 'Resend' to get a new code.
                              </p>
                            </div>
                          )}
                          
                          {otpStatus.otp_active && otpStatus.remaining_seconds <= 60 && (
                            <div className="bg-yellow-900/20 border border-yellow-500/50 rounded-lg p-2">
                              <p className="text-xs text-yellow-400 text-center">
                                ⚠️ Code expires soon. Have your code ready!
                              </p>
                            </div>
                          )}
                          
                          {otpStatus.otp_active && otpStatus.remaining_seconds <= 30 && (
                            <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-2 animate-pulse">
                              <p className="text-xs text-red-400 text-center">
                                🚨 Code expires very soon! Enter it now!
                              </p>
                            </div>
                          )}
                        </>
                      )}

                      {/* Error state */}
                      {otpStatusError && (
                        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-2">
                          <p className="text-xs text-red-400 text-center">
                            ❌ {otpStatusError}
                          </p>
                        </div>
                      )} 
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <>
                {/* Enhanced Messages with Eva Features */}
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className="max-w-[90%] relative">
                      <div
                        className={`rounded-lg relative ${
                          message.type === 'user'
                            ? 'bg-gray-800 text-white border border-gray-600 p-4'
                            : message.type === 'system'
                            ? 'bg-gray-700 text-white text-sm border border-gray-600 p-4'
                            : 'bg-yellow-500 text-black border-2 border-yellow-400 animate-pulse-border p-0' // Remove padding for bot
                        }`}
                      >
                        <div className="text-sm">
                          {message.messageType === 'voice' && message.type === 'user' && (
                            <span className="inline-flex items-center mr-2 text-xs text-yellow-400">
                              <Mic className="w-3 h-3 mr-1" />
                            </span>
                          )}
                          
                          {/* Use structured parsing for bot messages */}
                          {message.type === 'bot' ? (
                            (() => {

                              // Check if this should use simple text formatting (like Bot 2)
                              const shouldUseSimpleFormat = (message.content.includes("Perfect,") || 
                                                            message.content.includes("Current Status:") ||
                                                            message.content.includes("escalated") ||
                                                            message.content.includes("routed") ||
                                                            message.content.includes("tracking ID") ||
                                                            message.content.includes("I've immediately escalated"));

                              if (shouldUseSimpleFormat) {
                                console.log('🎯 Using simple text formatting like Bot 2');
                                return <SimpleTextMessage content={message.content} messageId={message.id} />;
                              }

                              // Check if this is an empathy message (Bot 2nd output)
                              const isEmpathyMessage = message.content.includes("What I'm doing right now") && 
                                                      !message.content.includes("**What I'm doing right now:**");
                              
                              if (isEmpathyMessage) {
                                console.log('🎯 Using empathy message formatting');
                                return (
                                  <div className="text-sm text-black space-y-2 p-4">
                                    {message.content.split('\n').map((line, lineIndex) => {
                                      const trimmed = line.trim();
                                      if (!trimmed) return <div key={lineIndex} className="h-2" />; // Spacing
                                      
                                      // Determine line type based on content patterns
                                      const isEmpathyStatement = !trimmed.includes('What I\'m doing') && 
                                                              !trimmed.includes('Analysis Status') && 
                                                              !trimmed.startsWith('•') &&
                                                              lineIndex < 3; // First few lines
                                      
                                      const isSectionHeader = trimmed.includes('What I\'m doing') || trimmed.includes('Analysis Status');
                                      const isBulletPoint = trimmed.startsWith('•');
                                      const isClosingStatement = !isSectionHeader && !isBulletPoint && !isEmpathyStatement;
                                      
                                      if (isEmpathyStatement) {
                                        return (
                                          <div key={lineIndex} className="leading-relaxed mb-3">
                                            {trimmed}
                                          </div>
                                        );
                                      }
                                      
                                      if (isSectionHeader) {
                                        return (
                                          <div key={lineIndex} className="font-semibold mt-4 mb-2">
                                            {trimmed}
                                          </div>
                                        );
                                      }
                                      
                                      if (isBulletPoint) {
                                        return (
                                          <div key={lineIndex} className="flex items-start space-x-2 ml-4">
                                            <span className="opacity-70 mt-1">•</span>
                                            <span className="leading-relaxed">{trimmed.substring(1).trim()}</span>
                                          </div>
                                        );
                                      }
                                      
                                      // Closing statement
                                      return (
                                        <div key={lineIndex} className="leading-relaxed mt-3 italic">
                                          {trimmed}
                                        </div>
                                      );
                                    })}
                                  </div>
                                );
                              }
                              
                              // First try complex parsing for triage responses
                              const complexParsed = parseComplexEvaResponse(message.content);
                              
                              if (complexParsed.messageStructure === 'complex') {
                                console.log('🎯 Using complex message structure');
                                return (
                                  <ComplexStructuredMessage 
                                    parsedResponse={complexParsed} 
                                    messageId={message.id}
                                    onSendMessage={handleSendMessage}
                                    onSetInputText={handleSetInputTextWithFocus}
                                  />
                                );
                              }
                              
                              // Check if this is a greeting message (contains time-based greeting)
                              const isGreeting = /^(Good\s+(morning|afternoon|evening),?\s+[^!]+!)/i.test(message.content);
                              
                              if (isGreeting) {
                                console.log('🎯 Using Microsoft Copilot card structure for greeting');
                                return <MicrosoftCopilotCard content={message.content} messageId={message.id} />;
                              }
                              
                              // Fall back to regular parsing for other messages
                              const { parsedMessage } = parseEvaMessage(message.content);
                              console.log('🎯 Using regular message structure');
                              
                              return parsedMessage ? (
                                <StructuredMessage parsedMessage={parsedMessage} messageId={message.id} />
                              ) : (
                                <div className="text-black">{message.content}</div>
                              );
                            })()
                          ) : (
                            <div>{message.content}</div>
                          )}
                          
                          {/* Show emotional state for Eva messages */}
                          {message.type === 'bot' && message.emotional_state && message.emotional_state !== 'neutral' && (
                            <div className="mt-2 text-xs opacity-70">
                              <span className="bg-black/20 px-2 py-1 rounded">
                                Emotion detected: {message.emotional_state}
                              </span>
                            </div>
                          )}
                          
                          {/* Show next steps if available */}
                          {message.type === 'bot' && message.next_steps && message.next_steps.length > 0 && (
                            <div className="mt-2 text-xs">
                              <div className="bg-black/20 p-2 rounded">
                                <strong>Next Steps:</strong>
                                <ul className="list-disc list-inside mt-1">
                                  {message.next_steps.map((step, index) => (
                                    <li key={index}>{step}</li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          )}
                          
                          {/* Show specialist assignments if available */}
                          {message.type === 'bot' && message.specialists_mentioned && message.specialists_mentioned.length > 0 && (
                            <div className="mt-2 text-xs">
                              <div className="bg-black/20 p-2 rounded">
                                <div className="flex items-center mb-1">
                                  <Users className="w-3 h-3 mr-1" />
                                  <strong>Specialist Assigned:</strong>
                                </div>
                                {message.specialists_mentioned.map((specialist, index) => (
                                  <div key={index} className="text-xs">
                                    <strong>{specialist.name}</strong> - {specialist.title}
                                    <br />
                                    {specialist.experience} experience, {specialist.success_rate} success rate
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {/* Enhanced voice and action controls - SINGLE TOGGLE BUTTON */}
                      {message.type === 'bot' && (
                        <VoiceToggleButton
                          message={message}
                          currentlyPlayingMessageId={currentlyPlayingMessageId}
                          onToggleVoice={toggleMessageVoice}
                          onCopyMessage={copyMessage}
                        />
                      )}
                    </div>
                  </div>
                ))}
                
                {showAnalysisAnimation && <TriageAnalysisAnimation />}
                {isProcessing && (
                  <div className="flex justify-start">
                    <div className="bg-yellow-500 text-black border-2 border-yellow-400 p-3 rounded-lg animate-pulse-border">
                      <div className="flex items-center space-x-2">
                        <Brain className="w-4 h-4 animate-pulse" />
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-black rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-black rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-black rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                        <span className="text-xs">Eva thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
            
            {/* Sequential message indicator */}
            {pendingSequentialMessage && (
              <div className="flex justify-start">
                <div className="bg-blue-500 text-white border-2 border-blue-400 p-3 rounded-lg animate-pulse-border">
                  <div className="flex items-center space-x-2">
                    <Clock className="w-4 h-4 animate-pulse" />
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-white rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                    <span className="text-xs">
                      Next message in {isVoiceMuted ? '10' : '5'} seconds...
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Attachments Menu */}
          {showAttachments && isAuthenticated && (
            <div className="absolute bottom-20 left-4 w-[50%]">
              <div className="bg-black/80 backdrop-blur-md rounded-lg shadow-xl border border-gray-700 overflow-hidden animate-in slide-in-from-bottom-2 duration-200">
                <div className="p-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                      setShowAttachments(false);
                    }}
                    className="w-full flex items-center px-3 py-2 text-sm text-white hover:bg-gray-700/50 rounded-md transition-colors"
                  >
                    <Image className="h-4 w-4 mr-3 text-yellow-400" />
                    <span className="text-gray-300">Upload Image</span>
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                      setShowAttachments(false);
                    }}
                    className="w-full flex items-center px-3 py-2 text-sm text-white hover:bg-gray-700/50 rounded-md transition-colors"
                  >
                    <FileText className="h-4 w-4 mr-3 text-yellow-400" />
                    <span className="text-gray-300">Upload Document</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Enhanced Input Area */}
          {isAuthenticated && (
            <div className="!p-3 bg-black border-t border-gray-700 mt-auto">
              <div className="flex items-end space-x-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowAttachments(!showAttachments);
                  }}
                  className="!h-8 !w-8 !p-0 text-yellow-400 hover:text-yellow-300 hover:bg-gray-800 transition-transform duration-200"
                >
                  <div className={`transition-transform duration-200 ${showAttachments ? 'rotate-45' : 'rotate-0'}`}>
                    <Plus className="h-4 w-4" />
                  </div>
                </Button>
                {inputText.length === 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleVoiceInput}
                    className={`!h-8 !w-8 !p-0 transition-all duration-300 rounded-full ${
                      isListening 
                        ? 'text-yellow-400 hover:text-yellow-300 bg-yellow-900/20 hover:bg-yellow-900/30 animate-pulse scale-110 shadow-lg shadow-yellow-400/50' 
                        : 'text-yellow-400 hover:text-yellow-300 hover:bg-gray-800 hover:scale-105'
                    }`}
                  >
                    <Mic className={`h-4 w-4 ${isListening ? 'animate-bounce' : ''}`} />
                  </Button>
                )}
                <textarea
                  ref={inputRef}
                  value={inputText}
                  onChange={handleInputChange}
                  placeholder={awaitingFeedback ? "Let me know if my classification was correct..." : "Type your message to Eva..."}
                  className="flex-1 bg-gray-800 border-gray-600 text-white placeholder:text-gray-400 focus:border-yellow-400 focus:ring-yellow-400 transition-all duration-200 resize-none rounded-md border px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 overflow-hidden [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
                  style={{ 
                    height: `${inputHeight}px`,
                    minHeight: '40px',
                    maxHeight: '120px',
                  }}
                  onKeyPress={handleKeyPress}
                  disabled={isProcessing}
                  rows={1}
                />
                <Button
                  onClick={() => handleSendMessage(inputText)}
                  className="bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-400 hover:to-yellow-500 text-black"
                  disabled={isProcessing || !inputText.trim()}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.pdf,.doc,.docx"
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default AuthenticatedEvaChat;