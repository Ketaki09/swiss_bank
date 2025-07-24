import { useState, useRef, useEffect, useCallback } from 'react';
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
import { CheckCircle2, ArrowRight, User, Shield, AlertTriangle, CreditCard, Eye, TrendingUp, Users2, MessageSquare, Target, Zap} from "lucide-react";

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
}

// Triage Analysis Animation Component
const TriageAnalysisAnimation = () => (
  <div className="flex justify-start">
    <div className="bg-blue-500 text-white border-2 border-blue-400 p-3 rounded-lg animate-pulse-border">
      <div className="flex items-center space-x-2">
        <Brain className="w-4 h-4 animate-pulse" />
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-white rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
          <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
        </div>
        <span className="text-xs">Triage analysis in progress...</span>
      </div>
    </div>
  </div>
);

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
    isLoading: otpStatusLoading, 
    error: otpStatusError,
    refreshStatus: refreshOTPStatus,
    formatTimeRemaining,
    getColorForStatus,
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
  const [isVoiceGloballyMuted, setIsVoiceGloballyMuted] = useState(false);
  const [showRepeatInfo, setShowRepeatInfo] = useState(false);
  
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
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to automatically play voice for bot messages
  const playVoiceForBotMessage = useCallback(async (content: string, messageId: string) => {
    if (!isVoiceGloballyMuted) {
      try {
        await voiceService.textToSpeech(content);
      } catch (error) {
        console.error('Error playing voice message:', error);
      }
    }
  }, [isVoiceGloballyMuted, voiceService]);

  // Function to repeat voice message
  const repeatVoiceMessage = useCallback(async (content: string) => {
    if (!isVoiceGloballyMuted) {
      try {
        speechSynthesis.cancel();
        await voiceService.textToSpeech(content);
      } catch (error) {
        console.error('Error repeating voice message:', error);
      }
    } else {
      toast.info('Voice is currently muted. Enable voice to hear messages.');
    }
  }, [isVoiceGloballyMuted, voiceService]);

  // Copy message function
  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content).then(() => {
      toast.success('Message copied to clipboard');
    }).catch(err => {
      console.error('Failed to copy message:', err);
      toast.error('Failed to copy message');
    });
  };

  // Enhanced parsing function with follow-up question extraction
  const parseEvaMessage = (content: string): { parsedMessage: ParsedMessage | null, followUpQuestions: string[] } => {
    console.log('🔍 Parsing content:', content.substring(0, 100) + '...');
    
    // Enhanced detection patterns that match the actual Eva output
    const hasStructure = content.includes('**What I\'m doing right now:**') || 
                        content.includes('**What happens next:**') || 
                        content.includes('**Your next actions:**') ||
                        content.includes('**What I\'m doing**') ||
                        content.includes('**What happens**') ||
                        content.includes('**Your actions**') ||
                        content.includes('**Next steps**') ||
                        content.includes('**Current status**');
    
    console.log('📋 Structure detected:', hasStructure);
    
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
    
    if (!hasStructure) {
      return { parsedMessage: null, followUpQuestions };
    }

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

    console.log('✅ Parsing complete:', {
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

  // Section styling with single icon only
  const getSectionStyling = (title: string) => {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('doing right now') || titleLower.includes('doing now') || titleLower.includes('current action')) {
      return {
        icon: 'zap',
        emoji: '🔧'
      };
    }
    
    if (titleLower.includes('happens next') || titleLower.includes('next step') || titleLower.includes('what happens')) {
      return {
        icon: 'clock',
        emoji: '⏰'
      };
    }
    
    if (titleLower.includes('your next actions') || titleLower.includes('your action') || titleLower.includes('next actions')) {
      return {
        icon: 'target',
        emoji: '🎯'
      };
    }
    
    if (titleLower.includes('current status') || titleLower.includes('status')) {
      return {
        icon: 'eye',
        emoji: '👁️'
      };
    }
    
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
      <div className="space-y-4">
        {/* Intro with emoji */}
        {parsedMessage.intro && (
          <div className="text-sm leading-relaxed flex items-start space-x-2">
            <span className="text-lg">📱</span>
            <span className="text-black">{parsedMessage.intro}</span>
          </div>
        )}

        {/* Structured Sections - Black theme with single icon */}
        {parsedMessage.sections.map((section, index) => (
          <div 
            key={`${messageId}-section-${index}`}
            className="bg-black/10 border border-black/20 rounded-lg p-3 space-y-3 transition-all duration-200 hover:shadow-md"
          >
            {/* Section Header with single emoji */}
            <div className="flex items-center space-x-2 mb-3">
              <span className="text-lg">{section.emoji}</span>
              <h4 className="font-semibold text-sm text-black">
                {section.title}
              </h4>
            </div>
            
            {/* Section Items - Black bullet points and text */}
            <div className="space-y-2">
              {section.items.map((item, itemIndex) => (
                <div 
                  key={`${messageId}-item-${index}-${itemIndex}`}
                  className="flex items-start space-x-3 text-sm group"
                >
                  <div className="mt-2 w-1.5 h-1.5 bg-black rounded-full flex-shrink-0 opacity-70 group-hover:opacity-100 transition-opacity" />
                  <span className="leading-relaxed text-black">{item}</span>
                </div>
              ))}
            </div>
          </div>
        ))}

        {/* Conclusion with emoji - Black text */}
        {parsedMessage.conclusion && (
          <div className="text-sm leading-relaxed bg-black/10 p-3 rounded border-l-4 border-black flex items-start space-x-2">
            <span className="text-lg">💡</span>
            <span className="text-black">{parsedMessage.conclusion}</span>
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
    let confidence = 0.8; // default value
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
        if (!isVoiceGloballyMuted) {
          setTimeout(() => {
            voiceService.textToSpeech(question);
          }, index * 500); // Stagger multiple questions
        }
      }, 1000 + (index * 1500)); // Delay for natural conversation flow
    });
  }, [isVoiceGloballyMuted, voiceService]);

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
            
            // Play greeting voice message if not globally muted
            if (!isVoiceGloballyMuted) {
              setTimeout(() => {
                voiceService.textToSpeech(data.greeting);
              }, 1500);
            }
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
  }, [isAuthenticated, customerData, messages.length, isVoiceGloballyMuted, voiceService, sessionId, evaStatus, addSystemMessage]);

  // Sequential message handling with voice-aware timing
  useEffect(() => {
    if (pendingSequentialMessage) {
      const delay = isVoiceDisabled ? 10000 : 5000; // 10s if voice disabled, 5s if enabled
      
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
            
            // Play voice if enabled
            if (!isVoiceGloballyMuted) {
              setTimeout(() => {
                playVoiceForBotMessage(data.response, botMessage.id);
              }, 500);
            }
          }
        } catch (error) {
          console.error('Error getting sequential message:', error);
          setPendingSequentialMessage(null);
        }
      }, delay);
      
      return () => clearTimeout(timer);
    }
  }, [pendingSequentialMessage, isVoiceDisabled, isVoiceGloballyMuted, sessionId, playVoiceForBotMessage]);

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
        await refreshOTPStatus(); // Manually refresh for immediate update
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
        
        // Play voice if enabled
        if (!isVoiceGloballyMuted) {
          setTimeout(() => {
            playVoiceForBotMessage(data.followup_response, followupMessage.id);
          }, 500);
        }
        
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
  }, [pendingClassification, awaitingFeedback, sessionId, playVoiceForBotMessage, addSystemMessage, isVoiceGloballyMuted]);

  // Also need to add the ClassificationConfirmationData interface at the top with other interfaces
  interface ClassificationConfirmationData {
    feedback_processed: boolean;
    feedback_type: string;
    learning_applied: boolean;
    followup_response: string;
    next_steps: string[];
  }

  // Handle classification confirmation
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
        classification_pending: data.classification_pending,
        requires_confirmation: data.requires_confirmation
      };
      
      setMessages(prev => [...prev, botMessage]);
      
      // 🔥 FIXED: Only show analysis animation for specific background processing cases
      const needsBackgroundProcessing = (
        data.stage === 'triage_analysis_initiated' && 
        data.background_processing === true &&
        data.response.includes('analyzing')
      );
      
      console.log('🎯 Needs background processing:', needsBackgroundProcessing, 'Stage:', data.stage);
      
      if (needsBackgroundProcessing) {
        console.log('🎯 Starting analysis animation - NO POLLING');
        setShowAnalysisAnimation(true);
        
        // 🔥 CRITICAL FIX: Just show animation, NO POLLING
        // The backend will handle the flow naturally in subsequent messages
        setTimeout(() => {
          setShowAnalysisAnimation(false);
          console.log('🎯 Analysis animation ended - waiting for next user message');
        }, 5000); // Show animation for 5 seconds, then stop
      }
      
      // Handle confirmation requirements
      if (data.requires_confirmation && data.classification_pending) {
        setPendingClassification(data.classification_pending);
        setAwaitingFeedback(true);
        
        setTimeout(() => {
          addSystemMessage(
            `🔍 I've analyzed your message and classified it as: "${data.classification_pending!.theme}". ` +
            `Is this correct? Please respond with "Yes, exactly right!" or "No, that's not quite right" to help me learn.`
          );
        }, 1000);
      }

      playVoiceForBotMessage(cleanedResponse, botMessage.id);

      // Handle sequential messages (keep this part)
      if (data.sequential_messages_active && data.next_message_in_seconds) {
        setPendingSequentialMessage({
          conversationId: sessionId!,
          nextMessageTime: Date.now() + (data.next_message_in_seconds * 1000)
        });
      }

      // Handle follow-up questions (keep this part)
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

  // Detect confirmation responses
  useEffect(() => {
  if (!awaitingFeedback || messages.length === 0) return;
  
  const lastMessage = messages[messages.length - 1];
  
  // 🔥 CRITICAL FIX: Only process USER messages, and only process each message once
  if (lastMessage.type === 'user' && !lastMessage.processed) {
    const content = lastMessage.content.toLowerCase();
    
    console.log('🎯 Checking for confirmation in user message:', content);
    
    // Mark message as processed to prevent duplicate processing
    setMessages(prev => prev.map(msg => 
      msg.id === lastMessage.id ? { ...msg, processed: true } : msg
    ));
    
    if (content.includes('yes') || content.includes('correct') || content.includes('exactly') || content.includes('right')) {
      console.log('✅ Positive confirmation detected');
      handleClassificationConfirmation('Yes, exactly right!');
    } else if (content.includes('no') || content.includes('wrong') || content.includes('not right') || content.includes('incorrect')) {
      console.log('❌ Negative confirmation detected');
      handleClassificationConfirmation('No, that\'s not quite right');
    } else if (content.includes('partially') || content.includes('sort of') || content.includes('close but')) {
      console.log('🔄 Partial confirmation detected');
      handleClassificationConfirmation('Partially correct, but not exactly');
    } else {
      console.log('🤷 No clear confirmation pattern detected');
    }
  }
}, [messages, awaitingFeedback, handleClassificationConfirmation]);

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

  // Handle muting/unmuting and global voice control
  const toggleGlobalVoiceMute = () => {
    const newMuteState = !isVoiceGloballyMuted;
    setIsVoiceGloballyMuted(newMuteState);
    setIsVoiceDisabled(newMuteState);
    
    if (newMuteState) {
      speechSynthesis.cancel();
      toast.info('Voice messages muted - using 10-second delays');
    } else {
      toast.info('Voice messages enabled');
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
            {/* Global Voice Mute Toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleGlobalVoiceMute}
              className="!h-6 !w-6 !p-0 hover:bg-gray-800 text-yellow-400 hover:text-yellow-300"
              title={isVoiceGloballyMuted ? 'Enable voice messages' : 'Mute voice messages'}
            >
              {isVoiceGloballyMuted ? (
                <VolumeX className="h-3 w-3" />
              ) : (
                <Volume2 className="h-3 w-3" />
              )}
            </Button>
            
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
                        className={`p-4 rounded-lg relative ${
                          message.type === 'user'
                            ? 'bg-gray-800 text-white border border-gray-600'
                            : message.type === 'system'
                            ? 'bg-gray-700 text-white text-sm border border-gray-600'
                            : 'bg-yellow-500 text-black border-2 border-yellow-400 animate-pulse-border'
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
                              const { parsedMessage } = parseEvaMessage(message.content);
                              console.log('🎯 Parsed message result:', parsedMessage);
                              
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
                      
                      {/* External action buttons positioned outside and to the right */}
                      {message.type === 'bot' && (
                        <div className="flex items-center space-x-2 mt-2 justify-end">
                          <button
                            onClick={() => repeatVoiceMessage(message.content)}
                            className="flex items-center justify-center w-5 h-5 bg-transparent hover:bg-yellow-400/20 text-gray-400 hover:text-yellow-400 rounded-md transition-all duration-300 border border-transparent hover:border-yellow-400/50 hover:shadow-lg hover:shadow-yellow-400/25 hover:scale-110"
                            title="Repeat message"
                          >
                            <RefreshCw className="w-2 h-2" />
                          </button>
                          
                          <button
                            onClick={() => copyMessage(message.content)}
                            className="flex items-center justify-center w-5 h-5 bg-transparent hover:bg-yellow-400/20 text-gray-400 hover:text-yellow-400 rounded-md transition-all duration-300 border border-transparent hover:border-yellow-400/50 hover:shadow-lg hover:shadow-yellow-400/25 hover:scale-110"
                            title="Copy message"
                          >
                            <Copy className="w-2 h-2" />
                          </button>
                        </div>
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
                      Next message in {isVoiceDisabled ? '10' : '5'} seconds...
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
              {/* Show awaiting feedback indicator */}
              {awaitingFeedback && (
                <div className="mb-2 p-2 bg-blue-900/20 border border-blue-500/50 rounded text-xs text-blue-400">
                  <div className="flex items-center space-x-2">
                    <Brain className="w-3 h-3 animate-pulse" />
                    <span>Awaiting your feedback to improve my learning...</span>
                  </div>
                </div>
              )}
              
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