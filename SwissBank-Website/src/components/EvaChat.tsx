import { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { X, Send, Mic, MicOff, Image, FileText, Plus, AlertCircle, CheckCircle, Volume2, VolumeX, Clock, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { getAuthService } from "../services/authService";
import { VoiceService } from "../services/VoiceService";
import { config } from "../lib/config";
import type { AuthResponse } from "../services/authService";
import { useOTPStatus } from '../hooks/useOTPStatus';

interface Message {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
  messageType?: 'text' | 'voice' | 'image' | 'document';
}

interface CustomerData {
  customer_id: string;
  name: string;
  email: string;
  phone?: string;
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
  
  // Voice-related state
  const [isListening, setIsListening] = useState(false);
  const [isVoiceGloballyMuted, setIsVoiceGloballyMuted] = useState(false); // Global voice mute state
  const [showRepeatInfo, setShowRepeatInfo] = useState(false); // Info popup for repeat functionality
  
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

  // Add greeting message when user becomes authenticated (for existing sessions)
  useEffect(() => {
    if (isAuthenticated && customerData && messages.length === 0) {
      // Only add greeting if no messages exist (to avoid duplicate greetings)
      const greetingMessage: Message = {
        id: `greeting_${Date.now()}`,
        type: 'bot',
        content: `Hello ${customerData.name || 'valued customer'}! I'm Eva, your banking assistant. How can I help you today?`,
        timestamp: new Date(),
        messageType: 'text'
      };
      setMessages([greetingMessage]);
      
      // Play greeting voice message if not globally muted
      if (!isVoiceGloballyMuted) {
        setTimeout(() => {
          voiceService.textToSpeech(greetingMessage.content);
        }, 500); // Small delay to ensure message is rendered
      }
      
      // Show repeat info popup for first-time users
      setTimeout(() => {
        setShowRepeatInfo(true);
        // Auto-hide after 5 seconds
        setTimeout(() => setShowRepeatInfo(false), 5000);
      }, 1000);
      
      // Show repeat info popup for first-time users
      setTimeout(() => {
        setShowRepeatInfo(true);
        // Auto-hide after 5 seconds
        setTimeout(() => setShowRepeatInfo(false), 5000);
      }, 1000);
    }
  }, [isAuthenticated, customerData, messages.length, isVoiceGloballyMuted, voiceService]);

  // Updated event listeners to not show any messages after auth
  useEffect(() => {
    const handleAuthSuccess = (data: AuthResponse) => {
      setIsAuthenticated(true);
      setAuthStep('authenticated');
      setCustomerData(data.customer_data);
      
      // Add automatic greeting message when authentication is successful
      const greetingMessage: Message = {
        id: `greeting_${Date.now()}`,
        type: 'bot',
        content: `Hello ${data.customer_data?.name || 'valued customer'}! I'm Eva, your banking assistant. How can I help you today?`,
        timestamp: new Date(),
        messageType: 'text'
      };
      setMessages(prev => [...prev, greetingMessage]);
      
      // Play greeting voice message if not globally muted
      if (!isVoiceGloballyMuted) {
        setTimeout(() => {
          voiceService.textToSpeech(greetingMessage.content);
        }, 500); // Small delay to ensure message is rendered
      }
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
    
    return styles[urgency] || styles.normal;
  };

  const generateEvaResponse = useCallback((userInput: string, inputType: string): string => {
    const responses = {
      greeting: `Hello ${customerData?.name || 'valued customer'}! I'm Eva, your banking assistant. How can I help you today?`,
      
      services: "Swiss Bank offers comprehensive Private Banking, Corporate Banking, Asset Management, Trading Services, and Wealth Management. Each service is designed with Swiss precision to meet your specific financial objectives.",
      
      complaint: "I understand you'd like to file a complaint. I can help you with that. Please describe your concern in detail, and I'll make sure it's properly documented and escalated to the appropriate department.",
      
      account: `${customerData?.name || 'valued customer'}, I can help you with account inquiries. For security reasons, I'll need to verify some additional details before accessing your specific account information.`,
      
      default: `Thank you for your message, ${customerData?.name || 'valued customer'}. As your Swiss Bank AI assistant, I'm here to provide personalized guidance. How can I assist you today?`
    };

    const input = userInput.toLowerCase();
    
    if (input.includes('hello') || input.includes('hi') || input.includes('greet')) {
      return responses.greeting;
    } else if (input.includes('service') || input.includes('offer')) {
      return responses.services;
    } else if (input.includes('complaint') || input.includes('problem') || input.includes('issue')) {
      return responses.complaint;
    } else if (input.includes('account') || input.includes('balance')) {
      return responses.account;
    } else {
      return responses.default;
    }
  }, [customerData]);

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
      // For voice messages, use simple local processing
      if (type === 'voice') {
        const voiceResponse = await voiceService.processVoiceInput(content);
        
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: voiceResponse,
          timestamp: new Date(),
          messageType: 'text'
        };
        
        setMessages(prev => [...prev, botMessage]);
      
      // Automatically play voice for fallback bot message
      playVoiceForBotMessage(botMessage.content, botMessage.id);
        
        // Automatically play voice for bot message
        playVoiceForBotMessage(voiceResponse, botMessage.id);
        
      } else {
        // Call backend chat API for text messages
        const formData = new FormData();
        formData.append('message', content);
        formData.append('session_id', sessionId!);

        const response = await fetch(`${config.backendUrl}/api/chat/message`, {
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
            content: data.response || 'I received your message and am processing it.',
            timestamp: new Date(),
            messageType: 'text'
          };
          setMessages(prev => [...prev, botMessage]);
          
          // Automatically play voice for bot message
          playVoiceForBotMessage(data.response || 'I received your message and am processing it.', botMessage.id);
        } else {
          throw new Error('Chat API request failed');
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      // Fallback to simulated response
      const botResponse = generateEvaResponse(content, type);
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: botResponse,
        timestamp: new Date(),
        messageType: 'text'
      };
      setMessages(prev => [...prev, botMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Updated voice input with timeout and animations
  const handleVoiceInput = async () => {
    if (!isAuthenticated) {
      toast.error('Please authenticate first');
      return;
    }

    // Check if speech recognition is supported
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
      if (error.message !== 'Speech recognition timeout') {
        toast.error('Voice input failed. Please check your microphone permissions and try again.');
      }
    } finally {
      setIsListening(false);
    }
  };

  // Handle muting/unmuting specific messages and global voice control
  const toggleGlobalVoiceMute = () => {
    const newMuteState = !isVoiceGloballyMuted;
    setIsVoiceGloballyMuted(newMuteState);
    
    if (newMuteState) {
      // If muting globally, stop any current speech
      speechSynthesis.cancel();
      toast.info('Voice messages muted');
    } else {
      toast.info('Voice messages enabled');
    }
  };

  // Function to automatically play voice for bot messages
  const playVoiceForBotMessage = useCallback(async (content: string, messageId: string) => {
    // Only play if not globally muted
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
        // Stop any current speech before starting new one
        speechSynthesis.cancel();
        await voiceService.textToSpeech(content);
      } catch (error) {
        console.error('Error repeating voice message:', error);
      }
    } else {
      toast.info('Voice is currently muted. Enable voice to hear messages.');
    }
  }, [isVoiceGloballyMuted, voiceService]);

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
    
    handleSendMessage(`Uploaded ${fileType}: ${fileName}`, fileType);
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
              <span className="font-semibold font-serif text-sm">Eva - AI Assistant</span>
              <div className="font-semibold font-serif text-xs text-yellow-400">Bank of Swiss</div>
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
                {/* Messages - REMOVED TIMESTAMPS */}
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] p-3 rounded-lg relative ${
                        message.type === 'user'
                          ? 'bg-gray-800 text-white border border-gray-600'
                          : message.type === 'system'
                          ? 'bg-gray-700 text-white text-sm border border-gray-600'
                          : 'bg-yellow-500 text-black border-2 border-yellow-400 animate-pulse-border'
                      }`}
                    >
                      <div className="text-sm pr-8 pb-6">
                        {message.messageType === 'voice' && message.type === 'user' && (
                          <span className="inline-flex items-center mr-2 text-xs text-yellow-400">
                            <Mic className="w-3 h-3 mr-1" />
                          </span>
                        )}
                        {message.content}
                      </div>
                      
                      {/* Repeat button for bot messages */}
                      {message.type === 'bot' && (
                        <>
                          <button
                            onClick={() => repeatVoiceMessage(message.content)}
                            className="absolute bottom-1 right-1 p-1 rounded-full hover:bg-black/10 transition-colors"
                            title="Repeat message"
                          >
                            <RefreshCw className="w-3 h-3 text-black opacity-70 hover:opacity-100" />
                          </button>
                          
                          {/* Show tip popup beside the first bot message */}
                          {message.id.startsWith('greeting_') && showRepeatInfo && (
                            <div className="absolute -right-2 top-0 ml-2 w-64 z-50">
                              <div className="bg-yellow-400/95 backdrop-blur-sm border border-yellow-500 rounded-lg p-2 shadow-lg animate-in fade-in-0 slide-in-from-right-2 duration-300">
                                <div className="flex items-start space-x-2">
                                  <AlertCircle className="w-3 h-3 text-black mt-0.5 flex-shrink-0" />
                                  <div className="text-black text-xs leading-relaxed">
                                    <strong>💡 Tip:</strong> Use the repeat button (🔄) on any message to hear it again!
                                  </div>
                                  <button
                                    onClick={() => setShowRepeatInfo(false)}
                                    className="ml-auto p-0.5 rounded-full hover:bg-black/10 transition-colors"
                                  >
                                    <X className="w-2 h-2 text-black" />
                                  </button>
                                </div>
                              </div>
                              {/* Arrow pointing to the repeat button */}
                              <div className="absolute left-0 top-3 -ml-1 w-2 h-2 bg-yellow-400 border-l border-b border-yellow-500 transform rotate-45"></div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                ))}
                
                {isProcessing && (
                  <div className="flex justify-start">
                    <div className="bg-yellow-500 text-black border-2 border-yellow-400 p-3 rounded-lg animate-pulse-border">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-black rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-black rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-black rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                )}
              </>
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

          {/* Input Area with smooth scrolling */}
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
                  placeholder="Type your message to Eva..."
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