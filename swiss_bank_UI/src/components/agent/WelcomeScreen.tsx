import React, { useState, useRef, useEffect } from 'react';
import { X } from 'lucide-react';
import type { GooglePickerData, GoogleDriveFile, OneDriveOptions, OneDriveResponse, OneDriveFile, OneDriveError } from '../../types/api-types';

interface WelcomeScreenProps {
  onSendMessage: (message: string, attachedFiles?: File[]) => void;
}

interface AttachedFile {
  file: File;
  id: string;
  preview?: string;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSendMessage }) => {
  const [message, setMessage] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [showAttachmentMenu, setShowAttachmentMenu] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);
  const [isGoogleDriveConnected, setIsGoogleDriveConnected] = useState(false);
  const [isOneDriveConnected, setIsOneDriveConnected] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const attachmentButtonRef = useRef<HTMLButtonElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // TASK 2: Calculate max height dynamically to prevent overlap with footer
  const calculateMaxHeight = () => {
    const viewportHeight = window.innerHeight;
    const footerHeight = 120;
    const topOffset = 200;
    return Math.min(viewportHeight * 0.4, viewportHeight - footerHeight - topOffset);
  };

  // Auto-resize textarea with dynamic max height constraint and scrollbar when needed
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxHeight = calculateMaxHeight();
      
      if (scrollHeight > maxHeight) {
        textareaRef.current.style.height = `${maxHeight}px`;
        textareaRef.current.style.overflowY = 'auto';
      } else {
        textareaRef.current.style.height = `${Math.max(scrollHeight, 60)}px`;
        textareaRef.current.style.overflowY = 'hidden';
      }
    }
  }, [message]);

  // Close attachment menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (attachmentButtonRef.current && !attachmentButtonRef.current.contains(event.target as Node)) {
        setShowAttachmentMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

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
    
    if (!message.trim()) {
      return;
    }

    const files = attachedFiles.map(af => af.file);
    onSendMessage(message.trim(), files);
    setMessage('');
    setAttachedFiles([]);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const toggleVoiceInput = () => {
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

  // TASK 1: Enhanced file upload handlers with actual functionality
  const handleFileUpload = () => {
    setShowAttachmentMenu(false);
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const newFiles: AttachedFile[] = Array.from(files).map(file => ({
        file,
        id: Math.random().toString(36).substr(2, 9),
        preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
      }));
      
      setAttachedFiles(prev => [...prev, ...newFiles]);
      console.log('Files attached:', newFiles.map(f => f.file.name));
    }
  };

  // TASK 1: Real Google Drive connection implementation
  const handleConnectGoogleDrive = async () => {
    setShowAttachmentMenu(false);
    
    try {
      if (!window.gapi) {
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://apis.google.com/js/api.js';
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Failed to load Google API'));
          document.head.appendChild(script);
        });
      }

      await new Promise<void>((resolve) => {
        window.gapi.load('auth2:picker', resolve);
      });

      await window.gapi.client.init({
        apiKey: process.env.REACT_APP_GOOGLE_API_KEY || 'YOUR_GOOGLE_API_KEY',
        clientId: process.env.REACT_APP_GOOGLE_CLIENT_ID || 'YOUR_GOOGLE_CLIENT_ID',
        discoveryDocs: ['https://www.googleapis.com/discovery/v1/apis/drive/v3/rest'],
        scope: 'https://www.googleapis.com/auth/drive.readonly'
      });

      const authInstance = window.gapi.auth2.getAuthInstance();
      
      if (!authInstance.isSignedIn.get()) {
        await authInstance.signIn();
      }

      const picker = new window.google.picker.PickerBuilder()
        .addView(window.google.picker.ViewId.DOCS)
        .setOAuthToken(authInstance.currentUser.get().getAuthResponse().access_token)
        .setCallback((data: GooglePickerData) => {
          if (data.action === window.google.picker.Action.PICKED) {
            const files = data.docs.map((doc: GoogleDriveFile) => ({
              id: doc.id,
              name: doc.name,
              mimeType: doc.mimeType,
              url: doc.url
            }));
            console.log('Google Drive files selected:', files);
            setMessage(prev => prev + `[Google Drive files: ${files.map(f => f.name).join(', ')}] `);
          }
        })
        .build();
      
      picker.setVisible(true);
      setIsGoogleDriveConnected(true);
      
    } catch (error) {
      console.error('Google Drive connection failed:', error);
      alert('Failed to connect to Google Drive. Please check your API configuration.');
    }
  };

  // TASK 1: Real OneDrive connection implementation
  const handleConnectOneDrive = async () => {
    setShowAttachmentMenu(false);
    
    try {
      if (!window.OneDrive) {
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://js.live.net/v7.2/OneDrive.js';
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Failed to load OneDrive SDK'));
          document.head.appendChild(script);
        });
      }

      const odOptions: OneDriveOptions = {
        clientId: process.env.REACT_APP_ONEDRIVE_CLIENT_ID || 'YOUR_ONEDRIVE_CLIENT_ID',
        action: 'query',
        multiSelect: true,
        openInNewWindow: true,
        success: (files: OneDriveResponse) => {
          console.log('OneDrive files selected:', files);
          const fileNames = files.value.map((file: OneDriveFile) => file.name);
          setMessage(prev => prev + `[OneDrive files: ${fileNames.join(', ')}] `);
          setIsOneDriveConnected(true);
        },
        error: (error: OneDriveError) => {
          console.error('OneDrive picker error:', error);
          alert('Failed to access OneDrive files.');
        }
      };

      window.OneDrive.open(odOptions);
      
    } catch (error) {
      console.error('OneDrive connection failed:', error);
      alert('Failed to connect to OneDrive. Please check your configuration.');
    }
  };

  // TASK 3: Remove attached file
  const removeAttachedFile = (id: string) => {
    setAttachedFiles(prev => {
      const removed = prev.find(f => f.id === id);
      if (removed?.preview) {
        URL.revokeObjectURL(removed.preview);
      }
      return prev.filter(f => f.id !== id);
    });
  };

  // Combined button logic: voice when empty, send when has text
  const hasText = message.trim().length > 0;

  const handleVoiceSendButton = () => {
    if (hasText) {
      handleSubmit({
        preventDefault: () => {},
      } as React.FormEvent<HTMLFormElement>);
    } else {
      toggleVoiceInput();
    }
  };

  // Enhanced attachment menu items with connection status
  const attachmentMenuItems = [
    {
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="opacity-70">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
          <polyline points="10 9 9 9 8 9"/>
        </svg>
      ),
      label: 'Upload files',
      onClick: handleFileUpload,
      description: 'Upload documents, images, or other files'
    },
    {
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 24 24" className="opacity-70">
          <path d="M12.01 1.485c-2.082 0-3.754.02-3.743.047.01.02 1.708 3.001 3.774 6.62l3.76 6.574h3.76c2.081 0 3.753-.02 3.742-.047-.005-.02-1.708-3.001-3.775-6.62l-3.76-6.574zm-4.76 1.73a789.828 789.861 0 0 0-3.63 6.319L0 15.868l1.89 3.298 1.885 3.297 3.62-6.335 3.618-6.33-1.88-3.287C8.1 4.704 7.255 3.22 7.25 3.214zm2.259 12.653-.203.348c-.114.198-.96 1.672-1.88 3.287a423.93 423.948 0 0 1-1.698 2.97c-.01.026 3.24.042 7.222.042h7.244l1.796-3.157c.992-1.734 1.85-3.23 1.906-3.323l.104-.167h-7.249z"/>
        </svg>
      ),
      label: `Google Drive ${isGoogleDriveConnected ? '✓' : ''}`,
      onClick: handleConnectGoogleDrive,
      description: isGoogleDriveConnected ? 'Access more files from Google Drive' : 'Connect to Google Drive'
    },
    {
      icon: (
        <svg stroke="currentColor" fill="currentColor" strokeWidth="0" version="1.1" viewBox="0 0 16 16" className="opacity-70" height="16" width="16">
          <path d="M5.482 12.944c-0.942-0.235-1.466-0.984-1.468-2.095-0-0.355 0.025-0.525 0.114-0.754 0.217-0.56 0.793-0.982 1.55-1.138 0.377-0.077 0.493-0.16 0.493-0.353 0-0.060 0.045-0.24 0.1-0.399 0.249-0.724 0.71-1.327 1.202-1.573 0.515-0.258 0.776-0.316 1.399-0.313 0.886 0.005 1.327 0.197 1.945 0.846l0.34 0.357 0.304-0.105c1.473-0.51 2.942 0.358 3.061 1.809l0.032 0.397 0.29 0.104c0.829 0.297 1.218 0.92 1.148 1.837-0.046 0.599-0.326 1.078-0.77 1.315l-0.209 0.112-4.638 0.009c-3.564 0.007-4.697-0.006-4.893-0.055v0zM1.613 12.281c-0.565-0.142-1.164-0.67-1.445-1.273-0.159-0.342-0.168-0.393-0.168-0.998 0-0.576 0.014-0.668 0.14-0.954 0.267-0.603 0.78-1.038 1.422-1.21 0.136-0.036 0.263-0.094 0.283-0.128s0.043-0.221 0.050-0.415c0.045-1.206 0.794-2.269 1.839-2.61 0.565-0.184 1.306-0.202 1.92 0.058 0.195 0.082 0.173 0.1 0.585-0.471 0.244-0.338 0.705-0.695 1.108-0.909 0.435-0.231 0.887-0.337 1.428-0.336 1.512 0.004 2.815 1.003 3.297 2.529 0.154 0.487 0.146 0.624-0.035 0.628-0.079 0.002-0.306 0.048-0.505 0.102l-0.361 0.099-0.329-0.348c-0.928-0.98-2.441-1.192-3.728-0.522-0.514 0.268-0.927 0.652-1.239 1.153-0.222 0.357-0.506 1.024-0.506 1.189 0 0.117-0.090 0.176-0.474 0.309-1.189 0.412-1.883 1.364-1.882 2.582 0 0.443 0.108 0.986 0.258 1.296 0.057 0.117 0.088 0.228 0.070 0.247-0.046 0.049-1.525 0.032-1.73-0.019v0z"/>
        </svg>
      ),
      label: `OneDrive ${isOneDriveConnected ? '✓' : ''}`,
      onClick: handleConnectOneDrive,
      description: isOneDriveConnected ? 'Access more files from OneDrive' : 'Connect to OneDrive'
    }
  ];

  // TASK 3: Format file size
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // TASK 3: Get file type icon
  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) return '🖼️';
    if (file.type.startsWith('video/')) return '🎥';
    if (file.type.startsWith('audio/')) return '🎵';
    if (file.type.includes('pdf')) return '📄';
    if (file.type.includes('word') || file.type.includes('document')) return '📝';
    if (file.type.includes('sheet') || file.type.includes('excel')) return '📊';
    if (file.type.includes('presentation') || file.type.includes('powerpoint')) return '📽️';
    if (file.type.includes('zip') || file.type.includes('rar')) return '📦';
    return '📄';
  };

  return (
    <div className="flex flex-col items-center w-full h-full p-2 mx-auto justify-center sm:p-4 sm:gap-9 isolate mt-16 sm:mt-0">
      <div className="flex flex-col items-center gap-6 h-[450px] w-full sm:pt-12 isolate">
        {/* Logo Section */}
        <div className="flex flex-col items-center justify-center w-full sm:px-4 px-2 gap-6 sm:gap-4 xl:w-4/5 flex-initial pb-0">
          <div className="flex items-center gap-4 justify-center">
            <div className="w-14 h-14 bg-black rounded-full flex items-center justify-center overflow-hidden border-2 border-yellow-400">
              <img 
                src="/Images_upload/bank_logo.png" 
                alt="Swiss Bank Logo" 
                className="w-10 h-10 object-contain"
              />
            </div>
            
            <div className="text-center">
              <h1 className="text-4xl font-bold text-yellow-400 mb-1 font-serif tracking-tight">
                Swiss Agent
              </h1>
            </div>
          </div>
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={handleFileChange}
          accept="*/*"
          multiple
        />

        {/* Input Area Container */}
        <div className="absolute bottom-0 mx-auto inset-x-0 sm:relative flex flex-col items-center w-full gap-1 sm:gap-5 sm:bottom-auto sm:inset-x-auto sm:max-w-full">
          <div className="flex flex-col-reverse items-center justify-between flex-1 w-full gap-0 sm:gap-3 sm:flex-col relative p-2 sm:p-0">
            
            {/* Chat Input Form */}
            <form onSubmit={handleSubmit} className="w-full text-base flex flex-col gap-2 items-center justify-center relative z-10 mt-2">
              <div className="flex flex-col gap-0 justify-center w-full relative items-center xl:w-4/5">
                
                {/* Main Input Container */}
                <div className="!box-content flex flex-col bg-black mx-2 md:mx-0 items-stretch transition-all duration-200 relative cursor-text z-10 rounded-2xl border border-yellow-400/30 shadow-[0_0.25rem_1.25rem_hsl(0_0%_0%/3.5%),0_0_0_0.5px_hsla(var(--border-300)/0.15)] hover:shadow-[0_0.25rem_1.25rem_hsl(0_0%_0%/3.5%),0_0_0_0.5px_hsla(var(--border-200)/0.3)] focus-within:shadow-[0_0.25rem_1.25rem_hsl(0_0%_0%/7.5%),0_0_0_0.5px_hsla(var(--border-200)/0.3)] hover:focus-within:shadow-[0_0.25rem_1.25rem_hsl(0_0%_0%/7.5%),0_0_0_0.5px_hsla(var(--border-200)/0.3)] max-w-4xl w-full">
                  
                  {/* Text Area Section */}
                  <div className="flex flex-col gap-4 m-4 sm:m-6">
                    <div className="relative">
                      <div className="w-full transition-opacity duration-200 relative">
                        {/* Placeholder */}
                        {!message && (
                          <div className="absolute top-0 left-0 flex items-start text-gray-400 pointer-events-none select-none text-lg leading-[1.5] pt-0"
                               style={{
                                 fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"',
                                 fontSize: '16px',
                                 lineHeight: '1.5'
                               }}>
                            What do you want to know?
                          </div>
                        )}
                        
                        {/* Textarea with enhanced scrollbar */}
                        <textarea
                          ref={textareaRef}
                          dir="auto"
                          aria-label="Ask Swiss Agent anything"
                          value={message}
                          onChange={(e) => setMessage(e.target.value)}
                          onKeyPress={handleKeyPress}
                          className="w-full bg-transparent focus:outline-none text-white resize-none border-0 p-0 leading-[1.5] scrollbar-thin scrollbar-thumb-yellow-400/30 scrollbar-track-transparent hover:scrollbar-thumb-yellow-400/50"
                          style={{ 
                            minHeight: '60px',
                            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"',
                            fontSize: '16px',
                            lineHeight: '1.5',
                            fontWeight: 400
                          }}
                        />
                      </div>
                    </div>
                    
                    {/* Bottom Controls Row */}
                    <div className="flex gap-3 w-full items-center">
                      <div className="relative flex-1 flex items-center gap-3 shrink min-w-0">
                        
                        {/* Attachment Button */}
                        <div className="relative shrink-0">
                          <button
                            ref={attachmentButtonRef}
                            type="button"
                            onClick={() => setShowAttachmentMenu(!showAttachmentMenu)}
                            className="inline-flex items-center justify-center relative shrink-0 select-none disabled:pointer-events-none disabled:opacity-50 border-0.5 transition-all h-10 min-w-10 rounded-lg px-[9px] group text-yellow-400 border-yellow-400/30 active:scale-[0.98] hover:text-yellow-300 hover:bg-gray-800/50"
                            aria-label="Open attachments menu"
                          >
                            <div className="flex flex-row items-center justify-center gap-1">
                              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 256 256">
                                <path d="M224,128a8,8,0,0,1-8,8H136v80a8,8,0,0,1-16,0V136H40a8,8,0,0,1,0-16h80V40a8,8,0,0,1,16,0v80h80A8,8,0,0,1,224,128Z"/>
                              </svg>
                            </div>
                          </button>

                          {/* Attachment Menu */}
                          {showAttachmentMenu && (
                            <>
                              <div 
                                className="fixed inset-0 z-40" 
                                onClick={() => setShowAttachmentMenu(false)}
                              />
                              
                              <div 
                                className="absolute max-w-[calc(100vw-16px)] bottom-12 z-50 bg-black backdrop-blur-md"
                                style={{
                                  width: '12rem',
                                  animation: 'fadeIn 0.15s ease-out',
                                  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"',
                                  border: '1px solid rgba(250, 204, 21, 0.3)',
                                  borderRadius: '1rem',
                                  boxShadow: '0 0.25rem 1.25rem hsl(0 0% 0% / 3.5%), 0 0 0 0.5px hsla(var(--border-300) / 0.15)'
                                }}
                              >
                                <div className="flex flex-col p-1">
                                  {attachmentMenuItems.map((item, index) => (
                                    <button
                                      key={index}
                                      onClick={item.onClick}
                                      className="group flex w-full text-left gap-2 py-2 px-3 text-white rounded-lg select-none items-center active:scale-[0.995] hover:bg-gray-800/60 hover:text-white transition-colors duration-100"
                                      title={item.description}
                                    >
                                      <div className="group/icon min-w-4 min-h-4 flex items-center justify-center text-gray-300 shrink-0 group-hover:text-white">
                                        {item.icon}
                                      </div>
                                      <div className="flex flex-col flex-1 min-w-0">
                                        <div className="flex flex-row items-center flex-1">
                                          <p className="text-gray-300 text-sm text-ellipsis break-words whitespace-nowrap min-w-0 overflow-hidden group-hover:text-white">
                                            {item.label}
                                          </p>
                                        </div>
                                      </div>
                                    </button>
                                  ))}
                                </div>
                              </div>
                            </>
                          )}
                        </div>

                        <div className="flex flex-row items-center gap-2 min-w-0"></div>
                      </div>

                      {/* Send Button */}
                      <div style={{ opacity: 1, transform: 'none' }}>
                        <button
                          type={hasText ? "submit" : "button"}
                          onClick={hasText ? undefined : handleVoiceSendButton}
                          className={`inline-flex items-center justify-center relative shrink-0 select-none disabled:pointer-events-none disabled:opacity-50 transition-colors h-10 w-10 rounded-md active:scale-95 !rounded-lg !h-10 !w-10 ${
                            hasText 
                              ? 'bg-yellow-500 text-black hover:bg-yellow-600'
                              : isListening
                                ? 'bg-red-600 text-white hover:bg-red-700'
                                : 'bg-yellow-500 text-black hover:bg-yellow-600'
                          }`}
                          aria-label={hasText ? "Send message" : isListening ? "Stop recording" : "Enter voice mode"}
                        >
                          {hasText ? (
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 256 256">
                              <path d="M208.49,120.49a12,12,0,0,1-17,0L140,69V216a12,12,0,0,1-24,0V69L64.49,120.49a12,12,0,0,1-17-17l72-72a12,12,0,0,1,17,0l72,72A12,12,0,0,1,208.49,120.49Z"/>
                            </svg>
                          ) : isListening ? (
                            <div className="flex items-center justify-center gap-0.5">
                              {Array.from({length: 6}).map((_, i) => (
                                <div 
                                  key={i}
                                  className="w-0.5 rounded-full bg-current animate-pulse" 
                                  style={{
                                    height: ['0.5rem', '0.9rem', '1.3rem', '0.8rem', '1.1rem', '0.5rem'][i], 
                                    animationDelay: `${i * 100}ms`, 
                                    animationDuration: '0.8s'
                                  }}
                                />
                              ))}
                            </div>
                          ) : (
                            <div className="flex items-center justify-center gap-0.5">
                              {Array.from({length: 6}).map((_, i) => (
                                <div 
                                  key={i}
                                  className="w-0.5 rounded-full bg-current animate-pulse" 
                                  style={{
                                    height: ['0.5rem', '0.9rem', '1.3rem', '0.8rem', '1.1rem', '0.5rem'][i], 
                                    animationDelay: `${i * 200}ms`, 
                                    animationDuration: '1.2s'
                                  }}
                                />
                              ))}
                            </div>
                          )}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* TASK 3: Attached Files Section - Positioned at bottom */}
                  {attachedFiles.length > 0 && (
                    <div className="border-t border-yellow-400/20 bg-gray-900/20 rounded-b-2xl p-4 -m-0">
                      <div className="flex flex-row overflow-x-auto gap-3 pb-1">
                        {attachedFiles.map((attachedFile) => (
                          <div
                            key={attachedFile.id}
                            className="relative flex-shrink-0 group"
                          >
                            <div className="rounded-lg border border-yellow-400/30 bg-black/50 p-3 min-w-[120px] max-w-[120px] h-[120px] flex flex-col justify-between hover:border-yellow-400/50 transition-colors">
                              {/* File Preview or Icon */}
                              <div className="flex-1 flex items-center justify-center mb-2">
                                {attachedFile.preview ? (
                                  <img
                                    src={attachedFile.preview}
                                    alt={attachedFile.file.name}
                                    className="max-w-full max-h-16 object-cover rounded"
                                  />
                                ) : (
                                  <div className="text-2xl">
                                    {getFileIcon(attachedFile.file)}
                                  </div>
                                )}
                              </div>
                              
                              {/* File Info */}
                              <div className="text-center">
                                <h3 className="text-xs text-white truncate font-medium mb-1">
                                  {attachedFile.file.name}
                                </h3>
                                <p className="text-xs text-gray-400">
                                  {formatFileSize(attachedFile.file.size)}
                                </p>
                              </div>
                              
                              {/* File Type Badge */}
                              <div className="absolute top-2 right-2">
                                <div className="bg-yellow-400/20 text-yellow-400 px-1.5 py-0.5 rounded text-xs font-medium uppercase">
                                  {attachedFile.file.type.split('/')[1]?.substring(0, 3) || 'file'}
                                </div>
                              </div>
                            </div>
                            
                            {/* Remove Button */}
                            <button
                              onClick={() => removeAttachedFile(attachedFile.id)}
                              className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center transition-colors opacity-0 group-hover:opacity-100"
                              title="Remove file"
                            >
                              <X className="w-3 h-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                      
                      {/* Attached Files Summary */}
                      <div className="mt-3 pt-2 border-t border-gray-700/50">
                        <p className="text-xs text-gray-400">
                          {attachedFiles.length} file{attachedFiles.length !== 1 ? 's' : ''} attached
                          {attachedFiles.length > 0 && (
                            <span className="ml-2">
                              ({attachedFiles.reduce((total, f) => total + f.file.size, 0) > 1024 * 1024 
                                ? `${(attachedFiles.reduce((total, f) => total + f.file.size, 0) / (1024 * 1024)).toFixed(1)} MB`
                                : `${Math.round(attachedFiles.reduce((total, f) => total + f.file.size, 0) / 1024)} KB`
                              })
                            </span>
                          )}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>

      {/* Add global styles for animations */}
      <style dangerouslySetInnerHTML={{
        __html: `
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(4px) scale(0.95);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
        
        /* Enhanced scrollbar styles */
        .scrollbar-thin::-webkit-scrollbar {
          width: 6px;
        }
        
        .scrollbar-thin::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .scrollbar-thumb-yellow-400\\/30::-webkit-scrollbar-thumb {
          background-color: rgba(250, 204, 21, 0.3);
          border-radius: 3px;
        }
        
        .hover\\:scrollbar-thumb-yellow-400\\/50:hover::-webkit-scrollbar-thumb {
          background-color: rgba(250, 204, 21, 0.5);
        }
        
        /* File attachment animations */
        .group:hover .opacity-0 {
          opacity: 1;
        }
      `}} />
    </div>
  );
};

export default WelcomeScreen;