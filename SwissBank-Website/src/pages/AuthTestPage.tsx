import React, { useState, useEffect, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { CheckCircle, AlertCircle, Loader2, RefreshCw, User, Mail, Phone } from "lucide-react";
import { getAuthService } from "../services/authService";
import type { SessionStatusResponse, AuthResponse } from "../services/authService";

// Define proper interfaces
interface SessionCreatedData {
  session_id?: string;
}

interface CustomerData {
  customer_id: string;
  name: string;
  email: string;
  phone?: string;
  account_status: string;
}

interface ContactVerifiedData {
  customer_name?: string;
  customer_data?: CustomerData;
  otp_method?: string;
}

interface OtpVerifiedData {
  success: boolean;
  message?: string;
}

interface AuthenticationSuccessData {
  success: boolean;
  customer_data?: CustomerData;
}

interface SessionData {
  session_id: string;
  state: string;
  contact_verified: boolean;
  authenticated: boolean;
  contact_attempts: number;
  max_contact_attempts: number;
  remaining_contact_attempts: number;
  preferred_otp_method?: string;
  created_at: string;
  last_activity: string;
  customer_data?: CustomerData;
}

const AuthTestPage = () => {
  const [authService] = useState(() => getAuthService('http://127.0.0.1:8001'));
  const [currentStep, setCurrentStep] = useState(1);
  const [sessionId, setSessionId] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [otp, setOtp] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [preferredMethod, setPreferredMethod] = useState<'email' | 'sms'>('email');
  const [sessionStatus, setSessionStatus] = useState<SessionData | null>(null);
  const [customerData, setCustomerData] = useState<CustomerData | null>(null);

  const resetMessages = () => {
    setError('');
    setSuccess('');
  };

  const fetchSessionStatus = useCallback(async () => {
    try {
      const status: SessionStatusResponse = await authService.getSessionStatus();
      if (status.success && status.data) {
        setSessionStatus(status.data);
        
        // Determine current step based on session state
        if (status.data.authenticated) {
          setCurrentStep(4);
        } else if (status.data.state === 'otp_verification') {
          setCurrentStep(3);
        } else if (status.data.contact_verified) {
          setCurrentStep(2);
        } else {
          setCurrentStep(2);
        }
      }
    } catch (err) {
      console.error('Failed to fetch session status:', err);
    }
  }, [authService]);

  useEffect(() => {
    // Listen to auth service events
    const handleSessionCreated = (data: SessionCreatedData) => {
      console.log('Session created:', data);
      setSessionId(data.session_id || authService.getSessionId() || '');
    };

    const handleContactVerified = (data: ContactVerifiedData) => {
      console.log('Contact verified:', data);
      setSuccess(`Contact verified for: ${data.customer_name || 'Customer'}`);
      setCustomerData(data.customer_data || null);
      setCurrentStep(3);
    };

    const handleOtpVerified = (data: OtpVerifiedData) => {
      console.log('OTP verified:', data);
      setSuccess('Authentication successful! You can now access the chat.');
      setCurrentStep(4);
    };

    const handleAuthenticationSuccess = (data: AuthenticationSuccessData) => {
      console.log('Authentication success:', data);
      fetchSessionStatus();
    };

    authService.on('sessionCreated', handleSessionCreated);
    authService.on('contactVerified', handleContactVerified);
    authService.on('otpVerified', handleOtpVerified);
    authService.on('authenticationSuccess', handleAuthenticationSuccess);

    // Check if there's an existing session
    const existingSessionId = authService.getSessionId();
    if (existingSessionId) {
      setSessionId(existingSessionId);
      fetchSessionStatus();
    }

    return () => {
      authService.off('sessionCreated', handleSessionCreated);
      authService.off('contactVerified', handleContactVerified);
      authService.off('otpVerified', handleOtpVerified);
      authService.off('authenticationSuccess', handleAuthenticationSuccess);
    };
  }, [authService, fetchSessionStatus]);

  // Step 1: Create Session
  const createSession = async () => {
    resetMessages();
    setLoading(true);
    
    try {
      const response = await authService.createSession();
      
      if (response.success && response.session_id) {
        setSessionId(response.session_id);
        setSuccess('Session created successfully!');
        setCurrentStep(2);
      } else {
        setError(authService.getUserFriendlyError(response));
      }
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError(authService.getUserFriendlyError(err as AuthResponse));
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Step 2: Verify Contact
  const verifyContact = async () => {
    resetMessages();
    setLoading(true);
    
    try {
      // Step 2a: Verify contact details
      const response = await authService.verifyContact(
        email || undefined, 
        phone || undefined, 
        preferredMethod
      );
      
      if (response.success) {
        setSuccess(`Contact verified for ${response.customer_name || 'customer'}! Sending OTP...`);
        setCustomerData(response.customer_data || null);
        
        // Step 2b: Initiate OTP - FIXED VERSION
        try {
          console.log('🔄 Initiating OTP for session:', sessionId);
          
          // Call the correct initiate-otp endpoint
          const formData = new FormData();
          formData.append('session_id', sessionId);
          
          const otpResponse = await fetch('http://127.0.0.1:8001/api/auth/initiate-otp', {
            method: 'POST',
            body: formData
          });
          
          console.log('🔍 OTP Response status:', otpResponse.status);
          
          if (otpResponse.ok) {
            const otpData = await otpResponse.json();
            console.log('✅ OTP Response data:', otpData);
            
            if (otpData.success) {
              setSuccess(`✅ OTP sent successfully! Check your ${otpData.otp_method} (${otpData.masked_contact})`);
              setCurrentStep(3);
            } else {
              setError(`❌ Failed to send OTP: ${otpData.message || 'Unknown error'}`);
              console.error('❌ OTP initiation failed:', otpData);
              setCurrentStep(3); // Still allow resend attempts
            }
          } else {
            const errorText = await otpResponse.text();
            console.error('❌ OTP HTTP error:', otpResponse.status, errorText);
            setError(`❌ OTP request failed (${otpResponse.status}). Try using the Resend button.`);
            setCurrentStep(3);
          }
          
        } catch (otpError) {
          console.error('❌ OTP initiation network error:', otpError);
          setError('Contact verified but network error occurred. Try the Resend button.');
          setCurrentStep(3);
        }
      } else {
        setError(authService.getUserFriendlyError(response));
      }
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError(authService.getUserFriendlyError(err as AuthResponse));
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Step 3: Verify OTP
  const verifyOTP = async () => {
    resetMessages();
    setLoading(true);
    
    try {
      const response = await authService.verifyOTP(otp);
      
      if (response.success) {
        setSuccess('Authentication successful! You can now access the chat.');
        setCurrentStep(4);
        fetchSessionStatus();
      } else {
        setError(authService.getUserFriendlyError(response));
      }
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError(authService.getUserFriendlyError(err as AuthResponse));
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Resend OTP
  const resendOTP = async () => {
    resetMessages();
    setLoading(true);
    
    try {
      const response = await authService.resendOTP();
      
      if (response.success) {
        setSuccess(`OTP resent via ${response.otp_method || preferredMethod}`);
      } else {
        setError(authService.getUserFriendlyError(response));
      }
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError(authService.getUserFriendlyError(err as AuthResponse));
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const resetTest = () => {
    setCurrentStep(1);
    setSessionId('');
    setEmail('');
    setPhone('');
    setOtp('');
    setSessionStatus(null);
    setCustomerData(null);
    resetMessages();
    authService.restartSession();
  };

  const getStepStatus = (step: number) => {
    if (currentStep > step) return 'completed';
    if (currentStep === step) return 'current';
    return 'pending';
  };

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-yellow-400">Swiss Bank Authentication Test</h1>
          <p className="text-gray-300">Test the complete authentication flow</p>
        </div>

        {/* Progress Indicator */}
        <div className="flex justify-center space-x-4 mb-8">
          {[1, 2, 3, 4].map((step) => {
            const status = getStepStatus(step);
            const stepNames = ['Create Session', 'Verify Contact', 'Verify OTP', 'Authenticated'];
            
            return (
              <div key={step} className="flex items-center">
                <div className={`
                  w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold
                  ${status === 'completed' ? 'bg-green-600 text-white' : 
                    status === 'current' ? 'bg-yellow-400 text-black' : 
                    'bg-gray-600 text-gray-300'}
                `}>
                  {status === 'completed' ? <CheckCircle className="w-4 h-4" /> : step}
                </div>
                <span className={`ml-2 text-sm ${status === 'current' ? 'text-yellow-400' : 'text-gray-400'}`}>
                  {stepNames[step - 1]}
                </span>
                {step < 4 && <div className="ml-4 w-8 h-px bg-gray-600"></div>}
              </div>
            );
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Main Authentication Flow */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader>
              <CardTitle className="text-yellow-400">Authentication Flow</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              
              {/* Step 1: Create Session */}
              <div className={`p-4 border rounded ${
                currentStep === 1 ? 'border-yellow-400 bg-yellow-400/10' : 
                currentStep > 1 ? 'border-green-500 bg-green-500/10' : 
                'border-gray-600'
              }`}>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-white">1. Create Session</h3>
                  {currentStep > 1 && <CheckCircle className="h-5 w-5 text-green-400" />}
                </div>
                {sessionId && (
                  <p className="text-sm text-gray-400 mt-1 font-mono">
                    {sessionId.substring(0, 30)}...
                  </p>
                )}
                {currentStep === 1 && (
                  <Button 
                    onClick={createSession} 
                    disabled={loading}
                    className="mt-2 w-full bg-yellow-400 text-black hover:bg-yellow-500"
                  >
                    {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                    Create Session
                  </Button>
                )}
              </div>

              {/* Step 2: Verify Contact */}
              <div className={`p-4 border rounded ${
                currentStep === 2 ? 'border-yellow-400 bg-yellow-400/10' : 
                currentStep > 2 ? 'border-green-500 bg-green-500/10' : 
                'border-gray-600'
              }`}>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-white">2. Verify Contact</h3>
                  {currentStep > 2 && <CheckCircle className="h-5 w-5 text-green-400" />}
                </div>
                {currentStep === 2 && (
                  <div className="mt-3 space-y-3">
                    <div>
                      <Label htmlFor="email" className="text-gray-300">Email</Label>
                      <Input
                        id="email"
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="customer@example.com"
                        className="bg-gray-800 border-gray-600 text-white"
                      />
                    </div>
                    <div>
                      <Label htmlFor="phone" className="text-gray-300">Phone (Optional)</Label>
                      <Input
                        id="phone"
                        type="tel"
                        value={phone}
                        onChange={(e) => setPhone(e.target.value)}
                        placeholder="+1234567890"
                        className="bg-gray-800 border-gray-600 text-white"
                      />
                    </div>
                    <div>
                      <Label className="text-gray-300">Preferred OTP Method</Label>
                      <div className="flex gap-2 mt-1">
                        <Button
                          variant={preferredMethod === 'email' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setPreferredMethod('email')}
                          className={preferredMethod === 'email' ? 'bg-yellow-400 text-black' : ''}
                        >
                          <Mail className="w-4 h-4 mr-1" />
                          Email
                        </Button>
                        <Button
                          variant={preferredMethod === 'sms' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setPreferredMethod('sms')}
                          className={preferredMethod === 'sms' ? 'bg-yellow-400 text-black' : ''}
                        >
                          <Phone className="w-4 h-4 mr-1" />
                          SMS
                        </Button>
                      </div>
                    </div>
                    <Button 
                      onClick={verifyContact} 
                      disabled={loading || !email}
                      className="w-full bg-yellow-400 text-black hover:bg-yellow-500"
                    >
                      {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                      Verify Contact
                    </Button>
                  </div>
                )}
              </div>

              {/* Step 3: Verify OTP */}
              <div className={`p-4 border rounded ${
                currentStep === 3 ? 'border-yellow-400 bg-yellow-400/10' : 
                currentStep > 3 ? 'border-green-500 bg-green-500/10' : 
                'border-gray-600'
              }`}>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-white">3. Verify OTP</h3>
                  {currentStep > 3 && <CheckCircle className="h-5 w-5 text-green-400" />}
                </div>
                {currentStep === 3 && (
                  <div className="mt-3 space-y-3">
                    <div>
                      <Label htmlFor="otp" className="text-gray-300">Enter OTP Code</Label>
                      <Input
                        id="otp"
                        type="text"
                        value={otp}
                        onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                        placeholder="123456"
                        maxLength={6}
                        className="bg-gray-800 border-gray-600 text-white text-center text-lg tracking-widest"
                      />
                    </div>
                    <div className="flex gap-2">
                      <Button 
                        onClick={verifyOTP} 
                        disabled={loading || otp.length !== 6}
                        className="flex-1 bg-yellow-400 text-black hover:bg-yellow-500"
                      >
                        {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                        Verify OTP
                      </Button>
                      <Button 
                        variant="outline"
                        onClick={resendOTP}
                        disabled={loading}
                        className="border-gray-600 text-white hover:bg-gray-800"
                      >
                        <RefreshCw className="w-4 h-4 mr-1" />
                        Resend
                      </Button>
                    </div>
                  </div>
                )}
              </div>

              {/* Step 4: Success */}
              {currentStep === 4 && (
                <div className="p-4 border border-green-500 bg-green-500/10 rounded">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-green-400" />
                    <h3 className="font-semibold text-green-400">Authentication Complete!</h3>
                  </div>
                  <p className="text-sm text-green-300 mt-1">
                    You can now access the chat interface and submit complaints.
                  </p>
                  {sessionId && (
                    <p className="text-xs text-green-400 mt-2 font-mono break-all">
                      Session Token: {sessionId}
                    </p>
                  )}
                </div>
              )}

              {/* Reset Button */}
              {currentStep > 1 && (
                <Button 
                  variant="outline" 
                  onClick={resetTest}
                  className="w-full border-gray-600 text-white hover:bg-gray-800"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Reset Test
                </Button>
              )}

            </CardContent>
          </Card>

          {/* Status Panel */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader>
              <CardTitle className="text-yellow-400">Session Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              
              {/* Session Information */}
              {sessionStatus && (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <Label className="text-gray-400">State</Label>
                      <Badge variant="outline" className="ml-2">
                        {sessionStatus.state}
                      </Badge>
                    </div>
                    <div>
                      <Label className="text-gray-400">Contact Verified</Label>
                      <Badge variant={sessionStatus.contact_verified ? "default" : "secondary"} className="ml-2">
                        {sessionStatus.contact_verified ? "Yes" : "No"}
                      </Badge>
                    </div>
                    <div>
                      <Label className="text-gray-400">Authenticated</Label>
                      <Badge variant={sessionStatus.authenticated ? "default" : "secondary"} className="ml-2">
                        {sessionStatus.authenticated ? "Yes" : "No"}
                      </Badge>
                    </div>
                    <div>
                      <Label className="text-gray-400">OTP Method</Label>
                      <Badge variant="outline" className="ml-2">
                        {sessionStatus.preferred_otp_method || "None"}
                      </Badge>
                    </div>
                  </div>
                  
                  <div className="text-xs text-gray-500 space-y-1">
                    <p>Created: {new Date(sessionStatus.created_at).toLocaleString()}</p>
                    <p>Last Activity: {new Date(sessionStatus.last_activity).toLocaleString()}</p>
                  </div>
                </div>
              )}

              {/* Customer Information */}
              {customerData && (
                <div className="border-t border-gray-700 pt-3">
                  <Label className="text-gray-400 text-sm">Customer Information</Label>
                  <div className="mt-2 space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <User className="w-4 h-4 text-gray-400" />
                      <span className="text-white">{customerData.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Mail className="w-4 h-4 text-gray-400" />
                      <span className="text-white">{customerData.email}</span>
                    </div>
                    {customerData.phone && (
                      <div className="flex items-center gap-2">
                        <Phone className="w-4 h-4 text-gray-400" />
                        <span className="text-white">{customerData.phone}</span>
                      </div>
                    )}
                    <div className="text-xs text-gray-500">
                      <p>Customer ID: {customerData.customer_id}</p>
                      <p>Status: {customerData.account_status}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Refresh Status */}
              <Button 
                variant="outline" 
                onClick={fetchSessionStatus}
                className="w-full border-gray-600 text-white hover:bg-gray-800"
                disabled={!sessionId}
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh Status
              </Button>

            </CardContent>
          </Card>

        </div>

        {/* Messages */}
        {error && (
          <Alert variant="destructive" className="bg-red-900/20 border-red-500">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-red-300">{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert className="bg-green-900/20 border-green-500">
            <CheckCircle className="h-4 w-4 text-green-400" />
            <AlertDescription className="text-green-300">{success}</AlertDescription>
          </Alert>
        )}

      </div>
    </div>
  );
};

export default AuthTestPage;
