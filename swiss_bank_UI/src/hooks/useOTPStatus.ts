// Create this file as src/hooks/useOTPStatus.ts

import { useState, useEffect, useCallback, useRef } from 'react';
import { config } from '../lib/config';

interface OTPStatus {
  otp_active: boolean;
  otp_initiated: boolean;
  expires_at: string | null;
  remaining_seconds: number;
  remaining_minutes: number;
  method: 'email' | 'sms';
  masked_contact: string;
  attempts_used: number;
  max_attempts: number;
  remaining_attempts: number;
  progress_percentage: number;
  status: 'not_initiated' | 'active' | 'active_warning' | 'expiring_soon' | 'expired';
  server_time: string;
  expiry_minutes: number;
}

interface UseOTPStatusReturn {
  otpStatus: OTPStatus | null;
  isLoading: boolean;
  error: string | null;
  refreshStatus: () => Promise<void>;
  formatTimeRemaining: (seconds: number) => string;
  getColorForStatus: (status: string) => string;
  getUrgencyLevel: (seconds: number) => 'normal' | 'warning' | 'urgent' | 'critical' | 'expired';
}

export const useOTPStatus = (sessionId: string | null): UseOTPStatusReturn => {
  const [otpStatus, setOtpStatus] = useState<OTPStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastFetchRef = useRef<number>(0);

  const refreshStatus = useCallback(async () => {
    if (!sessionId) {
      setOtpStatus(null);
      setError(null);
      return;
    }

    // Prevent too frequent API calls
    const now = Date.now();
    if (now - lastFetchRef.current < 1000) { // Minimum 1 second between calls
      return;
    }
    lastFetchRef.current = now;

    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(
        `${config.backendUrl}/api/auth/otp-status/${sessionId}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        setOtpStatus(result.data);
      } else {
        setError(result.message || 'Failed to get OTP status');
        setOtpStatus(null);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch OTP status';
      setError(errorMessage);
      console.error('Error fetching OTP status:', err);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  // Format time remaining in human-readable format
  const formatTimeRemaining = useCallback((seconds: number): string => {
    if (seconds <= 0) return 'Expired';
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes > 0) {
      return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    } else {
      return `0:${remainingSeconds.toString().padStart(2, '0')}`;
    }
  }, []);

  // Get color for status
  const getColorForStatus = useCallback((status: string): string => {
    const colorMap: Record<string, string> = {
      'not_initiated': '#6B7280', // gray-500
      'active': '#10B981',        // emerald-500
      'active_warning': '#EAB308', // yellow-500
      'expiring_soon': '#F59E0B',  // amber-500
      'expired': '#DC2626',       // red-600
    };
    
    return colorMap[status] || '#6B7280';
  }, []);

  // Get urgency level
  const getUrgencyLevel = useCallback((seconds: number): 'normal' | 'warning' | 'urgent' | 'critical' | 'expired' => {
    if (seconds <= 0) return 'expired';
    if (seconds <= 30) return 'critical';
    if (seconds <= 60) return 'urgent';
    if (seconds <= 180) return 'warning';
    return 'normal';
  }, []);

  // Set up polling when sessionId changes or OTP is active
  useEffect(() => {
    if (!sessionId) {
      // Clear interval if no session
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setOtpStatus(null);
      return;
    }

    // Initial fetch
    refreshStatus();

    // Set up polling interval
    const setupPolling = () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }

      intervalRef.current = setInterval(() => {
        refreshStatus();
      }, 1000); // Poll every second for real-time updates
    };

    setupPolling();

    // Cleanup on unmount or sessionId change
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [sessionId, refreshStatus]);

  // Adjust polling frequency based on OTP status
  useEffect(() => {
    if (!otpStatus || !sessionId) return;

    // Clear existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    let pollInterval: number;

    if (!otpStatus.otp_active) {
      // If OTP is not active, poll less frequently
      pollInterval = 5000; // Every 5 seconds
    } else if (otpStatus.remaining_seconds <= 60) {
      // If less than 1 minute remaining, poll more frequently
      pollInterval = 500; // Every 0.5 seconds
    } else if (otpStatus.remaining_seconds <= 180) {
      // If less than 3 minutes remaining, poll normally
      pollInterval = 1000; // Every 1 second
    } else {
      // If more than 3 minutes remaining, poll less frequently
      pollInterval = 2000; // Every 2 seconds
    }

    intervalRef.current = setInterval(() => {
      refreshStatus();
    }, pollInterval);

  }, [otpStatus, sessionId, refreshStatus]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return {
    otpStatus,
    isLoading,
    error,
    refreshStatus,
    formatTimeRemaining,
    getColorForStatus,
    getUrgencyLevel,
  };
};

// Optional: WebSocket version for real-time updates (if you prefer WebSocket over polling)
export const useOTPStatusWebSocket = (sessionId: string | null): UseOTPStatusReturn => {
  const [otpStatus, setOtpStatus] = useState<OTPStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const refreshStatus = useCallback(async () => {
    // For WebSocket version, this would manually fetch current status
    if (!sessionId) return;

    try {
      setIsLoading(true);
      const response = await fetch(`${config.backendUrl}/api/auth/otp-status/${sessionId}`);
      const result = await response.json();
      
      if (result.success) {
        setOtpStatus(result.data);
      } else {
        setError(result.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status');
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const formatTimeRemaining = useCallback((seconds: number): string => {
    if (seconds <= 0) return 'Expired';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return minutes > 0 
      ? `${minutes}:${remainingSeconds.toString().padStart(2, '0')}` 
      : `0:${remainingSeconds.toString().padStart(2, '0')}`;
  }, []);

  const getColorForStatus = useCallback((status: string): string => {
    const colorMap: Record<string, string> = {
      'not_initiated': '#6B7280',
      'active': '#10B981',
      'active_warning': '#EAB308',
      'expiring_soon': '#F59E0B',
      'expired': '#DC2626',
    };
    return colorMap[status] || '#6B7280';
  }, []);

  const getUrgencyLevel = useCallback((seconds: number): 'normal' | 'warning' | 'urgent' | 'critical' | 'expired' => {
    if (seconds <= 0) return 'expired';
    if (seconds <= 30) return 'critical';
    if (seconds <= 60) return 'urgent';
    if (seconds <= 180) return 'warning';
    return 'normal';
  }, []);

  // WebSocket connection management
  useEffect(() => {
    if (!sessionId) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setOtpStatus(null);
      return;
    }

    // Create WebSocket connection
    const wsUrl = `${config.backendUrl.replace('http', 'ws')}/ws/otp-status/${sessionId}`;
    
    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('OTP Status WebSocket connected');
        setError(null);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.success) {
            setOtpStatus(data.data);
          } else {
            setError(data.message);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Real-time connection failed');
      };

      wsRef.current.onclose = () => {
        console.log('OTP Status WebSocket disconnected');
        wsRef.current = null;
      };

    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setError('Failed to establish real-time connection');
    }

    // Cleanup on unmount or sessionId change
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [sessionId]);

  return {
    otpStatus,
    isLoading,
    error,
    refreshStatus,
    formatTimeRemaining,
    getColorForStatus,
    getUrgencyLevel,
  };
};