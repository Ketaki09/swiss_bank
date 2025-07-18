// src/services/authService.ts - Browser Compatible Implementation

type EventListener = (...args: unknown[]) => void;

class SimpleEventEmitter {
  private events: { [key: string]: EventListener[] } = {};

  on(event: string, listener: EventListener): void {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(listener);
  }

  off(event: string, listener: EventListener): void {
    if (!this.events[event]) return;
    this.events[event] = this.events[event].filter(l => l !== listener);
  }

  emit(event: string, ...args: unknown[]): void {
    if (!this.events[event]) return;
    this.events[event].forEach(listener => {
      try {
        listener(...args);
      } catch (error) {
        console.error('Event listener error:', error);
      }
    });
  }

  removeAllListeners(): void {
    this.events = {};
  }
}

// ==================== TYPE DEFINITIONS - Backend Aligned ====================

export interface AuthSession {
  session_id: string;
  state: SessionState;
  authenticated: boolean;
  contact_verified: boolean;
  customer_data?: CustomerData;
  contact_attempts: number;
  max_contact_attempts: number;
  remaining_contact_attempts: number;
  preferred_otp_method?: OTPMethod;
  contact_email?: string;
  contact_phone?: string;
  otp_auth_key?: string;
  expires_in_minutes: number;
  created_at: string;
  last_activity: string;
  locked_at?: string;
  lockout_remaining_minutes?: number;
  contact_verified_at?: string;
  otp_initiated_at?: string;
  otp_resent_at?: string;
  authenticated_at?: string;
}

export interface CustomerData {
  customer_id: string;
  name: string;
  email: string;
  phone: string;
  is_verified: boolean;
  account_status: 'active' | 'suspended' | 'pending';
  created_at: string;
  last_login?: string;
}

export interface AuthResponse {
  success: boolean;
  message: string;
  data?: unknown;
  error_code?: ErrorCode;
  retry_allowed?: boolean;
  technical_error?: boolean;
  action_required?: ActionRequired;
  retry_after_minutes?: number;
  current_attempt?: number;
  remaining_attempts?: number;
  max_attempts?: number;
  suggestions?: string[];
  session_id?: string;
  state?: SessionState;
  customer_name?: string;
  otp_method?: OTPMethod;
  masked_email?: string;
  masked_phone?: string;
  expires_in_minutes?: number;
  expires_in?: number;
  sent_to?: string;
  customer_data?: CustomerData;
}

export interface SessionStatusResponse extends AuthResponse {
  data?: {
    session_id: string;
    state: SessionState;
    contact_verified: boolean;
    authenticated: boolean;
    contact_attempts: number;
    max_contact_attempts: number;
    remaining_contact_attempts: number;
    preferred_otp_method?: OTPMethod;
    created_at: string;
    last_activity: string;
    customer_data?: CustomerData;
  };
}

interface VerifyContactPayload {
  session_id: string;
  email?: string;
  phone?: string;
  preferred_otp_method?: OTPMethod;
}

interface VerifyOTPPayload {
  session_id: string;
  otp: string;
}

interface ResendOTPPayload {
  session_id: string;
}

interface SessionStatusPayload {
  session_id: string;
}

// Backend-aligned error codes
export const ERROR_CODES = {
  // Session errors
  INVALID_SESSION: 'INVALID_SESSION',
  SESSION_EXPIRED: 'SESSION_EXPIRED',
  SESSION_CREATION_FAILED: 'SESSION_CREATION_FAILED',
  SESSION_LOCKED: 'SESSION_LOCKED',
  SESSION_INVALID: 'SESSION_INVALID',
  
  // Contact verification errors
  CUSTOMER_NOT_FOUND: 'CUSTOMER_NOT_FOUND',
  INVALID_INPUT: 'INVALID_INPUT',
  INVALID_EMAIL_FORMAT: 'INVALID_EMAIL_FORMAT',
  INVALID_PHONE_FORMAT: 'INVALID_PHONE_FORMAT',
  EMAIL_REQUIRED: 'EMAIL_REQUIRED',
  PHONE_REQUIRED: 'PHONE_REQUIRED',
  MAX_ATTEMPTS_EXCEEDED: 'MAX_ATTEMPTS_EXCEEDED',
  INVALID_OTP_METHOD: 'INVALID_OTP_METHOD',
  
  // OTP errors
  OTP_NOT_INITIATED: 'OTP_NOT_INITIATED',
  OTP_EXPIRED: 'OTP_EXPIRED',
  INVALID_OTP: 'INVALID_OTP',
  
  // Service errors (technical - retryable)
  SERVICE_ERROR: 'SERVICE_ERROR',
  DATABASE_ERROR: 'DATABASE_ERROR',
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR',
  SEND_FAILED: 'SEND_FAILED',
  RESEND_FAILED: 'RESEND_FAILED',
  
  // State errors
  INVALID_STATE: 'INVALID_STATE',
  
  // Client-side errors
  NO_SESSION: 'NO_SESSION',
  VALIDATION_ERROR: 'VALIDATION_ERROR'
} as const;

export type ErrorCode = typeof ERROR_CODES[keyof typeof ERROR_CODES];
export type SessionState = 'contact_verification' | 'otp_verification' | 'authenticated' | 'locked' | 'expired';
export type OTPMethod = 'email' | 'sms';
export type ActionRequired = 'restart' | 'verify_contact' | 'verify_otp' | 'wait' | 'contact_support';

// ==================== STORAGE MANAGER ====================

interface StorageManager {
  get<T>(key: string): T | null;
  set<T>(key: string, value: T, expiry?: number): void;
  remove(key: string): void;
  clear(): void;
}

class SimpleStorageManager implements StorageManager {
  private readonly prefix = 'eva_auth_';

  get<T>(key: string): T | null {
    try {
      const item = localStorage.getItem(`${this.prefix}${key}`);
      if (!item) return null;

      const parsed = JSON.parse(item);
      
      if (parsed.expiry && Date.now() > parsed.expiry) {
        this.remove(key);
        return null;
      }
      
      return parsed.value;
    } catch (error) {
      console.error('Storage get error:', error);
      return null;
    }
  }

  set<T>(key: string, value: T, expiry?: number): void {
    try {
      const item = {
        value,
        expiry: expiry ? Date.now() + expiry : null,
        timestamp: Date.now()
      };
      
      localStorage.setItem(`${this.prefix}${key}`, JSON.stringify(item));
    } catch (error) {
      console.error('Storage set error:', error);
    }
  }

  remove(key: string): void {
    localStorage.removeItem(`${this.prefix}${key}`);
  }

  clear(): void {
    const keys = Object.keys(localStorage).filter(key => key.startsWith(this.prefix));
    keys.forEach(key => localStorage.removeItem(key));
  }
}

// ==================== RETRY MANAGER - Backend Aligned ====================

class RetryManager {
  private static readonly DEFAULT_MAX_RETRIES = 3;
  private static readonly DEFAULT_BASE_DELAY = 1000;
  private static readonly TECHNICAL_ERROR_CODES = new Set<string>([
    ERROR_CODES.SERVICE_ERROR,
    ERROR_CODES.DATABASE_ERROR,
    ERROR_CODES.NETWORK_ERROR,
    ERROR_CODES.TIMEOUT_ERROR,
    ERROR_CODES.SEND_FAILED,
    ERROR_CODES.RESEND_FAILED
  ]);

  static async withRetry<T>(
    operation: () => Promise<T>,
    options: {
      maxRetries?: number;
      baseDelay?: number;
      onRetry?: (attempt: number, error: unknown) => void;
    } = {}
  ): Promise<T> {
    const {
      maxRetries = this.DEFAULT_MAX_RETRIES,
      baseDelay = this.DEFAULT_BASE_DELAY,
      onRetry
    } = options;

    let lastError: unknown;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;

        // Only retry technical errors, as per backend logic
        if (attempt === maxRetries || !this.shouldRetry(error)) {
          throw error;
        }

        const delay = baseDelay * Math.pow(2, attempt);
        onRetry?.(attempt + 1, error);
        await this.delay(delay);
      }
    }

    throw lastError;
  }

  private static shouldRetry(error: unknown): boolean {
    // Check if it's a technical error response
    if (
      typeof error === 'object' &&
      error !== null &&
      'technical_error' in error &&
      (error as { technical_error: boolean }).technical_error
    ) {
      return true;
    }

    // Check error code
    if (
      typeof error === 'object' &&
      error !== null &&
      'error_code' in error
    ) {
      const errorCode = (error as { error_code: string }).error_code;
      return this.TECHNICAL_ERROR_CODES.has(errorCode);
    }

    // Network errors
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return true;
    }

    return false;
  }

  private static delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ==================== MAIN AUTH SERVICE CLASS ====================

class BackendAlignedAuthService extends SimpleEventEmitter {
  private readonly baseUrl: string;
  private readonly storage: StorageManager;
  private sessionId: string | null = null;
  private initialized = false;
  private sessionCheckInterval: number | null = null;

  // Configuration constants aligned with backend
  private readonly SESSION_CHECK_INTERVAL = 60 * 1000; // 1 minute
  private readonly REQUEST_TIMEOUT = 30000; // 30 seconds

  constructor(baseUrl = 'http://localhost:8001') {
    super();
    this.baseUrl = baseUrl;
    this.storage = new SimpleStorageManager();
    this.initialize();
  }

  // ==================== INITIALIZATION ====================

  private async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      this.sessionId = this.storage.get<string>('session_id');

      if (this.sessionId) {
        await this.validateExistingSession();
      }

      this.startSessionCheck();
      this.setupEventListeners();
      this.initialized = true;

      this.emit('initialized', { hasSession: !!this.sessionId });
    } catch (error) {
      console.error('Auth service initialization failed:', error);
      this.clearAuth();
    }
  }

  private async validateExistingSession(): Promise<void> {
    try {
      const status = await this.getSessionStatus();
      if (!status.success || status.error_code === ERROR_CODES.SESSION_EXPIRED) {
        this.clearAuth();
      } else {
        this.emit('sessionRestored', status);
      }
    } catch (error) {
      console.error('Session validation failed:', error);
      this.clearAuth();
    }
  }

  private startSessionCheck(): void {
    if (this.sessionCheckInterval) return;

    this.sessionCheckInterval = window.setInterval(async () => {
      if (this.sessionId) {
        try {
          await this.validateExistingSession();
        } catch (error) {
          console.warn('Session check failed:', error);
        }
      }
    }, this.SESSION_CHECK_INTERVAL);
  }

  private setupEventListeners(): void {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && this.sessionId) {
        this.validateExistingSession();
      }
    });

    window.addEventListener('online', () => {
      this.emit('connectionRestored');
      if (this.sessionId) {
        this.validateExistingSession();
      }
    });

    window.addEventListener('offline', () => {
      this.emit('connectionLost');
    });
  }

  // ==================== VALIDATION METHODS ====================

  private validateEmailFormat(email: string): boolean {
    if (!email) return false;
    const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    return pattern.test(email) && email.length <= 254;
  }

  private validatePhoneFormat(phone: string): boolean {
    if (!phone) return false;
    const cleanPhone = phone.replace(/\D/g, '');
    return cleanPhone.length >= 10 && cleanPhone.length <= 15;
  }

  private validateOTPFormat(otp: string): boolean {
    return !!(otp && /^\d{6}$/.test(otp.trim()));
  }

  private validateContactInput(email?: string, phone?: string): AuthResponse | null {
    if (!email && !phone) {
      return this.createErrorResponse(
        'Please provide either email or phone number',
        ERROR_CODES.INVALID_INPUT
      );
    }

    if (email && phone) {
      return this.createErrorResponse(
        'Please provide either email or phone number, not both.',
        ERROR_CODES.INVALID_INPUT
      );
    }

    if (email && !this.validateEmailFormat(email)) {
      return this.createErrorResponse(
        'Please enter a valid email address.',
        ERROR_CODES.INVALID_EMAIL_FORMAT
      );
    }

    if (phone && !this.validatePhoneFormat(phone)) {
      return this.createErrorResponse(
        'Please enter a valid phone number.',
        ERROR_CODES.INVALID_PHONE_FORMAT
      );
    }

    return null;
  }

  // ==================== REQUEST HANDLING ====================

  private async handleResponse<T>(response: Response): Promise<T> {
    let data: Partial<AuthResponse>;
    
    try {
      const text = await response.text();
      data = text ? JSON.parse(text) : {};
    } catch (error) {
      throw this.createErrorResponse(
        'Invalid response from server',
        ERROR_CODES.NETWORK_ERROR,
        true,
        true
      );
    }

    const responseData: AuthResponse = {
      success: response.ok,
      message: data.message || (response.ok ? 'Success' : 'Request failed'),
      ...data
    };

    // Update session ID if provided
    if (responseData.session_id && responseData.session_id !== this.sessionId) {
      this.updateSession(responseData.session_id);
    }

    if (!response.ok) {
      // Handle specific HTTP status codes
      switch (response.status) {
        case 401:
          this.clearAuth();
          this.emit('authenticationRequired');
          break;
        case 429:
          responseData.retry_allowed = true;
          break;
        case 500:
        case 502:
        case 503:
        case 504:
          responseData.technical_error = true;
          responseData.retry_allowed = true;
          break;
      }
      
      throw responseData;
    }

    return responseData as T;
  }

  private handleNetworkError(error: unknown): AuthResponse {
    console.error('Network error:', error);
    
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return this.createErrorResponse(
        'Network connection failed. Please check your internet connection.',
        ERROR_CODES.NETWORK_ERROR,
        true,
        true
      );
    }

    return this.createErrorResponse(
      'Request failed. Please try again.',
      ERROR_CODES.NETWORK_ERROR,
      true,
      true
    );
  }

  // ==================== SESSION MANAGEMENT ====================

  private updateSession(sessionId: string): void {
    this.sessionId = sessionId;
    this.storage.set('session_id', sessionId, 24 * 60 * 60 * 1000); // 24 hours
    this.emit('sessionUpdated', { sessionId });
  }

  private clearAuth(): void {
    this.sessionId = null;
    this.storage.remove('session_id');
    
    if (this.sessionCheckInterval) {
      clearInterval(this.sessionCheckInterval);
      this.sessionCheckInterval = null;
    }
    
    this.emit('authCleared');
  }

  private validateSession(): boolean {
    if (!this.sessionId) {
      console.warn('No active session found');
      return false;
    }
    return true;
  }

  // ==================== UTILITY METHODS ====================

  private createErrorResponse(
    message: string,
    errorCode: ErrorCode,
    retryAllowed = false,
    technicalError = false,
    additionalData?: Record<string, unknown>
  ): AuthResponse {
    const response: AuthResponse = {
      success: false,
      message,
      error_code: errorCode,
      retry_allowed: retryAllowed,
      technical_error: technicalError
    };

    if (additionalData) {
      Object.assign(response, additionalData);
    }

    return response;
  }

  private createSuccessResponse(message: string, additionalData?: Partial<AuthResponse>): AuthResponse {
    return {
      success: true,
      message,
      ...additionalData
    };
  }

  // ==================== PUBLIC API METHODS - Backend Aligned ====================

  async createSession(): Promise<AuthResponse> {
    return RetryManager.withRetry(async () => {
      const response = await fetch(`${this.baseUrl}/api/auth/session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ip_address: '127.0.0.1',
          user_agent: 'web-client'
        })
      });

      const result = await this.handleResponse<AuthResponse>(response);

      if (result.success && result.session_id) {
        this.updateSession(result.session_id);
        this.emit('sessionCreated', result);
        console.log('✅ Session created successfully:', this.sessionId);
      }

      return result;
    }, {
      onRetry: (attempt, error) => {
        console.warn(`Session creation attempt ${attempt} failed:`, error);
        this.emit('retryAttempt', { operation: 'createSession', attempt, error });
      }
    });
  }

  async verifyContact(email?: string, phone?: string, preferredMethod?: OTPMethod): Promise<AuthResponse> {
    if (!this.validateSession()) {
      return this.createErrorResponse(
        'No active session. Please refresh and try again.',
        ERROR_CODES.NO_SESSION
      );
    }

    const validationError = this.validateContactInput(email, phone);
    if (validationError) {
      return validationError;
    }

    return RetryManager.withRetry(async () => {
      // Use FormData as backend expects form data
      const formData = new FormData();
      formData.append('session_id', this.sessionId!);
      
      if (email) {
        formData.append('email', email.trim().toLowerCase());
      }
      if (phone) {
        formData.append('phone', phone.trim());
      }
      if (preferredMethod) {
        formData.append('preferred_otp_method', preferredMethod);
      }

      const response = await fetch(`${this.baseUrl}/api/auth/verify-contact`, {
        method: 'POST',
        body: formData
      });

      const result = await this.handleResponse<AuthResponse>(response);

      if (result.success) {
        this.emit('contactVerified', result);
        console.log('✅ Contact verified successfully');
      }

      return result;
    }, {
      onRetry: (attempt, error) => {
        console.warn(`Contact verification attempt ${attempt} failed:`, error);
        this.emit('retryAttempt', { operation: 'verifyContact', attempt, error });
      }
    });
  }

  async verifyOTP(otp: string): Promise<AuthResponse> {
    if (!this.validateSession()) {
      return this.createErrorResponse(
        'No active session. Please refresh and try again.',
        ERROR_CODES.NO_SESSION
      );
    }

    if (!this.validateOTPFormat(otp)) {
      return this.createErrorResponse(
        'Please enter a valid 6-digit OTP.',
        ERROR_CODES.INVALID_OTP
      );
    }

    return RetryManager.withRetry(async () => {
      // Use FormData as backend expects form data
      const formData = new FormData();
      formData.append('session_id', this.sessionId!);
      formData.append('otp', otp.trim());

      const response = await fetch(`${this.baseUrl}/api/auth/verify-otp`, {
        method: 'POST',
        body: formData
      });

      const result = await this.handleResponse<AuthResponse>(response);

      if (result.success) {
        this.emit('otpVerified', result);
        this.emit('authenticationSuccess', result);
        console.log('✅ OTP verified successfully - User authenticated');
      }

      return result;
    }, {
      onRetry: (attempt, error) => {
        console.warn(`OTP verification attempt ${attempt} failed:`, error);
        this.emit('retryAttempt', { operation: 'verifyOTP', attempt, error });
      }
    });
  }

  async resendOTP(): Promise<AuthResponse> {
    if (!this.validateSession()) {
      return this.createErrorResponse(
        'No active session. Please refresh and try again.',
        ERROR_CODES.NO_SESSION
      );
    }

    return RetryManager.withRetry(async () => {
      // Use FormData as backend expects form data
      const formData = new FormData();
      formData.append('session_id', this.sessionId!);

      const response = await fetch(`${this.baseUrl}/api/auth/resend-otp`, {
        method: 'POST',
        body: formData
      });

      const result = await this.handleResponse<AuthResponse>(response);

      if (result.success) {
        this.emit('otpResent', result);
        console.log('✅ OTP resent successfully');
      }

      return result;
    }, {
      onRetry: (attempt, error) => {
        console.warn(`OTP resend attempt ${attempt} failed:`, error);
        this.emit('retryAttempt', { operation: 'resendOTP', attempt, error });
      }
    });
  }

  async getSessionStatus(): Promise<SessionStatusResponse> {
    if (!this.validateSession()) {
      return this.createErrorResponse(
        'No active session found.',
        ERROR_CODES.NO_SESSION
      ) as SessionStatusResponse;
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/auth/session/${this.sessionId}`, {
        method: 'GET'
      });

      const result = await this.handleResponse<SessionStatusResponse>(response);
      return result;
    } catch (error) {
      console.error('Session status check failed:', error);
      return error as SessionStatusResponse;
    }
  }

  async restartSession(): Promise<AuthResponse> {
    this.clearAuth();
    this.emit('sessionRestarted');
    return this.createSession();
  }
  
  async initiateOTP(): Promise<AuthResponse> {
    if (!this.validateSession()) {
      return this.createErrorResponse(
        'No active session. Please refresh and try again.',
        ERROR_CODES.NO_SESSION
      );
    }

    return RetryManager.withRetry(async () => {
      // Use FormData as backend expects form data
      const formData = new FormData();
      formData.append('session_id', this.sessionId!);

      const response = await fetch(`${this.baseUrl}/api/auth/initiate-otp`, {
        method: 'POST',
        body: formData
      });

      const result = await this.handleResponse<AuthResponse>(response);

      if (result.success) {
        this.emit('otpInitiated', result);
        console.log('✅ OTP initiated successfully');
      }

      return result;
    }, {
      onRetry: (attempt, error) => {
        console.warn(`OTP initiation attempt ${attempt} failed:`, error);
        this.emit('retryAttempt', { operation: 'initiateOTP', attempt, error });
      }
    });
  }

  // ==================== STATE MANAGEMENT METHODS ====================

  isAuthenticated(): boolean {
    return !!this.sessionId;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  getSessionId(): string | null {
    return this.sessionId;
  }

  // ==================== ERROR HANDLING UTILITIES ====================

  getErrorMessage(errorCode: ErrorCode): string {
    const errorMessages: Record<ErrorCode, string> = {
      [ERROR_CODES.INVALID_SESSION]: 'Your session has expired. Please refresh and try again.',
      [ERROR_CODES.SESSION_EXPIRED]: 'Your session has expired. Please start a new session.',
      [ERROR_CODES.SESSION_CREATION_FAILED]: 'Failed to create session. Please try again.',
      [ERROR_CODES.SESSION_LOCKED]: 'Your session is locked. Please wait before trying again.',
      [ERROR_CODES.SESSION_INVALID]: 'Invalid session. Please start a new session.',
      
      [ERROR_CODES.CUSTOMER_NOT_FOUND]: 'No customer found with the provided information.',
      [ERROR_CODES.INVALID_INPUT]: 'Please provide valid input.',
      [ERROR_CODES.INVALID_EMAIL_FORMAT]: 'Please enter a valid email address.',
      [ERROR_CODES.INVALID_PHONE_FORMAT]: 'Please enter a valid phone number.',
      [ERROR_CODES.EMAIL_REQUIRED]: 'Email address is required.',
      [ERROR_CODES.PHONE_REQUIRED]: 'Phone number is required.',
      [ERROR_CODES.MAX_ATTEMPTS_EXCEEDED]: 'Maximum attempts exceeded. Please try again later.',
      [ERROR_CODES.INVALID_OTP_METHOD]: 'Invalid OTP method specified.',
      
      [ERROR_CODES.OTP_NOT_INITIATED]: 'OTP not initiated. Please request an OTP first.',
      [ERROR_CODES.OTP_EXPIRED]: 'OTP has expired. Please request a new one.',
      [ERROR_CODES.INVALID_OTP]: 'Invalid OTP. Please check and try again.',
      
      [ERROR_CODES.SERVICE_ERROR]: 'Service temporarily unavailable. Please try again.',
      [ERROR_CODES.DATABASE_ERROR]: 'Database error occurred. Please try again.',
      [ERROR_CODES.NETWORK_ERROR]: 'Network error. Please check your connection.',
      [ERROR_CODES.TIMEOUT_ERROR]: 'Request timed out. Please try again.',
      [ERROR_CODES.SEND_FAILED]: 'Failed to send verification code. Please try again.',
      [ERROR_CODES.RESEND_FAILED]: 'Failed to resend verification code. Please try again.',
      
      [ERROR_CODES.INVALID_STATE]: 'Invalid authentication state.',
      
      [ERROR_CODES.NO_SESSION]: 'No active session found.',
      [ERROR_CODES.VALIDATION_ERROR]: 'Validation error occurred.'
    };

    return errorMessages[errorCode] || 'An unexpected error occurred.';
  }

  getUserFriendlyError(response: AuthResponse): string {
    if (response.message) {
      return response.message;
    }
    
    if (response.error_code) {
      return this.getErrorMessage(response.error_code);
    }
    
    return 'An unexpected error occurred. Please try again.';
  }

  shouldShowRetryButton(response: AuthResponse): boolean {
    return response.retry_allowed === true || 
           response.technical_error === true;
  }

  isRetryableError(response: AuthResponse): boolean {
    return RetryManager['shouldRetry'](response);
  }

  // ==================== CLEANUP ====================

  destroy(): void {
    if (this.sessionCheckInterval) {
      clearInterval(this.sessionCheckInterval);
      this.sessionCheckInterval = null;
    }
    
    this.removeAllListeners();
    this.initialized = false;
    
    console.log('✅ Auth service destroyed');
  }
}

// ==================== SINGLETON EXPORT ====================

let authServiceInstance: BackendAlignedAuthService | null = null;

export const getAuthService = (baseUrl?: string): BackendAlignedAuthService => {
  if (!authServiceInstance) {
    authServiceInstance = new BackendAlignedAuthService(baseUrl);
  }
  return authServiceInstance;
};

export const destroyAuthService = (): void => {
  if (authServiceInstance) {
    authServiceInstance.destroy();
    authServiceInstance = null;
  }
};

// Export the class and types
export {
  BackendAlignedAuthService
};

// Default export
export default getAuthService;
