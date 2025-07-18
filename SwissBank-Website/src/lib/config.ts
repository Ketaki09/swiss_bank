// File: Swissbank_website/src/lib/config.ts

interface AppConfig {
  claudeApiKey: string;
  backendUrl: string;
}

export const config: AppConfig = {
  claudeApiKey: import.meta.env.VITE_CLAUDE_API_KEY || '',
  backendUrl: import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8001'
};

// Validation function to ensure required environment variables are set
export const validateConfig = (): void => {
  if (!config.claudeApiKey) {
    console.warn('VITE_CLAUDE_API_KEY is not set in environment variables');
  }
  
  if (!config.backendUrl) {
    console.warn('VITE_BACKEND_URL is not set, using default: http://127.0.0.1:8001');
  }
};

