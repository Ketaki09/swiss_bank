// File: Swissbank_website/src/services/VoiceService.ts

// Embedded type declarations for Web Speech API
declare global {
  interface SpeechRecognitionEvent extends Event {
    readonly resultIndex: number;
    readonly results: SpeechRecognitionResultList;
  }

  interface SpeechRecognitionErrorEvent extends Event {
    readonly error: 
      | 'no-speech'
      | 'aborted'
      | 'audio-capture'
      | 'network'
      | 'not-allowed'
      | 'service-not-allowed'
      | 'bad-grammar'
      | 'language-not-supported';
    readonly message?: string;
  }

  interface SpeechRecognitionResultList {
    readonly length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
  }

  interface SpeechRecognitionResult {
    readonly length: number;
    readonly isFinal: boolean;
    item(index: number): SpeechRecognitionAlternative;
    [index: number]: SpeechRecognitionAlternative;
  }

  interface SpeechRecognitionAlternative {
    readonly transcript: string;
    readonly confidence: number;
  }

  interface SpeechRecognition extends EventTarget {
    continuous: boolean;
    grammars: SpeechGrammarList;
    interimResults: boolean;
    lang: string;
    maxAlternatives: number;
    serviceURI: string;
    abort(): void;
    start(): void;
    stop(): void;
    onaudioend: ((this: SpeechRecognition, ev: Event) => void) | null;
    onaudiostart: ((this: SpeechRecognition, ev: Event) => void) | null;
    onend: ((this: SpeechRecognition, ev: Event) => void) | null;
    onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => void) | null;
    onnomatch: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => void) | null;
    onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => void) | null;
    onsoundend: ((this: SpeechRecognition, ev: Event) => void) | null;
    onsoundstart: ((this: SpeechRecognition, ev: Event) => void) | null;
    onspeechend: ((this: SpeechRecognition, ev: Event) => void) | null;
    onspeechstart: ((this: SpeechRecognition, ev: Event) => void) | null;
    onstart: ((this: SpeechRecognition, ev: Event) => void) | null;
  }

  interface SpeechGrammarList {
    readonly length: number;
    addFromString(string: string, weight?: number): void;
    addFromURI(src: string, weight?: number): void;
    item(index: number): SpeechGrammar;
    [index: number]: SpeechGrammar;
  }

  interface SpeechGrammar {
    src: string;
    weight: number;
  }

  interface Window {
    SpeechRecognition?: {
      new(): SpeechRecognition;
    };
    webkitSpeechRecognition?: {
      new(): SpeechRecognition;
    };
    SpeechGrammarList?: {
      new(): SpeechGrammarList;
    };
    webkitSpeechGrammarList?: {
      new(): SpeechGrammarList;
    };
  }
}

// Voice Recognition Service - No LLM needed!
export class VoiceService {
  private recognition: SpeechRecognition | null = null;
  private isListening = false;
  private customerData: { name?: string; email?: string } | null = null;

  constructor() {
    this.initializeRecognition();
  }

  setCustomerData(customerData: { name?: string; email?: string } | null): void {
    this.customerData = customerData;
  }

  private initializeRecognition(): void {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = false;
      this.recognition.interimResults = false;
      this.recognition.lang = 'en-US';
    }
  }

  async startListening(): Promise<string> {
    return new Promise((resolve, reject) => {
      if (!this.recognition) {
        reject(new Error('Speech recognition not supported in this browser'));
        return;
      }

      this.recognition.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = event.results[0][0].transcript;
        resolve(transcript);
      };

      this.recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        reject(new Error(`Speech recognition error: ${event.error}`));
      };

      this.recognition.onend = () => {
        this.isListening = false;
      };

      this.isListening = true;
      this.recognition.start();
    });
  }

  stopListening(): void {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.isListening = false;
    }
  }

  getIsListening(): boolean {
    return this.isListening;
  }

  async processVoiceInput(transcript: string): Promise<string> {
    // Simple, fast local processing - no API calls needed!
    return this.getResponseForInput(transcript);
  }

  private getResponseForInput(transcript: string): string {
    const input = transcript.toLowerCase();
    const customerName = this.customerData?.name || 'valued customer';
    
    // Banking-specific responses with customer name
    const responses = {
      greeting: `Hello ${customerName}! I'm Eva, your Swiss Bank AI assistant. How can I help you today?`,
      services: `${customerName}, Swiss Bank offers Private Banking, Corporate Banking, Asset Management, Trading Services, and Wealth Management. Which service interests you?`,
      account: `${customerName}, I can help you with account inquiries. For security, I'll need to verify some details before accessing your account information.`,
      balance: `${customerName}, I can help you check your account balance. Please note that for security reasons, I'll need additional verification.`,
      transfer: `${customerName}, I can assist you with transfers. What type of transfer would you like to make?`,
      complaint: `${customerName}, I understand you'd like to file a complaint. Please describe your concern, and I'll ensure it's properly documented.`,
      help: `${customerName}, I'm here to help with all your banking needs. You can ask me about accounts, transfers, services, or file complaints.`,
      default: `Thank you for your message, ${customerName}. As your Swiss Bank AI assistant, I'm here to provide personalized banking assistance. How may I help you?`
    };

    // Simple keyword matching
    if (input.includes('hello') || input.includes('hi') || input.includes('hey') || input.includes('greet')) {
      return responses.greeting;
    } else if (input.includes('service') || input.includes('offer') || input.includes('product')) {
      return responses.services;
    } else if (input.includes('account') && !input.includes('balance')) {
      return responses.account;
    } else if (input.includes('balance') || input.includes('money')) {
      return responses.balance;
    } else if (input.includes('transfer') || input.includes('send') || input.includes('payment')) {
      return responses.transfer;
    } else if (input.includes('complaint') || input.includes('problem') || input.includes('issue') || input.includes('wrong')) {
      return responses.complaint;
    } else if (input.includes('help') || input.includes('assist') || input.includes('support')) {
      return responses.help;
    } else {
      return responses.default;
    }
  }

  async textToSpeech(text: string): Promise<void> {
    if ('speechSynthesis' in window) {
      // Cancel any ongoing speech
      speechSynthesis.cancel();
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      
      // Try to use a pleasant voice
      const voices = speechSynthesis.getVoices();
      const preferredVoice = voices.find(voice => 
        voice.name.includes('Google') || 
        voice.name.includes('Microsoft') ||
        (voice.lang.includes('en-US') && voice.name.includes('Female'))
      );
      
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      speechSynthesis.speak(utterance);
    }
  }

  isSupported(): boolean {
    return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  }
}
