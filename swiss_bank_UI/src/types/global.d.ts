// types/global.d.ts
// Add these type declarations to your project

// Google Drive API Types
interface GoogleDriveFile {
  id: string;
  name: string;
  mimeType: string;
  url: string;
  iconUrl?: string;
  embedUrl?: string;
}

interface GooglePickerData {
  action: string;
  docs: GoogleDriveFile[];
}

interface GooglePickerView {
  setMimeTypes: (types: string) => GooglePickerView;
  setSelectFolderEnabled: (enabled: boolean) => GooglePickerView;
}

interface GooglePickerBuilder {
  addView: (view: GooglePickerView) => GooglePickerBuilder;
  setOAuthToken: (token: string) => GooglePickerBuilder;
  setCallback: (callback: (data: GooglePickerData) => void) => GooglePickerBuilder;
  setOrigin: (origin: string) => GooglePickerBuilder;
  build: () => GooglePicker;
}

interface GooglePicker {
  setVisible: (visible: boolean) => void;
}

// OneDrive API Types
interface OneDriveFile {
  id: string;
  name: string;
  size: number;
  webUrl: string;
  downloadUrl?: string;
  thumbnails?: Array<{
    medium?: {
      url: string;
      width: number;
      height: number;
    };
  }>;
}

interface OneDriveResponse {
  value: OneDriveFile[];
}

interface OneDriveOptions {
  clientId: string;
  action: 'query' | 'share' | 'download';
  multiSelect: boolean;
  openInNewWindow: boolean;
  success: (files: OneDriveResponse) => void;
  error: (error: OneDriveError) => void;
}

interface OneDriveError {
  code: string;
  message: string;
}

declare global {
  interface Window {
    gapi: {
      load: (apis: string, callback: () => void) => void;
      client: {
        init: (config: {
          apiKey: string;
          clientId: string;
          discoveryDocs: string[];
          scope: string;
        }) => Promise<void>;
      };
      auth2: {
        getAuthInstance: () => {
          isSignedIn: {
            get: () => boolean;
          };
          signIn: () => Promise<void>;
          currentUser: {
            get: () => {
              getAuthResponse: () => {
                access_token: string;
              };
            };
          };
        };
      };
    };
    
    google: {
      picker: {
        PickerBuilder: new () => GooglePickerBuilder;
        ViewId: {
          DOCS: GooglePickerView;
          SPREADSHEETS: GooglePickerView;
          PRESENTATIONS: GooglePickerView;
          FORMS: GooglePickerView;
          IMAGES: GooglePickerView;
          VIDEOS: GooglePickerView;
          FOLDERS: GooglePickerView;
        };
        Action: {
          PICKED: 'picked';
          CANCEL: 'cancel';
        };
      };
    };

    OneDrive: {
      open: (options: OneDriveOptions) => void;
    };

    webkitSpeechRecognition: new () => SpeechRecognition;
    SpeechRecognition: new () => SpeechRecognition;
  }

  interface SpeechRecognition extends EventTarget {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    start(): void;
    stop(): void;
    onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => void) | null;
    onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => void) | null;
    onend: ((this: SpeechRecognition, ev: Event) => void) | null;
  }

  interface SpeechRecognitionEvent extends Event {
    results: SpeechRecognitionResultList;
  }

  interface SpeechRecognitionErrorEvent extends Event {
    error: string;
  }

  interface SpeechRecognitionResultList {
    length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
  }

  interface SpeechRecognitionResult {
    length: number;
    item(index: number): SpeechRecognitionAlternative;
    [index: number]: SpeechRecognitionAlternative;
    isFinal: boolean;
  }

  interface SpeechRecognitionAlternative {
    transcript: string;
    confidence: number;
  }
}

export {};