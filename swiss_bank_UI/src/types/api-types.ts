// src/types/api-types.ts
// Exportable types for Google Drive and OneDrive APIs

// Google Drive API Types
export interface GoogleDriveFile {
  id: string;
  name: string;
  mimeType: string;
  url: string;
  iconUrl?: string;
  embedUrl?: string;
}

export interface GooglePickerData {
  action: string;
  docs: GoogleDriveFile[];
}

export interface GooglePickerView {
  setMimeTypes: (types: string) => GooglePickerView;
  setSelectFolderEnabled: (enabled: boolean) => GooglePickerView;
}

export interface GooglePickerBuilder {
  addView: (view: GooglePickerView) => GooglePickerBuilder;
  setOAuthToken: (token: string) => GooglePickerBuilder;
  setCallback: (callback: (data: GooglePickerData) => void) => GooglePickerBuilder;
  setOrigin: (origin: string) => GooglePickerBuilder;
  build: () => GooglePicker;
}

export interface GooglePicker {
  setVisible: (visible: boolean) => void;
}

// OneDrive API Types
export interface OneDriveFile {
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

export interface OneDriveResponse {
  value: OneDriveFile[];
}

export interface OneDriveOptions {
  clientId: string;
  action: 'query' | 'share' | 'download';
  multiSelect: boolean;
  openInNewWindow: boolean;
  success: (files: OneDriveResponse) => void;
  error: (error: OneDriveError) => void;
}

export interface OneDriveError {
  code: string;
  message: string;
}
