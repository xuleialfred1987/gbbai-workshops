import type { RefObject } from 'react';

import type { ICaption, Conversation } from 'src/types/chat';

// ----------------------------------------------------------------------

export type ResizeHandle = 'bottom-right' | 'bottom-left' | 'bottom';

export type VideoRef =
  | ((instance: HTMLVideoElement | null) => void)
  | RefObject<HTMLVideoElement | null>
  | null
  | undefined;

export type CameraContainerProps = {
  videoRef: VideoRef;
  onClose?: () => void;
  onStreamStarted?: (stream: MediaStream) => void;
  onStreamStopped?: () => void;
  isFloating?: boolean;
  onToggleFloating?: () => void;
  title?: React.ReactNode;
  // Additional props for full camera functionality
  conversation?: Conversation;
  isListening?: boolean;
  isSpeaking?: boolean;
  onSend?: Function;
  onCaptureFrame?: () => { dataUri: string } | undefined;
  processImageCaption?: (caption: ICaption) => void;
  resourceName?: string;
  returnToHomepage?: () => void;
  onCameraCapture?: (base64Image: string) => void;
  messageCount?: number;
};

export type PaperLikeBoxProps = {
  isFloating: boolean;
  size: { width: number; height: number };
  header: React.ReactNode;
  content: React.ReactNode;
  handles: boolean;
  sx?: Record<string, unknown>;
  onPointerDown?: (event: React.PointerEvent<HTMLDivElement>) => void;
};

export type FloatingHeaderProps = {
  title?: React.ReactNode;
  isFloating: boolean;
  isDragging?: boolean;
  onPointerDown?: (event: React.PointerEvent<HTMLDivElement>) => void;
  onToggleFloating?: () => void;
  onClose?: () => void;
  onDoubleClick?: () => void;
  onFlipCamera?: () => void;
};

// ----------------------------------------------------------------------

export const MIN_WIDTH = 320;
export const MIN_HEIGHT = 220;
export const DEFAULT_SIZE = { width: 420, height: 320 };
export const DEFAULT_POSITION = { top: 120, left: 96 };
