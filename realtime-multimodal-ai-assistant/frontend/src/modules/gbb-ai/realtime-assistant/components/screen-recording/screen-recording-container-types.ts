// ----------------------------------------------------------------------

export type RecordingOptionId = 'entire-screen' | 'window' | 'tab';

export type RecordingOption = {
  id: RecordingOptionId;
  label: string;
  description: string;
  icon: string;
  constraints: DisplayMediaStreamOptions;
};

export type ScreenCaptureMeta = {
  optionId?: RecordingOptionId;
  label?: string;
};

export type ScreenCaptureHandler = (base64Image: string, meta?: ScreenCaptureMeta) => void;

export type ScreenRecordingContainerProps = {
  onStreamAvailable?: (stream: MediaStream, label: string) => void;
  onStreamEnded?: () => void;
  onClose?: () => void;
  initialOptionId?: RecordingOptionId | null;
  initialStream?: MediaStream | null;
  isStreamPending?: boolean;
  onInitialOptionHandled?: () => void;
  isFloating?: boolean;
  onToggleFloating?: () => void;
  onScreenCapture?: ScreenCaptureHandler;
  messageCount?: number;
};

export type FloatingSize = { width: number; height: number };
export type FloatingPosition = { top: number; left: number };

export type ResizeHandle = 'bottom-right' | 'bottom-left' | 'bottom';

// ----------------------------------------------------------------------

export const MIN_WIDTH = 320;
export const MIN_HEIGHT = 220;
export const DEFAULT_SIZE: FloatingSize = { width: 420, height: 320 };
export const DEFAULT_POSITION: FloatingPosition = { top: 96, left: 96 };

export const SCREEN_FLOATING_ROOT_ID = 'screen-recording-floating-root';
export const SCREEN_HEADER_TITLE = 'Screen Recording';

// ----------------------------------------------------------------------

type DisplaySurfaceConstraint = MediaTrackConstraints & {
  displaySurface?: 'monitor' | 'window' | 'application' | 'browser';
  monitorTypeSurfaces?: 'include' | 'exclude';
  surfaceSwitching?: 'include' | 'exclude';
};

const constraintWithSurface = (
  surface: 'monitor' | 'window' | 'browser'
): DisplayMediaStreamOptions => ({
  audio: true,
  video: {
    cursor: 'always',
    displaySurface: surface,
    surfaceSwitching: 'include',
    ...(surface === 'monitor' ? { monitorTypeSurfaces: 'include' } : {}),
  } as DisplaySurfaceConstraint,
});

export const createRecordingOptions = (): RecordingOption[] => [
  {
    id: 'entire-screen',
    label: 'Entire Screen',
    description: 'Record any monitor that is currently connected.',
    icon: 'mdi:monitor-share',
    constraints: constraintWithSurface('monitor'),
  },
  {
    id: 'window',
    label: 'Application Window',
    description: 'Pick a specific application window to share.',
    icon: 'mdi:application',
    constraints: constraintWithSurface('window'),
  },
  {
    id: 'tab',
    label: 'Browser Tab',
    description: 'Share a single browser tab with optional audio.',
    icon: 'mdi:tab',
    constraints: constraintWithSurface('browser'),
  },
];
