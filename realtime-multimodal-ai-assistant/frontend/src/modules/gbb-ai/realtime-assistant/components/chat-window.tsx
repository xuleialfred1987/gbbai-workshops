import { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';

import uuidv4 from 'src/utils/uuidv4';

import {
  Message,
  ICaption,
  SendMessage,
  Conversation,
  IRtConfiguration,
  SendTextFuncProps,
} from 'src/types/chat';

import ConversationContainer from './conversation-container';
import FloatingCameraContainer from './camera/camera-container';
import ScreenRecordingContainer, { type RecordingOptionId } from './screen-recording/screen-recording-container';

// ----------------------------------------------------------------------

const USER = { id: 'user', name: 'Admin' };
const ASSISTANTS = [{ id: 'assistant', name: 'GPT' }];

// ----------------------------------------------------------------------

type Props = {
  messages: Message[];
  onUpdateMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  configurations: IRtConfiguration;
  chatMode: string;
  isRecording: boolean;
  isSessionActive: boolean;
  onToggleListening: (active?: boolean) => Promise<void>;
  onSendText: (text: string) => void;
  onSendImage: (files: File[]) => Promise<void>;
  onSendCameraImage?: (base64Image: string) => void;
  isSpeaking?: boolean;
  onTruncateAudio: () => void;
  processImageCaption: (caption: ICaption) => void;
  onSetAudioMuted?: (muted: boolean) => void;
  onStopAudioPlayback?: () => void;
};

export default function ChatWindow({
  configurations,
  messages,
  onUpdateMessages,
  chatMode,
  isRecording,
  isSessionActive,
  onToggleListening,
  onSendText,
  onSendImage,
  onSendCameraImage,
  isSpeaking,
  onTruncateAudio,
  processImageCaption,
  onSetAudioMuted,
  onStopAudioPlayback,
}: Props) {
  const playerRef = useRef<HTMLVideoElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const [openCopilot, setOpenCopilot] = useState(false);
  const [videoMode, setVideoMode] = useState('stream');
  const [isScreenRecorderOpen, setScreenRecorderOpen] = useState(false);
  const [selectedRecordingOption, setSelectedRecordingOption] = useState<RecordingOptionId | null>(
    null
  );
  const [selectedRecordingStream, setSelectedRecordingStream] = useState<MediaStream | null>(null);
  const [isRecordingStreamPending, setRecordingStreamPending] = useState(false);
  const [isScreenRecorderFloating, setScreenRecorderFloating] = useState(false);
  const [isCameraPanelOpen, setCameraPanelOpen] = useState(false);
  const [isCameraPanelFloating, setCameraPanelFloating] = useState(false);
  const [leftPaneWidth, setLeftPaneWidth] = useState(40); // percentage when docked
  const [isDraggingDivider, setIsDraggingDivider] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const rafIdRef = useRef<number | null>(null);

  const handleSwitchCopilot = useCallback(() => {
    setOpenCopilot((prev) => !prev);
  }, []);

  const handleSetVideoMode = useCallback((mode: string) => {
    setVideoMode(mode);
  }, []);

  const handleRequestScreenRecorder = useCallback(
    (optionId: RecordingOptionId, stream: MediaStream | null, isPending: boolean) => {
      setSelectedRecordingOption(optionId);
      setSelectedRecordingStream(stream ?? null);
      setRecordingStreamPending(isPending);

      if (!isPending && !stream) {
        setSelectedRecordingOption(null);
      }
    },
    []
  );

  const handleToggleScreenRecorder = useCallback(() => {
    setScreenRecorderOpen((prev) => {
      const next = !prev;

      if (!next) {
        setScreenRecorderFloating(false);
        setSelectedRecordingOption(null);
        setRecordingStreamPending(false);
        setSelectedRecordingStream((prevStream) => {
          if (prevStream) {
            prevStream.getTracks().forEach((track) => track.stop());
          }
          return null;
        });
      }

      return next;
    });
  }, []);

  const handleCloseScreenRecorder = useCallback(() => {
    setScreenRecorderOpen(false);
    setScreenRecorderFloating(false);
    setSelectedRecordingOption(null);
    setRecordingStreamPending(false);
    setSelectedRecordingStream((prevStream) => {
      if (prevStream) {
        prevStream.getTracks().forEach((track) => track.stop());
      }
      return null;
    });
  }, []);

  const closeAllRealtimePanels = useCallback(() => {
    setScreenRecorderOpen(false);
    setScreenRecorderFloating(false);
    setSelectedRecordingOption(null);
    setRecordingStreamPending(false);
    setSelectedRecordingStream((prevStream) => {
      if (prevStream) {
        prevStream.getTracks().forEach((track) => track.stop());
      }
      return null;
    });
    setCameraPanelOpen(false);
    setCameraPanelFloating(false);
    setVideoMode('none');
  }, []);

  const onSendMessage = useCallback(
    (conversation: SendMessage) => {
      const {
        messageId,
        message,
        contentType,
        sources,
        function_calls,
        createdAt,
        senderId,
        mode,
        ddb_uuid,
        log_timestamp,
        attachments,
      } = conversation;

      const newMessage = {
        id: messageId,
        body: message,
        contentType,
        sources,
        function_calls,
        createdAt,
        senderId,
        mode,
        chatMode,
        ddb_uuid,
        log_timestamp,
        attachments,
      } as Message;

      onUpdateMessages((prev) => {
        const existedMsgIndex = prev.findIndex((msg) => msg.id === messageId);
        if (!message) return [...prev];
        if (existedMsgIndex === -1) {
          return [...prev, newMessage];
        }
        const newArr = [...prev];
        newArr[existedMsgIndex] = newMessage;
        return newArr;
      });
    },
    [onUpdateMessages, chatMode]
  );

  const conversation = {
    id: '1',
    participants: [USER, ASSISTANTS[0]],
    type: 'ONE_TO_ONE',
    unreadCount: 0,
    messages,
  } as Conversation;

  const handleSendText = useCallback(
    ({
      content,
      senderId,
      mode,
      sources = [],
      function_calls = [],
      msgId = undefined,
      uuid = undefined,
      timestamp = undefined,
      attachments = [],
    }: SendTextFuncProps) => {
      onSendMessage({
        conversationId: 'realtime-assistant',
        messageId: msgId === undefined ? uuidv4() : msgId,
        message: content,
        contentType: 'text',
        sources,
        function_calls,
        createdAt: new Date(),
        senderId,
        mode,
        chatMode,
        ddb_uuid: uuid,
        log_timestamp: timestamp,
        attachments,
      });
    },
    [onSendMessage, chatMode]
  );

  const captureVideoFrame = useCallback((video: any) => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d')?.drawImage(video, 0, 0);
    const dataUri = canvas.toDataURL('image/jpeg');
    return { dataUri };
  }, []);

  const captureStream = useCallback(() => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        const dataUri = canvas.toDataURL('image/jpeg');
        return { dataUri };
      }
      return undefined;
    }
    return undefined;
  }, [videoRef]);

  const captureFrame = useCallback(() => {
    if (videoMode === 'none') {
      return undefined;
    }
    if (videoMode === 'stream') {
      return captureStream();
    }
    if (playerRef.current) {
      const frame = captureVideoFrame((playerRef.current as any).getInternalPlayer());
      return frame;
    }
    return undefined;
  }, [videoMode, captureStream, captureVideoFrame, playerRef]);

  const handleInitialOptionHandled = useCallback(() => {
    setSelectedRecordingOption(null);
    setSelectedRecordingStream(null);
    setRecordingStreamPending(false);
  }, []);

  const handleRecordingStreamEnded = useCallback(() => {
    setSelectedRecordingStream(null);
    setSelectedRecordingOption(null);
    setScreenRecorderOpen(false);
    setRecordingStreamPending(false);
    setScreenRecorderFloating(false);
  }, []);

  const handleToggleScreenRecorderFloating = useCallback(() => {
    setScreenRecorderFloating((prev) => !prev);
  }, []);

  const handleToggleCameraPanel = useCallback(() => {
    setCameraPanelOpen((prev) => {
      const next = !prev;
      if (next) {
        setVideoMode('stream');
      } else {
        setCameraPanelFloating(false);
        if (!selectedRecordingStream) {
          setVideoMode((current) => (current === 'stream' ? 'none' : current));
        }
      }
      return next;
    });
  }, [selectedRecordingStream]);

  const handleCloseCameraPanel = useCallback(() => {
    setCameraPanelOpen(false);
    setCameraPanelFloating(false);
    if (!selectedRecordingStream) {
      setVideoMode((current) => (current === 'stream' ? 'none' : current));
    }
  }, [selectedRecordingStream]);

  const handleToggleCameraPanelFloating = useCallback(() => {
    setCameraPanelFloating((prev) => !prev);
  }, []);

  const isScreenDocked = isScreenRecorderOpen && !isScreenRecorderFloating;
  const isCameraDocked = isCameraPanelOpen && !isCameraPanelFloating;
  const hasDockedPanel = isScreenDocked || isCameraDocked;

  const handleDividerMouseDown = useCallback(
    (event: React.MouseEvent) => {
      if (!hasDockedPanel) {
        return;
      }
      event.preventDefault();
      setIsDraggingDivider(true);
    },
    [hasDockedPanel]
  );

  const handleDividerMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!isDraggingDivider || !containerRef.current || !hasDockedPanel) {
        return;
      }

      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }

      rafIdRef.current = requestAnimationFrame(() => {
        if (!containerRef.current) {
          return;
        }

        const containerRect = containerRef.current.getBoundingClientRect();
        const containerWidth = containerRect.width;
        if (containerWidth <= 0) {
          rafIdRef.current = null;
          return;
        }

        const minLeftPx = 260;
        const minRightPx = 320;
        const mouseX = event.clientX - containerRect.left;

        const maxLeftPx = Math.max(containerWidth - minRightPx, minLeftPx);
        const clampedPx = Math.min(Math.max(mouseX, minLeftPx), maxLeftPx);
        const newLeftPercent = (clampedPx / containerWidth) * 100;

        const minLeftPercent = Math.min((minLeftPx / containerWidth) * 100, 100);
        const maxLeftPercent = Math.min(
          Math.max(100 - (minRightPx / containerWidth) * 100, minLeftPercent),
          100
        );
        const boundedPercent = Math.min(Math.max(newLeftPercent, minLeftPercent), maxLeftPercent);

        setLeftPaneWidth(boundedPercent);
        rafIdRef.current = null;
      });
    },
    [hasDockedPanel, isDraggingDivider]
  );

  const handleDividerMouseUp = useCallback(() => {
    setIsDraggingDivider(false);
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
  }, []);

  const handleDividerDoubleClick = useCallback(() => {
    setLeftPaneWidth(30);
  }, []);

  useEffect(() => {
    if (!isDraggingDivider) {
      return undefined;
    }

    const handleMove = (event: MouseEvent) => {
      handleDividerMouseMove(event);
    };

    const handleUp = () => {
      handleDividerMouseUp();
    };

    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [handleDividerMouseMove, handleDividerMouseUp, isDraggingDivider]);

  useEffect(() => {
    if (!hasDockedPanel && isDraggingDivider) {
      handleDividerMouseUp();
    }
  }, [handleDividerMouseUp, hasDockedPanel, isDraggingDivider]);

  useEffect(() => {
    if (!isSessionActive) {
      closeAllRealtimePanels();
    }
  }, [closeAllRealtimePanels, isSessionActive]);

  return (
    <Stack
      ref={containerRef}
      direction={{ xs: 'column', lg: 'row' }}
      spacing={{ xs: 2, md: 3, lg: 0 }}
      sx={{
        mx: 'auto',
        width: 1,
        height: 1,
        overflow: 'visible',
        alignItems: 'stretch',
        position: 'relative',
        maxWidth: hasDockedPanel ? '100%' : 840,
      }}
    >
      <Box
        sx={{
          flexGrow: 1,
          position: 'relative',
          minHeight: { xs: 560, md: 640 },
          flexBasis: {
            lg: hasDockedPanel ? `${leftPaneWidth}%` : '100%',
          },
          maxWidth: {
            lg: hasDockedPanel ? `${leftPaneWidth}%` : '100%',
          },
          width: { xs: '100%', lg: hasDockedPanel ? `${leftPaneWidth}%` : '100%' },
          transition: isDraggingDivider
            ? 'none'
            : 'flex-basis 0.3s ease, max-width 0.3s ease, width 0.3s ease',
        }}
      >
        <ConversationContainer
          open={openCopilot}
          conversation={conversation}
          onSwitchOpenCopilot={handleSwitchCopilot}
          isListening={isRecording}
          isSessionActive={isSessionActive}
    isSpeaking={isSpeaking}
          // Recorded video stream
          playerRef={playerRef}
          onCaptureFrame={captureFrame}
          // Live video stream
          videoRef={videoRef}
          onSetVideoMode={handleSetVideoMode}
          // Input box
          onSend={handleSendText}
          onToggleListening={onToggleListening}
          onSendText={onSendText}
          onSendImage={onSendImage}
          onSendCameraImage={onSendCameraImage}
          resourceName={configurations[`rt-Deployment`]}
          onTruncateAudio={onTruncateAudio}
          processImageCaption={processImageCaption}
          onToggleScreenRecorder={handleToggleScreenRecorder}
          onRequestScreenRecorder={handleRequestScreenRecorder}
          isScreenRecorderOpen={isScreenRecorderOpen}
          onToggleCameraPanel={handleToggleCameraPanel}
          isCameraPanelOpen={isCameraPanelOpen}
          onSetAudioMuted={onSetAudioMuted}
          onStopAudioPlayback={onStopAudioPlayback}
        />
      </Box>

      {hasDockedPanel && (
        <Box
          onMouseDown={handleDividerMouseDown}
          onDoubleClick={handleDividerDoubleClick}
          sx={{
            display: { xs: 'none', lg: 'flex' },
            alignSelf: 'stretch',
            width: 8,
            cursor: 'col-resize',
            position: 'relative',
            mx: { lg: 0 },
            transition: 'background-color 0.2s ease',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: '50%',
              bottom: 'auto',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: 2,
              height: '33%',
              borderRadius: 1,
              backgroundColor: 'primary.main',
              opacity: isDraggingDivider ? 1 : 0,
              transition: 'opacity 0.2s ease',
            },
            '&:hover::before': {
              backgroundColor: 'primary.main',
              opacity: 1,
            },
          }}
        />
      )}

      <Box
        sx={{
          display: hasDockedPanel ? 'flex' : 'none',
          flexDirection: 'column',
          flexGrow: 1,
          flexBasis: {
            lg: hasDockedPanel ? `${100 - leftPaneWidth}%` : '0%',
          },
          maxWidth: {
            lg: hasDockedPanel ? `${100 - leftPaneWidth}%` : '0%',
          },
          width: {
            xs: '100%',
            lg: hasDockedPanel ? `${100 - leftPaneWidth}%` : '0%',
          },
          minWidth: { lg: hasDockedPanel ? 320 : 0 },
          minHeight: { xs: 560, md: 640 },
          transition: isDraggingDivider
            ? 'none'
            : 'flex-basis 0.3s ease, max-width 0.3s ease, width 0.3s ease',
        }}
      >
        <Stack spacing={2} sx={{ flexGrow: 1, height: '100%' }}>
          {isScreenRecorderOpen && (
            <Box
              sx={{
                flexGrow: 1,
                minHeight: 320,
                height: '100%',
                display: isScreenDocked ? 'flex' : 'none',
              }}
            >
              <ScreenRecordingContainer
                onClose={handleCloseScreenRecorder}
                initialOptionId={selectedRecordingOption}
                initialStream={selectedRecordingStream}
                isStreamPending={isRecordingStreamPending}
                onInitialOptionHandled={handleInitialOptionHandled}
                onStreamEnded={handleRecordingStreamEnded}
                isFloating={isScreenRecorderFloating}
                onToggleFloating={handleToggleScreenRecorderFloating}
                onScreenCapture={onSendCameraImage}
                messageCount={messages.length}
              />
            </Box>
          )}

          {isCameraPanelOpen && (
            <Box
              sx={{
                p: 0.5,
                flexGrow: 1,
                minHeight: 320,
                height: '100%',
                display: isCameraDocked ? 'flex' : 'none',
              }}
            >
              <FloatingCameraContainer
                videoRef={videoRef}
                onClose={handleCloseCameraPanel}
                isFloating={isCameraPanelFloating}
                onToggleFloating={handleToggleCameraPanelFloating}
                onCameraCapture={onSendCameraImage}
                title="Camera"
                messageCount={messages.length}
              />
            </Box>
          )}
        </Stack>
      </Box>
    </Stack>
  );
}
