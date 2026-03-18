import { Icon } from '@iconify/react';
import roundSend from '@iconify/icons-ic/round-send';
import { useRef, useState, useEffect, useCallback } from 'react';

import { useTheme } from '@mui/material/styles';
import { Box, Input, IconButton } from '@mui/material';

import { MultiFilePreview } from 'src/widgets/upload';
import ImageGallery, { useImageGallery } from 'src/widgets/overlay';

import './chat-message-input-rt.css';
import ChatMessageInputTalkBtn from './talk-button';
import ImageUploadButton from './image-upload-button';
import CameraPanelButton from './camera/camera-panel-button';
import ScreenRecordingButton from './screen-recording/screen-recording-button';
import { type RecordingOptionId } from './screen-recording/screen-recording-container';

// ----------------------------------------------------------------------

export type ChatView = 'camera' | 'audio' | 'video';

export type ChatViewButton = {
  value: ChatView;
  label: string;
  icon: string;
};

const joinClasses = (...classNames: Array<string | false | undefined>) =>
  classNames.filter(Boolean).join(' ');

type ChatMessageInputProps = {
  onSend: Function;
  onCaptureFrame: () => { dataUri: string } | undefined;
  setSpeakingMode?: React.Dispatch<React.SetStateAction<boolean>>;
  onToggleListening: (active?: boolean) => Promise<void>;
  onSendText: (text: string) => void;
  onSendImage: (files: File[]) => Promise<void>; // Back to simple signature
  resourceName?: string;
  isRecording?: boolean;
  isSessionActive?: boolean;
  isSpeaking?: boolean;
  onTruncateAudio: () => void;
  // onSelectView: (view: ActiveChatView) => void;
  onToggleScreenRecorder: () => void;
  onRequestScreenRecorder: (
    optionId: RecordingOptionId,
    stream: MediaStream | null,
    isPending: boolean
  ) => void;
  isScreenRecorderOpen: boolean;
  onToggleCameraPanel?: () => void;
  isCameraPanelOpen?: boolean;
  onSetAudioMuted?: (muted: boolean) => void;
  onStopAudioPlayback?: () => void;
};

export default function ChatMessageInput({
  onSend,
  onCaptureFrame,
  setSpeakingMode,
  onToggleListening,
  onSendText,
  onSendImage,
  resourceName,
  isRecording,
  isSessionActive = false,
  isSpeaking,
  onTruncateAudio,
  // onSelectView,
  onToggleScreenRecorder,
  onRequestScreenRecorder,
  isScreenRecorderOpen,
  onToggleCameraPanel,
  isCameraPanelOpen = false,
  onSetAudioMuted,
  onStopAudioPlayback,
}: ChatMessageInputProps) {
  const theme = useTheme();
  const photoInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // const [lines, setLines] = useState(0);
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState<any[]>([]);
  const [inputFocus, setInputFocus] = useState(false);
  const [isMuted, setIsMuted] = useState(false);

  const images = files.map((file) => ({ src: file.preview }));
  const hasFiles = files.length > 0;
  const hasMessage = message.trim().length > 0;
  const canSend = isSessionActive && (hasMessage || hasFiles);
  const isInputDisabled = !isSessionActive;
  const isCompactRecordingLayout = Boolean(isRecording && !inputFocus);
  const rootClassName = joinClasses(
    'rt-message-input',
    isRecording && 'is-recording',
    inputFocus && 'is-focused',
    hasFiles && 'has-files',
    isCompactRecordingLayout && 'is-compact'
  );

  const lightbox = useImageGallery(images);

  // Effect to maintain focus when switching between layouts
  useEffect(() => {
    if (inputFocus && inputRef.current) {
      // Use setTimeout to ensure the DOM has updated
      setTimeout(() => {
        inputRef.current?.focus();
      }, 0);
    }
  }, [inputFocus, isRecording]);

  // Reset mute state when recording stops
  useEffect(() => {
    if (!isRecording) {
      setIsMuted(false);
    }
  }, [isRecording]);

  useEffect(() => {
    if (!isSessionActive) {
      setFiles([]);
    }
  }, [isSessionActive]);

  // const handleToggleView = (view: ChatView) => {
  //   const nextView: ActiveChatView = activeView === view ? 'chat' : view;
  //   onSelectView(nextView);
  // };

  const handleInputFocus = (flag: boolean) => {
    setInputFocus(flag);
  };

  const handleToggleMute = () => {
    setIsMuted((prev) => {
      const newMutedState = !prev;
      // Call the actual mute function from the audio recorder
      if (onSetAudioMuted) {
        onSetAudioMuted(newMutedState);
      }
      return newMutedState;
    });
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      handleSend();
    }
  };

  const handleSend = async () => {
    if (!isSessionActive) {
      return;
    }

    // Send images if there are any
    if (files.length > 0) {
      try {
        // Send images to WebSocket/realtime API
        await onSendImage(files);

        // Create UI message using onSend with both text and images
        onSend({
          content: message.trim() || 'Sent image(s)',
          senderId: 'user',
          mode: 'new',
          attachments: files,
          sources: [],
          status: 'completed',
        });

        // If there's also text, send it separately to WebSocket
        if (message.trim().length > 0) {
          onSendText(message);
        }
      } catch (error) {
        console.error('Error sending images:', error);
      }
    } else if (message.trim().length > 0) {
      // Only send text message if no images and there is text
      onSendText(message);

      onSend({
        content: message,
        senderId: 'user',
        mode: 'new',
        attachments: [],
        sources: [],
        status: 'completed',
      });
    }

    // Clear the input and files
    setMessage('');
    setFiles([]);
    // setLines(1);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []).filter((file): file is File =>
      file.type.startsWith('image/')
    );

    const newFiles = selectedFiles.map((file: File) => {
      const enhancedFile = Object.assign(file, {
        preview: URL.createObjectURL(file),
        path: file.name, // Add path for fileFormat detection
      });

      return enhancedFile;
    });

    setFiles([...files, ...newFiles]);

    // Reset the input value so the same file can be selected again
    if (event.target) {
      event.target.value = '';
    }
  };

  const handleUploadImages = useCallback(async () => {
    if (!isSessionActive) {
      return;
    }

    photoInputRef.current?.click();
  }, [isSessionActive]);

  const handleToggleScreenRecorderClick = useCallback(() => {
    if (!isSessionActive) {
      return;
    }

    onToggleScreenRecorder();
  }, [isSessionActive, onToggleScreenRecorder]);

  const handleRequestScreenRecorderWithSession = useCallback(
    (optionId: RecordingOptionId, stream: MediaStream | null, isPending: boolean) => {
      if (!isSessionActive) {
        return;
      }

      onRequestScreenRecorder(optionId, stream, isPending);
    },
    [isSessionActive, onRequestScreenRecorder]
  );

  const handleToggleCameraPanelClick = useCallback(() => {
    if (!isSessionActive) {
      return;
    }

    onToggleCameraPanel?.();
  }, [isSessionActive, onToggleCameraPanel]);

  const handleRemoveFile = useCallback(
    (inputFile: File | string) => {
      const filtered = files.filter((file) => file !== inputFile);
      setFiles(filtered);
    },
    [files]
  );

  const handleOpenLightbox = (img: string) => {
    lightbox.onOpen(img);
  };

  // let borderRadius;
  // if (files.length > 0) {
  //   borderRadius = 1;
  // } else if (lines < 2) {
  //   borderRadius = 20;
  // } else if (lines === 2) {
  //   borderRadius = 2.5;
  // } else if (lines === 3) {
  //   borderRadius = 2;
  // } else {
  //   borderRadius = 1.5;
  // }

  return (
    <Box className={rootClassName} data-theme-mode={theme.palette.mode}>
      {/* File preview section - separate row above text input */}
      {hasFiles && (
        <Box className="rt-message-input__preview-row">
          <Box className="rt-message-input__preview-list">
            <MultiFilePreview
              thumbnail
              files={files}
              onRemove={(file) => handleRemoveFile(file)}
              onClick={handleOpenLightbox}
              imageView
            />
          </Box>
        </Box>
      )}

      {/* Reorganized layout when recording and not focused */}
      {isCompactRecordingLayout ? (
        <Box className="rt-message-input__compact-row">
          {/* Talk button */}
          <ChatMessageInputTalkBtn onToggleRT={onToggleListening} isActive={isRecording} />

          {/* Mute button - only show when recording */}
          <IconButton
            onClick={handleToggleMute}
            size="small"
            className={joinClasses(
              'rt-message-input__icon-button',
              'rt-message-input__mute-button',
              isMuted && 'is-muted'
            )}
          >
            <Icon icon="ph:microphone-slash-fill" width={18} height={18} />
          </IconButton>

          {/* Speaker mute button - stop model audio playback */}
          {onStopAudioPlayback && isSpeaking && (
            <IconButton
              onClick={onStopAudioPlayback}
              size="small"
              className="rt-message-input__icon-button rt-message-input__speaker-button"
            >
              <Icon icon="ph:speaker-slash-fill" width={18} height={18} />
            </IconButton>
          )}

          {/* Input */}
          <Input
            className="rt-message-input__input rt-message-input__input--compact llm-input"
            multiline
            maxRows={1}
            fullWidth
            disabled={isInputDisabled}
            value={message}
            disableUnderline
            onBlur={() => handleInputFocus(false)}
            onFocus={() => handleInputFocus(true)}
            onKeyDown={handleKeyDown}
            onChange={(e) => {
              setMessage(e.target.value);
            }}
            placeholder={
              isSessionActive
                ? 'Text, talk, share screen, or show me'
                : 'Start a talk session to send messages'
            }
          />

          {/* Send button - show when there are files */}
          {hasFiles && (
            <IconButton
              disabled={!canSend}
              onClick={handleSend}
              className={joinClasses(
                'rt-message-input__icon-button',
                'rt-message-input__send-button',
                canSend && 'is-active'
              )}
            >
              <Icon icon={roundSend} width={16} height={16} />
            </IconButton>
          )}

          {/* Three icon buttons */}
          <Box
            className={joinClasses(
              'rt-message-input__action-group',
              'rt-message-input__action-group--right',
              !isSessionActive && 'is-disabled'
            )}
          >
            <ImageUploadButton onSelectOption={handleUploadImages} />
            <ScreenRecordingButton
              onRequestScreenRecorder={handleRequestScreenRecorderWithSession}
              onToggleScreenRecorder={handleToggleScreenRecorderClick}
              isScreenRecorderOpen={isScreenRecorderOpen}
            />
            {onToggleCameraPanel && (
              <CameraPanelButton
                onToggleCameraPanel={handleToggleCameraPanelClick}
                isCameraPanelOpen={isCameraPanelOpen}
              />
            )}
          </Box>
        </Box>
      ) : (
        <>
          {/* First Row: Text Input */}
          <Box
            className={joinClasses('rt-message-input__input-row', isRecording && 'is-recording')}
          >
            <Input
              inputRef={inputRef}
              className="rt-message-input__input llm-input"
              multiline
              maxRows={isRecording ? 1 : 4}
              fullWidth
              disabled={isInputDisabled}
              value={message}
              disableUnderline
              onBlur={() => handleInputFocus(false)}
              onFocus={() => handleInputFocus(true)}
              onKeyDown={handleKeyDown}
              onChange={(e) => {
                setMessage(e.target.value);
              }}
              placeholder={
                isSessionActive
                  ? 'Text, talk, share screen, or show me'
                  : 'Start a talk session to send messages'
              }
            />

            <IconButton
              disabled={!canSend}
              onClick={handleSend}
              className={joinClasses(
                'rt-message-input__icon-button',
                'rt-message-input__send-button',
                canSend && 'is-active'
              )}
            >
              <Icon icon={roundSend} width={16} height={16} />
            </IconButton>
          </Box>

          {/* Second Row: Function Buttons */}
          <Box className="rt-message-input__actions-row">
            {/* Left Side: Phone Call Button */}
            <Box className="rt-message-input__action-group rt-message-input__action-group--left">
              <ChatMessageInputTalkBtn onToggleRT={onToggleListening} isActive={isRecording} />

              {/* Mute button - only show when recording */}
              {isRecording && (
                <IconButton
                  onClick={handleToggleMute}
                  size="small"
                  className={joinClasses(
                    'rt-message-input__icon-button',
                    'rt-message-input__mute-button',
                    isMuted && 'is-muted'
                  )}
                >
                  <Icon icon="ph:microphone-slash-fill" width={18} height={18} />
                </IconButton>
              )}

              {/* Speaker mute button - stop model audio playback */}
              {onStopAudioPlayback && isSpeaking && (
                <IconButton
                  onClick={onStopAudioPlayback}
                  size="small"
                  className="rt-message-input__icon-button rt-message-input__speaker-button"
                >
                  <Icon icon="ph:speaker-slash-fill" width={18} height={18} />
                </IconButton>
              )}
            </Box>

            {/* Right Side: Other Function Buttons */}
            <Box
              className={joinClasses(
                'rt-message-input__action-group',
                'rt-message-input__action-group--right',
                !isSessionActive && 'is-disabled'
              )}
            >
              {/* Upload button */}
              <ImageUploadButton onSelectOption={handleUploadImages} />

              <ScreenRecordingButton
                onRequestScreenRecorder={handleRequestScreenRecorderWithSession}
                onToggleScreenRecorder={handleToggleScreenRecorderClick}
                isScreenRecorderOpen={isScreenRecorderOpen}
              />

              {onToggleCameraPanel && (
                <CameraPanelButton
                  onToggleCameraPanel={handleToggleCameraPanelClick}
                  isCameraPanelOpen={isCameraPanelOpen}
                />
              )}
            </Box>
          </Box>
        </>
      )}

      {/* Hidden file input */}
      <input
        ref={photoInputRef}
        type="file"
        multiple
        accept="image/*"
        onChange={handleFileChange}
        hidden
      />

      <ImageGallery
        index={lightbox.selected}
        slides={images}
        open={lightbox.open}
        close={lightbox.onClose}
      />
    </Box>
  );
}
