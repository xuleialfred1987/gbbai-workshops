import { useRef, useMemo, useState, useEffect, useCallback, type ReactNode } from 'react';

// mui
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import ButtonBase from '@mui/material/ButtonBase';
import { alpha, useTheme } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';

// project imports
import useMessagesScroll from 'src/hooks/messages-scroll';

import Iconify from 'src/widgets/iconify';
import Scroller from 'src/widgets/scroller';

import { ICaption, Conversation, type Message } from 'src/types/chat';

import ChatMessageItem from './chat-message-item';
import ChatMessageInput from './chat-message-input-rt';
import { type RecordingOptionId } from './screen-recording/screen-recording-container';

// ----------------------------------------------------------------------

type Props = {
  open: boolean;
  conversation: Conversation;
  playerRef: any;
  // Video stream
  videoRef: any;
  onSwitchOpenCopilot: () => void;
  isSpeaking?: boolean;
  isListening?: boolean;
  isSessionActive: boolean;
  // Input box
  onSend: Function;
  onCaptureFrame: () => { dataUri: string } | undefined;
  setSpeakingMode?: React.Dispatch<React.SetStateAction<boolean>>;
  onToggleListening: (active?: boolean) => Promise<void>;
  onSendText: (text: string) => void;
  onSendImage: (files: File[]) => Promise<void>;
  onSendCameraImage?: (base64Image: string) => void; // Add camera image callback
  onSetVideoMode: (mode: string) => void;
  resourceName?: string;
  onTruncateAudio: () => void;
  processImageCaption: (caption: ICaption) => void;
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

type StarterOption = {
  id: string;
  icon: string;
  description: string;
  prompt: string;
};

export default function ConversationContainer({
  open,
  conversation,
  playerRef,
  onSwitchOpenCopilot,
  isSpeaking,
  isListening,
  isSessionActive,
  // Video stream
  videoRef,
  // Input box
  onSend,
  onCaptureFrame,
  setSpeakingMode,
  onToggleListening,
  onSendText,
  onSendImage,
  onSendCameraImage,
  onSetVideoMode,
  resourceName,
  onTruncateAudio,
  processImageCaption,
  onToggleScreenRecorder,
  onRequestScreenRecorder,
  isScreenRecorderOpen,
  onToggleCameraPanel,
  isCameraPanelOpen = false,
  onSetAudioMuted,
  onStopAudioPlayback,
}: Props) {
  const theme = useTheme();
  const isSmScreen = useMediaQuery(theme.breakpoints.down(1200));

  const filteredMessages = useMemo(
    () => conversation.messages.filter((message) => message.senderId !== 'image-captioning'),
    [conversation.messages]
  );

  const hasMessages = filteredMessages.length > 0;

  const starterOptions = useMemo<StarterOption[]>(
    () => [
      {
        id: 'voice-chat',
        icon: 'mingcute:mic-ai-fill',
        description: 'Click the mic button below to start talking with the assistant.',
        prompt: 'I want to talk with you using voice.',
      },
      {
        id: 'camera-insight',
        icon: 'solar:videocamera-record-bold-duotone',
        description: 'Activate camera and inquire about object identification.',
        prompt: 'Please activate the camera and help me identify the object in front of it.',
      },
      {
        id: 'screen-recording',
        icon: 'fluent:share-screen-start-16-filled',
        description: 'Start screen recording to capture and analyze screen content.',
        prompt: "Please start screen recording to help me analyze what's happening on my screen.",
      },
    ],
    []
  );

  const handleStarterSelect = useCallback(
    async (option: StarterOption) => {
      const ensureListening = async () => {
        if (isListening) {
          return true;
        }

        try {
          await onToggleListening(true);
          return true;
        } catch (error) {
          // eslint-disable-next-line no-console
          console.error('Failed to activate realtime listening from starter option', error);
          return false;
        }
      };

      if (option.id === 'voice-chat') {
        await ensureListening();
        return;
      }

      if (option.id === 'camera-insight') {
        if (onSetVideoMode) {
          onSetVideoMode('stream');
        }

        if (!isCameraPanelOpen) {
          onToggleCameraPanel?.();
        }

        await ensureListening();
        return;
      }

      if (option.id === 'screen-recording') {
        const listeningReady = await ensureListening();

        if (!listeningReady) {
          return;
        }

        if (!navigator.mediaDevices?.getDisplayMedia) {
          alert('Screen recording is not supported in this browser.');
          return;
        }

        const openedBySelection = !isScreenRecorderOpen;

        if (openedBySelection) {
          onToggleScreenRecorder();
        }

        onRequestScreenRecorder('entire-screen', null, true);

        try {
          const stream = await navigator.mediaDevices.getDisplayMedia({
            audio: true,
            video: {
              cursor: 'always',
              displaySurface: 'monitor',
              monitorTypeSurfaces: 'include',
              surfaceSwitching: 'include',
            } as MediaTrackConstraints & {
              displaySurface?: 'monitor';
              monitorTypeSurfaces?: 'include';
              surfaceSwitching?: 'include';
            },
          });

          onRequestScreenRecorder('entire-screen', stream, false);
        } catch (error) {
          onRequestScreenRecorder('entire-screen', null, false);

          if (openedBySelection) {
            onToggleScreenRecorder();
          }
        }
      }
    },
    [
      isCameraPanelOpen,
      isListening,
      isScreenRecorderOpen,
      onRequestScreenRecorder,
      onSetVideoMode,
      onToggleCameraPanel,
      onToggleListening,
      onToggleScreenRecorder,
    ]
  );

  const renderChatInput = useCallback(
    () => (
      <ChatMessageInput
        onSend={onSend}
        onCaptureFrame={onCaptureFrame}
        onToggleListening={onToggleListening}
        onSendText={onSendText}
        onSendImage={onSendImage}
        resourceName={resourceName}
        isRecording={isListening}
        isSessionActive={isSessionActive}
        isSpeaking={isSpeaking}
        setSpeakingMode={setSpeakingMode}
        onTruncateAudio={onTruncateAudio}
        onToggleScreenRecorder={onToggleScreenRecorder}
        onRequestScreenRecorder={onRequestScreenRecorder}
        isScreenRecorderOpen={isScreenRecorderOpen}
        onToggleCameraPanel={onToggleCameraPanel}
        isCameraPanelOpen={isCameraPanelOpen}
        onSetAudioMuted={onSetAudioMuted}
        onStopAudioPlayback={onStopAudioPlayback}
      />
    ),
    [
      onSend,
      onCaptureFrame,
      onToggleListening,
      onSendText,
      onSendImage,
      resourceName,
      isListening,
      isSessionActive,
      isSpeaking,
      setSpeakingMode,
      onTruncateAudio,
      onToggleScreenRecorder,
      onRequestScreenRecorder,
      isScreenRecorderOpen,
      onToggleCameraPanel,
      isCameraPanelOpen,
      onSetAudioMuted,
      onStopAudioPlayback,
    ]
  );

  const shouldShowInput = hasMessages;
  const content: ReactNode = hasMessages ? (
    <ChatMessageList conversation={conversation} messages={filteredMessages} />
  ) : (
    <WelcomeContent
      renderChatInput={renderChatInput}
      starterOptions={starterOptions}
      onStarterSelect={handleStarterSelect}
      isRecording={isListening || false}
    />
  );

  return (
    <Box
      sx={{
        height: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        transition: 'transform 0.3s ease-in-out',
        transform: !isSmScreen && open ? 'translateX(-32px)' : 'none',
      }}
    >
      <Box
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minHeight: 0,
        }}
      >
        {content}
      </Box>

      {shouldShowInput && (
        <Box
          sx={{
            p: { xs: 2, md: 0.5 },
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <Box
            sx={{
              width: '100%',
              maxWidth: isListening ? { xs: 480, sm: 560 } : '100%',
              transition: 'max-width 0.4s ease-in-out',
            }}
          >
            {renderChatInput()}
          </Box>
        </Box>
      )}
    </Box>
  );
}

// ----------------------------------------------------------------------

type ChatMessageListProps = {
  conversation: Conversation;
  messages: Message[];
};

type RenderableMessage = {
  message: Message;
  extraFunctionCalls: Message['function_calls'];
};

function ChatMessageList({ conversation, messages }: ChatMessageListProps) {
  const { messagesEndRef } = useMessagesScroll(messages);

  const hasMessages = messages.length > 0;
  const renderableMessages = useMemo<RenderableMessage[]>(() => {
    const accumulator: RenderableMessage[] = [];
    let pendingFunctionMessages: Message[] = [];

    messages.forEach((message) => {
      const isStandaloneFunctionMessage =
        message.body?.startsWith('(SYS)function') && (message.function_calls?.length ?? 0) > 0;

      if (!isStandaloneFunctionMessage) {
        const canAttachPendingToCurrentAssistant =
          message.senderId === 'assistant' && !message.body?.startsWith('(SYS)');

        if (pendingFunctionMessages.length > 0 && !canAttachPendingToCurrentAssistant) {
          pendingFunctionMessages.forEach((pendingMessage) => {
            accumulator.push({
              message: pendingMessage,
              extraFunctionCalls: [],
            });
          });
          pendingFunctionMessages = [];
        }

        accumulator.push({
          message,
          extraFunctionCalls: canAttachPendingToCurrentAssistant
            ? pendingFunctionMessages.flatMap((pendingMessage) => pendingMessage.function_calls || [])
            : [],
        });

        if (canAttachPendingToCurrentAssistant) {
          pendingFunctionMessages = [];
        }

        return;
      }

      const previousRenderable = accumulator[accumulator.length - 1];
      const previousMessage = previousRenderable?.message;
      const canAttachToPreviousAssistant =
        previousMessage &&
        previousMessage.senderId === 'assistant' &&
        !previousMessage.body?.startsWith('(SYS)');

      if (canAttachToPreviousAssistant) {
        previousRenderable.extraFunctionCalls = [
          ...(previousRenderable.extraFunctionCalls || []),
          ...(message.function_calls || []),
        ];
        return;
      }

      pendingFunctionMessages.push(message);
    });

    if (pendingFunctionMessages.length > 0) {
      pendingFunctionMessages.forEach((pendingMessage) => {
        accumulator.push({
          message: pendingMessage,
          extraFunctionCalls: [],
        });
      });
    }

    return accumulator;
  }, [messages]);

  return (
    <Scroller
      ref={messagesEndRef}
      sx={{
        px: 1.25,
        width: 1,
        height: 1,
        minHeight: 0,
        flexGrow: 1,
        overflowX: 'hidden',
      }}
    >
      <Stack
        spacing={1}
        sx={{
          p: { xs: 2, md: 0.5 },
          minHeight: '100%',
          justifyContent: hasMessages ? 'flex-end' : 'center',
        }}
      >
        {hasMessages ? (
          renderableMessages.map(({ message, extraFunctionCalls }, index) => (
            <ChatMessageItem
              key={message.id}
              message={message}
              conversation={conversation}
              isLastMessage={index === renderableMessages.length - 1}
              extraFunctionCalls={extraFunctionCalls}
              onOpenLightbox={() => {}}
            />
          ))
        ) : (
          <Box
            sx={{
              flexGrow: 1,
              minHeight: 240,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              color: 'text.secondary',
              px: { xs: 2, md: 4 },
            }}
          >
            <Typography variant="body2">
              Start the conversation by sending a message or choosing an action above.
            </Typography>
          </Box>
        )}

        <Box sx={{ height: 8, flexShrink: 0 }} />
      </Stack>
    </Scroller>
  );
}

type WelcomeContentProps = {
  renderChatInput: () => ReactNode;
  starterOptions: StarterOption[];
  onStarterSelect: (option: StarterOption) => void;
  isRecording: boolean;
};

function WelcomeContent({
  renderChatInput,
  starterOptions,
  onStarterSelect,
  isRecording,
}: WelcomeContentProps) {
  const theme = useTheme();
  const starterStackRef = useRef<HTMLDivElement | null>(null);
  const [isWideStarterStack, setIsWideStarterStack] = useState(false);

  useEffect(() => {
    const updateLayout = () => {
      const node = starterStackRef.current;
      if (!node) {
        return;
      }
      const { width } = node.getBoundingClientRect();
      setIsWideStarterStack(width >= 600);
    };

    updateLayout();

    const cleanups: Array<() => void> = [];
    const node = starterStackRef.current;

    if (node && typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(() => updateLayout());
      observer.observe(node);
      cleanups.push(() => observer.disconnect());
    } else {
      window.addEventListener('resize', updateLayout);
      cleanups.push(() => window.removeEventListener('resize', updateLayout));
    }

    return () => {
      cleanups.forEach((cleanup) => cleanup());
    };
  }, []);

  return (
    <Stack
      spacing={isRecording ? 0 : 4}
      alignItems="center"
      justifyContent={isRecording ? 'flex-end' : 'center'}
      sx={{
        flexGrow: 1,
        minHeight: { xs: 480, md: 500 },
        px: { xs: 0, md: 5 },
        py: { xs: 1, md: 5 },
        pb: isRecording ? { xs: 2, md: 0.5 } : { xs: 1, md: 5 },
        transform: { xs: 'none', sm: isRecording ? 'none' : 'translateY(-40px)' },
        transition: 'all 0.4s ease-in-out',
      }}
    >
      <Box
        component="img"
        src="/assets/images/chatbots/chatbot-2.jpg"
        alt="Realtime Assistant"
        sx={{
          width: 64,
          height: 64,
          minWidth: 64,
          minHeight: 64,
          borderRadius: '50%',
          objectFit: 'cover',
          boxShadow: 2,
          opacity: isRecording ? 0 : 1,
          maxHeight: isRecording ? 0 : 64,
          transition: 'opacity 0.3s ease-in-out, max-height 0.3s ease-in-out',
        }}
      />

      <Stack
        spacing={1}
        alignItems="center"
        textAlign="center"
        sx={{
          opacity: isRecording ? 0 : 1,
          maxHeight: isRecording ? 0 : 200,
          overflow: 'hidden',
          transition: 'opacity 0.3s ease-in-out, max-height 0.3s ease-in-out',
        }}
      >
        <Typography variant="h4">Your multimodal assistant awaits</Typography>
      </Stack>

      <Box
        sx={{
          width: '100%',
          maxWidth: isRecording ? { xs: 480, sm: 560 } : { xs: 520, sm: 740 },
          transition: 'max-width 0.4s ease-in-out',
        }}
      >
        {renderChatInput()}
      </Box>

      <Stack
        spacing={2}
        alignItems="center"
        sx={{
          width: '100%',
          maxWidth: { xs: 580, sm: 756 },
          opacity: isRecording ? 0 : 1,
          maxHeight: isRecording ? 0 : 1000,
          overflow: 'hidden',
          transition: 'opacity 0.3s ease-in-out, max-height 0.4s ease-in-out',
        }}
      >
        <Typography
          variant="body2"
          color="text.disabled"
          fontSize={13}
          fontWeight={500}
          sx={{ alignSelf: 'flex-start', ml: 1 }}
        >
          Get started
        </Typography>

        <Stack
          ref={starterStackRef}
          direction={isWideStarterStack ? 'row' : 'column'}
          spacing={2}
          sx={{
            px: 1,
            pb: 1.25,
            width: '100%',
            alignItems: { xs: 'stretch', sm: 'stretch' },
            flexWrap: isWideStarterStack ? 'wrap' : 'nowrap',
          }}
        >
          {starterOptions.map((option) => (
            <Paper
              key={option.id}
              component={ButtonBase}
              onClick={() => onStarterSelect(option)}
              type="button"
              sx={{
                width: '100%',
                px: 1.5,
                py: 2,
                pb: 1.25,
                borderRadius: 2,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
                justifyContent: 'center',
                gap: 1.5,
                textAlign: 'left',
                backgroundColor:
                  theme.palette.mode === 'light'
                    ? alpha(theme.palette.grey[500], 0.08)
                    : alpha(theme.palette.common.white, 0.06),
                transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                color: 'text.primary',
                flex: { xs: '0 0 auto', sm: 1 },
                '&:hover': {
                  boxShadow: `0px 4px 8px ${alpha(
                    theme.palette.mode === 'light'
                      ? theme.palette.grey[900]
                      : theme.palette.common.black,
                    theme.palette.mode === 'light' ? 0.08 : 0.3
                  )}`,
                  // transform: 'translateY(-2px)',
                },
              }}
            >
              <Iconify
                icon={option.icon}
                width={20}
                height={20}
                color={theme.palette.text.secondary}
              />
              <Typography variant="body2" color="text.secondary" fontSize={13}>
                {option.description}
              </Typography>
            </Paper>
          ))}
        </Stack>
      </Stack>
    </Stack>
  );
}
