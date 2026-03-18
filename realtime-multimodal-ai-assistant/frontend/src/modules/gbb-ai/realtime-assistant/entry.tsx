import { ReadyState } from 'react-use-websocket';
import { useRef, useState, useEffect, useCallback } from 'react';

// mui
import Box from '@mui/material/Box';
import { useTheme } from '@mui/material/styles';
import Container from '@mui/material/Container';
import useMediaQuery from '@mui/material/useMediaQuery';

// project imports
import { useBoolean } from 'src/hooks/boolean';
import { getStorage, setStorage } from 'src/hooks/local-storage';

import getInitialResourceName from 'src/utils/aoai-resources';

import { getRtConfiguration } from 'src/api/gpt/index';

import { useSnackbar } from 'src/widgets/notification';
import { useSettingsContext } from 'src/widgets/settings';
import { RtConfigEntry } from 'src/widgets/realtime-configuration';

import { SendMessage, IRtConfiguration } from 'src/types/chat';

import ChatWindow from './components/chat-window';
import useRealTime from './components/hooks/useRealtime';
import useAudioPlayer from './components/hooks/useAudioPlayer';
import useToolActivity from './components/hooks/useToolActivity';
import useBackendHealth from './components/hooks/useBackendHealth';
import useAudioRecorder from './components/hooks/useAudioRecorder';
import useRealtimeMessages from './components/hooks/useRealtimeMessages';
import useAssistantSpeaking from './components/hooks/useAssistantSpeaking';
import RealtimeAssistantTopBar from './components/realtime-assistant-topbar';
import useRealtimeAssistantActions from './components/hooks/useRealtimeAssistantActions';

// ----------------------------------------------------------------------

type Props = {
  id: string;
};

export default function RealtimeAssistant({ id }: Props) {
  const chatMode = 'open-chat';
  const settings = useSettingsContext();
  const { enqueueSnackbar } = useSnackbar();

  const [isRecording, setIsRecording] = useState(false);
  const [isTalkModeEnabled, setIsTalkModeEnabled] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [wsError, setWsError] = useState<string | null>(null);
  const [userInitiatedTalking, setUserInitiatedTalking] = useState(false);
  const [wsConnecting, setWsConnecting] = useState(false);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const wsReadyStateRef = useRef(ReadyState.UNINSTANTIATED);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    },
    []
  );

  const theme = useTheme();
  const isXsScreen = useMediaQuery(theme.breakpoints.down('sm'));

  const initialResourceName = getInitialResourceName() || 'rubicon';

  const localConfigurations: IRtConfiguration = getStorage(id);

  const initialConfigurations: IRtConfiguration = getRtConfiguration(initialResourceName);
  const [configurations, setConfigurations] = useState<IRtConfiguration>({
    ...initialConfigurations,
    ...localConfigurations,
  });

  const { play: playAudio, stop: stopAudioPlayer, isPlaying } = useAudioPlayer();
  const {
    messages,
    setMessages,
    onSendMessage,
    clearMessages,
    handleConversationItemCreated,
    conversationItemPositionsRef,
  } = useRealtimeMessages(chatMode);
  const { clearToolActivity, handleToolResponse, handleToolStatus } = useToolActivity({
    chatMode,
    setMessages,
    conversationItemPositionsRef,
  });
  const { isAssistantSpeaking, manuallyStoppedRef, setAssistantSpeaking } =
    useAssistantSpeaking(isPlaying);

  useEffect(() => {
    if (audioError) enqueueSnackbar(audioError, { variant: 'error' });
  }, [audioError, enqueueSnackbar]);

  useEffect(() => {
    if (wsError && userInitiatedTalking) enqueueSnackbar(wsError, { variant: 'error' });
  }, [wsError, userInitiatedTalking, enqueueSnackbar]);

  const {
    readyState,
    startSession,
    addUserText,
    addUserImage,
    addUserAudio,
    addImageCaption,
    responseCancel,
    closeSession,
    forceReconnect,
    syncConversationTranscript,
  } = useRealTime({
    configurations,
    enableInputAudioTranscription: true,
    onWebSocketOpen: () => {
      setWsError(null);
      setWsConnecting(false);
    },
    onWebSocketClose: () => {
      setWsConnecting(false);
      setIsRecording(false);
      setIsTalkModeEnabled(false);
      setAssistantSpeaking(false);
      setIsSessionActive(false);
    },
    onWebSocketError: (event) => {
      setWsConnecting(false);
      // Show immediate error feedback when WebSocket fails
      const errorMessage = 'Connection failed. Please check if the backend is running.';
      setWsError(errorMessage);
      enqueueSnackbar(errorMessage, { variant: 'error' });
      setIsRecording(false);
      setAssistantSpeaking(false);
      setIsSessionActive(false);
    },
    onReceivedError: (message) => {
      if (
        message.error.code === 'response_cancel_not_active' ||
        message.error.code === 'conversation_already_has_active_response' ||
        message.error.code === 'missing_required_parameter'
      ) {
        // Don't show snackbar for these expected errors
      }
      // setWsError('An error occurred');
      // setIsRecording(false);
    },
    onReceivedResponseAudioDone: () => {
      manuallyStoppedRef.current = false; // Reset the flag when response is done
      setAssistantSpeaking(false, 1200);
    },
    onReceivedResponseAudioDelta: (message) => {
      // Ignore audio if user manually stopped playback
      if (manuallyStoppedRef.current) {
        return;
      }
      // Only play audio response when talk mode is enabled
      if (isTalkModeEnabled) {
        setAssistantSpeaking(true);
        playAudio(message.delta);
      }
    },
    onReceivedInputAudioBufferSpeechStarted: () => {
      // Reset manual stop flag when user starts speaking
      manuallyStoppedRef.current = false;
      // Stop local audio playback only
      stopAudioPlayer();
    },
    onReceivedInputAudioTranscriptionCompleted: (message) => {
      if (message && message.transcript) {
        const msg = {
          conversationId: message.event_id,
          messageId: message.item_id,
          message: message.transcript,
          contentType: 'text',
          createdAt: new Date(),
          senderId: 'user',
          mode: 'new',
          chatMode: 'open-chat',
          realtimeItemId: message.item_id,
        } as SendMessage;
        onSendMessage(msg);
      }
    },
    onReceivedResponseAudioTranscriptDelta: (message) => {
      if (message && message.delta) {
        const msg = {
          conversationId: message.event_id,
          messageId: message.item_id,
          message: message.delta,
          contentType: 'text',
          sources: [],
          function_calls: [],
          createdAt: new Date(),
          senderId: 'assistant',
          mode: 'new',
          chatMode: 'open-chat',
          realtimeItemId: message.item_id,
        } as SendMessage;
        onSendMessage(msg);
      }
    },
    onReceivedExtensionMiddleTierToolStatus: handleToolStatus,
    onReceivedExtensionMiddleTierToolResponse: (message) => {
      if (!message || typeof message !== 'object' || message.tool_name === 'image_caption') return;
      handleToolResponse(message);
    },
    onReceivedConversationItemCreated: handleConversationItemCreated,
  });
  const isBackendAvailable = useBackendHealth({ readyState, wsConnecting });

  useEffect(() => {
    if (!isSessionActive || readyState === ReadyState.UNINSTANTIATED) {
      return;
    }

    const transcript = messages
      .filter(
        (message) =>
          message.body &&
          message.body !== '(SYS)function' &&
          (message.senderId === 'user' || message.senderId === 'assistant')
      )
      .slice(-40)
      .map((message) => ({
        role: (message.senderId === 'assistant' ? 'assistant' : 'user') as 'user' | 'assistant',
        text: message.body.trim(),
        created_at: new Date(message.createdAt).toISOString(),
      }))
      .filter((message) => message.text);

    syncConversationTranscript(transcript);
  }, [isSessionActive, messages, readyState, syncConversationTranscript]);

  useEffect(() => {
    wsReadyStateRef.current = readyState;

    if (readyState === ReadyState.OPEN || readyState === ReadyState.CLOSED) {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    }
  }, [readyState]);

  const isChipConnected = readyState === ReadyState.OPEN && isSessionActive;
  const isChipConnecting =
    !isChipConnected &&
    (readyState === ReadyState.CONNECTING || readyState === ReadyState.CLOSING || wsConnecting);
  const isBackendUnavailable =
    isBackendAvailable === false && !isChipConnected && !isChipConnecting;
  const isChipDisconnected =
    !isChipConnected &&
    !isChipConnecting &&
    !isBackendUnavailable &&
    readyState !== ReadyState.UNINSTANTIATED &&
    (isSessionActive || userInitiatedTalking || Boolean(wsError));
  const isBackendReady =
    isBackendAvailable === true && !isChipConnected && !isChipConnecting && !isChipDisconnected;

  const ensureRealtimeSession = useCallback(() => {
    setIsSessionActive(true);
    setWsError(null);
    if (readyState !== ReadyState.OPEN) {
      setWsConnecting(true);
    }
    startSession();
  }, [readyState, startSession]);

  const requireActiveSession = useCallback(() => {
    if (isSessionActive) {
      return true;
    }

    enqueueSnackbar('Start a talk session first.', { variant: 'info' });
    return false;
  }, [enqueueSnackbar, isSessionActive]);

  const {
    start: startAudioRecording,
    stop: stopAudioRecording,
    setMuted: setAudioMuted,
  } = useAudioRecorder({
    onAudioRecorded: addUserAudio,
    onError: (error) => {
      setAudioError(error.message);
      setIsRecording(false);
    },
  });

  const {
    onToggleListening,
    handleTruncateAudio,
    handleStopAssistantAudio,
    onSendImage,
    onSendUserText,
    onSendCameraImage,
    processImageCaption,
    handleClearHistory,
    handleRenewSession,
  } = useRealtimeAssistantActions({
    isRecording,
    isSessionActive,
    wsConnecting,
    setAudioError,
    setWsError,
    setUserInitiatedTalking,
    setIsTalkModeEnabled,
    setIsRecording,
    setIsSessionActive,
    setWsConnecting,
    setAssistantSpeaking,
    startAudioRecording,
    stopAudioRecording,
    stopAudioPlayer,
    ensureRealtimeSession,
    requireActiveSession,
    closeSession,
    responseCancel,
    clearToolActivity,
    addUserImage,
    addUserText,
    addImageCaption,
    onSendMessage,
    clearMessages,
    forceReconnect,
    startSession,
    enqueueSnackbar,
    manuallyStoppedRef,
    wsReadyStateRef,
    reconnectTimeoutRef,
  });

  const handleUpdateConfigs = (_config: IRtConfiguration) => {
    setConfigurations(_config);
    setStorage(id, _config);
  };

  const config = useBoolean();

  const boxHeight = isXsScreen ? 'calc(100vh - 150px)' : 'calc(100vh - 158px)';

  return (
    <Container maxWidth={settings.themeStretch ? 'xl' : 'lg'} sx={{ mb: -8, height: boxHeight }}>
      <RealtimeAssistantTopBar
        isChipConnected={isChipConnected}
        isChipDisconnected={isChipDisconnected}
        isChipConnecting={isChipConnecting}
        isRecording={isRecording}
        isBackendReady={isBackendReady}
        isBackendUnavailable={isBackendUnavailable}
        onRenewSession={handleRenewSession}
        onClearHistory={handleClearHistory}
        onOpenConfig={config.onTrue}
      />

      <Box
        zIndex={2}
        sx={{
          mt: 2.5,
          mx: 'auto',
          // width: { xs: '90%', sm: '80%' },
          height: 1,
          display: 'flex',
          position: 'relative',
          overflow: 'visible',
        }}
      >
        <ChatWindow
          messages={messages}
          onUpdateMessages={setMessages}
          configurations={configurations}
          chatMode={chatMode}
          isRecording={isRecording}
          isSessionActive={isSessionActive}
          onToggleListening={onToggleListening}
          onSendText={onSendUserText}
          onSendImage={onSendImage}
          onSendCameraImage={onSendCameraImage}
          isSpeaking={isAssistantSpeaking}
          onTruncateAudio={handleTruncateAudio}
          processImageCaption={processImageCaption}
          onSetAudioMuted={setAudioMuted}
          onStopAudioPlayback={handleStopAssistantAudio}
        />
      </Box>

      <RtConfigEntry
        open={config.value}
        callerId={id}
        onClose={config.onFalse}
        configurations={configurations}
        onUpdate={handleUpdateConfigs}
      />
    </Container>
  );
}
