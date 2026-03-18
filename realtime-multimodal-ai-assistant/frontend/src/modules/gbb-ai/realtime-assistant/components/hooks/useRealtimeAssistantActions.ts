import { ReadyState } from 'react-use-websocket';
import { useCallback, type Dispatch, type SetStateAction, type MutableRefObject } from 'react';

import { convertBase64 } from 'src/api/gpt/api';

import type { ICaption, SendMessage } from 'src/types/chat';

type Props = {
  isRecording: boolean;
  isSessionActive: boolean;
  wsConnecting: boolean;
  setAudioError: Dispatch<SetStateAction<string | null>>;
  setWsError: Dispatch<SetStateAction<string | null>>;
  setUserInitiatedTalking: Dispatch<SetStateAction<boolean>>;
  setIsTalkModeEnabled: Dispatch<SetStateAction<boolean>>;
  setIsRecording: Dispatch<SetStateAction<boolean>>;
  setIsSessionActive: Dispatch<SetStateAction<boolean>>;
  setWsConnecting: Dispatch<SetStateAction<boolean>>;
  setAssistantSpeaking: (shouldSpeak: boolean, graceMs?: number) => void;
  startAudioRecording: () => Promise<boolean>;
  stopAudioRecording: () => Promise<void>;
  stopAudioPlayer: () => void | Promise<void>;
  ensureRealtimeSession: () => void;
  requireActiveSession: () => boolean;
  closeSession: () => void;
  responseCancel: () => void;
  clearToolActivity: () => void;
  addUserImage: (base64Image: string, fileType: string) => void;
  addUserText: (text: string) => void;
  addImageCaption: (caption: string) => void;
  onSendMessage: (message: SendMessage) => void;
  clearMessages: () => void;
  forceReconnect: () => void;
  startSession: () => void;
  enqueueSnackbar: (message: string, options?: any) => void;
  manuallyStoppedRef: MutableRefObject<boolean>;
  wsReadyStateRef: MutableRefObject<ReadyState>;
  reconnectTimeoutRef: MutableRefObject<ReturnType<typeof setTimeout> | null>;
};

export default function useRealtimeAssistantActions({
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
}: Props) {
  const onToggleListening = useCallback(
    async (active?: boolean) => {
      setAudioError(null);
      setWsError(null);
      const shouldStartListening = typeof active === 'boolean' ? active : !isRecording;

      if (shouldStartListening) {
        setUserInitiatedTalking(true);
        setIsTalkModeEnabled(true);

        try {
          ensureRealtimeSession();
          const started = await startAudioRecording();
          if (!started) {
            setAudioError('Failed to start audio recording');
            return;
          }

          setIsRecording(true);
        } catch (error) {
          setAudioError(error instanceof Error ? error.message : 'Failed to start recording');
          setIsSessionActive(false);
          closeSession();
        }

        return;
      }

      setIsTalkModeEnabled(false);
      setAssistantSpeaking(false);
      setUserInitiatedTalking(false);
      setIsRecording(false);
      setIsSessionActive(false);

      try {
        responseCancel();
        await stopAudioRecording();
        await stopAudioPlayer();
      } catch {
        // Silently handle any errors during stop.
      } finally {
        closeSession();
      }
    },
    [
      closeSession,
      ensureRealtimeSession,
      isRecording,
      responseCancel,
      setAssistantSpeaking,
      setAudioError,
      setIsRecording,
      setIsSessionActive,
      setIsTalkModeEnabled,
      setUserInitiatedTalking,
      setWsError,
      startAudioRecording,
      stopAudioPlayer,
      stopAudioRecording,
    ]
  );

  const handleTruncateAudio = useCallback(() => {
    stopAudioPlayer();
    setAssistantSpeaking(false);
  }, [setAssistantSpeaking, stopAudioPlayer]);

  const handleStopAssistantAudio = useCallback(() => {
    manuallyStoppedRef.current = true;
    responseCancel();
    stopAudioPlayer();
    setAssistantSpeaking(false);
  }, [manuallyStoppedRef, responseCancel, setAssistantSpeaking, stopAudioPlayer]);

  const onSendImage = useCallback(
    async (files: File[]) => {
      if (!files || files.length === 0 || !requireActiveSession()) {
        return;
      }

      manuallyStoppedRef.current = false;

      try {
        ensureRealtimeSession();

        const imagePromises = files.map(async (file) => {
          const base64Data = (await convertBase64(file, 1080, 1)) as string;
          const base64Image = base64Data.split(',')[1];
          const fileType = file.type.split('/')[1] || 'png';

          return {
            base64Image,
            fileType,
          };
        });

        const processedImages = await Promise.all(imagePromises);
        processedImages.forEach(({ base64Image, fileType }) => {
          addUserImage(base64Image, fileType);
        });
      } catch {
        enqueueSnackbar('Failed to send images', { variant: 'error' });
      }
    },
    [
      addUserImage,
      enqueueSnackbar,
      ensureRealtimeSession,
      manuallyStoppedRef,
      requireActiveSession,
    ]
  );

  const onSendUserText = useCallback(
    async (text: string) => {
      if (!requireActiveSession()) {
        return;
      }

      setUserInitiatedTalking(true);
      manuallyStoppedRef.current = false;
      ensureRealtimeSession();
      await stopAudioPlayer();
      addUserText(text);
    },
    [
      addUserText,
      ensureRealtimeSession,
      manuallyStoppedRef,
      requireActiveSession,
      setUserInitiatedTalking,
      stopAudioPlayer,
    ]
  );

  const onSendCameraImage = useCallback(
    (base64Image: string) => {
      if (!isSessionActive) {
        return;
      }

      ensureRealtimeSession();
      addUserImage(base64Image, 'jpeg');
    },
    [addUserImage, ensureRealtimeSession, isSessionActive]
  );

  const processImageCaption = useCallback(
    (captionItem: ICaption) => {
      if (!captionItem || !isSessionActive) {
        return;
      }

      ensureRealtimeSession();
      const message = {
        messageId: captionItem.timestamp.toString(),
        message: captionItem.caption,
        contentType: 'text',
        createdAt: new Date(),
        senderId: 'image-captioning',
        mode: 'new',
        chatMode: 'open-chat',
        sources: [],
        status: 'completed',
      } as SendMessage;

      onSendMessage(message);
      addImageCaption(captionItem.caption);
    },
    [addImageCaption, ensureRealtimeSession, isSessionActive, onSendMessage]
  );

  const handleClearHistory = useCallback(() => {
    clearMessages();
    clearToolActivity();
  }, [clearMessages, clearToolActivity]);

  const handleRenewSession = useCallback(() => {
    if (wsConnecting) {
      return;
    }

    setIsSessionActive(true);
    setWsConnecting(true);
    setWsError(null);
    setAudioError(null);
    setUserInitiatedTalking(false);
    setIsRecording(false);
    setIsTalkModeEnabled(true);
    setAssistantSpeaking(false);

    forceReconnect();
    startSession();

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    reconnectTimeoutRef.current = setTimeout(() => {
      if (wsReadyStateRef.current !== ReadyState.OPEN) {
        setWsConnecting(false);
        const errorMessage = 'Connection timeout.';
        setWsError(errorMessage);
        enqueueSnackbar(errorMessage, { variant: 'error' });
      }
    }, 5000);
  }, [
    enqueueSnackbar,
    forceReconnect,
    reconnectTimeoutRef,
    setAssistantSpeaking,
    setAudioError,
    setIsRecording,
    setIsSessionActive,
    setIsTalkModeEnabled,
    setUserInitiatedTalking,
    setWsConnecting,
    setWsError,
    startSession,
    wsConnecting,
    wsReadyStateRef,
  ]);

  return {
    handleClearHistory,
    handleRenewSession,
    handleStopAssistantAudio,
    handleTruncateAudio,
    onSendCameraImage,
    onSendImage,
    onSendUserText,
    onToggleListening,
    processImageCaption,
  };
}