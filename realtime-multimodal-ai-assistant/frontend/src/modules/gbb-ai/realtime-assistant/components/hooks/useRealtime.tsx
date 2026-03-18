import useWebSocket, { ReadyState } from 'react-use-websocket';
import { useRef, useState, useEffect, useCallback } from 'react';

import { REALTIME_API } from 'src/config-global';

import { IRtConfiguration } from 'src/types/chat';

import {
  Message,
  ResponseDone,
  InputTextCommand,
  ResponseAudioDelta,
  SessionUpdateCommand,
  ResponseCancelCommand,
  WebSocketErrorPayload,
  ConversationItemCreated,
  ResponseAudioTranscriptDelta,
  InputAudioBufferClearCommand,
  InputAudioBufferAppendCommand,
  ConversationItemDeleteCommand,
  ExtensionMiddleTierToolStatus,
  ConversationItemTruncateCommand,
  ExtensionMiddleTierToolResponse,
  ResponseInputAudioTranscriptionCompleted,
} from '../types/realtime-events';

type Parameters = {
  useDirectAoaiApi?: boolean;
  aoaiEndpointOverride?: string;
  aoaiApiKeyOverride?: string;
  aoaiModelOverride?: string;

  enableInputAudioTranscription?: boolean;
  onWebSocketOpen?: () => void;
  onWebSocketClose?: () => void;
  onWebSocketError?: (event: Event) => void;
  onWebSocketMessage?: (event: MessageEvent<any>) => void;

  onReceivedResponseAudioDone?: (message: Message) => void;
  onReceivedResponseAudioDelta?: (message: ResponseAudioDelta) => void;
  onReceivedInputAudioBufferSpeechStarted?: (message: Message) => void;
  onReceivedResponseDone?: (message: ResponseDone) => void;
  onReceivedConversationItemCreated?: (message: ConversationItemCreated) => void;
  onReceivedExtensionMiddleTierToolStatus?: (message: ExtensionMiddleTierToolStatus) => void;
  onReceivedExtensionMiddleTierToolResponse?: (message: ExtensionMiddleTierToolResponse) => void;
  onReceivedResponseAudioTranscriptDelta?: (message: ResponseAudioTranscriptDelta) => void;
  onReceivedInputAudioTranscriptionCompleted?: (
    message: ResponseInputAudioTranscriptionCompleted
  ) => void;
  onReceivedError?: (message: WebSocketErrorPayload) => void;
  configurations?: IRtConfiguration;
};

export default function useRealTime({
  useDirectAoaiApi,
  aoaiEndpointOverride,
  aoaiApiKeyOverride,
  aoaiModelOverride,
  enableInputAudioTranscription,
  onWebSocketOpen,
  onWebSocketClose,
  onWebSocketError,
  onWebSocketMessage,
  onReceivedResponseDone,
  onReceivedResponseAudioDone,
  onReceivedResponseAudioDelta,
  onReceivedConversationItemCreated,
  onReceivedExtensionMiddleTierToolStatus,
  onReceivedResponseAudioTranscriptDelta,
  onReceivedInputAudioBufferSpeechStarted,
  onReceivedExtensionMiddleTierToolResponse,
  onReceivedInputAudioTranscriptionCompleted,
  onReceivedError,
  configurations,
}: Parameters) {
  const mountedRef = useRef(true);
  const [isConnected, setIsConnected] = useState(false);
  const [shouldConnect, setShouldConnect] = useState(false);
  const [sessionInitialized, setSessionInitialized] = useState(false);
  const [reconnectKey, setReconnectKey] = useState(0);
  const reconnectRequestedRef = useRef(false);

  // ------------------------------------------------------------------
  // 1. queue for messages produced while the socket is not yet OPEN
  // ------------------------------------------------------------------
  const queuedRef = useRef<unknown[]>([]);

  const flushQueue = useCallback((sender: (msg: unknown) => void) => {
    queuedRef.current.forEach(sender);
    queuedRef.current = [];
  }, []);

  useEffect(
    () => () => {
      mountedRef.current = false;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  // Get voice host to determine which WebSocket endpoint to use
  const voiceHost = configurations?.['rt-Voice host'] || 'gpt-realtime';

  // Determine WebSocket endpoint based on configuration
  let wsEndpoint: string;
  if (useDirectAoaiApi) {
    wsEndpoint = `${aoaiEndpointOverride}/openai/realtime?api-key=${aoaiApiKeyOverride}&deployment=${aoaiModelOverride}&api-version=2024-10-01-preview&t=${reconnectKey}`;
  } else {
    wsEndpoint = `${REALTIME_API}?voice_host=${voiceHost}&t=${reconnectKey}`;
  }

  const { sendJsonMessage, getWebSocket, readyState } = useWebSocket(
    // Always connect, but control reconnection behavior
    wsEndpoint,
    {
      onOpen: () => {
        setIsConnected(true);
        flushQueue(sendJsonMessage);
        // Always call onWebSocketOpen when connection opens
        reconnectRequestedRef.current = false;
        onWebSocketOpen?.();
      },
      onClose: () => {
        setIsConnected(false);
        if (reconnectRequestedRef.current) {
          return;
        }
        // Only call onWebSocketClose if session was active
        if (sessionInitialized) {
          onWebSocketClose?.();
        }
      },
      onError: (event) => {
        setIsConnected(false);
        // Only show errors if user has initiated a session
        if (sessionInitialized) {
          console.error('WebSocket error during session:', event);
          onWebSocketError?.(event);
        } else {
          // Silently log connection errors when no session is active
          console.debug('WebSocket connection error (no active session):', event);
        }
      },
      onMessage: (event) => onMessageReceived(event),
      shouldReconnect: (_closeEvent) =>
        mountedRef.current && (shouldConnect || reconnectRequestedRef.current),
      reconnectAttempts: 5,
      reconnectInterval: 1000, // Fast reconnection
    },
    shouldConnect
  );

  const safeSendJsonMessage = useCallback(
    (data: unknown) => {
      if (readyState === ReadyState.OPEN) {
        try {
          sendJsonMessage(data);
        } catch (err) {
          console.error('Error sending message:', err);
          const errorPayload: WebSocketErrorPayload = {
            type: 'error',
            event_id: null, // Or generate a unique ID if possible
            error: {
              code: 'send_error', // Custom code for send failure
              message: err instanceof Error ? err.message : 'Failed to send message',
              param: null,
              type: 'internal_client_error', // Custom type
            },
          };
          onReceivedError?.(errorPayload);
        }
        return;
      }

      queuedRef.current.push(data);
    },
    [readyState, sendJsonMessage, onReceivedError]
  );

  // Monitor configuration changes (like voice choice) and update session if connected
  const previousConfigRef = useRef<IRtConfiguration>();
  useEffect(() => {
    const currentVoiceChoice = configurations?.['rt-Voice choice'];
    const previousVoiceChoice = previousConfigRef.current?.['rt-Voice choice'];

    if (
      sessionInitialized &&
      isConnected &&
      previousVoiceChoice &&
      currentVoiceChoice !== previousVoiceChoice
    ) {
      // Send updated session configuration
      const command = {
        type: 'session.update',
        session: {
          voice: currentVoiceChoice?.toLowerCase() || 'alloy',
        },
      };

      safeSendJsonMessage(command);
    }

    previousConfigRef.current = configurations;
  }, [configurations, sessionInitialized, isConnected, safeSendJsonMessage]);

  const startSession = () => {
    if (!shouldConnect) {
      setShouldConnect(true);
    }

    // Mark that we've explicitly started a session
    if (!sessionInitialized) {
      setSessionInitialized(true);
    }

    // Send the session configuration
    const command: SessionUpdateCommand = {
      type: 'session.update',
      session: {
        turn_detection: {
          type: 'server_vad',
          threshold:
            configurations && configurations['rt-VAD Threshold']
              ? parseFloat(configurations['rt-VAD Threshold'])
              : 0.55,
          prefix_padding_ms:
            configurations && configurations['rt-VAD Prefix Padding (ms)']
              ? parseInt(configurations['rt-VAD Prefix Padding (ms)'], 10)
              : 180,
          silence_duration_ms:
            configurations && configurations['rt-VAD Silence Duration (ms)']
              ? parseInt(configurations['rt-VAD Silence Duration (ms)'], 10)
              : 220,
        } as any, // Type assertion to allow additional VAD parameters
        modalities: ['text', 'audio'],
        ...(configurations && {
          temperature: configurations['rt-Temperature']
            ? parseFloat(configurations['rt-Temperature'])
            : 0.7,
          voice: configurations['rt-Voice choice']?.toLowerCase() || 'alloy',
          max_response_output_tokens: configurations['rt-Max response']
            ? parseInt(configurations['rt-Max response'], 10)
            : 800,
          // Override with configuration if explicitly set
          ...(configurations['rt-Disable audio'] && { modalities: ['text'] }),
          instructions: configurations['rt-System message'],
        }),
      },
    };

    if (enableInputAudioTranscription) {
      command.session.input_audio_transcription = {
        model: 'whisper-1',
      };
    }

    safeSendJsonMessage(command);
  };

  const closeSession = () => {
    setShouldConnect(false);
    setSessionInitialized(false);
    reconnectRequestedRef.current = false;
    queuedRef.current = [];
    const ws = getWebSocket();
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      // Force close with code 1000 (normal closure)
      ws.close(1000, 'Session closed by user');
    }
  };

  const forceReconnect = () => {
    // Change the URL to force a new connection
    setReconnectKey((prev) => prev + 1);
    reconnectRequestedRef.current = true;

    // Close the current connection
    const ws = getWebSocket();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close(1000, 'Manual reconnection');
    }

    // Reset states to allow fresh connection
    setIsConnected(false);

    // Clear the queue
    queuedRef.current = [];
  };

  const addUserText = (value: string) => {
    const command: InputTextCommand = {
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: value,
          },
        ],
      },
    };

    safeSendJsonMessage(command);
    sendJsonMessage({
      type: 'response.create',
    });
  };

  const addUserImage = (base64Image: string, imageFormat: string = 'png') => {
    const command = {
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_image',
            image_url: `data:image/${imageFormat};base64,${base64Image}`,
          },
        ],
      },
    };

    safeSendJsonMessage(command);
  };

  const addImageCaption = (value: string) => {
    const command: InputTextCommand = {
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: value,
          },
        ],
      },
    };

    safeSendJsonMessage(command);
  };

  const addAssistantText = (value: string) => {
    const command: InputTextCommand = {
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'input_text',
            text: value,
          },
        ],
      },
    };

    safeSendJsonMessage(command);
  };

  const addFunctionCallOutput = (value: Object) => {
    safeSendJsonMessage(value);
    safeSendJsonMessage({
      type: 'response.create',
    });
  };

  const syncConversationTranscript = (
    transcript: { role: 'user' | 'assistant'; text: string; created_at?: string }[]
  ) => {
    safeSendJsonMessage({
      type: 'extension.transcript.update',
      transcript,
    });
  };

  const addUserAudio = (base64Audio: string) => {
    const command: InputAudioBufferAppendCommand = {
      type: 'input_audio_buffer.append',
      audio: base64Audio,
    };
    safeSendJsonMessage(command);
  };

  const responseCancel = () => {
    const command: ResponseCancelCommand = {
      type: 'response.cancel',
    };

    safeSendJsonMessage(command);
  };

  const inputAudioBufferClear = () => {
    const command: InputAudioBufferClearCommand = {
      type: 'input_audio_buffer.clear',
    };

    safeSendJsonMessage(command);
  };

  const conversationItemTruncate = (
    itemId: string,
    contentIndex: number = 0,
    audioEndMs: number = 0
  ) => {
    // Default contentIndex to 0
    const command: ConversationItemTruncateCommand = {
      type: 'conversation.item.truncate',
      item_id: itemId,
      content_index: contentIndex,
      audio_end_ms: audioEndMs,
    };
    safeSendJsonMessage(command);
  };

  const conversationItemDelete = (itemId: string) => {
    const command: ConversationItemDeleteCommand = {
      type: 'conversation.item.delete',
      item_id: itemId,
    };
    safeSendJsonMessage(command);
  };

  const onMessageReceived = (event: MessageEvent<any>) => {
    onWebSocketMessage?.(event);

    let message: Message;
    try {
      message = JSON.parse(event.data);
    } catch (e) {
      console.error('Failed to parse JSON message:', e);
      throw e;
    }

    switch (message.type) {
      case 'response.done':
        onReceivedResponseDone?.(message as ResponseDone);
        break;
      case 'response.audio.delta':
        onReceivedResponseAudioDelta?.(message as ResponseAudioDelta);
        break;
      case 'response.audio_transcript.delta':
        onReceivedResponseAudioTranscriptDelta?.(message as ResponseAudioTranscriptDelta);
        break;
      case 'input_audio_buffer.speech_started':
        onReceivedInputAudioBufferSpeechStarted?.(message);
        break;
      case 'conversation.item.input_audio_transcription.completed':
        onReceivedInputAudioTranscriptionCompleted?.(
          message as ResponseInputAudioTranscriptionCompleted
        );
        break;
      case 'extension.middle_tier_tool_response':
        onReceivedExtensionMiddleTierToolResponse?.(message as ExtensionMiddleTierToolResponse);
        break;
      case 'extension.middle_tier_tool_status':
        onReceivedExtensionMiddleTierToolStatus?.(message as ExtensionMiddleTierToolStatus);
        break;
      case 'response.audio.done':
        onReceivedResponseAudioDone?.(message);
        break;
      case 'error':
        onReceivedError?.(message as WebSocketErrorPayload);
        break;
      case 'conversation.item.created':
        onReceivedConversationItemCreated?.(message as ConversationItemCreated);
        break;
      case 'response.created':
        break;
      case 'rate_limits.updated':
        break;
      case 'session.created':
        break;
      case 'session.updated':
        break;
      case 'response.output_item.added':
        break;
      case 'response.output_item.done':
        break;
      case 'input_audio_buffer.cleared':
        break;
      case 'audio_transcript.done':
        break;
      case 'response.audio_transcript.done':
        break;
      case 'response.content_part.done':
        break;
      case 'response.content_part.added':
        break;
      default:
        // Handle unexpected message types
        // console.warn(`Unhandled message type: ${message.type}`);
        break;
    }
  };

  return {
    isConnected,
    readyState,
    startSession,
    addUserText,
    addUserImage,
    addUserAudio,
    addImageCaption,
    inputAudioBufferClear,
    responseCancel,
    closeSession,
    forceReconnect,
    addAssistantText,
    addFunctionCallOutput,
    syncConversationTranscript,
    conversationItemTruncate,
    conversationItemDelete,
  };
}
