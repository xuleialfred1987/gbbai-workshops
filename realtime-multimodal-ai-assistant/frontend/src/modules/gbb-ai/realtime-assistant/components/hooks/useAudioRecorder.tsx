import { useRef, useState, useEffect, useCallback } from 'react';

import { Recorder } from '../audio/recorder';

// Optimized buffer size for real-time audio
// 1920 bytes = 960 samples = 40ms at 24kHz (better latency)
// Aligns with common WebRTC audio packet sizes
const BUFFER_SIZE = 1920;

type Parameters = {
  onAudioRecorded: (base64: string) => void;
  onError?: (error: Error) => void;
};

export default function useAudioRecorder({ onAudioRecorded, onError }: Parameters) {
  const audioRecorder = useRef<Recorder>();
  const [isInitialized, setIsInitialized] = useState(false);
  const bufferRef = useRef<Uint8Array>(new Uint8Array()); // Make buffer persistent across renders

  const appendToBuffer = useCallback((newData: Uint8Array) => {
    const buffer = bufferRef.current;
    const newBuffer = new Uint8Array(buffer.length + newData.length);
    newBuffer.set(buffer);
    newBuffer.set(newData, buffer.length);
    bufferRef.current = newBuffer;
  }, []);

  const handleAudioData = useCallback(
    (data: Iterable<number>) => {
      try {
        const uint8Array = new Uint8Array(data);
        appendToBuffer(uint8Array);

        const buffer = bufferRef.current;
        if (buffer.length >= BUFFER_SIZE) {
          const toSend = new Uint8Array(buffer.slice(0, BUFFER_SIZE));
          bufferRef.current = new Uint8Array(buffer.slice(BUFFER_SIZE));

          let binaryString = '';
          for (let i = 0; i < toSend.length; i += 1) {
            binaryString += String.fromCharCode(toSend[i]);
          }
          const base64 = btoa(binaryString);

          onAudioRecorded(base64);
        }
      } catch (error) {
        onError?.(error instanceof Error ? error : new Error('Unknown audio processing error'));
      }
    },
    [appendToBuffer, onAudioRecorded, onError]
  );

  useEffect(() => {
    if (audioRecorder.current) {
      audioRecorder.current.onDataAvailable = handleAudioData;
    }
  }, [handleAudioData]);

  const start = useCallback(async () => {
    try {
      // Check if browser supports required audio APIs
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Audio recording not supported in this browser');
      }

      if (!window.AudioContext) {
        throw new Error('AudioContext not supported in this browser');
      }

      if (!audioRecorder.current) {
        audioRecorder.current = new Recorder(handleAudioData);
      } else {
        audioRecorder.current.onDataAvailable = handleAudioData;
      }

      bufferRef.current = new Uint8Array();

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      await audioRecorder.current.start(stream);
      setIsInitialized(true);
      return true;
    } catch (error) {
      onError?.(error instanceof Error ? error : new Error('Failed to start recording'));
      return false;
    }
  }, [handleAudioData, onError]);

  const stop = useCallback(async () => {
    try {
      if (audioRecorder.current) {
        await audioRecorder.current.stop();
      }
      setIsInitialized(false);
      // Clear buffer on stop to prevent stale data
      bufferRef.current = new Uint8Array();
    } catch {
      // Silently handle stop errors
    }
  }, []);

  const setMuted = useCallback((muted: boolean) => {
    if (audioRecorder.current) {
      audioRecorder.current.setMuted(muted);
    }
  }, []);

  useEffect(
    () => () => {
      stop().catch(() => {
        // Ignore cleanup errors
      });
    },
    [stop]
  );

  return { start, stop, setMuted, isInitialized };
}
