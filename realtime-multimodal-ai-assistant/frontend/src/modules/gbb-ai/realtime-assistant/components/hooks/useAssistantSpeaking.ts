import { useRef, useState, useEffect, useCallback } from 'react';

export default function useAssistantSpeaking(isPlaying: boolean) {
  const [isAssistantSpeaking, setIsAssistantSpeaking] = useState(false);
  const assistantSpeakingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isPlayingRef = useRef(false);
  const manuallyStoppedRef = useRef(false);

  const setAssistantSpeaking = useCallback((shouldSpeak: boolean, graceMs = 0) => {
    if (assistantSpeakingTimeoutRef.current) {
      clearTimeout(assistantSpeakingTimeoutRef.current);
      assistantSpeakingTimeoutRef.current = null;
    }

    if (shouldSpeak) {
      setIsAssistantSpeaking(true);
      return;
    }

    if (graceMs > 0) {
      assistantSpeakingTimeoutRef.current = setTimeout(() => {
        if (!isPlayingRef.current) {
          setIsAssistantSpeaking(false);
        }
        assistantSpeakingTimeoutRef.current = null;
      }, graceMs);
      return;
    }

    setIsAssistantSpeaking(false);
  }, []);

  useEffect(
    () => () => {
      if (assistantSpeakingTimeoutRef.current) {
        clearTimeout(assistantSpeakingTimeoutRef.current);
        assistantSpeakingTimeoutRef.current = null;
      }
    },
    []
  );

  useEffect(() => {
    isPlayingRef.current = isPlaying;

    if (isPlaying) {
      setIsAssistantSpeaking(true);
      return;
    }

    if (!assistantSpeakingTimeoutRef.current) {
      setIsAssistantSpeaking(false);
    }
  }, [isPlaying, setAssistantSpeaking]);

  return {
    isAssistantSpeaking,
    manuallyStoppedRef,
    setAssistantSpeaking,
  };
}