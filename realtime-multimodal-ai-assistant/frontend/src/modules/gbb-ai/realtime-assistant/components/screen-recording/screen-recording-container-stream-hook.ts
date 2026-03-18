import { useRef, useMemo, useState, useEffect, useCallback } from 'react';

import { convertBase64 } from 'src/api/gpt/api';

import { createRecordingOptions } from './screen-recording-container-types';
import type {
  RecordingOption,
  ScreenCaptureHandler,
  ScreenRecordingContainerProps,
} from './screen-recording-container-types';

// ----------------------------------------------------------------------

type UseScreenRecordingArgs = Pick<
  ScreenRecordingContainerProps,
  | 'initialOptionId'
  | 'initialStream'
  | 'isStreamPending'
  | 'onInitialOptionHandled'
  | 'onStreamAvailable'
  | 'onStreamEnded'
  | 'onScreenCapture'
  | 'messageCount'
>;

export function useScreenRecording({
  initialOptionId = null,
  initialStream = null,
  isStreamPending = false,
  onInitialOptionHandled,
  onStreamAvailable,
  onStreamEnded,
  onScreenCapture,
  messageCount = 0,
}: UseScreenRecordingArgs) {
  const recordingOptions = useMemo(createRecordingOptions, []);

  const [selectedOption, setSelectedOption] = useState<RecordingOption | null>(null);
  const [selectedStream, setSelectedStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const selectedStreamRef = useRef<MediaStream | null>(null);
  const selectedOptionRef = useRef<RecordingOption | null>(null);
  const previousMessageCountRef = useRef<number>(messageCount);
  const pendingCaptureRef = useRef(false);
  const captureTimeoutRef = useRef<number | null>(null);

  const onStreamAvailableRef = useRef(onStreamAvailable);
  const onStreamEndedRef = useRef(onStreamEnded);
  const onInitialOptionHandledRef = useRef(onInitialOptionHandled);
  const onScreenCaptureRef = useRef<ScreenCaptureHandler | undefined>(onScreenCapture);

  useEffect(() => {
    onStreamAvailableRef.current = onStreamAvailable;
    onStreamEndedRef.current = onStreamEnded;
    onInitialOptionHandledRef.current = onInitialOptionHandled;
    onScreenCaptureRef.current = onScreenCapture;
  });

  useEffect(() => {
    selectedOptionRef.current = selectedOption;
  }, [selectedOption]);

  const clearScheduledCapture = useCallback(() => {
    if (captureTimeoutRef.current !== null) {
      window.clearTimeout(captureTimeoutRef.current);
      captureTimeoutRef.current = null;
    }
  }, []);

  const isVideoReady = useCallback(
    (video: HTMLVideoElement) =>
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      video.videoWidth > 0 &&
      video.videoHeight > 0,
    []
  );

  useEffect(() => {
    selectedStreamRef.current = selectedStream;
    if (!selectedStream && videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (!selectedStream) {
      pendingCaptureRef.current = false;
      clearScheduledCapture();
    }
  }, [clearScheduledCapture, selectedStream]);

  const captureFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !onScreenCaptureRef.current || !isVideoReady(video)) {
      return false;
    }

    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return false;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const rawDataUrl = canvas.toDataURL('image/jpeg', 0.85);
      const optimizedDataUrl = (await convertBase64(rawDataUrl, 640, 0.8)) as string;
      const base64Image = optimizedDataUrl.includes(',')
        ? optimizedDataUrl.split(',')[1]
        : optimizedDataUrl;

      if (!base64Image) {
        return false;
      }

      onScreenCaptureRef.current(base64Image, {
        optionId: selectedOptionRef.current?.id,
        label: selectedOptionRef.current?.label,
      });

      return true;
    } catch (captureError) {
      console.error('Screen capture failed:', captureError);
      return false;
    }
  }, [isVideoReady]);

  const scheduleCapture = useCallback(
    (delay = 0) => {
      clearScheduledCapture();

      captureTimeoutRef.current = window.setTimeout(async () => {
        captureTimeoutRef.current = null;

        if (!pendingCaptureRef.current || !selectedStreamRef.current) {
          return;
        }

        const didCapture = await captureFrame();
        if (didCapture) {
          pendingCaptureRef.current = false;
          return;
        }

        if (pendingCaptureRef.current && selectedStreamRef.current) {
          scheduleCapture(100);
        }
      }, delay);
    },
    [captureFrame, clearScheduledCapture]
  );

  useEffect(() => {
    if (!selectedStreamRef.current) {
      return undefined;
    }

    pendingCaptureRef.current = true;
    scheduleCapture();

    return undefined;
  }, [scheduleCapture, selectedStream]);

  // Capture a frame when message count changes
  useEffect(() => {
    if (messageCount === previousMessageCountRef.current) {
      return undefined;
    }

    previousMessageCountRef.current = messageCount;

    if (!selectedStreamRef.current) {
      return undefined;
    }

    pendingCaptureRef.current = true;
    scheduleCapture(300);

    return () => {
      pendingCaptureRef.current = false;
      clearScheduledCapture();
    };
  }, [clearScheduledCapture, messageCount, scheduleCapture]);

  const setupVideoEvents = useCallback((videoElement: HTMLVideoElement) => {
    const handleVideoReady = () => {
      if (pendingCaptureRef.current) {
        scheduleCapture();
      }
    };

    videoElement.onloadedmetadata = handleVideoReady;
    videoElement.oncanplay = handleVideoReady;
    videoElement.onplaying = handleVideoReady;
  }, [scheduleCapture]);

  const assignVideoRef = useCallback(
    (node: HTMLVideoElement | null) => {
      videoRef.current = node;

      if (node && selectedStreamRef.current && node.srcObject !== selectedStreamRef.current) {
        node.srcObject = selectedStreamRef.current;
        node.play().catch(() => undefined);
        setupVideoEvents(node);
      }
    },
    [setupVideoEvents]
  );

  const stopStream = useCallback(() => {
    pendingCaptureRef.current = false;
    clearScheduledCapture();

    const stream = selectedStreamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      selectedStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.onloadedmetadata = null;
      videoRef.current.oncanplay = null;
      videoRef.current.onplaying = null;
    }

    setSelectedStream(null);
    setSelectedOption(null);
    onStreamEndedRef.current?.();
  }, [clearScheduledCapture]);

  useEffect(() => {
    if (!initialOptionId) {
      if (!isStreamPending && !selectedStreamRef.current) {
        setSelectedOption(null);
        setSelectedStream(null);
      }
      return;
    }

    const option = recordingOptions.find((item) => item.id === initialOptionId) ?? null;
    setSelectedOption(option);

    if (initialStream) {
      setError(null);
      setSelectedStream(initialStream);
      selectedStreamRef.current = initialStream;
      onStreamAvailableRef.current?.(initialStream, option?.label ?? 'Screen share');
      onInitialOptionHandledRef.current?.();
      return;
    }

    if (isStreamPending) {
      setError(null);
      setSelectedStream(null);
      selectedStreamRef.current = null;
      return;
    }

    if (!option) {
      onInitialOptionHandledRef.current?.();
      return;
    }

    if (selectedStreamRef.current) {
      onInitialOptionHandledRef.current?.();
      return;
    }

    setSelectedOption(null);
    setSelectedStream(null);
    selectedStreamRef.current = null;
    onInitialOptionHandledRef.current?.();
  }, [initialOptionId, initialStream, isStreamPending, recordingOptions]);

  useEffect(() => {
    const videoElement = videoRef.current;
    const stream = selectedStreamRef.current;

    if (!videoElement || !stream) {
      return () => {};
    }

    videoElement.srcObject = stream;
    videoElement.play().catch(() => undefined);
    setupVideoEvents(videoElement);

    const handleEnded = () => {
      stopStream();
    };

    const videoTracks = stream.getVideoTracks();
    videoTracks.forEach((track) => {
      track.addEventListener('ended', handleEnded, { once: true });
    });

    return () => {
      videoTracks.forEach((track) => {
        track.removeEventListener('ended', handleEnded);
      });
    };
  }, [setupVideoEvents, stopStream, selectedStream]);

  useEffect(
    () => () => {
      stopStream();
    },
    [stopStream]
  );

  const isLoadingStream = isStreamPending && !selectedStream;

  return {
    recordingOptions,
    selectedOption,
    selectedStream,
    error,
    setError,
    isLoadingStream,
    assignVideoRef,
    stopStream,
  };
}
