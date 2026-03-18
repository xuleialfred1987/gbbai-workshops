import { useRef, useState, useEffect, useCallback, type MutableRefObject } from 'react';

import { convertBase64 } from 'src/api/gpt/api';

import type { VideoRef } from './camera-container-types';

// ----------------------------------------------------------------------

export function useCameraStream(
  facingMode: 'user' | 'environment',
  videoRef: VideoRef,
  onStreamStarted?: (stream: MediaStream) => void,
  onStreamStopped?: () => void,
  onCameraCapture?: (base64Image: string) => void,
  messageCount: number = 0
) {
  const localVideoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onCameraCaptureRef = useRef(onCameraCapture);
  const onStreamStoppedRef = useRef(onStreamStopped);
  const onStreamStartedRef = useRef(onStreamStarted);
  const previousMessageCountRef = useRef<number>(messageCount);
  const pendingCaptureRef = useRef(false);
  const captureTimeoutRef = useRef<number | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Update refs to avoid re-renders
  useEffect(() => {
    onCameraCaptureRef.current = onCameraCapture;
    onStreamStoppedRef.current = onStreamStopped;
    onStreamStartedRef.current = onStreamStarted;
  });

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

  const captureFrame = useCallback(async () => {
    if (!localVideoRef.current || !onCameraCaptureRef.current) {
      return false;
    }

    const video = localVideoRef.current;
    if (!isVideoReady(video)) {
      return false;
    }

    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      const ctx = canvas.getContext('2d');
      if (!ctx || !isVideoReady(video)) {
        return false;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const rawDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      const optimizedDataUrl = (await convertBase64(rawDataUrl, 640, 0.6)) as string;
      const base64Image = optimizedDataUrl.includes(',')
        ? optimizedDataUrl.split(',')[1]
        : optimizedDataUrl;

      if (!base64Image) {
        return false;
      }

      onCameraCaptureRef.current(base64Image);

      return true;
    } catch (captureError) {
      console.error('Camera capture failed:', captureError);
      return false;
    }
  }, [isVideoReady]);

  const scheduleCapture = useCallback(
    (delay = 0) => {
      clearScheduledCapture();

      captureTimeoutRef.current = window.setTimeout(async () => {
        captureTimeoutRef.current = null;

        if (!pendingCaptureRef.current || !streamRef.current) {
          return;
        }

        const didCapture = await captureFrame();
        if (didCapture) {
          pendingCaptureRef.current = false;
          return;
        }

        if (pendingCaptureRef.current && streamRef.current) {
          scheduleCapture(100);
        }
      }, delay);
    },
    [captureFrame, clearScheduledCapture]
  );

  // Capture a frame when message count changes
  useEffect(() => {
    if (messageCount === previousMessageCountRef.current) {
      return undefined;
    }

    previousMessageCountRef.current = messageCount;

    if (!streamRef.current) {
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
      localVideoRef.current = node;

      if (node && streamRef.current && node.srcObject !== streamRef.current) {
        node.srcObject = streamRef.current;
        node.play().catch(() => undefined);
        setupVideoEvents(node);
      }

      // Handle external videoRef prop
      if (videoRef) {
        if (typeof videoRef === 'function') {
          videoRef(node);
        } else if ('current' in videoRef) {
          (videoRef as MutableRefObject<HTMLVideoElement | null>).current = node;
        }
      }
    },
    [videoRef, setupVideoEvents]
  );

  const stopStream = useCallback(() => {
    pendingCaptureRef.current = false;
    clearScheduledCapture();

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      onStreamStoppedRef.current?.();
    }

    if (localVideoRef.current) {
      localVideoRef.current.srcObject = null;
      localVideoRef.current.onloadedmetadata = null;
      localVideoRef.current.oncanplay = null;
      localVideoRef.current.onplaying = null;
    }
  }, [clearScheduledCapture]);

  const startStream = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not supported in this browser.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Clean up existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: true,
      });

      streamRef.current = stream;

      // If video element exists, assign stream and setup events
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
        localVideoRef.current.play().catch(() => undefined);
        setupVideoEvents(localVideoRef.current);
      }

      pendingCaptureRef.current = true;
      scheduleCapture();

      onStreamStartedRef.current?.(stream);
    } catch (err) {
      console.error('Camera access failed:', err);
      setError('Could not access the camera. Please check permissions.');
    } finally {
      setIsLoading(false);
    }
  }, [facingMode, scheduleCapture, setupVideoEvents]);

  useEffect(() => {
    startStream();
    return () => {
      stopStream();
    };
  }, [facingMode, startStream, stopStream]);

  return {
    assignVideoRef,
    isLoading,
    error,
    startStream,
    stopStream,
  };
}
