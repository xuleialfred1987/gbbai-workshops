import { useRef, useState, useEffect, useCallback } from 'react';

import { Player } from '../audio/player';

const SAMPLE_RATE = 24000;

export default function useAudioPlayer() {
  const isMounted = useRef(true);
  const audioPlayer = useRef<Player>();
  const [isPlaying, setIsPlaying] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const playbackTimerRef = useRef<NodeJS.Timeout | null>(null);
  const playbackEndTimeRef = useRef<number>(0); // Track when playback should end
  const effectiveSampleRateRef = useRef<number>(SAMPLE_RATE);
  // Internal readiness + queue management
  const playerReadyRef = useRef(false);
  const initPromiseRef = useRef<Promise<boolean> | null>(null);
  const chunkQueueRef = useRef<Int16Array[]>([]);
  const flushingRef = useRef(false);

  // ------------------------------------------------------------
  // mark unmount
  // ------------------------------------------------------------
  useEffect(
    () => () => {
      isMounted.current = false;
      if (playbackTimerRef.current) clearTimeout(playbackTimerRef.current);
    },
    []
  );

  // helper that is safe after unmount
  const safeSetIsPlaying = useCallback((v: boolean) => {
    if (isMounted.current) setIsPlaying(v);
  }, []);

  // ------------------------------------------------------------
  // initialisation
  // ------------------------------------------------------------
  const initAudioPlayer = useCallback(async (force = false): Promise<boolean> => {
    // Fast path (reuse existing ready instance unless forcing re-init)
    if (!force && playerReadyRef.current && audioPlayer.current) {
      // Attempt resume if context got suspended by browser policy
      try {
        const ctx: AudioContext | undefined = (audioPlayer.current as any)?.audioContext;
        if (ctx && ctx.state === 'suspended') {
          await ctx.resume();
        }
      } catch {
        /* ignore */
      }
      return true;
    }
    // Reuse in‑flight init
    if (initPromiseRef.current) return initPromiseRef.current;

    initPromiseRef.current = (async () => {
      try {
        if (!window.AudioContext) {
          console.warn('AudioContext not supported');
          return false;
        }

        if (force || !audioPlayer.current) {
          if (audioPlayer.current) {
            try {
              audioPlayer.current.stop();
            } catch {
              /* ignore */
            }
          }
          audioPlayer.current = new Player();
          await audioPlayer.current.init(SAMPLE_RATE);
        }

        // Lightweight readiness check (single tick + property presence)
        await new Promise((r) => setTimeout(r, 10));
        const hasCtx = !!(audioPlayer.current as any)?.audioContext;
        const hasNode = !!(audioPlayer.current as any)?.playbackNode;
        if (!hasCtx || !hasNode) {
          console.warn('Player internal nodes not ready yet, soft retry');
          // one retry after 40ms
          await new Promise((r) => setTimeout(r, 40));
        }

        const finalCtx = !!(audioPlayer.current as any)?.audioContext;
        const finalNode = !!(audioPlayer.current as any)?.playbackNode;
        if (!finalCtx || !finalNode) {
          console.error('Player readiness check failed');
          return false;
        }

        const actualSampleRate = audioPlayer.current?.getSampleRate?.() ?? SAMPLE_RATE;
        effectiveSampleRateRef.current = actualSampleRate || SAMPLE_RATE;
        playerReadyRef.current = true;
        if (isMounted.current) setIsInitialized(true);
        return true;
      } catch (e) {
        console.error('Player initialization failed:', e);
        audioPlayer.current = undefined;
        playerReadyRef.current = false;
        if (isMounted.current) setIsInitialized(false);
        return false;
      } finally {
        initPromiseRef.current = null;
      }
    })();

    return initPromiseRef.current;
  }, []);

  // Flush queued PCM chunks once player ready
  const flushQueue = useCallback(() => {
    if (flushingRef.current) return;
    if (!playerReadyRef.current || !audioPlayer.current) return;
    if (chunkQueueRef.current.length === 0) return;
    flushingRef.current = true;

    const queue = chunkQueueRef.current;
    chunkQueueRef.current = [];

    let totalSamples = 0;
    queue.forEach((pcm) => {
      try {
        audioPlayer.current?.play(pcm);
        totalSamples += pcm.length;
      } catch (e) {
        console.error('Failed to play queued chunk', e);
      }
    });

    if (totalSamples > 0) {
      const sampleRate = effectiveSampleRateRef.current || SAMPLE_RATE;
      const durationMs = (totalSamples * 1000) / sampleRate;
      const now = Date.now();

      // Calculate the new end time by extending from current end time or now
      const newEndTime = Math.max(now, playbackEndTimeRef.current) + durationMs;
      playbackEndTimeRef.current = newEndTime;

      // Set playing state
      safeSetIsPlaying(true);

      // Clear existing timer and set a new one for the accumulated end time
      if (playbackTimerRef.current) clearTimeout(playbackTimerRef.current);
      const timeUntilEnd = newEndTime - now;
      playbackTimerRef.current = setTimeout(() => {
        safeSetIsPlaying(false);
        playbackEndTimeRef.current = 0;
      }, timeUntilEnd);
    }

    flushingRef.current = false;
    // In case new chunks arrived during flush
    if (chunkQueueRef.current.length > 0) flushQueue();
  }, [safeSetIsPlaying]);

  // run once
  useEffect(() => {
    initAudioPlayer().catch(() => {
      // Audio player initialization failed - will retry on first play
    });

    return () => {
      if (audioPlayer.current) {
        try {
          audioPlayer.current.stop();
        } catch {
          // Ignore cleanup errors
        }
      }
    };
  }, [initAudioPlayer]);

  // ------------------------------------------------------------
  // reset
  // ------------------------------------------------------------
  const reset = useCallback(async () => {
    try {
      if (playbackTimerRef.current) {
        clearTimeout(playbackTimerRef.current);
        playbackTimerRef.current = null;
      }
      chunkQueueRef.current = [];
      // Don't destroy the player, just ensure it's ready for the next stream
      await initAudioPlayer();
      if (isMounted.current) setIsPlaying(false);
    } catch {
      if (isMounted.current) setIsInitialized(false);
    }
  }, [initAudioPlayer]);

  // ------------------------------------------------------------
  // play
  // ------------------------------------------------------------
  const play = useCallback(
    async (base64: string) => {
      try {
        // Decode first (fast) and enqueue
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i += 1) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const pcm = new Int16Array(bytes.buffer);
        chunkQueueRef.current.push(pcm);

        // Kick off init (only once) and then flush
        const ready = await initAudioPlayer();
        if (!ready) {
          console.warn('Audio player not ready yet; chunk queued.');
          return;
        }
        flushQueue();
      } catch (error) {
        console.error('Error in play function:', error);
      }
    },
    [initAudioPlayer, flushQueue]
  );

  // ------------------------------------------------------------
  // stop
  // ------------------------------------------------------------
  const stop = useCallback(async (): Promise<void> => {
    try {
      if (playbackTimerRef.current) {
        clearTimeout(playbackTimerRef.current);
        playbackTimerRef.current = null;
      }
      chunkQueueRef.current = [];
      flushingRef.current = false;
      if (audioPlayer.current) {
        audioPlayer.current.stop();
      }
      safeSetIsPlaying(false);
    } catch {
      // Silently handle stop errors
    }
  }, [safeSetIsPlaying]);

  return { reset, play, stop, isPlaying, isInitialized };
}
