import { useRef, useState, useEffect, useCallback } from 'react';

import type { ResizeHandle } from './camera-container-types';
import { MIN_WIDTH, MIN_HEIGHT, DEFAULT_SIZE, DEFAULT_POSITION } from './camera-container-types';

// ----------------------------------------------------------------------

export function useFloatingLayout(isFloating: boolean) {
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [floatingPosition, setFloatingPosition] = useState(DEFAULT_POSITION);
  const [floatingSize, setFloatingSize] = useState(DEFAULT_SIZE);

  const floatingPositionRef = useRef(floatingPosition);
  const floatingSizeRef = useRef(floatingSize);
  const dragStateRef = useRef<{ pointerId: number; offsetX: number; offsetY: number } | null>(null);
  const resizeStateRef = useRef<{
    handle: ResizeHandle;
    pointerId: number;
    startX: number;
    startY: number;
    startWidth: number;
    startHeight: number;
    startLeft: number;
    startTop: number;
  } | null>(null);

  useEffect(() => {
    floatingPositionRef.current = floatingPosition;
    floatingSizeRef.current = floatingSize;
  }, [floatingPosition, floatingSize]);

  const clamp = useCallback((value: number, min: number, max: number) => {
    if (Number.isNaN(value)) {
      return min;
    }
    return Math.min(Math.max(value, min), max);
  }, []);

  const resetFloatingLayout = useCallback(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const margin = 16;
    const maxLeft = Math.max(margin, window.innerWidth - DEFAULT_SIZE.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - DEFAULT_SIZE.height - margin);

    setFloatingSize((prev) => {
      if (prev.width === DEFAULT_SIZE.width && prev.height === DEFAULT_SIZE.height) {
        return prev;
      }
      return DEFAULT_SIZE;
    });

    setFloatingPosition({
      left: clamp(DEFAULT_POSITION.left, margin, maxLeft),
      top: clamp(DEFAULT_POSITION.top, margin, maxTop),
    });
  }, [clamp]);

  useEffect(() => {
    if (!isFloating) {
      setIsDragging(false);
      setIsResizing(false);
      dragStateRef.current = null;
      resizeStateRef.current = null;
    }
  }, [isFloating]);

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!isFloating) {
        return;
      }

      const target = event.target as HTMLElement;
      const handleType = target.getAttribute('data-resize-handle') as ResizeHandle | null;

      if (handleType) {
        setIsResizing(true);
        resizeStateRef.current = {
          handle: handleType,
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          startWidth: floatingSizeRef.current.width,
          startHeight: floatingSizeRef.current.height,
          startLeft: floatingPositionRef.current.left,
          startTop: floatingPositionRef.current.top,
        };
        (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
      }
    },
    [isFloating]
  );

  const handleHeaderPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!isFloating || event.button !== 0) {
        return;
      }

      const target = event.target as HTMLElement | null;
      if (target?.closest('button, [role="button"], a, input, textarea')) {
        return;
      }

      setIsDragging(true);
      dragStateRef.current = {
        pointerId: event.pointerId,
        offsetX: event.clientX - floatingPositionRef.current.left,
        offsetY: event.clientY - floatingPositionRef.current.top,
      };
      (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
      event.preventDefault();
    },
    [isFloating]
  );

  // Pointer move and resize logic
  useEffect(() => {
    if (!isFloating || (!isDragging && !isResizing)) {
      return undefined;
    }

    const handlePointerMove = (event: PointerEvent) => {
      if (dragStateRef.current) {
        const { offsetX, offsetY } = dragStateRef.current;
        const currentSize = floatingSizeRef.current;
        const margin = 16;
        const maxLeft = window.innerWidth - currentSize.width - margin;
        const maxTop = window.innerHeight - currentSize.height - margin;
        const nextLeft = clamp(event.clientX - offsetX, margin, Math.max(margin, maxLeft));
        const nextTop = clamp(event.clientY - offsetY, margin, Math.max(margin, maxTop));

        setFloatingPosition((prev) => {
          if (prev.left === nextLeft && prev.top === nextTop) {
            return prev;
          }
          return { left: nextLeft, top: nextTop };
        });
      } else if (resizeStateRef.current) {
        const { handle, startX, startY, startWidth, startHeight, startLeft, startTop } =
          resizeStateRef.current;
        const deltaX = event.clientX - startX;
        const deltaY = event.clientY - startY;
        const margin = 16;
        const maxHeight = Math.max(MIN_HEIGHT, window.innerHeight - margin - startTop);

        let nextWidth = startWidth;
        let nextLeft = startLeft;

        if (handle === 'bottom-right') {
          const maxWidth = Math.max(MIN_WIDTH, window.innerWidth - margin - startLeft);
          nextWidth = clamp(startWidth + deltaX, MIN_WIDTH, maxWidth);
        } else if (handle === 'bottom-left') {
          const rightEdge = startLeft + startWidth;
          const maxLeft = Math.max(margin, rightEdge - MIN_WIDTH);
          const rawLeft = clamp(startLeft + deltaX, margin, maxLeft);
          const maxWidthFromLeft = Math.max(MIN_WIDTH, window.innerWidth - margin - rawLeft);
          const widthFromRight = rightEdge - rawLeft;
          nextWidth = clamp(widthFromRight, MIN_WIDTH, maxWidthFromLeft);
          nextLeft = rightEdge - nextWidth;
        }

        const nextHeight = clamp(startHeight + deltaY, MIN_HEIGHT, maxHeight);

        setFloatingSize((prev) => {
          if (prev.width === nextWidth && prev.height === nextHeight) {
            return prev;
          }
          return { width: nextWidth, height: nextHeight };
        });

        if (handle === 'bottom-left') {
          setFloatingPosition((prev) => {
            if (prev.left === nextLeft) {
              return prev;
            }
            return { ...prev, left: nextLeft };
          });
        }
      }
    };

    const handlePointerUp = () => {
      setIsDragging(false);
      setIsResizing(false);
      dragStateRef.current = null;
      resizeStateRef.current = null;
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp, { once: false });

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [clamp, isDragging, isFloating, isResizing]);

  // Window resize handling
  useEffect(() => {
    if (!isFloating) {
      return undefined;
    }

    const margin = 16;

    const handleWindowResize = () => {
      setFloatingPosition((prev) => {
        const currentSize = floatingSizeRef.current;
        const maxLeft = Math.max(margin, window.innerWidth - currentSize.width - margin);
        const maxTop = Math.max(margin, window.innerHeight - currentSize.height - margin);
        return {
          left: clamp(prev.left, margin, maxLeft),
          top: clamp(prev.top, margin, maxTop),
        };
      });

      setFloatingSize((prev) => {
        const maxWidth = Math.max(
          MIN_WIDTH,
          window.innerWidth - floatingPositionRef.current.left - margin
        );
        const maxHeight = Math.max(
          MIN_HEIGHT,
          window.innerHeight - floatingPositionRef.current.top - margin
        );
        return {
          width: clamp(prev.width, MIN_WIDTH, maxWidth),
          height: clamp(prev.height, MIN_HEIGHT, maxHeight),
        };
      });
    };

    window.addEventListener('resize', handleWindowResize);
    handleWindowResize();

    return () => {
      window.removeEventListener('resize', handleWindowResize);
    };
  }, [clamp, isFloating]);

  return {
    isDragging,
    isResizing,
    floatingPosition,
    floatingSize,
    resetFloatingLayout,
    handlePointerDown,
    handleHeaderPointerDown,
  };
}
