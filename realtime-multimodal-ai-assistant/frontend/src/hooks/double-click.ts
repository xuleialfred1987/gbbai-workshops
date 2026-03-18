import React, { useRef, useCallback } from 'react';

// ----------------------------------------------------------------------

interface DoubleClickOptions {
  timeout?: number;
  click?: (e: React.SyntheticEvent) => void;
  doubleClick: (e: React.SyntheticEvent) => void;
}

export function useDoubleClick({ click, doubleClick, timeout = 250 }: DoubleClickOptions) {
  // Store the timer reference
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Handler function to manage click events
  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLElement>) => {
      // Clean up any existing timer
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }

      // Handle single click with delay
      if (click && event.detail === 1) {
        timerRef.current = setTimeout(() => {
          click(event);
          timerRef.current = null;
        }, timeout);
      }

      // Handle double click immediately
      if (event.detail !== 0 && event.detail % 2 === 0) {
        doubleClick(event);
      }
    },
    [doubleClick, click, timeout]
  );

  return handleClick;
}
