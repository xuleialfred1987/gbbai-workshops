import { useScroll } from 'framer-motion';
import { useMemo, useState, useEffect, useCallback } from 'react';

// ----------------------------------------------------------------------

interface ScrollConfiguration extends Omit<ScrollOptions, 'container' | 'target'> {
  container?: React.RefObject<HTMLElement>;
  target?: React.RefObject<HTMLElement>;
}

/**
 * Hook that tracks whether scroll position exceeds a specific threshold
 *
 * @param top - Threshold value in pixels
 * @param options - Additional scroll configuration options
 * @returns Boolean indicating if scroll position exceeds threshold
 */
export function useOffSetTop(top = 0, options?: ScrollConfiguration): boolean {
  // Initialize state
  const [isExceedingThreshold, setIsExceedingThreshold] = useState<boolean>(false);

  // Get scroll position from framer-motion
  const { scrollY } = useScroll(options);

  // Handler for scroll position changes
  const handleScrollChange = useCallback(() => {
    const unsubscribe = scrollY.on('change', (currentScrollHeight) => {
      // Update state based on threshold comparison
      setIsExceedingThreshold(currentScrollHeight > top);
    });

    // Return cleanup function
    return () => unsubscribe();
  }, [scrollY, top]);

  // Setup scroll listener
  useEffect(() => {
    const cleanup = handleScrollChange();
    return cleanup;
  }, [handleScrollChange]);

  // Memoize result to prevent unnecessary re-renders
  return useMemo(() => isExceedingThreshold, [isExceedingThreshold]);
}
