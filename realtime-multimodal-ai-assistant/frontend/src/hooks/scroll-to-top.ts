import { useEffect, useCallback } from 'react';
import { useLocation } from 'react-router-dom';

// ----------------------------------------------------------------------

/**
 * Custom hook that automatically scrolls the window to the top
 * whenever the route pathname changes
 *
 * @returns void - This hook doesn't return any value
 */
export function useScrollToTop(): null {
  // Get the current location object from react-router
  const location = useLocation();

  // Extract only the pathname property
  const { pathname } = location;

  // Define scroll function
  const resetScrollPosition = useCallback(() => {
    try {
      // Reset scroll position to top left corner
      window.scrollTo({
        top: 0,
        left: 0,
        behavior: 'auto',
      });
    } catch (error) {
      // Fallback for older browsers
      window.scrollTo(0, 0);
    }
  }, []);

  // Run effect when pathname changes
  useEffect(() => {
    // Reset scroll position when route changes
    resetScrollPosition();

    // No cleanup needed
  }, [pathname, resetScrollPosition]);

  // Return null as this hook doesn't provide any values
  return null;
}
