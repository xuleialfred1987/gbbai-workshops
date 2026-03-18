import { useMemo } from 'react';
import { useLocation } from 'react-router-dom';

// ----------------------------------------------------------------------

/**
 * Custom hook that tracks and returns the current route path
 * @returns The current pathname from the URL
 */
export function useRoutePath(): string {
  const { pathname } = useLocation();

  // Memoize the pathname to avoid unnecessary re-renders
  const currentPath = useMemo(() => pathname, [pathname]);

  return currentPath;
}
