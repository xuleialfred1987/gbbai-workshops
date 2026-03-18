import { useMemo } from 'react';
import { useSearchParams as useReactRouterSearchParams } from 'react-router-dom';

// ----------------------------------------------------------------------

/**
 * Provides a memoized version of search parameters from the URL
 * @returns URLSearchParams object that stays consistent between renders
 */
export function useSearchParams(): URLSearchParams {
  // Extract the search parameters from React Router
  const [urlQueryParameters] = useReactRouterSearchParams();

  // Optimize performance by preventing unnecessary re-renders
  const memoizedParameters = useMemo(() => urlQueryParameters, [urlQueryParameters]);

  return memoizedParameters;
}
