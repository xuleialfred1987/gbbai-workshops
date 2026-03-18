import { matchPath, useLocation } from 'react-router-dom';

// ----------------------------------------------------------------------

export function useCurrentLink(targetPath: string, useDeepComparison = true): boolean {
  const location = useLocation();

  if (!targetPath) {
    return false;
  }

  // For deep comparison, extract first 3 segments of the path
  const extractPathSegments = (path: string): string => {
    const segments = path.split('/').filter(Boolean);
    return `/${segments.slice(0, 3).join('/')}`;
  };

  const truncatedPath = useDeepComparison
    ? extractPathSegments(location.pathname)
    : location.pathname;

  return !!matchPath(
    {
      path: targetPath,
      end: false,
    },
    useDeepComparison ? truncatedPath : location.pathname
  );
}
