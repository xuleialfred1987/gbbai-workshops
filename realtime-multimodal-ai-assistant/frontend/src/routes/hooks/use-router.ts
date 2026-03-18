import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';

// ----------------------------------------------------------------------

/**
 * Provides a set of navigation utilities for React Router
 * @returns Navigation utility object
 */
export function useRouter(): {
  back: () => void;
  forward: () => void;
  reload: () => void;
  push: (path: string) => void;
  replace: (path: string) => void;
} {
  const navigateFunction = useNavigate();

  const navigationControls = useMemo(
    () => ({
      back: (): void => navigateFunction(-1),
      forward: (): void => navigateFunction(1),
      reload: (): void => window.location.reload(),
      push: (path: string): void => navigateFunction(path),
      replace: (path: string): void => navigateFunction(path, { replace: true }),
    }),
    [navigateFunction]
  );

  return navigationControls;
}
