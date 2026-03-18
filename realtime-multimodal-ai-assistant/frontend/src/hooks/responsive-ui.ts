// mui
import useMediaQuery from '@mui/material/useMediaQuery';
import { useTheme, Breakpoint } from '@mui/material/styles';

// ----------------------------------------------------------------------

export type Query = 'up' | 'down' | 'between' | 'only';
export type Value = Breakpoint | number;
type ReturnType = boolean;

// ----------------------------------------------------------------------

/**
 * Hook for responsive UI queries
 */
export function useResponsiveUI(query: Query, start?: Value, end?: Value): ReturnType {
  const theme = useTheme();

  // Always call all hooks unconditionally
  const upMatch = useMediaQuery(theme.breakpoints.up(start as Value));
  const downMatch = useMediaQuery(theme.breakpoints.down(start as Value));
  const betweenMatch = useMediaQuery(theme.breakpoints.between(start as Value, end as Value));
  const onlyMatch = useMediaQuery(theme.breakpoints.only(start as Breakpoint));

  // Return appropriate value based on query type
  switch (query) {
    case 'up':
      return upMatch;
    case 'down':
      return downMatch;
    case 'between':
      return betweenMatch;
    case 'only':
    default:
      return onlyMatch;
  }
}

/**
 * Hook to determine current breakpoint width
 */
export function useWidth(): Breakpoint {
  const theme = useTheme();

  // Define media queries for all breakpoints
  const xl = useMediaQuery(theme.breakpoints.up('xl'));
  const lg = useMediaQuery(theme.breakpoints.up('lg'));
  const md = useMediaQuery(theme.breakpoints.up('md'));
  const sm = useMediaQuery(theme.breakpoints.up('sm'));

  // Return the first matching breakpoint
  if (xl) return 'xl';
  if (lg) return 'lg';
  if (md) return 'md';
  if (sm) return 'sm';
  return 'xs';
}
