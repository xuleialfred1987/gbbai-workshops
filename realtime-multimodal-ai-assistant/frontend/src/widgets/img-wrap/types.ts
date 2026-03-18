import type { LazyLoadImageProps } from 'react-lazy-load-image-component';

// mui
import type { BoxProps } from '@mui/material/Box';

// ----------------------------------------------------------------------

// Map of supported aspect ratios
const SUPPORTED_RATIOS = {
  STANDARD: '4/3',
  PORTRAIT: '3/4',
  WIDE: '6/4',
  TALL: '4/6',
  WIDESCREEN: '16/9',
  VERTICAL: '9/16',
  ULTRAWIDE: '18/9',
  ULTRATALL: '9/18',
  CINEMATIC: '21/9',
  SUPERVERTICAL: '9/21',
  SQUARE: '1/1',
} as const;

// Create union type from the values of SUPPORTED_RATIOS
export type ImageRatio = (typeof SUPPORTED_RATIOS)[keyof typeof SUPPORTED_RATIOS];

// Main component props interface
export type ImageBlockProps = BoxProps &
  LazyLoadImageProps & {
    overlay?: string;
    ratio?: ImageRatio;
    disabledEffect?: boolean;
  };

// Export constants for external use
export const aspectRatios = SUPPORTED_RATIOS;
