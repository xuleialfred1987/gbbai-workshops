import { ReactNode } from 'react';
import type { Props as SimpleBarProps } from 'simplebar-react';

// mui
import type { Theme, SxProps } from '@mui/material/styles';

// ----------------------------------------------------------------------

// Extended interface for the Scroller component
export interface ScrollerProps extends SimpleBarProps {
  /**
   * Child elements to be rendered inside the Scroller
   */
  children?: ReactNode;

  /**
   * MUI system props for additional styling
   */
  sx?: SxProps<Theme>;
}
