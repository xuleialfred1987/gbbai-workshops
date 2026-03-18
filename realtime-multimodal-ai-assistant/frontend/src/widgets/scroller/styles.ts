import SimpleBar from 'simplebar-react';

// mui
import { alpha, styled as muiStyled } from '@mui/material/styles';

// ----------------------------------------------------------------------

// Container wrapper for Scroller component
const ScrollerContainer = muiStyled('div')(() => ({
  flexGrow: 1,
  overflow: 'hidden',
  height: '100%',
  position: 'relative',
}));

// Enhanced SimpleBar with custom styling
const CustomScroller = muiStyled(SimpleBar)(({ theme }) => ({
  maxHeight: '100%',

  // Style the Scroller handle
  '& .simplebar-Scroller': {
    '&::before': {
      background: alpha(theme.palette.grey[600], 0.48),
      borderRadius: '6px',
    },

    // Style the Scroller when visible
    '&.simplebar-visible::before': {
      opacity: 1,
    },
  },

  // Ensure proper z-index
  '& .simplebar-mask': {
    zIndex: 'inherit',
  },
}));

export const StyledRootScroller = ScrollerContainer;
export const StyledScroller = CustomScroller;
