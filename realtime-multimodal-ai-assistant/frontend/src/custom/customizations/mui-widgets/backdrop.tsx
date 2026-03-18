import { alpha, Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

const createBackdropStyles = (theme: Theme) => ({
  root: {
    background: alpha(theme.palette.common.black, 0.8),
  },
  invisible: {
    backgroundColor: 'transparent',
  },
});

export const backdrop = (theme: Theme) => ({
  MuiBackdrop: {
    styleOverrides: createBackdropStyles(theme),
  },
});
