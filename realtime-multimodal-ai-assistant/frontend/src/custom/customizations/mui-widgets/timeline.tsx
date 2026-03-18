import type { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export const timeline = (theme: Theme) => ({
  MuiTimelineDot: {
    styleOverrides: {
      root: {
        // Remove default shadow effect
        boxShadow: 'none',
      },
    },
  },

  MuiTimelineConnector: {
    styleOverrides: {
      root: {
        // Use theme's divider color for connector
        backgroundColor: theme.palette.divider,
      },
    },
  },
});
