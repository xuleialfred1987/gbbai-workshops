import type { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

// Global style overrides for MUI components
export function cssBaseline(theme: Theme) {
  const globalStyles = {
    MuiCssBaseline: {
      styleOverrides: {
        // Reset box model for all elements
        '*': {
          boxSizing: 'border-box',
        },

        // Document root styles
        html: {
          width: '100%',
          height: '100%',
          margin: 0,
          padding: 0,
          WebkitOverflowScrolling: 'touch',
        },

        // Body container styles
        body: {
          width: '100%',
          height: '100%',
          margin: 0,
          padding: 0,
        },

        // Application root containers
        '#root, #__next': {
          height: '100%',
          width: '100%',
        },

        // Image handling
        img: {
          display: 'inline-block',
          verticalAlign: 'bottom',
          maxWidth: '100%',
        },

        // Form element customizations
        input: {
          '&[type=number]': {
            MozAppearance: 'textfield',

            // Hide spinner buttons
            '&::-webkit-outer-spin-button, &::-webkit-inner-spin-button': {
              WebkitAppearance: 'none',
              margin: 0,
            },
          },
        },
      },
    },
  };

  return globalStyles;
}
