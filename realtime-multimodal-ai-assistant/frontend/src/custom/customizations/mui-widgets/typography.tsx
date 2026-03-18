import type { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

/**
 * Creates typography-related style overrides for Material UI components
 */
interface TypographyOverrides {
  MuiTypography: {
    styleOverrides: {
      paragraph: Record<string, unknown>;
      gutterBottom: Record<string, unknown>;
    };
  };
}

/**
 * Generates typography style customizations based on the provided theme
 * @param theme - The MUI theme object
 * @returns Typography style overrides object
 */
export function typography(theme: Theme): TypographyOverrides {
  return {
    MuiTypography: {
      styleOverrides: {
        paragraph: {
          marginBottom: theme.spacing(2),
        },
        gutterBottom: {
          marginBottom: theme.spacing(1),
        },
      },
    },
  };
}
