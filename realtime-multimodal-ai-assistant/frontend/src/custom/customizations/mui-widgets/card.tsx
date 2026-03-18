import { alpha, Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function card(theme: Theme) {
  const isLightTheme = theme.palette.mode === 'light';

  // Shadow configuration
  const shadowBaseColor = isLightTheme ? theme.palette.grey[400] : theme.palette.common.black;

  // Component customizations
  return {
    // Card component styling
    MuiCard: {
      styleOverrides: {
        root: {
          position: 'relative',
          zIndex: 0,
          borderRadius: Number(theme.shape.borderRadius) * 1.25,
          boxShadow: generateElevationShadow(shadowBaseColor),
        },
      },
    },

    // Card header styling
    MuiCardHeader: {
      styleOverrides: {
        root: applySpacing(theme, 3, 3, 0),
      },
    },

    // Card content styling
    MuiCardContent: {
      styleOverrides: {
        root: applySpacing(theme, 3),
      },
    },
  };
}

/**
 * Creates a custom shadow effect with specified color
 */
function generateElevationShadow(baseColor: string): string {
  const ambientShadow = alpha(baseColor, 0.32);
  const directionalShadow = alpha(baseColor, 0.14);

  return [`0 0 3px 0 ${ambientShadow}`, `0 2px 6px -3px ${directionalShadow}`].join(', ');
}

/**
 * Helper to apply consistent spacing to components
 */
function applySpacing(
  theme: Theme,
  vertical: number,
  horizontalLeft?: number,
  horizontalRight?: number
): object {
  if (horizontalLeft === undefined) {
    // Apply equal spacing on all sides
    return { padding: theme.spacing(vertical) };
  }

  if (horizontalRight === undefined) {
    // Apply equal horizontal spacing
    return { padding: theme.spacing(vertical, horizontalLeft) };
  }

  // Apply custom spacing pattern
  return { padding: theme.spacing(vertical, horizontalLeft, horizontalRight) };
}
