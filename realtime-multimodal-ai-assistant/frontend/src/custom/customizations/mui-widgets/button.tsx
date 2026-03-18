import { Theme, alpha } from '@mui/material/styles';
import { ButtonProps, buttonClasses } from '@mui/material/Button';

// ----------------------------------------------------------------------

// Extend Button interface to support 'soft' variant
declare module '@mui/material/Button' {
  interface ButtonPropsVariantOverrides {
    soft: true;
  }
}

// Available color options
const AVAILABLE_COLORS = ['primary', 'secondary', 'info', 'success', 'warning', 'error'] as const;

export function button(theme: Theme) {
  const isDarkMode = theme.palette.mode !== 'light';

  // Create style overrides for the MuiButton component
  return {
    MuiButton: {
      styleOverrides: {
        root: ({ ownerState }: { ownerState: ButtonProps }) => {
          // Extract button properties
          const { color, variant, size } = ownerState;
          const isInheritColor = color === 'inherit';

          // Build style configuration
          return [
            // Base styles
            getBaseStyles(theme, isDarkMode, variant, isInheritColor),

            // Color-specific styles
            ...generateColorStyles(theme, isDarkMode, color, variant),

            // Size variations
            getSizeStyles(variant, size),

            // Disabled state
            getDisabledStyles(theme, variant),
          ];
        },
      },
    },
  };
}

// Helper functions to generate style fragments
function getBaseStyles(
  theme: Theme,
  isDarkMode: boolean,
  variant?: string,
  isInheritColor?: boolean
) {
  const baseStyle = {
    borderRadius: 6,
  };

  if (!isInheritColor) return baseStyle;

  // Style variations based on variant
  switch (variant) {
    case 'contained':
      return {
        ...baseStyle,
        color: isDarkMode ? theme.palette.grey[800] : theme.palette.common.white,
        backgroundColor: isDarkMode ? theme.palette.common.white : theme.palette.grey[800],
        '&:hover': {
          backgroundColor: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[700],
        },
      };

    case 'outlined':
      return {
        ...baseStyle,
        borderColor: alpha(theme.palette.grey[500], 0.32),
        '&:hover': {
          backgroundColor: theme.palette.action.hover,
          borderColor: 'currentColor',
          boxShadow: '0 0 0 0.5px currentColor',
        },
      };

    case 'text':
      return {
        ...baseStyle,
        '&:hover': {
          backgroundColor: theme.palette.action.hover,
        },
      };

    case 'soft':
      return {
        ...baseStyle,
        color: theme.palette.text.primary,
        backgroundColor: alpha(theme.palette.grey[500], 0.08),
        '&:hover': {
          backgroundColor: alpha(theme.palette.grey[500], 0.24),
        },
      };

    default:
      return baseStyle;
  }
}

function generateColorStyles(theme: Theme, isDarkMode: boolean, color?: string, variant?: string) {
  return AVAILABLE_COLORS.map((colorOption) => {
    if (color !== colorOption) return {};

    const styles = {};

    // Add variant-specific color styles
    if (variant === 'contained') {
      Object.assign(styles, {
        '&:hover': {
          // boxShadow: theme.customShadows[colorOption],
        },
      });
    }

    if (variant === 'soft') {
      Object.assign(styles, {
        color: theme.palette[colorOption][isDarkMode ? 'light' : 'dark'],
        backgroundColor: alpha(theme.palette[colorOption].main, 0.16),
        '&:hover': {
          backgroundColor: alpha(theme.palette[colorOption].main, 0.32),
        },
      });
    }

    return styles;
  });
}

function getSizeStyles(variant?: string, size?: string) {
  const isPaddingReduced = variant === 'text';

  switch (size) {
    case 'small':
      return {
        height: 30,
        fontSize: 13,
        paddingLeft: isPaddingReduced ? 4 : 8,
        paddingRight: isPaddingReduced ? 4 : 8,
      };

    case 'large':
      return {
        height: 48,
        fontSize: 15,
        paddingLeft: isPaddingReduced ? 10 : 16,
        paddingRight: isPaddingReduced ? 10 : 16,
      };

    // Medium size (default)
    default:
      return {
        paddingLeft: isPaddingReduced ? 8 : 12,
        paddingRight: isPaddingReduced ? 8 : 12,
      };
  }
}

function getDisabledStyles(theme: Theme, variant?: string) {
  return {
    [`&.${buttonClasses.disabled}`]: {
      ...(variant === 'soft' && {
        backgroundColor: theme.palette.action.disabledBackground,
      }),
    },
  };
}
