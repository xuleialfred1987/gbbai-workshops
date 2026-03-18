import { alpha, Theme } from '@mui/material/styles';
import { PaginationProps } from '@mui/material/Pagination';
import { paginationItemClasses } from '@mui/material/PaginationItem';

// ----------------------------------------------------------------------

// Type augmentations for Material-UI Pagination component
declare module '@mui/material/Pagination' {
  interface PaginationPropsColorOverrides {
    info: true;
    success: true;
    warning: true;
    error: true;
  }

  interface PaginationPropsVariantOverrides {
    soft: true;
  }
}

// Available theme colors
type ThemeColor = 'primary' | 'secondary' | 'info' | 'success' | 'warning' | 'error';

// ----------------------------------------------------------------------

/**
 * Creates custom pagination styling for the theme
 */
export function pagination(theme: Theme) {
  // Determine if we're in light or dark mode
  const isLightMode = theme.palette.mode === 'light';

  // Style generator for pagination component
  const generatePaginationStyles = ({ color, variant }: PaginationProps) => {
    // Base styles for all pagination items
    const baseStyles = {
      [`& .${paginationItemClasses.root}`]: {
        // Selected item styling
        [`&.${paginationItemClasses.selected}`]: {
          fontWeight: theme.typography.fontWeightSemiBold,
        },
      },
    };

    // Variant-specific styles
    const variantStyles = (() => {
      // Outlined variant
      if (variant === 'outlined') {
        return {
          [`& .${paginationItemClasses.root}`]: {
            borderColor: alpha(theme.palette.grey[500], 0.24),
            [`&.${paginationItemClasses.selected}`]: {
              borderColor: 'currentColor',
            },
          },
        };
      }

      // Text variant with standard color
      if (variant === 'text' && color === 'standard') {
        return {
          [`& .${paginationItemClasses.root}`]: {
            [`&.${paginationItemClasses.selected}`]: {
              color: isLightMode ? theme.palette.common.white : theme.palette.grey[800],
              backgroundColor: theme.palette.text.primary,
              '&:hover': {
                backgroundColor: isLightMode ? theme.palette.grey[700] : theme.palette.grey[100],
              },
            },
          },
        };
      }

      return {};
    })();

    // Color-specific styles for the custom 'soft' variant
    const colorStyles = (() => {
      if (variant !== 'soft' || color === 'standard') {
        return {};
      }

      const themeColor = color as ThemeColor;

      return {
        [`& .${paginationItemClasses.root}`]: {
          [`&.${paginationItemClasses.selected}`]: {
            color: theme.palette[themeColor][isLightMode ? 'dark' : 'light'],
            backgroundColor: alpha(theme.palette[themeColor].main, 0.08),
            '&:hover': {
              backgroundColor: alpha(theme.palette[themeColor].main, 0.16),
            },
          },
        },
      };
    })();

    // Merge all styles
    return { ...baseStyles, ...variantStyles, ...colorStyles };
  };

  // Return theme component override
  return {
    MuiPagination: {
      styleOverrides: {
        root: ({ ownerState }: { ownerState: PaginationProps }) =>
          generatePaginationStyles(ownerState),
      },
    },
  };
}
