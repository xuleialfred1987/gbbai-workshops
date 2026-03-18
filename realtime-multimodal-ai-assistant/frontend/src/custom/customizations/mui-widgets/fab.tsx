import { Theme, alpha } from '@mui/material/styles';
import { FabProps, fabClasses } from '@mui/material/Fab';

// ----------------------------------------------------------------------

// Extend MUI Fab component with custom variants
declare module '@mui/material/Fab' {
  interface FabPropsVariantOverrides {
    outlined: true;
    outlinedExtended: true;
    soft: true;
    softExtended: true;
  }
}

// Available color options
const AVAILABLE_COLORS = ['primary', 'secondary', 'info', 'success', 'warning', 'error'] as const;

export function fab(theme: Theme) {
  const isDarkMode = theme.palette.mode !== 'light';

  // Style generator function
  function generateStyles(ownerState: FabProps) {
    // Determine button type
    const isDefault = ownerState.color === 'default';
    const isInherit = ownerState.color === 'inherit';

    // Determine variants
    const variants = {
      isCircular: ownerState.variant === 'circular',
      isExtended: ownerState.variant === 'extended',
      isOutlined: ownerState.variant === 'outlined',
      isOutlinedExtended: ownerState.variant === 'outlinedExtended',
      isSoft: ownerState.variant === 'soft',
      isSoftExtended: ownerState.variant === 'softExtended',
    };

    const isFilled = variants.isCircular || variants.isExtended;
    const isOutlinedType = variants.isOutlined || variants.isOutlinedExtended;
    const isSoftType = variants.isSoft || variants.isSoftExtended;
    const isExtendedType =
      variants.isExtended || variants.isOutlinedExtended || variants.isSoftExtended;

    // Base styles for all variants
    const baseStyles = {
      '&:hover, &:active': {
        boxShadow: 'none',
      },
    };

    // Filled variant styles
    const filledStyles = isFilled
      ? {
          ...((isDefault || isInherit) && {
            boxShadow: theme.customShadows.z8,
          }),
          ...(isInherit && {
            backgroundColor: theme.palette.text.primary,
            color: isDarkMode ? theme.palette.grey[800] : theme.palette.common.white,
            '&:hover': {
              backgroundColor: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[700],
            },
          }),
        }
      : {};

    // Outlined variant styles
    const outlinedStyles = isOutlinedType
      ? {
          boxShadow: 'none',
          backgroundColor: 'transparent',
          ...((isDefault || isInherit) && {
            border: `solid 1px ${alpha(theme.palette.grey[500], 0.32)}`,
          }),
          ...(isDefault && {
            ...(isDarkMode && {
              color: theme.palette.text.secondary,
            }),
          }),
          '&:hover': {
            borderColor: 'currentColor',
            boxShadow: '0 0 0 0.5px currentColor',
            backgroundColor: theme.palette.action.hover,
          },
        }
      : {};

    // Soft variant styles
    const softStyles = isSoftType
      ? {
          boxShadow: 'none',
          ...(isDefault && {
            color: theme.palette.grey[800],
            backgroundColor: theme.palette.grey[300],
            '&:hover': {
              backgroundColor: theme.palette.grey[400],
            },
          }),
          ...(isInherit && {
            backgroundColor: alpha(theme.palette.grey[500], 0.08),
            '&:hover': {
              backgroundColor: alpha(theme.palette.grey[500], 0.24),
            },
          }),
        }
      : {};

    // Specific color styles for each variant
    const colorStyles = AVAILABLE_COLORS.map((color) => {
      if (ownerState.color !== color) return {};

      return {
        // Filled color styles
        ...(isFilled && {
          boxShadow: theme.customShadows[color],
          '&:hover': {
            backgroundColor: theme.palette[color].dark,
          },
        }),
        // Outlined color styles
        ...(isOutlinedType && {
          color: theme.palette[color].main,
          border: `solid 1px ${alpha(theme.palette[color].main, 0.48)}`,
          '&:hover': {
            backgroundColor: alpha(theme.palette[color].main, 0.08),
          },
        }),
        // Soft color styles
        ...(isSoftType && {
          color: theme.palette[color][isDarkMode ? 'light' : 'dark'],
          backgroundColor: alpha(theme.palette[color].main, 0.16),
          '&:hover': {
            backgroundColor: alpha(theme.palette[color].main, 0.32),
          },
        }),
      };
    });

    // Styles for disabled state
    const disabledStyles = {
      [`&.${fabClasses.disabled}`]: {
        ...(isOutlinedType && {
          backgroundColor: 'transparent',
          border: `solid 1px ${theme.palette.action.disabledBackground}`,
        }),
      },
    };

    // Size styles for extended variants
    const sizeStyles = isExtendedType
      ? {
          width: 'auto',
          '& svg': {
            marginRight: theme.spacing(1),
          },
          ...(ownerState.size === 'small' && {
            height: 34,
            minHeight: 34,
            borderRadius: 17,
            padding: theme.spacing(0, 1),
          }),
          ...(ownerState.size === 'medium' && {
            height: 40,
            minHeight: 40,
            borderRadius: 20,
            padding: theme.spacing(0, 2),
          }),
          ...(ownerState.size === 'large' && {
            height: 48,
            minHeight: 48,
            borderRadius: 24,
            padding: theme.spacing(0, 2),
          }),
        }
      : {};

    // Combine all styles
    return [
      { ...baseStyles, ...filledStyles, ...outlinedStyles, ...softStyles },
      ...colorStyles,
      disabledStyles,
      sizeStyles,
    ];
  }

  // Return the customized component
  return {
    MuiFab: {
      styleOverrides: {
        root: ({ ownerState }: { ownerState: FabProps }) => generateStyles(ownerState),
      },
    },
  };
}
