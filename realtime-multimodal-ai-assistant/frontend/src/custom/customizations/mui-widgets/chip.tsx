import { alpha, Theme } from '@mui/material/styles';
import { ChipProps, chipClasses } from '@mui/material/Chip';

// ----------------------------------------------------------------------

// Add custom 'soft' variant support
declare module '@mui/material/Chip' {
  interface ChipPropsVariantOverrides {
    soft: true;
  }
}

// Theme colors
const THEME_COLORS = ['primary', 'secondary', 'info', 'success', 'warning', 'error'] as const;

/**
 * MUI Chip component theme customization
 */
export const chip = (theme: Theme) => ({
  MuiChip: {
    styleOverrides: {
      root: ({ ownerState }: { ownerState: ChipProps }) => [
        createBaseStyles(theme, ownerState),
        ...createColorStyles(theme, ownerState),
        createDisabledStateStyles(theme, ownerState),
        { fontWeight: 500 },
      ],
    },
  },
});

/**
 * Creates the base styling for chips
 */
function createBaseStyles(theme: Theme, ownerState: ChipProps) {
  const { color, variant } = ownerState;
  const isDefaultColor = color === 'default';
  const isDarkMode = theme.palette.mode === 'dark';

  // Common delete icon styling
  const deleteIconStyles = {
    [`& .${chipClasses.deleteIcon}`]: {
      opacity: 0.48,
      color: 'currentColor',
      '&:hover': {
        opacity: 1,
        color: 'currentColor',
      },
    },
  };

  // Return early if not default color
  if (!isDefaultColor) return deleteIconStyles;

  const avatarStyles = {
    [`& .${chipClasses.avatar}`]: {
      color: theme.palette.text.primary,
    },
  };

  // Handle different variants
  switch (variant) {
    case 'filled':
      return {
        ...deleteIconStyles,
        ...avatarStyles,
        color: isDarkMode ? theme.palette.grey[800] : theme.palette.common.white,
        backgroundColor: theme.palette.text.primary,
        '&:hover': {
          backgroundColor: isDarkMode ? theme.palette.grey[100] : theme.palette.grey[700],
        },
        [`& .${chipClasses.icon}`]: {
          color: isDarkMode ? theme.palette.grey[800] : theme.palette.common.white,
        },
      };

    case 'outlined':
      return {
        ...deleteIconStyles,
        ...avatarStyles,
        border: `solid 1px ${alpha(theme.palette.grey[500], 0.32)}`,
      };

    case 'soft':
      return {
        ...deleteIconStyles,
        ...avatarStyles,
        color: theme.palette.text.primary,
        backgroundColor: alpha(theme.palette.grey[500], 0.16),
        '&:hover': {
          backgroundColor: alpha(theme.palette.grey[500], 0.32),
        },
      };

    default:
      return { ...deleteIconStyles, ...avatarStyles };
  }
}

/**
 * Creates color-specific styling based on theme colors
 */
function createColorStyles(theme: Theme, ownerState: ChipProps) {
  const isDarkMode = theme.palette.mode === 'dark';

  return THEME_COLORS.map((colorName) => {
    // Skip if color doesn't match
    if (ownerState.color !== colorName) return {};

    const baseStyles = {
      [`& .${chipClasses.avatar}`]: {
        color: theme.palette[colorName].lighter,
        backgroundColor: theme.palette[colorName].dark,
      },
    };

    // Add soft variant styling
    if (ownerState.variant === 'soft') {
      return {
        ...baseStyles,
        color: theme.palette[colorName][isDarkMode ? 'light' : 'dark'],
        backgroundColor: alpha(theme.palette[colorName].main, 0.16),
        '&:hover': {
          backgroundColor: alpha(theme.palette[colorName].main, 0.32),
        },
      };
    }

    return baseStyles;
  });
}

/**
 * Creates disabled state styling
 */
function createDisabledStateStyles(theme: Theme, ownerState: ChipProps) {
  // Common disabled styles
  const commonStyles = {
    opacity: 1,
    color: theme.palette.action.disabled,
    [`& .${chipClasses.icon}`]: {
      color: theme.palette.action.disabled,
    },
    [`& .${chipClasses.avatar}`]: {
      color: theme.palette.action.disabled,
      backgroundColor: theme.palette.action.disabledBackground,
    },
  };

  // Determine variant-specific disabled styles
  const variantStyles = (() => {
    if (ownerState.variant === 'filled' || ownerState.variant === 'soft') {
      return {
        backgroundColor: theme.palette.action.disabledBackground,
      };
    }

    if (ownerState.variant === 'outlined') {
      return {
        borderColor: theme.palette.action.disabledBackground,
      };
    }

    // Default case for any other variants
    return {};
  })();

  // Return combined styles
  return {
    [`&.${chipClasses.disabled}`]: {
      ...commonStyles,
      ...variantStyles,
    },
  };
}
