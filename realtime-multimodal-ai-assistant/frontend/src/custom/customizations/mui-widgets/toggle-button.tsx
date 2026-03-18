import { Theme } from '@mui/material/styles';
import { ToggleButtonProps, toggleButtonClasses } from '@mui/material/ToggleButton';

// ----------------------------------------------------------------------

// Available color options for the toggle button
const AVAILABLE_COLORS = ['primary', 'secondary', 'info', 'success', 'warning', 'error'] as const;

export function toggleButton(theme: Theme) {
  // Generate toggle button base styling
  const createToggleButtonStyles = (props: { ownerState: ToggleButtonProps }) => {
    const { ownerState } = props;

    // Base styles for selected state
    const baseSelectedStyles = {
      [`&.${toggleButtonClasses.selected}`]: {
        boxShadow: '0 0 0 0.5px currentColor',
        borderColor: 'currentColor',
      },
    };

    // Disabled state styles
    const disabledStyles = {
      [`&.${toggleButtonClasses.disabled}`]: {
        [`&.${toggleButtonClasses.selected}`]: {
          backgroundColor: theme.palette.action.selected,
          borderColor: theme.palette.action.disabledBackground,
          color: theme.palette.action.disabled,
        },
      },
    };

    // Generate color-specific hover states
    const colorSpecificStyles = AVAILABLE_COLORS.reduce((styles, colorName) => {
      if (ownerState.color === colorName) {
        return {
          ...styles,
          '&:hover': {
            backgroundColor: `${theme.palette[colorName].main}${Math.round(theme.palette.action.hoverOpacity * 255).toString(16)}`,
            borderColor: `${theme.palette[colorName].main}7A`, // ~48% opacity
          },
        };
      }
      return styles;
    }, {});

    return [baseSelectedStyles, colorSpecificStyles, disabledStyles];
  };

  // Return component styling configuration
  return {
    MuiToggleButton: {
      styleOverrides: {
        root: createToggleButtonStyles,
      },
    },
    MuiToggleButtonGroup: {
      styleOverrides: {
        root: {
          backgroundColor: theme.palette.background.paper,
          border: `solid 1px ${theme.palette.grey[500]}14`, // ~8% opacity
          borderRadius: theme.shape.borderRadius,
        },
        grouped: {
          margin: 4,
          borderColor: 'transparent',
          borderRadius: theme.shape.borderRadius,

          // Remove default box shadow when selected
          [`&.${toggleButtonClasses.selected}`]: {
            boxShadow: 'none',
          },

          // Ensure proper border radius on all items
          '&:not(:first-of-type), &:not(:last-of-type)': {
            borderRadius: theme.shape.borderRadius,
          },
        },
      },
    },
  };
}
