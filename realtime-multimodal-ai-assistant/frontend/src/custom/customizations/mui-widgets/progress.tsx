import { Theme } from '@mui/material/styles';
import { LinearProgressProps, linearProgressClasses } from '@mui/material/LinearProgress';

// ----------------------------------------------------------------------

// Available color options for styling
const AVAILABLE_COLORS = ['primary', 'secondary', 'info', 'success', 'warning', 'error'] as const;

export function progress(theme: Theme) {
  // Generate style overrides for the LinearProgress component
  return {
    MuiLinearProgress: {
      styleOverrides: {
        root: ({ ownerState }: { ownerState: LinearProgressProps }) => {
          // Base styles applied to all variants
          const baseStyles = {
            borderRadius: 4,
            [`& .${linearProgressClasses.bar}`]: {
              borderRadius: 4,
            },
            ...(ownerState.variant === 'buffer'
              ? {
                  backgroundColor: 'transparent',
                }
              : {}),
          };

          // Apply color-specific styling based on the color prop
          const colorVariants = AVAILABLE_COLORS.reduce((styles, color) => {
            if (ownerState.color === color) {
              return {
                ...styles,
                backgroundColor: theme.palette[color].main
                  ? `${theme.palette[color].main}3d` // 3d = 24% opacity in hex
                  : undefined,
              };
            }
            return styles;
          }, {});

          // Combine base styles with color variants
          return {
            ...baseStyles,
            ...colorVariants,
          };
        },
      },
    },
  };
}
