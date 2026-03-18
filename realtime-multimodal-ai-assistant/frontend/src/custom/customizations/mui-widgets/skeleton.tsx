import { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function skeleton(theme: Theme) {
  // Extract values from theme for consistency
  const neutralBackground = theme.palette.background.neutral;
  const borderRadiusMultiplier = 2;

  // Define styling overrides
  const rootStyles = {
    backgroundColor: neutralBackground,
  };

  const roundedStyles = {
    borderRadius: theme.shape.borderRadius * borderRadiusMultiplier,
  };

  // Component customization
  const skeletonCustomization = {
    MuiSkeleton: {
      styleOverrides: {
        root: rootStyles,
        rounded: roundedStyles,
      },
    },
  };

  return skeletonCustomization;
}
