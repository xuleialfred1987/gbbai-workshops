import { alpha, Theme } from '@mui/material/styles';
import { ratingClasses } from '@mui/material/Rating';
import { svgIconClasses } from '@mui/material/SvgIcon';

// ----------------------------------------------------------------------

export function rating(theme: Theme) {
  // Icon size configurations for different rating sizes
  const iconSizes = {
    small: { width: 20, height: 20 },
    medium: { width: 24, height: 24 },
    large: { width: 28, height: 28 },
  };

  // Empty icon color with transparency
  const emptyIconColor = alpha(theme.palette.grey[500], 0.48);

  // Create the component style overrides
  const ratingOverrides = {
    MuiRating: {
      styleOverrides: {
        // Base styling for the rating component
        root: {
          [`&.${ratingClasses.disabled}`]: {
            opacity: 0.48,
          },
        },

        // Empty icon styling
        iconEmpty: {
          color: emptyIconColor,
        },

        // Size variants styling
        sizeSmall: {
          [`& .${svgIconClasses.root}`]: iconSizes.small,
        },

        sizeMedium: {
          [`& .${svgIconClasses.root}`]: iconSizes.medium,
        },

        sizeLarge: {
          [`& .${svgIconClasses.root}`]: iconSizes.large,
        },
      },
    },
  };

  return ratingOverrides;
}
