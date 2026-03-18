import { Theme } from '@mui/material/styles';
import { sliderClasses } from '@mui/material/Slider';

// ----------------------------------------------------------------------

export function slider(theme: Theme) {
  // Determine if we're in light mode
  const isLightMode = theme.palette.mode === 'light';

  // Select appropriate background color based on mode
  const valueLabelBgColor = isLightMode ? theme.palette.grey[800] : theme.palette.grey[700];

  // Configure component style overrides
  const sliderCustomStyles = {
    MuiSlider: {
      styleOverrides: {
        // Base slider styling
        root: {
          [`&.${sliderClasses.disabled}`]: {
            color: theme.palette.action.disabled,
          },
        },

        // Track styling
        rail: {
          opacity: 0.32,
        },

        // Label styling
        markLabel: {
          fontSize: 13,
          color: theme.palette.text.disabled,
        },

        // Value label styling
        valueLabel: {
          borderRadius: 8,
          backgroundColor: valueLabelBgColor,
        },
      },
    },
  };

  return sliderCustomStyles;
}
