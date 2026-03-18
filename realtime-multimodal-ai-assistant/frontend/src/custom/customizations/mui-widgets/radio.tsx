import { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function radio(theme: Theme) {
  // Define typography styles
  const labelTypography = { ...theme.typography.body2 };

  // Define spacing values
  const radioSpacing = theme.spacing(1);

  // Construct and return component customizations
  const componentOverrides = {
    MuiFormControlLabel: {
      styleOverrides: {
        label: labelTypography,
      },
    },

    MuiRadio: {
      styleOverrides: {
        root: {
          padding: radioSpacing,
        },
      },
    },
  };

  return componentOverrides;
}
