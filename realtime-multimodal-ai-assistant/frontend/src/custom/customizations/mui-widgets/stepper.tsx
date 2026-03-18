import { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function stepper(theme: Theme) {
  // Define connector line styles
  const connectorLineStyles = {
    borderColor: theme.palette.divider,
  };

  // Construct component overrides
  const componentOverrides = {
    MuiStepConnector: {
      styleOverrides: {
        line: connectorLineStyles,
      },
    },
  };

  return componentOverrides;
}
