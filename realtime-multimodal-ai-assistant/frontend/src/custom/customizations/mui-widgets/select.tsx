import { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function select(theme: Theme) {
  // Common icon properties for select components
  const iconStyles = {
    right: 10,
    width: 18,
    height: 18,
    top: 'calc(50% - 9px)',
  };

  // Component override definitions
  const selectOverrides = {
    MuiSelect: {
      styleOverrides: {
        icon: { ...iconStyles },
      },
    },
    MuiNativeSelect: {
      styleOverrides: {
        icon: { ...iconStyles },
      },
    },
  };

  return selectOverrides;
}
