import { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function tooltip(theme: Theme) {
  // Determine background color based on theme mode
  const backgroundColor =
    theme.palette.mode === 'light' ? theme.palette.grey[800] : theme.palette.grey[600];

  // Define component style overrides
  const customStyles = {
    MuiTooltip: {
      styleOverrides: {
        tooltip: { backgroundColor },
        arrow: { color: backgroundColor },
      },
    },
  };

  return customStyles;
}
