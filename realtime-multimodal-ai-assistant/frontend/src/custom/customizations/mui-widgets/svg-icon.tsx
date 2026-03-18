import { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function svgIcon(theme: Theme) {
  const LARGE_ICON_DIMENSION = 32;

  const overrides = {
    MuiSvgIcon: {
      styleOverrides: {
        fontSizeLarge: {
          width: LARGE_ICON_DIMENSION,
          height: LARGE_ICON_DIMENSION,
          fontSize: 'inherit',
        },
      },
    },
  };

  return overrides;
}
