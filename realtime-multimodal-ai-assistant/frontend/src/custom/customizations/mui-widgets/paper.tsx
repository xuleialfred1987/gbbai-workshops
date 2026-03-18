import { Theme, alpha } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function paper(theme: Theme) {
  const noBackgroundImage = {
    backgroundImage: 'none',
  };

  const outlinedBorder = {
    borderColor: alpha(theme.palette.grey[500], 0.16),
  };

  return {
    MuiPaper: {
      styleOverrides: {
        root: noBackgroundImage,
        outlined: outlinedBorder,
      },
    },
  };
}
