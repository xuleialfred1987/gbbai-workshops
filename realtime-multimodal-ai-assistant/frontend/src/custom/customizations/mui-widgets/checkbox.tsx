import { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export const checkbox = (theme: Theme) => ({
  MuiCheckbox: {
    styleOverrides: {
      // Apply custom padding to the checkbox component
      root: () => ({
        padding: `${theme.spacing(1.15)}`,

        // Additional styles could be added here in the future
        // without changing the actual padding value
      }),
    },
  },
});
