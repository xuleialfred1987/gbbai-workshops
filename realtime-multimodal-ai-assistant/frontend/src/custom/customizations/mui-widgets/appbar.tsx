import { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

type AppBarConfig = {
  MuiAppBar: {
    styleOverrides: {
      root: Record<string, unknown>;
    };
  };
};

export const appBar = (theme: Theme): AppBarConfig => ({
  MuiAppBar: {
    styleOverrides: {
      root: {
        boxShadow: 'none',
        backgroundColor: 'transparent',
        backgroundImage: 'none',
        transition: theme.transitions.create(['background-color']),
      },
    },
  },
});
