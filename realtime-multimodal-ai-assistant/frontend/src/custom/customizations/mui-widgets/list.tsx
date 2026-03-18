import { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function list(theme: Theme) {
  // Common styles that can be shared across components
  const sharedStyles = {
    minWidth: 'auto',
    marginRight: theme.spacing(2),
  };

  // Zero margin styles
  const noMarginStyles = {
    margin: 0,
  };

  // Component overrides grouped into a single object
  const componentOverrides = {
    MuiListItemIcon: {
      styleOverrides: {
        root: {
          color: 'inherit',
          ...sharedStyles,
        },
      },
    },
    MuiListItemAvatar: {
      styleOverrides: {
        root: {
          ...sharedStyles,
        },
      },
    },
    MuiListItemText: {
      styleOverrides: {
        root: noMarginStyles,
        multiline: noMarginStyles,
      },
    },
  };

  return componentOverrides;
}
