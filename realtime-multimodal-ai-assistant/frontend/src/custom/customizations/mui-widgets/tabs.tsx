import { Theme } from '@mui/material/styles';
import { tabClasses } from '@mui/material/Tab';

// ----------------------------------------------------------------------

export function tabs(theme: Theme) {
  const tabsStyleConfig = {
    MuiTabs: {
      styleOverrides: {
        indicator: {
          backgroundColor: theme.palette.text.primary,
        },
        scrollButtons: {
          width: 48,
          borderRadius: '50%',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          opacity: 1,
          padding: 0,
          minWidth: 48,
          minHeight: 48,
          fontWeight: theme.typography.fontWeightSemiBold,

          // Setting styles for tabs except the last one
          '&:not(:last-of-type)': {
            [theme.breakpoints.up('sm')]: {
              marginRight: theme.spacing(5),
            },
            marginRight: theme.spacing(3),
          },

          // Styling for non-selected tabs
          [`&.${tabClasses.root}:not(.${tabClasses.selected})`]: {
            color: theme.palette.text.secondary,
          },
        },
      },
    },
  };

  return tabsStyleConfig;
}
