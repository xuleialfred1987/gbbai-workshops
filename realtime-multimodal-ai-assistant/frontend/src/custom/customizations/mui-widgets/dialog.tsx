import { Theme } from '@mui/material/styles';
import { DialogProps } from '@mui/material/Dialog';

// ----------------------------------------------------------------------

export const dialog = (theme: Theme) => ({
  MuiDialog: {
    styleOverrides: {
      paper: ({ ownerState }: { ownerState: DialogProps }) => {
        const baseStyles = {
          boxShadow: theme.customShadows.dialog,
          borderRadius: theme.shape.borderRadius,
        };

        if (!ownerState.fullScreen) {
          Object.assign(baseStyles, {
            margin: theme.spacing(2),
          });
        }

        return baseStyles;
      },
      paperFullScreen: {
        borderRadius: 'unset',
      },
    },
  },

  MuiDialogTitle: {
    styleOverrides: {
      root: {
        paddingTop: theme.spacing(3),
        paddingBottom: theme.spacing(3),
        paddingLeft: theme.spacing(3),
        paddingRight: theme.spacing(3),
      },
    },
  },

  MuiDialogContent: {
    styleOverrides: {
      root: {
        paddingLeft: theme.spacing(3),
        paddingRight: theme.spacing(3),
        paddingTop: 0,
        paddingBottom: 0,
      },
      dividers: {
        borderTopWidth: 0,
        borderBottom: `1px dashed ${theme.palette.divider}`,
        paddingBottom: theme.spacing(3),
      },
    },
  },

  MuiDialogActions: {
    styleOverrides: {
      root: {
        padding: theme.spacing(3),
        '& > :not(:first-of-type)': {
          marginLeft: theme.spacing(1.5),
        },
      },
    },
  },
});
