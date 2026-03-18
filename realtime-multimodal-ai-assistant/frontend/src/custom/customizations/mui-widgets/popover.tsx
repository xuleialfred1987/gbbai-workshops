// mui
import { Theme } from '@mui/material/styles';
import { listClasses } from '@mui/material/List';

// project imports
import { paper } from '../../css';

// ----------------------------------------------------------------------

export const popover = (theme: Theme) => ({
  MuiPopover: {
    styleOverrides: {
      paper: {
        ...paper({ theme, dropdown: true }),
        [`& .${listClasses.root}`]: {
          paddingTop: '0px',
          paddingBottom: '0px',
        },
      },
    },
  },
});
