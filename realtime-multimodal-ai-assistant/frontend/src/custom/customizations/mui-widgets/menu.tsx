// mui
import type { Theme } from '@mui/material/styles';

// project imports
import { menuItem } from '../../css';

// ----------------------------------------------------------------------

export const menu = (theme: Theme) => ({
  MuiMenuItem: {
    styleOverrides: {
      root: { ...menuItem(theme) },
    },
  },
});
