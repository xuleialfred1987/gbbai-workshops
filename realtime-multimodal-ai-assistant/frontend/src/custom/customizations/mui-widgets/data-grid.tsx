// mui
import { paperClasses } from '@mui/material/Paper';
import { Theme, alpha } from '@mui/material/styles';
import { buttonClasses } from '@mui/material/Button';
import { listClasses, listItemIconClasses } from '@mui/material';
import { tablePaginationClasses } from '@mui/material/TablePagination';

// project imports
import { paper } from '../../css';

// ----------------------------------------------------------------------

// Helper function to generate common spacing
const spacing = (theme: Theme, factor: number) => theme.spacing(factor);

// Helper function for dashed borders
const dashedBorder = (theme: Theme, position: 'top' | 'bottom') => ({
  [`border${position.charAt(0).toUpperCase() + position.slice(1)}`]: `1px dashed ${theme.palette.divider}`,
});

// Helper function for paper styles
const getPaperStyles = (theme: Theme) => ({
  ...paper({ theme, dropdown: true }),
  padding: 0,
});

// Main export function
export function dataGrid(theme: Theme) {
  return {
    MuiDataGrid: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          borderWidth: 0,
          [`& .${tablePaginationClasses.root}`]: { borderTop: 0 },
          [`& .${tablePaginationClasses.toolbar}`]: { height: 'auto' },
        },
        cell: dashedBorder(theme, 'bottom'),
        selectedRowCount: { whiteSpace: 'nowrap' },
        columnSeparator: { color: theme.palette.divider },
        toolbarContainer: {
          padding: spacing(theme, 2),
          backgroundColor: theme.palette.background.neutral,
          ...dashedBorder(theme, 'bottom'),
        },
        paper: getPaperStyles(theme),
        menu: {
          [`& .${paperClasses.root}`]: getPaperStyles(theme),
          [`& .${listClasses.root}`]: {
            padding: 0,
            [`& .${listItemIconClasses.root}`]: {
              minWidth: 0,
              marginRight: spacing(theme, 2),
            },
          },
        },
        columnHeaders: {
          borderRadius: 0,
          backgroundColor: theme.palette.background.neutral,
        },
        panelHeader: { padding: spacing(theme, 2) },
        panelFooter: {
          padding: spacing(theme, 2),
          justifyContent: 'flex-end',
          borderTop: `1px dashed ${theme.palette.divider}`,
          [`& .${buttonClasses.root}`]: {
            '&:first-of-type': {
              border: `1px solid ${alpha(theme.palette.grey[500], 0.24)}`,
            },
            '&:last-of-type': {
              marginLeft: spacing(theme, 1.5),
              color: theme.palette.background.paper,
              backgroundColor: theme.palette.text.primary,
            },
          },
        },
        filterForm: { padding: spacing(theme, 2) },
        filterFormValueInput: { marginLeft: spacing(theme, 2) },
        filterFormColumnInput: { marginLeft: spacing(theme, 2) },
        filterFormOperatorInput: { marginLeft: spacing(theme, 2) },
      },
    },
  };
}
