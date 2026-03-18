import { Theme } from '@mui/material/styles';
import { tableRowClasses } from '@mui/material/TableRow';
import { tableCellClasses } from '@mui/material/TableCell';

// ----------------------------------------------------------------------

export function table(theme: Theme) {
  // Define style configurations for each component
  const styleConfigs = {
    container: {
      position: 'relative',
    },

    row: {
      [`&.${tableRowClasses.selected}`]: {
        backgroundColor: `${theme.palette.primary.dark}0a`, // alpha 0.04 equivalent
        '&:hover': {
          backgroundColor: `${theme.palette.primary.dark}14`, // alpha 0.08 equivalent
        },
      },
      '&:last-of-type': {
        [`& .${tableCellClasses.root}`]: {
          borderColor: 'transparent',
        },
      },
    },

    cell: {
      default: {
        borderBottomStyle: 'dashed',
      },
      header: {
        fontSize: 14,
        color: theme.palette.text.primary,
        fontWeight: theme.typography.fontWeightSemiBold,
        position: 'relative',
        '&::after': {
          content: '""',
          position: 'absolute',
          left: 0,
          bottom: 0,
          width: '100%',
          borderBottom: `1.5px solid ${theme.palette.background.neutral}`,
        },
      },
      stickyHeader: {
        backgroundColor: theme.palette.background.paper,
        backgroundImage: `linear-gradient(
          to bottom,
          ${theme.palette.background.neutral} 0%,
          ${theme.palette.background.neutral} 100%
        )`,
      },
      checkbox: {
        paddingLeft: theme.spacing(1),
      },
    },

    pagination: {
      wrapper: {
        width: '100%',
      },
      toolbar: {
        height: 64,
      },
      actions: {
        marginRight: 8,
      },
      select: {
        paddingLeft: 8,
        '&:focus': {
          borderRadius: theme.shape.borderRadius,
        },
      },
      selectIcon: {
        right: 2,
        width: 18,
        height: 18,
      },
    },
  };

  // Construct and return the component style overrides
  return {
    MuiTableContainer: {
      styleOverrides: {
        root: styleConfigs.container,
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: styleConfigs.row,
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: styleConfigs.cell.default,
        head: styleConfigs.cell.header,
        stickyHeader: styleConfigs.cell.stickyHeader,
        paddingCheckbox: styleConfigs.cell.checkbox,
      },
    },
    MuiTablePagination: {
      styleOverrides: {
        root: styleConfigs.pagination.wrapper,
        toolbar: styleConfigs.pagination.toolbar,
        actions: styleConfigs.pagination.actions,
        select: styleConfigs.pagination.select,
        selectIcon: styleConfigs.pagination.selectIcon,
      },
    },
  };
}
