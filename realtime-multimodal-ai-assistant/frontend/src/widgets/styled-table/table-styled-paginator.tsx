import { ReactElement } from 'react';

// mui
import { Theme, SxProps } from '@mui/material/styles';
import type { TablePaginationProps } from '@mui/material/TablePagination';
import { Box, Switch, TablePagination, FormControlLabel } from '@mui/material';

// ----------------------------------------------------------------------

type TableStyledPaginatorProps = TablePaginationProps & {
  dense?: boolean;
  onChangeDense?: (event: React.ChangeEvent<HTMLInputElement>) => void;
  sx?: SxProps<Theme>;
};

/**
 * Customized table pagination component with optional dense mode toggle
 */
export default function TableStyledPaginator({
  dense,
  onChangeDense,
  rowsPerPageOptions = [5, 10, 25],
  sx,
  ...other
}: TableStyledPaginatorProps): ReactElement {
  const renderDenseToggle = () => {
    if (!onChangeDense) return null;

    return (
      <FormControlLabel
        control={<Switch size="small" checked={Boolean(dense)} onChange={onChangeDense} />}
        label="Dense"
        sx={{
          padding: '1.25rem 0 1.25rem 1.75rem',
          position: { sm: 'absolute' },
          top: 0,
        }}
      />
    );
  };

  return (
    <Box
      sx={{
        display: 'flex',
        position: 'relative',
        alignItems: 'center',
        ...(sx || {}),
      }}
    >
      <TablePagination
        component="div"
        rowsPerPageOptions={rowsPerPageOptions}
        sx={{
          borderTopColor: 'transparent',
          width: '100%',
        }}
        {...other}
      />

      {renderDenseToggle()}
    </Box>
  );
}
