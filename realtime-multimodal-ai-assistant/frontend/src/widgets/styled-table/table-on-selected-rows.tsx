import { ReactElement } from 'react';

// mui
import type { StackProps } from '@mui/material/Stack';
import { Card, Stack, Checkbox, useTheme, Typography } from '@mui/material';

// ----------------------------------------------------------------------

interface Props extends StackProps {
  dense?: boolean;
  action?: React.ReactNode;
  rowCount: number;
  numSelected: number;
  checkboxMl?: number;
  denseCheckboxMl?: number;
  onSelectAllRows: (checked: boolean) => void;
  newTop?: number;
}

/**
 * Displays an action bar when table rows are selected
 */
function TableOnSelectedRows({
  dense,
  action,
  rowCount,
  numSelected,
  checkboxMl = 2,
  denseCheckboxMl = 4.5,
  onSelectAllRows,
  sx,
  newTop = 3,
  ...other
}: Props): ReactElement | null {
  const theme = useTheme();

  // Don't render anything if no rows are selected
  if (numSelected <= 0) {
    return null;
  }

  // Calculate if all rows or just some rows are selected
  const isAllSelected = !!rowCount && numSelected === rowCount;
  const isSomeSelected = !!numSelected && numSelected < rowCount;

  // Handle checkbox change
  const handleSelectAllClick = (event: React.ChangeEvent<HTMLInputElement>) => {
    onSelectAllRows(event.target.checked);
  };

  return (
    <Card
      elevation={3}
      sx={{
        position: 'absolute',
        zIndex: 9,
        top: newTop,
        left: 15,
        borderRadius: 0.75,
        py: 0.25,
        px: 0.5,
        backgroundColor: theme.palette.background.default,
        boxShadow: theme.customShadows.card,
        ...sx
      }}
    >
      <Stack direction="row" alignItems="center" spacing={1} {...other}>
        <Checkbox
          size="small"
          indeterminate={isSomeSelected}
          checked={isAllSelected}
          onChange={handleSelectAllClick}
          sx={{
            ml: dense ? denseCheckboxMl : checkboxMl,
            p: 0.75,
          }}
        />

        <Typography
          variant="subtitle2"
          sx={{
            flexGrow: 1,
            color: 'primary.main',
            ml: 1.25,
            mr: 1.5,
          }}
        >
          {numSelected} selected
        </Typography>

        {action !== undefined && action}
      </Stack>
    </Card>
  );
}

export default TableOnSelectedRows;
