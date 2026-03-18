import React from 'react';

// mui
import { Theme, SxProps } from '@mui/material/styles';
import { Box, Checkbox, TableRow, TableCell, TableHead, TableSortLabel } from '@mui/material';

// ----------------------------------------------------------------------

// Screen reader accessibility style
const visuallyHidden = {
  position: 'absolute',
  overflow: 'hidden',
  clip: 'rect(0 0 0 0)',
  height: '1px',
  width: '1px',
  whiteSpace: 'nowrap',
  border: 0,
  padding: 0,
  margin: -1,
} as const;

// Component type definition
type Props = {
  order?: 'asc' | 'desc';
  orderBy?: string;
  headLabel: any[];
  rowCount?: number;
  dense?: boolean;
  numSelected?: number;
  checkboxPl?: number;
  checkboxMl?: number;
  visible?: boolean;
  onSort?: (id: string) => void;
  onSelectAllRows?: (checked: boolean) => void;
  sx?: SxProps<Theme>;
};

export default function TableStyledHeader({
  order,
  orderBy,
  rowCount = 0,
  headLabel,
  dense = false,
  visible = true,
  numSelected = 0,
  checkboxPl = 2,
  checkboxMl = 4.5,
  onSort,
  onSelectAllRows,
  sx,
}: Props) {
  // Early visibility check
  if (!visible) return null;

  // Helper functions
  const isIndeterminate = () => !!numSelected && numSelected < rowCount;
  const isAllSelected = () => !!rowCount && numSelected === rowCount;
  const isCellSorted = (cellId: string) => orderBy === cellId;

  // Event handlers
  const handleSelectAll = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (onSelectAllRows) {
      onSelectAllRows(event.target.checked);
    }
  };

  const handleSortClick = (cellId: string) => {
    if (onSort) {
      onSort(cellId);
    }
  };

  // Render selection checkbox cell if needed
  const renderSelectionCell = () => {
    if (!onSelectAllRows) return null;

    return (
      <TableCell
        padding="checkbox"
        sx={{
          pl: checkboxPl,
          maxWidth: 40,
          minWidth: 40,
        }}
      >
        <Checkbox
          indeterminate={isIndeterminate()}
          checked={isAllSelected()}
          onChange={handleSelectAll}
          sx={{ ml: dense ? checkboxMl : 0 }}
        />
      </TableCell>
    );
  };

  // Render column cells
  const renderColumnCells = () =>
    headLabel.map((headCell) => (
      <TableCell
        key={headCell.id}
        align={headCell.align || 'left'}
        sortDirection={isCellSorted(headCell.id) ? order : false}
        sx={{
          width: headCell.width,
          minWidth: headCell.minWidth,
          maxWidth: headCell.maxWidth,
          py: 1.25,
        }}
      >
        {onSort ? (
          <TableSortLabel
            hideSortIcon
            active={isCellSorted(headCell.id)}
            direction={isCellSorted(headCell.id) ? order : 'asc'}
            onClick={() => handleSortClick(headCell.id)}
          >
            {headCell.label}

            {isCellSorted(headCell.id) && (
              <Box sx={{ ...visuallyHidden }}>
                {order === 'desc' ? 'sorted descending' : 'sorted ascending'}
              </Box>
            )}
          </TableSortLabel>
        ) : (
          headCell.label
        )}
      </TableCell>
    ));

  return (
    <TableHead sx={sx}>
      <TableRow>
        {renderSelectionCell()}
        {renderColumnCells()}
      </TableRow>
    </TableHead>
  );
}
