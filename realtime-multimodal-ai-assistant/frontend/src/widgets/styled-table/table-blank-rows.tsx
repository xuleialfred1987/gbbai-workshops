import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';

// ----------------------------------------------------------------------

type TableBlankRowsProps = {
  emptyRows: number;
  height?: number;
};

const TableBlankRows = ({ emptyRows, height }: TableBlankRowsProps) => {
  if (emptyRows <= 0) return null;

  const computedStyle = height ? { height: height * emptyRows } : {};

  return (
    <TableRow sx={computedStyle}>
      <TableCell colSpan={9} />
    </TableRow>
  );
};

export default TableBlankRows;
