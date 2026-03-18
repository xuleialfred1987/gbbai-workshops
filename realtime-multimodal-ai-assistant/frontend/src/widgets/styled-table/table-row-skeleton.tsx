import { Box } from '@mui/material';
import Skeleton from '@mui/material/Skeleton';
import TableCell from '@mui/material/TableCell';
import TableRow, { TableRowProps } from '@mui/material/TableRow';

// ----------------------------------------------------------------------

/**
 * Skeleton loader for table rows while data is loading
 */
export default function TableLoadingSkeleton(props: TableRowProps) {
  return (
    <TableRow {...props}>
      <TableCell colSpan={12} sx={{ p: 2 }}>
        <Box display="flex" alignItems="center" gap={2}>
          {/* Avatar/icon placeholder */}
          <Skeleton variant="rounded" width={48} height={48} />

          {/* Content placeholders with decreasing widths */}
          <Skeleton width="100%" height={12} />
          <Skeleton width={180} height={12} />
          <Skeleton width={160} height={12} />
          <Skeleton width={140} height={12} />
          <Skeleton width={120} height={12} />
        </Box>
      </TableCell>
    </TableRow>
  );
}
