// mui
import { TableRow, TableCell } from '@mui/material';
import { Theme, SxProps } from '@mui/material/styles';

// project imports
import { VoidKb } from '../void-layout';

// ----------------------------------------------------------------------

interface TableNoContentProps {
  notFound: boolean;
  sx?: SxProps<Theme>;
  title?: string;
  description?: string;
}

/**
 * Component to display when table has no content
 */
export default function TableNoContent({
  notFound,
  sx,
  title = 'No data',
  description,
}: TableNoContentProps) {
  if (!notFound) {
    return (
      <TableRow>
        <TableCell colSpan={12} sx={{ p: 0 }} />
      </TableRow>
    );
  }

  return (
    <TableRow>
      <TableCell colSpan={12}>
        <VoidKb
          filled
          title={title}
          description={description}
          sx={{
            paddingY: 10,
            ...(sx || {}),
          }}
        />
      </TableCell>
    </TableRow>
  );
}
