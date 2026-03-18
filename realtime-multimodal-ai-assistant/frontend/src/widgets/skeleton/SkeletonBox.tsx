import { Box, Stack, Skeleton, CardProps } from '@mui/material';

// ----------------------------------------------------------------------

interface Props extends CardProps {
  showStack?: boolean;
}

export default function SkeletonBox({ showStack = true, sx }: Props) {
  return (
    <Box sx={{ ...sx }}>
      <Stack spacing={2} sx={{ p: 2 }}>
        <Skeleton variant="text" sx={{ width: 0.5 }} />
        {showStack && (
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Skeleton variant="text" sx={{ width: 100 }} />
            <Skeleton variant="text" sx={{ width: 40 }} />
          </Stack>
        )}
      </Stack>
      <Skeleton variant="rectangular" sx={{ paddingBottom: '500%' }} />
    </Box>
  );
}
