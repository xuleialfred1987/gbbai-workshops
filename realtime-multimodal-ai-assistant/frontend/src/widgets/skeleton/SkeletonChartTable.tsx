import { Card, Grid, Skeleton, CardProps } from '@mui/material';

// ----------------------------------------------------------------------

interface Props extends CardProps {
  rowNumber?: number;
}

export default function SkeletonChartTable({ rowNumber = 7, sx }: Props) {
  return (
    <Card sx={{ ...sx, px: 2.5, py: 4 }}>
      <Grid container spacing={3}>
        <Grid item xs={1} md={1} lg={1}>
          {[...Array(rowNumber)].map((_, index) => (
            <Skeleton
              key={`a${index}`}
              variant="circular"
              width={40}
              height={40}
              sx={{ mb: index === rowNumber - 1 ? 0 : 3 }}
            />
          ))}
        </Grid>
        <Grid item xs={6} md={6} lg={6} sx={{ ml: -2 }}>
          {[...Array(rowNumber)].map((_, index) => (
            <Skeleton
              key={`b${index}`}
              variant="rectangular"
              width="100%"
              height={40}
              sx={{ mb: index === rowNumber - 1 ? 0 : 3, borderRadius: 1 }}
            />
          ))}
        </Grid>
        <Grid item xs={4} md={4} lg={4}>
          {[...Array(rowNumber)].map((_, index) => (
            <Skeleton
              key={`c${index}`}
              variant="rectangular"
              width="100%"
              height={40}
              sx={{ mb: index === rowNumber - 1 ? 0 : 3, borderRadius: 1 }}
            />
          ))}
        </Grid>
        <Grid item xs={1} md={1} lg={1}>
          {[...Array(rowNumber)].map((_, index) => (
            <Skeleton
              key={`d${index}`}
              variant="rectangular"
              width="100%"
              height={40}
              sx={{ mb: index === rowNumber - 1 ? 0 : 3, borderRadius: 1 }}
            />
          ))}
        </Grid>
      </Grid>
    </Card>
  );
}
