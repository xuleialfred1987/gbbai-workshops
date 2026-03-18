import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Avatar from '@mui/material/Avatar';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

// ----------------------------------------------------------------------

type Props = {
  data: any;
};

export default function FunctionBingSearchHandler({ data }: Props) {
  const theme = useTheme();
  return (
    <Stack spacing={0.75} sx={{ width: '100%', px: 0.15, mt: 2, pb: 1 }}>
      <Typography variant="subtitle2" fontSize={13} color="grey.400">
        Sources
      </Typography>
      <Stack
        spacing={1.5}
        sx={{
          width: '100%',
          overflowX: 'auto',
          '&::-webkit-scrollbar': { display: 'none' },
          msOverflowStyle: 'none', // IE and Edge
          scrollbarWidth: 'none', // Firefox
        }}
      >
        <Stack direction="row" spacing={1.5} sx={{ minWidth: 'min-content' }}>
          {data.map((store: any, index: number) => {
            const { title, link, snippet } = store;
            return (
              <Card
                key={index}
                sx={{
                  px: 1,
                  p: 1.25,
                  width: 200,
                  borderRadius: 1.5,
                  boxShadow: 'none',
                  bgcolor: `${alpha(theme.palette.grey[300], 0.12)}`,
                  color: 'common.white',
                }}
                onClick={() => window.open(link, '_blank')}
              >
                <Stack spacing={1.25} sx={{ p: 0.5, px: 0.25, pb: 0, width: '100%' }}>
                  <Stack direction="row" spacing={1} alignItems="center" justifyContent="start">
                    <Avatar
                      alt={getFaviconUrl(link)}
                      src={getFaviconUrl(link)}
                      sx={{ width: 20, height: 20 }}
                    />
                    <Typography
                      variant="subtitle2"
                      color="grey.300"
                      textTransform="capitalize"
                      sx={{
                        fontSize: 13,
                        maxWidth: '130px', // Limit width to ensure ellipsis appears when needed
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        flexShrink: 1, // Allow it to shrink if needed
                      }}
                    >
                      {title}
                      {/* {getCompanyNameFromUrl(link)} */}
                    </Typography>
                  </Stack>
                  <Typography
                    variant="body2"
                    sx={{
                      fontSize: 13,
                      display: '-webkit-box',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      WebkitBoxOrient: 'vertical',
                      WebkitLineClamp: 2,
                    }}
                  >
                    {snippet}
                  </Typography>
                </Stack>
              </Card>
            );
          })}
        </Stack>
      </Stack>
    </Stack>
  );
}

function getFaviconUrl(url: string) {
  try {
    const { origin } = new URL(url);
    return `${origin}/favicon.ico`;
  } catch (error) {
    return '';
  }
}

// function getCompanyNameFromUrl(url: string) {
//   try {
//     const { hostname } = new URL(url);
//     const parts = hostname.split('.');
//     if (parts.length === 3) {
//       return parts[1];
//     }
//     if (parts.length > 3) {
//       return parts.slice(-3, -2)[0];
//     }
//     return '';
//   } catch (error) {
//     return '';
//   }
// }
