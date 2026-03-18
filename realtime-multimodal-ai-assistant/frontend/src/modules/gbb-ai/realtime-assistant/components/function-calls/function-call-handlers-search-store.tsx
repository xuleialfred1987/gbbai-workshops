import Stack from '@mui/material/Stack';
import Rating from '@mui/material/Rating';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

import { bgBlur } from 'src/custom/css';

import Image from 'src/widgets/img-wrap';
import Iconify from 'src/widgets/iconify';

// ----------------------------------------------------------------------

type Props = {
  data: any;
};

export default function FunctionPhoneStoreHandler({ data }: Props) {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  
  return (
    <Stack spacing={1.5} sx={{ width: 0.5, minWidth: '300px', p: 0.5, px: 0, pt: 0, mt: 2 }}>
      {data.map((store: any, index: number) => {
        const { title, address, rating, reviews, image } = store;
        return (
          <Stack
            key={index}
            direction="row"
            justifyContent="space-between"
            alignItems="center"
            sx={{
              p: 1.25,
              px: 1.5,
              bgcolor: isDarkMode 
                ? alpha(theme.palette.grey[800], 0.6)
                : alpha(theme.palette.grey[100], 0.8),
              border: `1px solid ${alpha(
                isDarkMode ? theme.palette.grey[700] : theme.palette.grey[300], 
                0.5
              )}`,
              borderRadius: 1.5,
              color: isDarkMode ? 'common.white' : 'text.primary',
              ...bgBlur({ 
                color: isDarkMode ? theme.palette.grey[900] : theme.palette.common.white, 
                opacity: isDarkMode ? 0.6 : 0.8 
              }),
              transition: theme.transitions.create(['background-color', 'border-color', 'box-shadow'], {
                duration: theme.transitions.duration.shorter,
              }),
              '&:hover': {
                bgcolor: isDarkMode 
                  ? alpha(theme.palette.grey[800], 0.8)
                  : alpha(theme.palette.grey[200], 0.9),
                borderColor: alpha(
                  isDarkMode ? theme.palette.grey[600] : theme.palette.grey[400], 
                  0.6
                ),
                boxShadow: theme.shadows[4],
              },
            }}
          >
            <Stack direction="row" alignItems="center" spacing={1}>
              <Image
                alt={title}
                src={image}
                sx={{ width: 64, height: 64, borderRadius: 0.75, mr: 0.25 }}
              />
              <Stack alignItems="flex-start" justifyContent="flex-start" sx={{ pt: 0 }}>
                <Typography
                  variant="body1"
                  noWrap
                  sx={{
                    fontSize: 14,
                    fontWeight: 600,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {title}
                </Typography>
                <Rating
                  value={rating}
                  readOnly
                  size="small"
                  sx={{ transform: 'scale(0.7)', ml: -2, mr: -2, my: 0.5 }}
                />
                <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mt: 0, ml: -0.2 }}>
                  <Iconify 
                    color={isDarkMode ? theme.palette.grey[400] : theme.palette.grey[600]} 
                    icon="mdi:location" 
                    width={16} 
                  />
                  <Typography 
                    variant="caption" 
                    color={isDarkMode ? theme.palette.grey[400] : theme.palette.grey[600]}
                  >
                    {address}
                  </Typography>
                  {/* <Typography
                    variant="caption"
                    color={`${theme.palette.grey[400]}`}
                    sx={{ mx: 0.5 }}
                  >
                    •
                  </Typography>
                  <Typography variant="caption" color={`${theme.palette.grey[400]}`}>
                    {distance}
                  </Typography> */}
                </Stack>
              </Stack>
            </Stack>
            <Typography
              variant="body2"
              noWrap
              sx={{ fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis' }}
            >{`${reviews} Reviews`}</Typography>
          </Stack>
        );
      })}
    </Stack>
  );
}
