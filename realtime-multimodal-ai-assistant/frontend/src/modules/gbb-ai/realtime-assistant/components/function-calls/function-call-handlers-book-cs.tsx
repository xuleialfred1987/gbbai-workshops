import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

import { fDateTimeYMdHm } from 'src/utils/time-formatter';

import { bgBlur } from 'src/custom/css';

import Image from 'src/widgets/img-wrap';
import Iconify from 'src/widgets/iconify';

// ----------------------------------------------------------------------

type Props = {
  data: any;
};

export default function FunctionBookCsHandler({ data }: Props) {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const { booking_id, customer, device, service, date, time, device_image, location } = data;

  let bookingDateTime = new Date();
  if (date && time) {
    bookingDateTime = new Date(`${date}T${time}`);
  } else if (date) {
    bookingDateTime = new Date(date);
  }

  return (
    <Stack
      spacing={3.5}
      alignItems="center"
      justifyContent="space-between"
      sx={{
        p: 2,
        pt: 2.5,
        px: 2,
        mt: 2,
        width: 0.5,
        minWidth: '300px',
        bgcolor: isDarkMode
          ? alpha(theme.palette.grey[800], 0.6)
          : alpha(theme.palette.grey[100], 0.8),
        border: `1px solid ${alpha(
          isDarkMode ? theme.palette.grey[700] : theme.palette.grey[300],
          0.5
        )}`,
        borderRadius: 1.75,
        color: isDarkMode ? 'common.white' : 'text.primary',
        ...bgBlur({
          color: isDarkMode ? theme.palette.grey[800] : theme.palette.common.white,
          opacity: isDarkMode ? 0.44 : 0.7,
        }),
        transition: theme.transitions.create(['background-color', 'border-color'], {
          duration: theme.transitions.duration.shorter,
        }),
      }}
    >
      <Stack alignItems="center" spacing={1}>
        <Iconify icon="solar:verified-check-bold" width={32} sx={{ color: 'primary.main' }} />
        <Typography variant="h6" sx={{ mr: 0 }}>
          Booking done
        </Typography>
        <Typography
          variant="caption"
          color={isDarkMode ? theme.palette.grey[300] : theme.palette.grey[600]}
          sx={{ mt: -0.25 }}
        >
          ID: {booking_id}
        </Typography>
      </Stack>

      <Stack
        rowGap={5}
        columnGap={5}
        flexWrap="wrap"
        direction="row"
        alignItems="center"
        sx={{
          color: isDarkMode ? theme.palette.grey[300] : theme.palette.grey[700],
          typography: 'caption',
        }}
      >
        <Stack direction="row" alignItems="center">
          <Iconify width={14} icon="lets-icons:date-fill" sx={{ mr: 1.25, flexShrink: 0 }} />
          {fDateTimeYMdHm(bookingDateTime)}
        </Stack>

        <Stack direction="row" alignItems="center">
          <Iconify
            width={16}
            icon="material-symbols:person-rounded"
            sx={{ mr: 0.85, flexShrink: 0 }}
          />
          {customer}
        </Stack>
      </Stack>

      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        sx={{
          p: 1.5,
          width: 1,
          borderRadius: 1,
          bgcolor: isDarkMode
            ? alpha(theme.palette.grey[700], 0.5)
            : alpha(theme.palette.grey[200], 0.6),
          color: isDarkMode ? 'common.white' : 'text.primary',
          ...bgBlur({
            color: isDarkMode ? theme.palette.grey[600] : theme.palette.grey[100],
            opacity: 0.44,
          }),
        }}
      >
        <Stack direction="row" alignItems="center" spacing={1}>
          <Image
            alt="galaxy"
            src={device_image}
            ratio="18/9"
            sx={{ width: 56, height: 56, borderRadius: 0.5, mr: 0.25 }}
          />
          <Stack spacing={1} alignItems="flex-start" justifyContent="flex-start" sx={{ pt: 0 }}>
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
              {device}
            </Typography>

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
                {location}
              </Typography>
            </Stack>
          </Stack>
        </Stack>
        <Chip
          size="small"
          color={service.toLowerCase() === 'repair' ? 'warning' : 'success'}
          variant="filled"
          icon={
            <Iconify
              icon={service.toLowerCase() === 'repair' ? 'hugeicons:repair' : 'hugeicons:repair'}
              sx={{ width: 13, height: 13 }}
            />
          }
          label={service}
          sx={{ height: 23, p: 0.75, pr: 0.25, textTransform: 'capitalize' }}
        />
      </Stack>
    </Stack>
  );
}
