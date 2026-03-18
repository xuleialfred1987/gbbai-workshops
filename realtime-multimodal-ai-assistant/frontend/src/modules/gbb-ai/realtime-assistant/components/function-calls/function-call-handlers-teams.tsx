import { useMemo } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

import { bgBlur } from 'src/custom/css';

import Iconify from 'src/widgets/iconify';
import ImageGallery, { useImageGallery } from 'src/widgets/overlay';

// ----------------------------------------------------------------------

type Props = {
  data?: {
    status: 'success' | 'error';
    message?: string;
    title?: string;
    has_image?: boolean;
    has_chart?: boolean;
    error_code?: number;
    error_message?: string;
  };
  originalMessage?: string;
  title?: string;
  imageUrl?: string;
};

export default function FunctionTeamsHandler({ data, originalMessage, title, imageUrl }: Props) {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const images = useMemo(() => (imageUrl ? [{ src: imageUrl }] : []), [imageUrl]);
  const lightbox = useImageGallery(images);

  const safeData = data ?? {
    status: 'error' as const,
    error_message: 'Teams response payload is missing.',
  };

  const isSuccess = safeData.status === 'success';
  const statusColor = isSuccess ? '#6264A7' : theme.palette.error.main;

  return (
    <Stack spacing={1.5} sx={{ width: 0.75, minWidth: '350px', p: 0.5, px: 0, pt: 0, mt: 2 }}>
      <Stack
        spacing={1.5}
        sx={{
          p: 2,
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
            opacity: isDarkMode ? 0.6 : 0.8,
          }),
          transition: theme.transitions.create(['background-color', 'border-color'], {
            duration: theme.transitions.duration.shorter,
          }),
        }}
      >
        {/* Status Header */}
        <Stack direction="row" alignItems="center" spacing={1.5}>
          {/* <Iconify
            icon={isSuccess ? 'mdi:check-circle' : 'mdi:alert-circle'}
            width={18}
            sx={{ color: statusColor }}
          /> */}
          <Iconify
            icon="mdi:microsoft-teams"
            width={20}
            sx={{ color: '#6264A7' }} // Microsoft Teams brand color
          />
          <Typography
            variant="subtitle1"
            sx={{
              fontSize: 15,
              fontWeight: 600,
              color: statusColor,
            }}
          >
            {isSuccess ? 'Teams Message Sent' : 'Failed to Send Message'}
          </Typography>
        </Stack>

        {/* Message Title */}
        {title && (
          <Stack direction="row" spacing={1}>
            <Typography
              variant="body2"
              sx={{
                fontSize: 13,
                fontWeight: 600,
                color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[700],
              }}
            >
              Title:
            </Typography>
            <Typography
              variant="body2"
              sx={{
                fontSize: 13,
                color: isDarkMode ? theme.palette.grey[300] : theme.palette.grey[800],
              }}
            >
              {title}
            </Typography>
          </Stack>
        )}

        {/* Original Message Preview */}
        {originalMessage && (
          <Stack spacing={0.5}>
            <Typography
              variant="body2"
              sx={{
                fontSize: 13,
                fontWeight: 600,
                color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[700],
              }}
            >
              Message:
            </Typography>
            <Typography
              variant="body2"
              sx={{
                fontSize: 13,
                color: isDarkMode ? theme.palette.grey[300] : theme.palette.grey[800],
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                maxHeight: '150px',
                overflow: 'auto',
                p: 1.5,
                bgcolor: isDarkMode
                  ? alpha(theme.palette.grey[900], 0.4)
                  : alpha(theme.palette.grey[200], 0.8),
                borderRadius: 1,
              }}
            >
              {originalMessage}
            </Typography>
          </Stack>
        )}

        {/* Image Preview */}
        {imageUrl && (
          <Stack spacing={0.5}>
            <Typography
              variant="body2"
              sx={{
                fontSize: 13,
                fontWeight: 600,
                color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[700],
              }}
            >
              {safeData.has_chart ? 'Chart:' : 'Image:'}
            </Typography>
            <Stack
              sx={{
                p: 1,
                bgcolor: isDarkMode
                  ? alpha(theme.palette.grey[900], 0.4)
                  : alpha(theme.palette.grey[50], 0.6),
                borderRadius: 1,
                overflow: 'hidden',
              }}
            >
              <Box
                component="img"
                alt={title || 'Teams attachment'}
                src={imageUrl}
                onClick={() => lightbox.onOpen(imageUrl)}
                sx={{
                  width: '100%',
                  height: 'auto',
                  maxHeight: '300px',
                  display: 'block',
                  objectFit: 'contain',
                  borderRadius: 0.75,
                  cursor: 'zoom-in',
                }}
              />
            </Stack>
          </Stack>
        )}

        {/* Attachments Info */}
        {isSuccess && (safeData.has_image || safeData.has_chart) && (
          <Stack direction="row" spacing={2} sx={{ pl: 0.75 }}>
            {safeData.has_image && (
              <Stack direction="row" alignItems="center" spacing={0.5}>
                <Iconify
                  icon="mdi:image"
                  width={16}
                  sx={{ color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[600] }}
                />
                <Typography
                  variant="caption"
                  sx={{
                    fontSize: 12,
                    color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[600],
                  }}
                >
                  Image attached
                </Typography>
              </Stack>
            )}
            {safeData.has_chart && (
              <Stack direction="row" alignItems="center" spacing={0.5}>
                <Iconify
                  icon="mdi:chart-bar"
                  width={16}
                  sx={{ color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[600] }}
                />
                <Typography
                  variant="caption"
                  sx={{
                    fontSize: 12,
                    color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[600],
                  }}
                >
                  Chart attached
                </Typography>
              </Stack>
            )}
          </Stack>
        )}

        {/* Error Details */}
        {!isSuccess && (
          <Stack spacing={0.5}>
            {safeData.error_code && (
              <Typography
                variant="caption"
                sx={{
                  fontSize: 12,
                  color: theme.palette.error.main,
                  fontWeight: 500,
                }}
              >
                Error Code: {safeData.error_code}
              </Typography>
            )}
            {safeData.error_message && (
              <Typography
                variant="caption"
                sx={{
                  fontSize: 12,
                  color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[700],
                  fontStyle: 'italic',
                }}
              >
                {safeData.error_message}
              </Typography>
            )}
          </Stack>
        )}

        {/* Success Message */}
        {/* {isSuccess && data.message && (
          <Stack direction="row" alignItems="center" spacing={0.5}>
            <Iconify
              icon="mdi:microsoft-teams"
              width={18}
              sx={{ color: '#6264A7' }} // Microsoft Teams brand color
            />
            <Typography
              variant="caption"
              sx={{
                fontSize: 12,
                color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[600],
                fontStyle: 'italic',
              }}
            >
              {data.message}
            </Typography>
          </Stack>
        )} */}
      </Stack>

      <ImageGallery
        index={lightbox.selected}
        slides={images}
        open={lightbox.open}
        close={lightbox.onClose}
      />
    </Stack>
  );
}
