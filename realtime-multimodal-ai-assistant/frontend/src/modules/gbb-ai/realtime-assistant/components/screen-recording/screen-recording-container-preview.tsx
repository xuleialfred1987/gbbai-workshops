import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';

import Iconify from 'src/widgets/iconify';

import type { RecordingOption } from './screen-recording-container-types';

// ----------------------------------------------------------------------

type Props = {
  assignVideoRef: (node: HTMLVideoElement | null) => void;
  isLoading: boolean;
  selectedStream: MediaStream | null;
  selectedOption: RecordingOption | null;
  error: string | null;
  minHeight: number;
};

export default function ScreenRecordingPreview({
  assignVideoRef,
  isLoading,
  selectedStream,
  selectedOption,
  error,
  minHeight,
}: Props) {
  let content: React.ReactNode;

  if (isLoading) {
    content = (
      <Stack
        sx={{
          position: 'absolute',
          inset: 0,
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1.5,
          textAlign: 'center',
          px: 3,
        }}
      >
        <CircularProgress color="inherit" size={36} thickness={4} />
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
          Preparing your {selectedOption?.label?.toLowerCase() ?? 'screen share'}…
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.72 }}>
          We&apos;ll start streaming as soon as the browser provides access.
        </Typography>
      </Stack>
    );
  } else if (selectedStream) {
    content = (
      <video
        ref={assignVideoRef}
        muted
        playsInline
        autoPlay
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          objectPosition: 'center',
          backgroundColor: 'black',
        }}
      />
    );
  } else {
    content = (
      <Stack
        sx={{
          position: 'absolute',
          inset: 0,
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          px: 2,
        }}
        spacing={1.5}
      >
        <Iconify icon="mdi:monitor-screenshot" width={48} height={48} />
        <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
          No active recording
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.68 }}>
          Choose a screen or window to preview it here.
        </Typography>
      </Stack>
    );
  }

  return (
    <Stack
      spacing={1}
      sx={{
        position: 'absolute',
        inset: 0,
        p: 1,
        pt: 0.5,
      }}
    >
      <Box
        sx={{
          position: 'relative',
          flexGrow: 1,
          minHeight,
          borderRadius: 0.825,
          bgcolor: 'grey.900',
          color: 'common.white',
          overflow: 'hidden',
        }}
      >
        {content}
      </Box>

      {error && (
        <Typography variant="caption" color="error" sx={{ mt: -0.5 }}>
          {error}
        </Typography>
      )}
    </Stack>
  );
}
