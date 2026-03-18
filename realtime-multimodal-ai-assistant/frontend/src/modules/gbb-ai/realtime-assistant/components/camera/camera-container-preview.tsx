import arrowIosBackFill from '@iconify/icons-eva/arrow-ios-back-fill';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CircularProgress from '@mui/material/CircularProgress';

import Iconify from 'src/widgets/iconify';

import type { Conversation } from 'src/types/chat';

// ----------------------------------------------------------------------

type Props = {
  assignVideoRef: (node: HTMLVideoElement | null) => void;
  isLoading: boolean;
  error: string | null;
  isFloating: boolean;
  conversation?: Conversation;
  isListening?: boolean;
  isSpeaking?: boolean;
  returnToHomepage?: () => void;
};

export default function CameraPreviewContent({
  assignVideoRef,
  isLoading,
  error,
  isFloating,
  conversation,
  isListening,
  isSpeaking,
  returnToHomepage,
}: Props) {
  if (isLoading) {
    return (
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
          Connecting to camera…
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.72 }}>
          Grant permissions to preview the live feed here.
        </Typography>
      </Stack>
    );
  }

  if (error) {
    return (
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
        <Iconify icon="mdi:camera-off" width={40} height={40} />
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
          Camera unavailable
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.72 }}>
          {error}
        </Typography>
      </Stack>
    );
  }

  return (
    <Box sx={{ position: 'relative', width: '100%', height: '100%' }}>
      <video
        ref={assignVideoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          backgroundColor: 'black',
        }}
      />

      {/* Back button - only show when not floating and returnToHomepage is provided */}
      {!isFloating && returnToHomepage && (
        <IconButton
          color="inherit"
          onClick={returnToHomepage}
          sx={{ position: 'absolute', top: 8, left: 8, zIndex: 20 }}
        >
          <Iconify icon={arrowIosBackFill} />
        </IconButton>
      )}
    </Box>
  );
}
