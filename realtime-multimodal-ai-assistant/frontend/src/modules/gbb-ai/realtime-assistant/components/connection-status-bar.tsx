import { alpha } from '@mui/material/styles';
import { Box, Stack, Tooltip, Typography, IconButton } from '@mui/material';

import Iconify from 'src/widgets/iconify';

// ----------------------------------------------------------------------

interface ConnectionStatusBarProps {
  isConnected: boolean;
  isDisconnected: boolean;
  isConnecting: boolean;
  isRecording?: boolean;
  isBackendReady?: boolean;
  isBackendUnavailable?: boolean;
  onRenewSession: () => void;
}

export default function ConnectionStatusBar({
  isConnected,
  isDisconnected,
  isConnecting,
  isRecording = false,
  isBackendReady = false,
  isBackendUnavailable = false,
  onRenewSession,
}: ConnectionStatusBarProps) {
  const getStatusText = () => {
    if (isBackendUnavailable) return 'Backend unavailable';
    if (isRecording) return 'Listening';
    if (isConnected) return 'Realtime active';
    if (isConnecting) return 'Starting session...';
    if (isDisconnected) return 'Connection lost';
    if (isBackendReady) return 'Backend ready';
    return 'Ready to connect';
  };

  //   const getStatusColor = () => {
  //     if (isDisconnected) return 'error';
  //     if (isConnected) return 'success';
  //     return 'info';
  //   };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 2,
        py: 0.5,
        borderRadius: 1,
        bgcolor: (theme) => {
          if (isBackendUnavailable) return alpha(theme.palette.error.main, 0.1);
          if (isConnected) return alpha(theme.palette.success.main, 0.1);
          if (isConnecting) return alpha(theme.palette.warning.main, 0.1);
          if (isDisconnected) return alpha(theme.palette.error.main, 0.1);
          if (isBackendReady) return alpha(theme.palette.info.main, 0.14);
          return alpha(theme.palette.info.main, 0.1);
        },
        border: 1,
        borderColor: (theme) => {
          if (isBackendUnavailable) return alpha(theme.palette.error.main, 0.2);
          if (isConnected) return alpha(theme.palette.success.main, 0.2);
          if (isConnecting) return alpha(theme.palette.warning.main, 0.2);
          if (isDisconnected) return alpha(theme.palette.error.main, 0.2);
          if (isBackendReady) return alpha(theme.palette.info.main, 0.28);
          return alpha(theme.palette.info.main, 0.2);
        },
        transition: 'all 0.2s ease-in-out',
        minWidth: 100,
      }}
    >
      <Stack direction="row" alignItems="center" spacing={1}>
        <Box
          sx={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            bgcolor: (theme) => {
              if (isBackendUnavailable) return theme.palette.error.main;
              if (isConnected) return theme.palette.success.main;
              if (isConnecting) return theme.palette.warning.main;
              if (isDisconnected) return theme.palette.error.main;
              if (isBackendReady) return theme.palette.info.main;
              return theme.palette.info.main;
            },
            animation:
              isConnected || isDisconnected || isConnecting || isBackendUnavailable
                ? 'pulse 2s infinite'
                : 'none',
            '@keyframes pulse': {
              '0%': { opacity: 1 },
              '50%': { opacity: 0.5 },
              '100%': { opacity: 1 },
            },
          }}
        />

        <Typography variant="caption" sx={{ fontSize: 11, fontWeight: 500 }}>
          {getStatusText()}
        </Typography>

        {(isDisconnected || isBackendUnavailable) && (
          <Tooltip title="Renew Session">
            <IconButton
              size="small"
              color="inherit"
              onClick={onRenewSession}
              sx={{
                p: 0.25,
                mr: -0.5,
                width: 18,
                height: 18,
              }}
            >
              <Iconify icon="solar:refresh-bold" width={16} height={16} />
            </IconButton>
          </Tooltip>
        )}
      </Stack>
    </Box>
  );
}
