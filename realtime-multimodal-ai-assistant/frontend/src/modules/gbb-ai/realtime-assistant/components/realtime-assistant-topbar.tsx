import { Icon } from '@iconify/react';
import { Link as RouterLink } from 'react-router-dom';
import arrowIosBackFill from '@iconify/icons-eva/arrow-ios-back-fill';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Tooltip from '@mui/material/Tooltip';
import { useTheme } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';

import { DEFAULT_PATH } from 'src/config-global';

import Iconify from 'src/widgets/iconify';
import SvgColor from 'src/widgets/svg-color';

import ConnectionStatusBar from './connection-status-bar';

type Props = {
  isChipConnected: boolean;
  isChipDisconnected: boolean;
  isChipConnecting: boolean;
  isRecording: boolean;
  isBackendReady: boolean;
  isBackendUnavailable: boolean;
  onRenewSession: () => void;
  onClearHistory: () => void;
  onOpenConfig: () => void;
};

export default function RealtimeAssistantTopBar({
  isChipConnected,
  isChipDisconnected,
  isChipConnecting,
  isRecording,
  isBackendReady,
  isBackendUnavailable,
  onRenewSession,
  onClearHistory,
  onOpenConfig,
}: Props) {
  const theme = useTheme();

  return (
    <Stack
      zIndex={10}
      direction="row"
      justifyContent="space-between"
      alignItems="center"
      sx={{ mb: 0, position: 'relative' }}
    >
      <Button
        to={DEFAULT_PATH}
        component={RouterLink}
        size="small"
        color="inherit"
        startIcon={<Icon icon={arrowIosBackFill} style={{ marginRight: '-5px' }} />}
        sx={{ display: 'flex' }}
      >
        Home
      </Button>

      <Stack direction="row" alignItems="center">
        <Box sx={{ mr: 1.5 }}>
          <ConnectionStatusBar
            isConnected={isChipConnected}
            isDisconnected={isChipDisconnected}
            isConnecting={isChipConnecting}
            isRecording={isRecording}
            isBackendReady={isBackendReady}
            isBackendUnavailable={isBackendUnavailable}
            onRenewSession={onRenewSession}
          />
        </Box>

        <Tooltip title="New chat">
          <IconButton
            size="small"
            color="default"
            onClick={onClearHistory}
            sx={{ width: 36, height: 36 }}
          >
            <Iconify
              icon="solar:pen-new-round-bold-duotone"
              width={20}
              height={20}
              color={theme.palette.text.secondary}
            />
          </IconButton>
        </Tooltip>

        <IconButton size="small" color="default" onClick={onOpenConfig} sx={{ width: 36, height: 36 }}>
          <SvgColor src="/assets/icons/ic_settings.svg" sx={{ width: 20, height: 20 }} />
        </IconButton>
      </Stack>
    </Stack>
  );
}