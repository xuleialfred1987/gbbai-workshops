import { useMemo, Fragment, useState, useEffect } from 'react';

import { alpha, useTheme } from '@mui/material/styles';
import {
  List,
  Stack,
  Divider,
  Tooltip,
  IconButton,
  Typography,
  ListItemIcon,
  ListItemText,
  ListItemButton,
} from '@mui/material';

import Iconify from 'src/widgets/iconify';
import StyledPopover from 'src/widgets/styled-popover/styled-popover';

import {
  type RecordingOption,
  createRecordingOptions,
  type RecordingOptionId,
} from './screen-recording-container';

// ----------------------------------------------------------------------

type Props = {
  onRequestScreenRecorder: (
    optionId: RecordingOptionId,
    stream: MediaStream | null,
    isPending: boolean
  ) => void;
  onToggleScreenRecorder: () => void;
  isScreenRecorderOpen: boolean;
  size?: 'small' | 'medium';
  width?: number;
  height?: number;
};

export default function ScreenRecordingButton({
  onRequestScreenRecorder,
  onToggleScreenRecorder,
  isScreenRecorderOpen,
  size = 'small',
  width = 36,
  height = 36,
}: Props) {
  const theme = useTheme();
  const isLight = theme.palette.mode === 'light';

  const [recordPopoverAnchor, setRecordPopoverAnchor] = useState<HTMLElement | null>(null);

  const recordingOptions = useMemo(createRecordingOptions, []);
  const isPopoverOpen = Boolean(recordPopoverAnchor);

  const handleScreenRecorderClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    if (isScreenRecorderOpen) {
      onToggleScreenRecorder();
      setRecordPopoverAnchor(null);
      return;
    }

    // Always show popover when screen recorder is not open
    setRecordPopoverAnchor(event.currentTarget);
  };

  const handlePopoverClose = () => {
    setRecordPopoverAnchor(null);
  };

  const handleRecordingOptionSelect = async (option: RecordingOption) => {
    // Close the popover first
    setRecordPopoverAnchor(null);

    // Check browser support
    if (!navigator.mediaDevices?.getDisplayMedia) {
      alert('Screen recording is not supported in this browser.');
      return;
    }

    const openedBySelection = !isScreenRecorderOpen;

    if (openedBySelection) {
      onToggleScreenRecorder();
    }

    onRequestScreenRecorder(option.id, null, true);

    try {
      const stream = await navigator.mediaDevices.getDisplayMedia(option.constraints);
      onRequestScreenRecorder(option.id, stream, false);
    } catch (err) {
      onRequestScreenRecorder(option.id, null, false);

      if (openedBySelection) {
        onToggleScreenRecorder();
      }
    }
  };

  useEffect(() => {
    if (isScreenRecorderOpen) {
      setRecordPopoverAnchor(null);
    }
  }, [isScreenRecorderOpen]);

  // Button styling based on state
  let buttonColor = theme.palette.text.secondary;
  if (isScreenRecorderOpen) {
    buttonColor = theme.palette.error.contrastText;
  } else if (isPopoverOpen) {
    buttonColor = theme.palette.primary.main;
  }

  let buttonBg: string | undefined = 'transparent';
  if (isScreenRecorderOpen) {
    buttonBg = theme.palette.error.main;
  } else if (isPopoverOpen) {
    buttonBg = alpha(theme.palette.primary.main, isLight ? 0.16 : 0.24);
  }

  const hoverBg = isScreenRecorderOpen
    ? theme.palette.error.dark
    : alpha(theme.palette.primary.main, isLight ? 0.24 : 0.32);

  return (
    <>
      <Tooltip
        title={isScreenRecorderOpen ? 'Close screen recorder' : 'Screen Record'}
        arrow
        disableInteractive
      >
        <IconButton
          size={size}
          onClick={handleScreenRecorderClick}
          sx={{
            width,
            height,
            bgcolor: buttonBg,
            color: buttonColor,
            '&:hover': {
              bgcolor: hoverBg,
            },
          }}
        >
          <Iconify
            icon="fluent:share-screen-start-24-regular"
            width={20}
            height={20}
            color="currentColor"
          />
        </IconButton>
      </Tooltip>

      <StyledPopover
        open={recordPopoverAnchor}
        onClose={handlePopoverClose}
        arrow="bottom-center"
        sx={{
          pt: 1.75,
          pb: 1.25,
          width: 300,
          transform: 'translate(-0px, -8px) !important',
        }}
      >
        <Stack spacing={1.5}>
          <Stack spacing={0.75} sx={{ p: 1.5, py: 0 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              Screen recording
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Choose what you want to share with the assistant.
            </Typography>
          </Stack>

          {/* <Divider sx={{ my: 0.5, mb: 0 }} /> */}

          <List disablePadding sx={{ p: 0.5 }}>
            {recordingOptions.map((option, index) => (
              <Fragment key={option.id}>
                <ListItemButton
                  onClick={() => handleRecordingOptionSelect(option)}
                  sx={{ p: 1, px: 1.25, borderRadius: 1 }}
                >
                  <ListItemIcon sx={{ minWidth: 22, color: 'primary.main' }}>
                    <Iconify icon={option.icon} width={18} height={18} color="currentColor" />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.25 }}>
                        {option.label}
                      </Typography>
                    }
                    secondary={
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{
                          lineHeight: 1.5,
                          display: 'block',
                          m: 0,
                          p: 0,
                        }}
                      >
                        {option.description}
                      </Typography>
                    }
                  />
                </ListItemButton>

                {index < recordingOptions.length - 1 && (
                  <Divider component="li" sx={{ mx: 1, my: 1 }} />
                )}
              </Fragment>
            ))}
          </List>
        </Stack>
      </StyledPopover>
    </>
  );
}
