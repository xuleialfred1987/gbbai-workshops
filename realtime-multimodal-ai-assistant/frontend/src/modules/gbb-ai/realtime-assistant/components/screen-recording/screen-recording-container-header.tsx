import type { PointerEvent as ReactPointerEvent } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';

import Iconify from 'src/widgets/iconify';

import { SCREEN_HEADER_TITLE } from './screen-recording-container-types';
import type { RecordingOption } from './screen-recording-container-types';

// ----------------------------------------------------------------------

type Props = {
  isFloating: boolean;
  isDragging: boolean;
  onPointerDown?: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onDoubleClick?: () => void;
  onToggleFloating?: () => void;
  onClose?: () => void;
  selectedOption: RecordingOption | null;
  hasActiveStream: boolean;
};

export default function ScreenRecordingHeader({
  isFloating,
  isDragging,
  onPointerDown,
  onDoubleClick,
  onToggleFloating,
  onClose,
  selectedOption,
  hasActiveStream,
}: Props) {
  let cursor: 'default' | 'grab' | 'grabbing' = 'default';
  if (isFloating) {
    cursor = isDragging ? 'grabbing' : 'grab';
  }

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        gap: 1,
        cursor,
      }}
      onPointerDown={onPointerDown}
      onDoubleClick={isFloating ? onDoubleClick : undefined}
    >
      <Box sx={{ pr: onToggleFloating || onClose ? 1 : 0 }}>
        {!hasActiveStream && (
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mt: 0.35 }}>
            {SCREEN_HEADER_TITLE}
          </Typography>
        )}

        {hasActiveStream && (
          <Stack spacing={0.25}>
            <Typography variant="subtitle2" fontSize={13} color="text.secondary">
              Currently sharing:
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {selectedOption?.label ?? ''}
            </Typography>
          </Stack>
        )}
      </Box>

      <Stack direction="row" spacing={0.75} alignItems="center">
        {onToggleFloating && (
          <Tooltip title={isFloating ? 'Dock recorder' : 'Float recorder'} arrow disableInteractive>
            <IconButton size="small" color="inherit" onClick={onToggleFloating}>
              <Iconify
                icon={isFloating ? 'mdi:dock-window' : 'mdi:arrow-expand-all'}
                width={18}
                height={18}
              />
            </IconButton>
          </Tooltip>
        )}

        {onClose && (
          <Tooltip title="Close panel" arrow disableInteractive>
            <IconButton size="small" color="inherit" onClick={onClose}>
              <Iconify icon="mdi:close" width={18} height={18} />
            </IconButton>
          </Tooltip>
        )}
      </Stack>
    </Box>
  );
}
