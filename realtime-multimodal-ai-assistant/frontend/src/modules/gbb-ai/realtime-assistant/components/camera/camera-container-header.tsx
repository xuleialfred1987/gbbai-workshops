import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';

import Iconify from 'src/widgets/iconify';

import type { FloatingHeaderProps } from './camera-container-types';

// ----------------------------------------------------------------------

export default function FloatingHeader({
  title,
  isFloating,
  isDragging,
  onPointerDown,
  onToggleFloating,
  onClose,
  onDoubleClick,
  onFlipCamera,
}: FloatingHeaderProps) {
  let headerCursor: 'default' | 'grab' | 'grabbing' = 'default';
  if (isFloating) {
    headerCursor = isDragging ? 'grabbing' : 'grab';
  }

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        gap: 1,
        cursor: headerCursor,
      }}
      onPointerDown={onPointerDown}
      onDoubleClick={isFloating ? onDoubleClick : undefined}
    >
      <Box sx={{ pr: onToggleFloating || onClose ? 1 : 0 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, mt: 0.35 }}>
          {title || 'Camera preview'}
        </Typography>
      </Box>

      <Stack direction="row" spacing={0.75} alignItems="center">
        {onFlipCamera && (
          <Tooltip title="Flip camera" arrow disableInteractive>
            <IconButton size="small" color="inherit" onClick={onFlipCamera}>
              <Iconify icon="mdi:camera-flip" width={16} height={16} />
            </IconButton>
          </Tooltip>
        )}

        {onToggleFloating && (
          <Tooltip title={isFloating ? 'Dock camera' : 'Float camera'} arrow disableInteractive>
            <IconButton size="small" color="inherit" onClick={onToggleFloating}>
              <Iconify
                icon={isFloating ? 'mdi:dock-window' : 'mdi:arrow-expand-all'}
                width={16}
                height={16}
              />
            </IconButton>
          </Tooltip>
        )}

        {onClose && (
          <Tooltip title="Close panel" arrow disableInteractive>
            <IconButton size="small" color="inherit" onClick={onClose}>
              <Iconify icon="mdi:close" width={16} height={16} />
            </IconButton>
          </Tooltip>
        )}
      </Stack>
    </Box>
  );
}
