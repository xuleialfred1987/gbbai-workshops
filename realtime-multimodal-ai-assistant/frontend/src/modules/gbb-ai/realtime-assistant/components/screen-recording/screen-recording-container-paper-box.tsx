import type { ReactNode } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';

import type { FloatingSize, FloatingPosition } from './screen-recording-container-types';

// ----------------------------------------------------------------------

type Props = {
  isFloating: boolean;
  position: FloatingPosition;
  size: FloatingSize;
  header: ReactNode;
  content: ReactNode;
  handles: boolean;
  onPointerDown?: (event: React.PointerEvent<HTMLDivElement>) => void;
};

export default function ScreenRecordingPaperBox({
  isFloating,
  position,
  size,
  header,
  content,
  handles,
  onPointerDown,
}: Props) {
  const borderRadius = 1.25;
  const innerRadius = 0.825;

  return (
    <Box
      sx={{
        position: isFloating ? 'fixed' : 'relative',
        top: isFloating ? position.top : undefined,
        left: isFloating ? position.left : undefined,
        width: isFloating ? size.width : '100%',
        height: isFloating ? size.height : '100%',
        zIndex: isFloating ? 1400 : 'auto',
        boxShadow: isFloating ? 24 : 12,
        borderRadius,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'rgba(15, 23, 42, 0.92)',
        border: '1px solid rgba(148, 163, 184, 0.18)',
        backdropFilter: isFloating ? 'blur(16px)' : 'none',
      }}
      onPointerDown={onPointerDown}
    >
      <Stack spacing={1} sx={{ p: 1.25, flexGrow: 1, color: 'common.white', minHeight: 0 }}>
        {header}
        <Box
          sx={{
            position: 'relative',
            flexGrow: 1,
            borderRadius: innerRadius,
            overflow: 'hidden',
            minHeight: 0,
          }}
        >
          {content}
        </Box>
      </Stack>

      {handles && (
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            pointerEvents: 'none',
            '& > span': {
              pointerEvents: 'auto',
              position: 'absolute',
              backgroundColor: 'transparent',
              zIndex: 2,
            },
          }}
        >
          <span
            data-resize-handle="bottom-right"
            style={{ right: 0, bottom: 0, width: 18, height: 18, cursor: 'nwse-resize' }}
          />
          <span
            data-resize-handle="bottom-left"
            style={{ left: 0, bottom: 0, width: 18, height: 18, cursor: 'nesw-resize' }}
          />
          <span
            data-resize-handle="bottom"
            style={{
              left: '50%',
              bottom: -2,
              width: '38%',
              height: 16,
              cursor: 'ns-resize',
              transform: 'translateX(-50%)',
            }}
          />
        </Box>
      )}
    </Box>
  );
}
