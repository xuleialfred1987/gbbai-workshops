import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';

import type { PaperLikeBoxProps } from './camera-container-types';

// ----------------------------------------------------------------------

export default function PaperLikeBox({
  isFloating,
  size,
  header,
  content,
  handles,
  sx,
  onPointerDown,
}: PaperLikeBoxProps) {
  const borderRadius = 1.25;
  const innerRadius = 0.825;

  const style: Record<string, unknown> = {
    width: isFloating ? size.width : '100%',
    height: isFloating ? size.height : '100%',
    borderRadius,
    boxShadow: isFloating ? '0 12px 40px rgba(15,23,42,0.32)' : '0 12px 36px rgba(15,23,42,0.18)',
    overflow: 'hidden',
    backgroundColor: isFloating ? 'rgba(15, 23, 42, 0.94)' : 'rgba(15, 23, 42, 0.9)',
    border: isFloating
      ? '1px solid rgba(148, 163, 184, 0.24)'
      : '1px solid rgba(148, 163, 184, 0.12)',
    backdropFilter: 'blur(24px)',
    position: isFloating ? 'fixed' : 'relative',
    display: 'flex',
    flexDirection: 'column',
  };

  return (
    <Box sx={{ ...style, ...sx }} onPointerDown={onPointerDown}>
      <Stack spacing={1} sx={{ p: 1.25, flex: 1, color: 'common.white', minHeight: 0 }}>
        {header}
        <Box
          sx={{
            position: 'relative',
            flex: 1,
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
