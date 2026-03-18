import { useRef, useEffect } from 'react';

// mui
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Dialog from '@mui/material/Dialog';
import Divider from '@mui/material/Divider';
import Backdrop from '@mui/material/Backdrop';
import { useTheme } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';

// project imports
import Iconify from 'src/widgets/iconify';

// ----------------------------------------------------------------------

// Extend window interface
declare global {
  interface Window {
    graphvizFullscreenPanzoom?: any;
  }
}

interface GraphVizFullscreenDialogProps {
  open: boolean;
  svgContent: string;
  onClose: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitView: () => void;
  onDownload: () => void;
}

export default function GraphVizFullscreenDialog({
  open,
  svgContent,
  onClose,
  onZoomIn,
  onZoomOut,
  onFitView,
  onDownload,
}: GraphVizFullscreenDialogProps) {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const panZoomRef = useRef<any>(null);

  // Initialize panzoom for fullscreen view
  useEffect(() => {
    if (!open || !svgContent) {
      return undefined;
    }

    // Wait for the next tick to ensure dialog is mounted
    const timer = setTimeout(() => {
      if (containerRef.current && svgContent) {
        containerRef.current.innerHTML = svgContent;
        const svgElement = containerRef.current.querySelector('svg');

        if (svgElement) {
          svgElement.style.width = '100%';
          svgElement.style.height = '100%';

          // eslint-disable-next-line import/no-extraneous-dependencies
          import('panzoom').then((panzoomModule) => {
            panZoomRef.current = panzoomModule.default(svgElement, {
              maxZoom: 10,
              minZoom: 0.1,
              bounds: true,
              boundsPadding: 0.1,
              smoothScroll: false,
            });

            // Expose panzoom instance globally after initialization
            window.graphvizFullscreenPanzoom = panZoomRef.current;
          });
        }
      }
    }, 200);

    return () => {
      clearTimeout(timer);
      if (panZoomRef.current) {
        panZoomRef.current.dispose();
        panZoomRef.current = null;
        window.graphvizFullscreenPanzoom = null;
      }
    };
  }, [svgContent, open]);

  return (
    <>
      {/* Backdrop with blur effect */}
      <Backdrop
        open={open}
        sx={{
          zIndex: (_theme) => _theme.zIndex.modal - 1,
          backdropFilter: 'blur(8px)',
          backgroundColor: 'rgba(0, 0, 0, 0.18)',
        }}
      />

      {/* Fullscreen Dialog */}
      <Dialog
        open={open}
        onClose={onClose}
        maxWidth={false}
        fullScreen
        sx={{
          '& .MuiDialog-paper': {
            bgcolor: 'transparent',
            boxShadow: 'none',
          },
        }}
      >
        {/* Full screen container */}
        <Box
          sx={{
            width: '100%',
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
            position: 'relative',
          }}
        >
          {/* Control toolbar */}
          <Box
            sx={{
              position: 'absolute',
              top: 32,
              right: 32,
              zIndex: 1300,
              bgcolor: 'background.neutral',
              borderRadius: 1,
              // boxShadow: theme.shadows[2],
              border: `1px solid ${theme.palette.divider}`,
              p: 0.25,
            }}
          >
            <Stack direction="row" spacing={0.5}>
              <IconButton
                onClick={onZoomIn}
                sx={{
                  '&:hover': {
                    bgcolor: 'action.hover',
                  },
                }}
              >
                <Iconify icon="eva:plus-circle-outline" width={20} />
              </IconButton>
              <Divider orientation="vertical" flexItem sx={{ borderStyle: 'dashed' }} />
              <IconButton
                onClick={onZoomOut}
                sx={{
                  '&:hover': {
                    bgcolor: 'action.hover',
                  },
                }}
              >
                <Iconify icon="eva:minus-circle-outline" width={20} />
              </IconButton>
              <Divider orientation="vertical" flexItem sx={{ borderStyle: 'dashed' }} />
              <IconButton
                onClick={onFitView}
                sx={{
                  '&:hover': {
                    bgcolor: 'action.hover',
                  },
                }}
              >
                <Iconify icon="eva:expand-outline" width={20} />
              </IconButton>
              <Divider orientation="vertical" flexItem sx={{ borderStyle: 'dashed' }} />
              <IconButton
                onClick={onDownload}
                sx={{
                  '&:hover': {
                    bgcolor: 'action.hover',
                  },
                }}
              >
                <Iconify icon="eva:download-outline" width={20} />
              </IconButton>
              <Divider orientation="vertical" flexItem sx={{ borderStyle: 'dashed' }} />
              <IconButton
                onClick={onClose}
                sx={{
                  '&:hover': {
                    bgcolor: 'action.hover',
                  },
                }}
              >
                <Iconify icon="eva:close-outline" width={20} />
              </IconButton>
            </Stack>
          </Box>

          {/* Control hints */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 24,
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 1300,
              bgcolor: 'background.paper',
              borderRadius: 1,
              px: 2,
              py: 1,
              fontSize: '14px',
              color: 'text.secondary',
              boxShadow: theme.shadows[4],
              pointerEvents: 'none',
            }}
          >
            Use mouse wheel to zoom • Drag to pan
          </Box>

          {/* Graph container */}
          <Box
            sx={{
              flex: 1,
              m: 3,
              position: 'relative',
              bgcolor: 'background.paper',
              borderRadius: 1.25,
              boxShadow: theme.shadows[8],
              overflow: 'hidden',
            }}
          >
            <Box
              ref={containerRef}
              sx={{
                width: '100%',
                height: '100%',
                cursor: 'grab',
                '&:active': {
                  cursor: 'grabbing',
                },
                '& svg': {
                  maxWidth: 'none !important',
                  maxHeight: 'none !important',
                },
              }}
            />
          </Box>
        </Box>
      </Dialog>
    </>
  );
}
