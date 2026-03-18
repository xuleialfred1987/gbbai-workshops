import { useRef, useState, useEffect } from 'react';

// mui
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import { useTheme } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';
// import CircularProgress from '@mui/material/CircularProgress';

// project imports
import Iconify from 'src/widgets/iconify';

import GraphVizFullscreenDialog from './graphviz-fullscreen-dialog';

// ----------------------------------------------------------------------

interface GraphVizRendererProps {
  dot: string;
}

/**
 * Validates if DOT code is syntactically complete
 */
const isCompleteDotSyntax = (dot: string): boolean => {
  // Remove comments and normalize whitespace
  const cleanDot = dot
    .replace(/\/\/.*$/gm, '') // Remove single-line comments
    .replace(/\/\*[\s\S]*?\*\//g, '') // Remove multi-line comments
    .replace(/\s+/g, ' ') // Normalize whitespace
    .trim();

  // Basic validation: must start with digraph/graph and have balanced braces
  const hasValidStart = /^(di)?graph\s+\w*\s*{/i.test(cleanDot);
  if (!hasValidStart) return false;

  // Count braces to ensure they're balanced
  let braceCount = 0;
  let inString = false;
  let escapeNext = false;

  for (let i = 0; i < cleanDot.length; i += 1) {
    const char = cleanDot[i];

    if (escapeNext) {
      escapeNext = false;
    } else if (char === '\\') {
      escapeNext = true;
    } else if (char === '"') {
      inString = !inString;
    } else if (!inString) {
      if (char === '{') {
        braceCount += 1;
      } else if (char === '}') {
        braceCount -= 1;
      }
    }
  }

  // DOT is complete if braces are balanced and we're not inside a string
  return braceCount === 0 && !inString;
};

/**
 * Custom renderer for GraphViz DOT language
 */
const GraphVizRenderer = ({ dot }: GraphVizRendererProps) => {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const panZoomRef = useRef<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [svgContent, setSvgContent] = useState<string>('');
  const [isFirstRender, setIsFirstRender] = useState(true);
  const [isWaitingForComplete, setIsWaitingForComplete] = useState(false);
  const renderTimeoutRef = useRef<NodeJS.Timeout>();
  const lastCompleteDotRef = useRef<string>('');
  const lastDotRef = useRef<string>('');

  useEffect(() => {
    let mounted = true;

    // Clear any existing timeout
    if (renderTimeoutRef.current) {
      clearTimeout(renderTimeoutRef.current);
    }

    const renderGraph = async (dotCode: string) => {
      try {
        // Dynamic import
        // eslint-disable-next-line import/no-extraneous-dependencies
        const { Graphviz } = await import('@hpcc-js/wasm');
        const graphViz = await Graphviz.load();
        const svg = await graphViz.dot(dotCode);

        if (!mounted) return;

        setSvgContent(svg);
        lastCompleteDotRef.current = dotCode;
        setError(null);
        setIsFirstRender(false);
        setIsWaitingForComplete(false);
      } catch (err: any) {
        if (mounted && !isWaitingForComplete) {
          setError(err.message || 'Failed to render graph');
          setIsFirstRender(false);
        }
      }
    };

    // Skip if the dot hasn't changed
    if (dot === lastDotRef.current) {
      return undefined;
    }

    lastDotRef.current = dot;

    // Check if DOT syntax is complete
    const isComplete = isCompleteDotSyntax(dot);

    if (isComplete) {
      // If complete and different from last rendered, render immediately
      if (dot !== lastCompleteDotRef.current) {
        renderGraph(dot);
      }
    } else {
      // Mark as waiting for complete syntax
      if (!isWaitingForComplete) {
        setIsWaitingForComplete(true);
      }

      // Set a timeout to try rendering after a delay (in case streaming stops)
      renderTimeoutRef.current = setTimeout(() => {
        if (mounted && dot.trim()) {
          renderGraph(dot);
        }
      }, 2000); // Wait 2 seconds of inactivity before attempting render
    }

    return () => {
      mounted = false;
      if (renderTimeoutRef.current) {
        clearTimeout(renderTimeoutRef.current);
      }
    };
  }, [dot, isWaitingForComplete]);

  // Initialize panzoom for normal view
  useEffect(() => {
    if (!containerRef.current || !svgContent) {
      return undefined;
    }

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
      });
    }

    return () => {
      if (panZoomRef.current) {
        panZoomRef.current.dispose();
        panZoomRef.current = null;
      }
    };
  }, [svgContent]);

  const handleZoomIn = () => {
    const pz = fullscreen ? window.graphvizFullscreenPanzoom : panZoomRef.current;
    if (pz) {
      const currentZoom = pz.getTransform().scale;
      pz.zoomAbs(0, 0, currentZoom * 1.2);
    }
  };

  const handleZoomOut = () => {
    const pz = fullscreen ? window.graphvizFullscreenPanzoom : panZoomRef.current;
    if (pz) {
      const currentZoom = pz.getTransform().scale;
      pz.zoomAbs(0, 0, currentZoom * 0.8);
    }
  };

  const handleFitView = () => {
    const pz = fullscreen ? window.graphvizFullscreenPanzoom : panZoomRef.current;
    if (pz) {
      pz.moveTo(0, 0);
      pz.zoomAbs(0, 0, 1);
    }
  };

  const handleFullscreen = () => {
    setFullscreen(true);
  };

  const handleCloseFullscreen = () => {
    setFullscreen(false);
  };

  const handleDownload = async () => {
    let svgElement: SVGElement | null = null;

    if (fullscreen) {
      // In fullscreen mode, find the SVG within the dialog's graph container
      const dialogContainer = document.querySelector('.MuiDialog-root');
      if (dialogContainer) {
        // Look for the SVG inside the graph container (not in the toolbar)
        const graphContainer = dialogContainer.querySelector(
          '[class*="MuiBox-root"]:last-child [class*="MuiBox-root"]:last-child'
        );
        if (graphContainer) {
          svgElement = graphContainer.querySelector('svg');
        }
      }
    } else {
      // In normal mode, use the containerRef
      svgElement = containerRef.current?.querySelector('svg') || null;
    }

    if (!svgElement) {
      console.error('SVG element not found');
      return;
    }

    try {
      // Clone the SVG to avoid modifying the original
      const clonedSvg = svgElement.cloneNode(true) as SVGElement;

      // Get the actual SVG dimensions from viewBox or width/height attributes
      let width = 800;
      let height = 600;

      // First try to get dimensions from viewBox
      const viewBox = svgElement.getAttribute('viewBox');
      if (viewBox) {
        const [, , vbWidth, vbHeight] = viewBox.split(' ').map(Number);
        if (vbWidth && vbHeight) {
          width = vbWidth;
          height = vbHeight;
        }
      }

      // If no viewBox, try width/height attributes
      if (!viewBox || width === 800) {
        const svgWidth = svgElement.getAttribute('width');
        const svgHeight = svgElement.getAttribute('height');
        if (svgWidth && svgHeight) {
          // Remove 'px' or other units if present
          width = parseFloat(svgWidth);
          height = parseFloat(svgHeight);
        }
      }

      // If still default dimensions, use getBoundingClientRect
      if (width === 800 || height === 600) {
        const rect = svgElement.getBoundingClientRect();
        if (rect.width && rect.height) {
          ({ width, height } = rect);
        }
      }

      // Create a canvas with proper dimensions
      const canvas = document.createElement('canvas');
      const scale = 4; // For higher resolution
      canvas.width = width * scale;
      canvas.height = height * scale;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        console.error('Failed to get canvas context');
        return;
      }

      // Set white background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Scale the context for higher resolution
      ctx.scale(scale, scale);

      // Ensure the cloned SVG has proper dimensions and namespace
      clonedSvg.setAttribute('width', String(width));
      clonedSvg.setAttribute('height', String(height));

      // Set xmlns if not already present
      if (!clonedSvg.hasAttribute('xmlns')) {
        clonedSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      }

      // If viewBox exists, ensure it's set correctly
      if (viewBox) {
        clonedSvg.setAttribute('viewBox', viewBox);
      } else {
        // Create a viewBox if none exists
        clonedSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      }

      // Convert SVG to data URL
      const svgData = new XMLSerializer().serializeToString(clonedSvg);
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
      const svgUrl = URL.createObjectURL(svgBlob);

      // Create an image from the SVG
      const img = new Image();

      // Set crossOrigin to handle any potential CORS issues
      img.crossOrigin = 'anonymous';

      img.onload = () => {
        // Draw the image on canvas with proper dimensions
        ctx.drawImage(img, 0, 0, width, height);

        // Convert canvas to PNG and download
        canvas.toBlob(
          (blob) => {
            if (blob) {
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `graph-${new Date().getTime()}.png`;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);

              // Clean up
              URL.revokeObjectURL(url);
            } else {
              console.error('Failed to create blob from canvas');
            }
          },
          'image/png',
          1.0
        );

        // Clean up
        URL.revokeObjectURL(svgUrl);
      };

      img.onerror = (e) => {
        console.error('Failed to load SVG as image:', e);
        URL.revokeObjectURL(svgUrl);

        // Fallback: Try direct SVG download
        const fallbackSvgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const fallbackSvgUrl = URL.createObjectURL(fallbackSvgBlob);
        const a = document.createElement('a');
        a.href = fallbackSvgUrl;
        a.download = `graph-${new Date().getTime()}.svg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(fallbackSvgUrl);
        console.log('Fallback: Downloaded as SVG instead');
      };

      img.src = svgUrl;
    } catch (err) {
      console.error('Failed to download graph:', err);
    }
  };

  // Determine what to display
  const showInitialLoading = isFirstRender && !svgContent && !error;
  // const showStreamingMessage = isWaitingForComplete && !svgContent && !isFirstRender;
  const showGraph = svgContent && !error;
  const showError = error && !isWaitingForComplete;
  const isButtonsDisabled = !showGraph;

  return (
    <>
      <Box
        sx={{
          my: 2,
          borderRadius: 1,
          border: `1px solid ${theme.palette.divider}`,
          bgcolor: theme.palette.background.paper,
          overflow: 'hidden',
        }}
      >
        {/* Toolbar */}
        <Stack
          direction="row"
          alignItems="center"
          sx={{
            px: 1,
            py: 0.5,
            borderBottom: `1px solid ${theme.palette.divider}`,
            bgcolor: theme.palette.background.neutral,
          }}
        >
          <Stack direction="row" spacing={0.5} sx={{ mr: 3 }}>
            <IconButton
              size="small"
              onClick={handleZoomIn}
              disabled={isButtonsDisabled}
              sx={{ color: 'text.secondary' }}
            >
              <Iconify icon="eva:plus-circle-outline" width={20} />
            </IconButton>
            <IconButton
              size="small"
              onClick={handleZoomOut}
              disabled={isButtonsDisabled}
              sx={{ color: 'text.secondary' }}
            >
              <Iconify icon="eva:minus-circle-outline" width={20} />
            </IconButton>
            <IconButton
              size="small"
              onClick={handleFitView}
              disabled={isButtonsDisabled}
              sx={{ color: 'text.secondary' }}
            >
              <Iconify icon="eva:expand-outline" width={20} />
            </IconButton>
            <IconButton
              size="small"
              onClick={handleFullscreen}
              disabled={isButtonsDisabled}
              sx={{ color: 'text.secondary' }}
            >
              <Iconify icon="gg:maximize" width={20} />
            </IconButton>
            <IconButton
              size="small"
              onClick={handleDownload}
              disabled={isButtonsDisabled}
              sx={{ color: 'text.secondary' }}
            >
              <Iconify icon="eva:download-outline" width={20} />
            </IconButton>
          </Stack>

          {/* Control hints */}
          {showGraph && (
            <Box
              sx={{
                ml: 'auto',
                fontSize: '12px',
                color: 'text.secondary',
              }}
            >
              {isWaitingForComplete
                ? 'Streaming... (showing last valid graph)'
                : 'Use mouse wheel to zoom • Drag to pan'}
            </Box>
          )}
        </Stack>

        {/* Graph Container */}
        <Box
          sx={{
            position: 'relative',
            height: 360,
            overflow: 'hidden',
          }}
        >
          {/* Only show these states when appropriate */}
          {/* <Box
            sx={{
              display: showInitialLoading ? 'flex' : 'none',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              bgcolor: 'background.default',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 2,
            }}
          >
            <CircularProgress size={40} />
          </Box> */}

          <Box
            sx={{
              display: showInitialLoading ? 'flex' : 'none',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              bgcolor: 'background.default',
              p: 3,
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 2,
            }}
          >
            {/* <CircularProgress size={30} sx={{ mb: 2 }} /> */}
            <Box sx={{ color: 'text.secondary', textAlign: 'center' }}>
              <Box sx={{ mb: 1 }}>Receiving graph data ...</Box>
            </Box>
          </Box>

          <Box
            sx={{
              display: showError ? 'block' : 'none',
              p: 2,
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 2,
              bgcolor: 'background.default',
              overflow: 'auto',
            }}
          >
            <Box sx={{ color: 'error.main', mb: 1 }}>Error rendering graph: {error}</Box>
            <Box
              sx={{
                p: 1,
                borderRadius: 1,
                backgroundColor: theme.palette.action.hover,
                fontFamily: 'monospace',
                fontSize: 12,
                overflow: 'auto',
                maxHeight: 300,
              }}
            >
              <pre>{dot}</pre>
            </Box>
          </Box>

          {/* Graph display - always present but may be dimmed */}
          <Box
            ref={containerRef}
            sx={{
              width: '100%',
              height: '100%',
              cursor: 'grab',
              opacity: isWaitingForComplete ? 0.6 : 1,
              transition: 'opacity 0.3s ease',
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

      {/* Fullscreen Dialog */}
      <GraphVizFullscreenDialog
        open={fullscreen}
        svgContent={svgContent}
        onClose={handleCloseFullscreen}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onFitView={handleFitView}
        onDownload={handleDownload}
      />
    </>
  );
};

export default GraphVizRenderer;
