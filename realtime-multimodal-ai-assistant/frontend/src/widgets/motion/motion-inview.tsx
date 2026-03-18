import React from 'react';
import { m, AnimatePresence } from 'framer-motion';

// mui
import { Box } from '@mui/material';

// project import
import { useResponsiveUI } from 'src/hooks/responsive-ui';

import { varSequence } from './patterns';

// ----------------------------------------------------------------------

interface ViewportAnimationProps {
  children: React.ReactNode;
  mobileAnimationsDisabled?: boolean;
  viewportTriggerAmount?: number;
  triggerOnce?: boolean;
  sx?: Record<string, any>;
  [x: string]: any; // For additional props
}

/**
 * Component that animates children when they enter the viewport
 * Uses the 'm' namespace for compatibility with LazyMotion
 */
export default function AnimatedViewportWrapper({
  children,
  mobileAnimationsDisabled = true,
  viewportTriggerAmount = 0.3,
  triggerOnce = true,
  sx = {},
  ...remainingProps
}: ViewportAnimationProps): JSX.Element {
  // Check if device is mobile
  const isMobileView = useResponsiveUI('down', 'sm');

  // Disable animations on mobile if specified
  if (isMobileView && mobileAnimationsDisabled) {
    return (
      <Box sx={{ ...sx }} {...remainingProps}>
        {children}
      </Box>
    );
  }

  return (
    <AnimatePresence>
      <Box
        component={m.div}
        initial="initial"
        whileInView="animate"
        viewport={{
          once: triggerOnce,
          amount: viewportTriggerAmount,
        }}
        variants={varSequence()}
        sx={{ ...sx }}
        {...remainingProps}
      >
        {children}
      </Box>
    </AnimatePresence>
  );
}
