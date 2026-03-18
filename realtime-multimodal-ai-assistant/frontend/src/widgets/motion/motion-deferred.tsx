import React from 'react';
import { m, domMax, LazyMotion } from 'framer-motion';

// ----------------------------------------------------------------------

/**
 * MotionDeferred - Component that lazily loads motion capabilities
 *
 * This wrapper uses Framer Motion's LazyMotion to defer loading
 * animation features until they're needed
 */
export const MotionDeferred: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => (
  <LazyMotion features={domMax} strict>
    <m.div
      className="motion-container"
      style={{
        height: '100%',
        position: 'relative',
      }}
    >
      {children}
    </m.div>
  </LazyMotion>
);
