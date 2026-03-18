import React from 'react';

// mui
import { Box } from '@mui/material';

// project imports
import { ScrollerProps } from './types';
import { StyledScroller, StyledRootScroller } from './styles';

// ----------------------------------------------------------------------

const ScrollerComponent: React.ForwardRefRenderFunction<HTMLDivElement, ScrollerProps> = (
  props,
  forwardedRef
) => {
  const { children, sx, ...restProps } = props;

  // Check if the device is mobile
  const isMobileDevice = React.useMemo(() => {
    if (typeof window === 'undefined') return false;

    const mobilePlatformRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
    return mobilePlatformRegex.test(navigator.userAgent);
  }, []);

  // Render native scrolling for mobile devices
  if (isMobileDevice) {
    return (
      <Box
        ref={forwardedRef}
        sx={{
          overflowY: 'auto',
          overflowX: 'hidden',
          ...sx,
        }}
        {...restProps}
      >
        {children}
      </Box>
    );
  }

  // Render custom Scroller for desktop
  return (
    <StyledRootScroller>
      <StyledScroller
        scrollableNodeProps={{
          ref: forwardedRef,
        }}
        clickOnTrack={false}
        sx={sx}
        {...restProps}
      >
        {children}
      </StyledScroller>
    </StyledRootScroller>
  );
};

const Scroller = React.memo(React.forwardRef<HTMLDivElement, ScrollerProps>(ScrollerComponent));

export default Scroller;
