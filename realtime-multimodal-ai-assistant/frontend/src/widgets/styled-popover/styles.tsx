// mui
import { alpha, styled } from '@mui/material/styles';

// project imports
import { bgBlur } from 'src/custom/css';

import { MenuPopoverArrowValue } from './types';

// ----------------------------------------------------------------------

export const StyledArrow = styled('span')<{ arrow: MenuPopoverArrowValue }>(({ arrow, theme }) => {
  // Constants
  const ARROW_DIMENSION = 14;
  const BORDER_RADIUS = ARROW_DIMENSION / 4;
  const OFFSET = -(ARROW_DIMENSION / 2) + 0.5;
  const SIDE_POSITION = 20;

  // Border styling
  const borderStyle = `solid 1px ${alpha(
    theme.palette.mode === 'light' ? theme.palette.grey[500] : theme.palette.common.black,
    0.12
  )}`;

  // Base arrow styling
  const baseStyles = {
    width: ARROW_DIMENSION,
    height: ARROW_DIMENSION,
    position: 'absolute' as const,
    borderBottomLeftRadius: BORDER_RADIUS,
    clipPath: 'polygon(0% 0%, 100% 100%, 0% 100%)',
    border: borderStyle,
    ...bgBlur({
      color: theme.palette.background.paper,
    }),
  };

  // Direction-based styling presets
  const directionStyles = {
    top: {
      position: OFFSET,
      rotation: 'rotate(135deg)',
    },
    bottom: {
      position: OFFSET,
      rotation: 'rotate(-45deg)',
    },
    left: {
      position: OFFSET,
      rotation: 'rotate(45deg)',
    },
    right: {
      position: OFFSET,
      rotation: 'rotate(-135deg)',
    },
  };

  // Alignment presets
  const alignmentStyles = {
    start: (direction: 'top' | 'bottom' | 'left' | 'right') => {
      const isVertical = direction === 'top' || direction === 'bottom';
      return isVertical ? { left: SIDE_POSITION } : { top: SIDE_POSITION };
    },
    center: (direction: 'top' | 'bottom' | 'left' | 'right') => {
      const isVertical = direction === 'top' || direction === 'bottom';
      return isVertical
        ? { left: 0, right: 0, margin: 'auto' }
        : { top: 0, bottom: 0, margin: 'auto' };
    },
    end: (direction: 'top' | 'bottom' | 'left' | 'right') => {
      const isVertical = direction === 'top' || direction === 'bottom';
      return isVertical ? { right: SIDE_POSITION } : { bottom: SIDE_POSITION };
    },
  };

  // Parse arrow position
  const [direction, alignment] = arrow.split('-') as [
    'top' | 'bottom' | 'left' | 'right',
    'left' | 'center' | 'right' | 'top' | 'bottom',
  ];

  // Map alignment values to consistent naming
  let normalizedAlignment: 'start' | 'center' | 'end';
  if (alignment === 'left' || alignment === 'top') normalizedAlignment = 'start';
  else if (alignment === 'center') normalizedAlignment = 'center';
  else normalizedAlignment = 'end';

  // Build direction-specific styles
  const directionStyle = {
    [direction]: directionStyles[direction].position,
    transform: directionStyles[direction].rotation,
  };

  // Combine styles
  return {
    ...baseStyles,
    ...directionStyle,
    ...alignmentStyles[normalizedAlignment](direction),
  };
});
