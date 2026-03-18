import { PopoverProps } from '@mui/material/Popover';

// ----------------------------------------------------------------------

// Position combinations for arrow placement
type DirectionHorizontal = 'left' | 'center' | 'right';
type DirectionVertical = 'top' | 'bottom';
type DirectionSide = 'left' | 'right';
type AlignmentVertical = 'top' | 'center' | 'bottom';

// Vertical positions with horizontal alignment
type VerticalPosition = `${DirectionVertical}-${DirectionHorizontal}`;

// Horizontal positions with vertical alignment
type HorizontalPosition = `${DirectionSide}-${AlignmentVertical}`;

// Union of all possible arrow positions
export type MenuPopoverArrowValue = VerticalPosition | HorizontalPosition;

/**
 * Extended popover props with custom arrow positioning
 *
 * @property open - Element that anchors the popover
 * @property arrow - Position of the arrow element
 * @property hiddenArrow - Whether to hide the arrow element
 */
export interface MenuPopoverProps extends Omit<PopoverProps, 'open'> {
  open: HTMLElement | null;
  arrow?: MenuPopoverArrowValue;
  hiddenArrow?: boolean;
}
