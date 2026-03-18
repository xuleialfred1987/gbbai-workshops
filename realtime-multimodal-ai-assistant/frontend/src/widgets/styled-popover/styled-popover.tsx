// mui
import { menuItemClasses } from '@mui/material/MenuItem';
import Popover, { PopoverOrigin } from '@mui/material/Popover';

// project imports
import { getPosition } from './utils';
import { StyledArrow } from './styles';
import { MenuPopoverProps } from './types';

// ----------------------------------------------------------------------

/**
 * Custom styled popover component with arrow positioning
 */
function StyledPopover(props: MenuPopoverProps) {
  const {
    open = null,
    children = null,
    arrow = 'top-right',
    hiddenArrow = false,
    sx = {},
    ...remainingProps
  } = props;

  // Calculate positioning based on arrow direction
  const positionConfig = getPosition(arrow);

  // Extract positioning data
  const {
    style: positionStyle,
    anchorOrigin: anchorPos,
    transformOrigin: transformPos,
  } = positionConfig;

  // Render component structure
  return (
    <Popover
      open={!!open}
      anchorEl={open}
      anchorOrigin={anchorPos as PopoverOrigin}
      transformOrigin={transformPos as PopoverOrigin}
      slotProps={{
        paper: {
          sx: {
            width: 'auto',
            overflow: 'inherit',
            ...positionStyle,
            [`& .${menuItemClasses.root}`]: {
              '& svg': {
                mr: 2,
                flexShrink: 0,
              },
            },
            ...(sx || {}),
          },
        },
      }}
      {...remainingProps}
    >
      {hiddenArrow ? null : <StyledArrow arrow={arrow} />}
      {children}
    </Popover>
  );
}

export default StyledPopover;
