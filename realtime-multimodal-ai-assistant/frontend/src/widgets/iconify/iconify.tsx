import React from 'react';
import { Icon as IconifyIcon } from '@iconify/react';

// mui
import { Box, BoxProps } from '@mui/material';

// project imports
import { IconifyProps } from './types';

// ----------------------------------------------------------------------

/**
 * Enhanced icon component that wraps Iconify with Material-UI Box
 */
type IconifyComponentProps = BoxProps & {
  icon: IconifyProps;
};

/**
 * IconifyComponent - A wrapper for Iconify icons with Material-UI styling
 *
 * @param props - Component properties
 * @param ref - Forwarded ref to access the underlying SVG element
 * @returns Styled icon component
 */
const IconifyComponent = React.forwardRef<SVGElement, IconifyComponentProps>((props, ref) => {
  const { icon, width = 20, sx = {}, ...restProps } = props;

  return (
    <Box
      ref={ref}
      component={IconifyIcon}
      icon={icon}
      sx={{
        width,
        height: width,
        ...sx,
      }}
      {...restProps}
    />
  );
});

export default IconifyComponent;
