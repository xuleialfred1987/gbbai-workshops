import React from 'react';

// mui
import { Box, Link, BoxProps } from '@mui/material';

// project imports
import { NavigationLink } from 'src/routes/components';

import { useSettingsContext } from 'src/widgets/settings';

// ----------------------------------------------------------------------

type LogoComponentProps = BoxProps & {
  disabledLink?: boolean;
  singleMode?: boolean;
};

const LogoComponent = React.forwardRef<HTMLDivElement, LogoComponentProps>((props, ref) => {
  const { disabledLink = false, singleMode = false, sx, ...otherProps } = props;

  // Get current theme settings
  const { themeLayout } = useSettingsContext();

  // Determine which logo to show
  const shouldShowSingleLogo = themeLayout === 'mini' || singleMode;
  const logoSrc = shouldShowSingleLogo ? '/logo/gbb_single.svg' : '/logo/gbb_full.svg';

  // Build the logo element
  const logoElement = (
    <Box
      ref={ref}
      component="img"
      src={logoSrc}
      sx={{
        width: 146,
        height: 40,
        cursor: 'pointer',
        ...sx,
      }}
      {...otherProps}
    />
  );

  // Return with or without link wrapper
  return disabledLink ? (
    logoElement
  ) : (
    <Link component={NavigationLink} path="/" sx={{ display: 'contents' }}>
      {logoElement}
    </Link>
  );
});

export type { LogoComponentProps as LogoProps };
export default LogoComponent;
