// mui
import Box, { BoxProps } from '@mui/material/Box';

// project imports
import { useResponsiveUI } from 'src/hooks/responsive-ui';

import { HEADER } from '../default-layout';

// ----------------------------------------------------------------------

const Main = ({ children, sx, ...other }: BoxProps): JSX.Element => {
  const lgUp = useResponsiveUI('up', 'lg');

  // Build responsive styles based on viewport
  const responsiveSx = lgUp
    ? {
        px: 0,
        py: `${HEADER.H_DESKTOP}px`,
        width: 'calc(100% - 0px)',
      }
    : {};

  return (
    <Box
      component="main"
      sx={{
        flexGrow: 1,
        minHeight: 1,
        display: 'flex',
        flexDirection: 'column',
        py: `${HEADER.H_DESKTOP}px`,
        ...responsiveSx,
        ...sx,
      }}
      {...other}
    >
      {children}
    </Box>
  );
};

export default Main;
