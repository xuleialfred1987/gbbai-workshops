// mui
import { alpha, Theme } from '@mui/material/styles';
import { DrawerProps, drawerClasses } from '@mui/material/Drawer';

// project imports
import { paper } from '../../css';

// ----------------------------------------------------------------------

export function drawer(theme: Theme) {
  const isLightMode = theme.palette.mode === 'light';

  // Shadow color based on theme mode
  const shadowColor = isLightMode ? theme.palette.grey[500] : theme.palette.common.black;

  // Shadow opacity
  const shadowOpacity = 0.24;

  // Generate shadow based on anchor position
  const createShadow = (direction: string) => {
    const xOffset = direction === 'left' ? '40px' : '-40px';
    return `${xOffset} 40px 80px -8px ${alpha(shadowColor, shadowOpacity)}`;
  };

  // Configure drawer component styles
  return {
    MuiDrawer: {
      styleOverrides: {
        root: (props: { ownerState: DrawerProps }) => {
          // Only apply custom styles to temporary drawers
          if (props.ownerState.variant !== 'temporary') {
            return {};
          }

          // Apply paper styles with appropriate shadow
          const { anchor } = props.ownerState;

          return {
            [`& .${drawerClasses.paper}`]: {
              ...paper({ theme }),
              ...(anchor === 'left' && { boxShadow: createShadow('left') }),
              ...(anchor === 'right' && { boxShadow: createShadow('right') }),
            },
          };
        },
      },
    },
  };
}
