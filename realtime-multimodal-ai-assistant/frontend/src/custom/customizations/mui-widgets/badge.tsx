import { Theme } from '@mui/material/styles';
import { BadgeProps, badgeClasses } from '@mui/material/Badge';

// ----------------------------------------------------------------------

// Extend Badge variants with custom status indicators
declare module '@mui/material/Badge' {
  interface BadgePropsVariantOverrides {
    online: true;
    offline: true;
    busy: true;
    alway: true;
    invisible: true;
  }
}

// Helper function to create badge styles
const createBadgeStyle = (backgroundColor: string, decorations = {}) => ({
  backgroundColor,
  width: 10,
  height: 10,
  minWidth: 'auto',
  padding: 0,
  zIndex: 9,
  [`&.${badgeClasses.invisible}`]: {
    transform: 'unset',
  },
  '&:before, &:after': {
    content: "''",
    borderRadius: 1,
    backgroundColor: 'white',
    ...decorations,
  },
});

// Define variant-specific styles
const getVariantStyles = (theme: Theme, variant?: string) => {
  const styles: Record<string, any> = {};

  if (variant === 'online') {
    styles[`& .${badgeClasses.badge}`] = createBadgeStyle(theme.palette.success.main);
  }

  if (variant === 'busy') {
    styles[`& .${badgeClasses.badge}`] = createBadgeStyle(theme.palette.error.main, {
      '&:before': { width: 6, height: 2 },
    });
  }

  if (variant === 'offline') {
    styles[`& .${badgeClasses.badge}`] = createBadgeStyle(theme.palette.text.disabled, {
      '&:before': {
        width: 6,
        height: 6,
        borderRadius: '50%',
      },
    });
  }

  if (variant === 'alway') {
    styles[`& .${badgeClasses.badge}`] = createBadgeStyle(theme.palette.warning.main, {
      '&:before': {
        width: 2,
        height: 4,
        transform: 'translateX(1px) translateY(-1px)',
      },
      '&:after': {
        width: 2,
        height: 4,
        transform: 'translateY(1px) rotate(125deg)',
      },
    });
  }

  if (variant === 'invisible') {
    styles[`& .${badgeClasses.badge}`] = {
      display: 'none',
    };
  }

  return styles;
};

// Main component customization function
export function badge(theme: Theme) {
  return {
    MuiBadge: {
      styleOverrides: {
        dot: {
          borderRadius: '50%',
        },
        root: ({ ownerState }: { ownerState: BadgeProps }) =>
          getVariantStyles(theme, ownerState.variant),
      },
    },
  };
}
