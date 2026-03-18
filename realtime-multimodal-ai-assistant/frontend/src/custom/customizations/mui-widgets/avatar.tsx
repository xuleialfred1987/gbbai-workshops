import { AvatarProps } from '@mui/material/Avatar';
import { alpha, Theme } from '@mui/material/styles';
import { AvatarGroupProps, avatarGroupClasses } from '@mui/material/AvatarGroup';

// ----------------------------------------------------------------------

const COLOR_MAP: Record<string, string> = {
  primary: 'acf',
  secondary: 'edh',
  info: 'ikl',
  success: 'mnp',
  warning: 'qst',
  error: 'vxy',
};

const COLORS = ['default', 'primary', 'secondary', 'info', 'success', 'warning', 'error'] as const;

const getColorVariant = (name: string): (typeof COLORS)[number] => {
  const initial = name.charAt(0).toLowerCase();
  const foundColor = (Object.entries(COLOR_MAP).find(([_, chars]) => chars.includes(initial)) ||
    [])[0];
  return (foundColor as (typeof COLORS)[number]) || 'default';
};

// Extend AvatarGroup variants
declare module '@mui/material/AvatarGroup' {
  interface AvatarGroupPropsVariantOverrides {
    compact: true;
  }
}

// ----------------------------------------------------------------------

const generateAvatarVariants = (theme: Theme) =>
  COLORS.map((color) => ({
    props: { color },
    style:
      color === 'default'
        ? {
            color: theme.palette.text.secondary,
            backgroundColor: alpha(theme.palette.grey[500], 0.24),
          }
        : {
            color: theme.palette[color].contrastText,
            backgroundColor: theme.palette[color].main,
          },
  }));

const avatarStyleOverrides = (theme: Theme) => ({
  rounded: {
    borderRadius: theme.shape.borderRadius * 1.5,
  },
  colorDefault: ({ ownerState }: { ownerState: AvatarProps }) => {
    const variantColor = getColorVariant(ownerState.alt || '');
    return ownerState.alt
      ? {
          color:
            variantColor === 'default'
              ? theme.palette.text.secondary
              : theme.palette[variantColor].contrastText,
          backgroundColor:
            variantColor === 'default'
              ? alpha(theme.palette.grey[500], 0.24)
              : theme.palette[variantColor].main,
        }
      : {};
  },
});

const avatarGroupStyleOverrides = (theme: Theme) => ({
  root: ({ ownerState }: { ownerState: AvatarGroupProps }) => ({
    justifyContent: 'flex-end',
    ...(ownerState.variant === 'compact' && {
      width: 40,
      height: 40,
      position: 'relative',
      [`& .${avatarGroupClasses.avatar}`]: {
        margin: 0,
        width: 28,
        height: 28,
        fontSize: 15,
        position: 'absolute',
        '&:first-of-type': {
          left: 0,
          bottom: 0,
          zIndex: 9,
        },
        '&:last-of-type': {
          top: 0,
          right: 0,
        },
      },
    }),
  }),
  avatar: {
    fontSize: 16,
    fontWeight: theme.typography.fontWeightSemiBold,
    '&:first-of-type': {
      fontSize: 12,
      color: theme.palette.primary.dark,
      backgroundColor: theme.palette.primary.lighter,
    },
  },
});

// Main export
export const avatar = (theme: Theme) => ({
  MuiAvatar: {
    variants: generateAvatarVariants(theme),
    styleOverrides: avatarStyleOverrides(theme),
  },
  MuiAvatarGroup: {
    styleOverrides: avatarGroupStyleOverrides(theme),
  },
});
