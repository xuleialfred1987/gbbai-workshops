import { alpha, Theme } from '@mui/material/styles';
import { dividerClasses } from '@mui/material/Divider';
import { checkboxClasses } from '@mui/material/Checkbox';
import { menuItemClasses } from '@mui/material/MenuItem';
import { autocompleteClasses } from '@mui/material/Autocomplete';

// ----------------------------------------------------------------------

export function paper({ theme, dropdown }: { theme: Theme; bgcolor?: string; dropdown?: boolean }) {
  return {
    backgroundPosition: dropdown ? 'top right, left bottom' : undefined,
    backgroundSize: dropdown ? '50%, 50%' : undefined,
    backgroundRepeat: dropdown ? 'no-repeat, no-repeat' : undefined,
    ...(dropdown && {
      borderRadius: Number(theme.shape.borderRadius) * 1.25,
      boxShadow: theme.customShadows.dropdown,
      padding: theme.spacing(0.5),
    }),
  };
}

// ----------------------------------------------------------------------

export function menuItem(theme: Theme) {
  const styles = {
    ...theme.typography.body2,
    borderRadius: Number(theme.shape.borderRadius) * 0.75,
    padding: theme.spacing(0.75, 1),
  };

  const hoverStyles = {
    backgroundColor: theme.palette.action.hover,
  };

  return {
    ...styles,
    '&:not(:last-of-type)': {
      marginBottom: 4,
    },
    [`& .${checkboxClasses.root}`]: {
      marginLeft: theme.spacing(-0.5),
      marginRight: theme.spacing(0.5),
      padding: theme.spacing(0.5),
    },
    [`&.${menuItemClasses.selected}`]: {
      backgroundColor: theme.palette.action.selected,
      fontWeight: theme.typography.fontWeightSemiBold,
      '&:hover': hoverStyles,
    },
    [`&.${autocompleteClasses.option}[aria-selected="true"]`]: {
      backgroundColor: theme.palette.action.selected,
      '&:hover': hoverStyles,
    },
    [`&+.${dividerClasses.root}`]: {
      margin: theme.spacing(0.5, 0),
    },
  };
}

// ----------------------------------------------------------------------

interface BlurEffectProps {
  blur?: number;
  opacity?: number;
  color?: string;
  imgUrl?: string;
}

export function bgBlur(props?: BlurEffectProps) {
  const defaultColor = '#000000';
  const defaultBlur = 6;
  const defaultOpacity = 0.8;

  const finalColor = props?.color || defaultColor;
  const finalBlur = props?.blur || defaultBlur;
  const finalOpacity = props?.opacity || defaultOpacity;

  const blurEffects = {
    WebkitBackdropFilter: `blur(${finalBlur}px)`,
    backdropFilter: `blur(${finalBlur}px)`,
    backgroundColor: alpha(finalColor, finalOpacity),
  };

  if (!props?.imgUrl) {
    return blurEffects;
  }

  return {
    position: 'relative',
    backgroundImage: `url(${props.imgUrl})`,
    '&:before': {
      ...blurEffects,
      content: '""',
      height: '100%',
      left: 0,
      position: 'absolute',
      top: 0,
      width: '100%',
      zIndex: 9,
    },
  } as const;
}

// ----------------------------------------------------------------------

interface GradientOptions {
  direction?: string;
  color?: string;
  startColor?: string;
  endColor?: string;
  imgUrl?: string;
  type?: 'linear' | 'radial';
  shape?: string; // For radial gradient: circle, ellipse
  size?: string; // For radial gradient: closest-side, farthest-corner, etc.
  position?: string; // For radial gradient: center, top left, etc.
}

export function bgGradient(props?: GradientOptions) {
  const type = props?.type || 'linear';
  const colorStart = props?.startColor || props?.color;
  const colorEnd = props?.endColor || props?.color;

  let gradientStyle = {};

  if (type === 'linear') {
    const direction = props?.direction || 'to bottom';
    gradientStyle = {
      background: `linear-gradient(${direction}, ${colorStart}, ${colorEnd})`,
    };
  } else if (type === 'radial') {
    const shape = props?.shape || 'circle';
    const size = props?.size || 'farthest-corner';
    const position = props?.position || 'center';
    gradientStyle = {
      background: `radial-gradient(${shape} ${size} at ${position}, ${colorStart}, ${colorEnd})`,
    };
  }

  if (!props?.imgUrl) {
    return gradientStyle;
  }

  // return {
  //   ...gradientStyle,
  //   background: `linear-gradient(${direction}, ${colorStart}, ${colorEnd}), url(${props.imgUrl})`,
  //   backgroundPosition: 'center center',
  //   backgroundRepeat: 'no-repeat',
  //   backgroundSize: 'cover',
  // };

  if (type === 'linear') {
    const direction = props?.direction || 'to bottom';
    return {
      ...gradientStyle,
      background: `linear-gradient(${direction}, ${colorStart}, ${colorEnd}), url(${props.imgUrl})`,
      backgroundPosition: 'center center',
      backgroundRepeat: 'no-repeat',
      backgroundSize: 'cover',
    };
  }

  // For radial gradient with image
  const shape = props?.shape || 'circle';
  const size = props?.size || 'farthest-corner';
  const position = props?.position || 'center';
  return {
    ...gradientStyle,
    background: `radial-gradient(${shape} ${size} at ${position}, ${colorStart}, ${colorEnd}), url(${props.imgUrl})`,
    backgroundPosition: 'center center',
    backgroundRepeat: 'no-repeat',
    backgroundSize: 'cover',
  };
}

// ----------------------------------------------------------------------

export function textGradient(value: string) {
  const gradientTextEffect = {
    background: `-webkit-linear-gradient(${value})`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  };

  return gradientTextEffect;
}

// ----------------------------------------------------------------------

export function boxGradient(value: string) {
  const gradientBoxEffect = {
    WebkitBackgroundClip: 'box',
    WebkitTextFillColor: 'transparent',
    background: `-webkit-linear-gradient(${value})`,
  };

  return gradientBoxEffect;
}

// ----------------------------------------------------------------------

export const hideScroll = {
  x: {
    '&::-webkit-scrollbar': {
      display: 'none',
    },
    msOverflowStyle: 'none',
    overflowX: 'scroll',
    scrollbarWidth: 'none',
  },
  y: {
    '&::-webkit-scrollbar': {
      display: 'none',
    },
    msOverflowStyle: 'none',
    overflowY: 'scroll',
    scrollbarWidth: 'none',
  },
} as const;
