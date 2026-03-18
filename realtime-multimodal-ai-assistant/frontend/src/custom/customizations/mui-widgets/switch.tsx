import { Theme, alpha } from '@mui/material/styles';
import { SwitchProps, switchClasses } from '@mui/material/Switch';

// ----------------------------------------------------------------------

export function switches(theme: Theme) {
  const isDarkTheme = theme.palette.mode !== 'light';

  const createSwitchStyles = (props: SwitchProps) => {
    // Base dimensions
    const dimensions = {
      standard: {
        containerWidth: 58,
        containerHeight: 38,
        thumbSize: 14,
        padding: '9px 13px 9px 12px',
        translation: 13,
        internalPadding: 12,
      },
      small: {
        containerWidth: 40,
        containerHeight: 24,
        thumbSize: 10,
        padding: '4px 8px 4px 7px',
        translation: 9,
        internalPadding: 7,
      },
    };

    const size = props.size === 'small' ? 'small' : 'standard';
    const { containerWidth, containerHeight, thumbSize, padding, translation, internalPadding } =
      dimensions[size];

    return {
      width: containerWidth,
      height: containerHeight,
      padding, // Using object shorthand here instead of padding: padding

      [`& .${switchClasses.thumb}`]: {
        boxShadow: 'none',
        width: thumbSize,
        height: thumbSize,
        color: theme.palette.common.white,
      },

      [`& .${switchClasses.track}`]: {
        borderRadius: 14,
        opacity: 1,
        backgroundColor: alpha(theme.palette.grey[500], 0.48),
      },

      [`& .${switchClasses.switchBase}`]: {
        left: 3,
        padding: internalPadding,

        [`&.${switchClasses.checked}`]: {
          transform: `translateX(${translation}px)`,

          [`&+.${switchClasses.track}`]: {
            opacity: 1,
          },
        },

        [`&.${switchClasses.disabled}`]: {
          [`& .${switchClasses.thumb}`]: {
            opacity: isDarkTheme ? 0.48 : 1,
          },

          [`&+.${switchClasses.track}`]: {
            opacity: 0.48,
          },
        },
      },
    };
  };

  return {
    MuiSwitch: {
      styleOverrides: {
        root: ({ ownerState }: { ownerState: SwitchProps }) => createSwitchStyles(ownerState),
      },
    },
  };
}
