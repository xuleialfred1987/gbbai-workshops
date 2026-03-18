import { Theme, alpha } from '@mui/material/styles';
import { AlertProps, alertClasses } from '@mui/material/Alert';

// ----------------------------------------------------------------------

export function alert(theme: Theme) {
  const isLightTheme = theme.palette.mode === 'light';
  const severityColors = ['info', 'success', 'warning', 'error'] as const;

  // Style configuration by variant type
  const variantStyles = {
    standard: (color: (typeof severityColors)[number]) => ({
      color: theme.palette[color][isLightTheme ? 'darker' : 'lighter'],
      backgroundColor: theme.palette[color][isLightTheme ? 'lighter' : 'darker'],
      [`& .${alertClasses.icon}`]: {
        color: theme.palette[color][isLightTheme ? 'main' : 'light'],
      },
    }),

    filled: (color: (typeof severityColors)[number]) => ({
      color: theme.palette[color].contrastText,
      backgroundColor: theme.palette[color].main,
    }),

    outlined: (color: (typeof severityColors)[number]) => ({
      backgroundColor: alpha(theme.palette[color].main, 0.08),
      color: theme.palette[color][isLightTheme ? 'dark' : 'light'],
      border: `solid 1px ${alpha(theme.palette[color].main, 0.16)}`,
      [`& .${alertClasses.icon}`]: {
        color: theme.palette[color].main,
      },
    }),
  };

  // Generate styles for each severity color based on the alert's variant
  const generateStylesByProps = (props: AlertProps) =>
    severityColors.map((color) => {
      if (props.severity !== color) return {};

      const variant = props.variant || 'standard';
      return variantStyles[variant](color);
    });

  return {
    MuiAlert: {
      styleOverrides: {
        root: ({ ownerState }: { ownerState: AlertProps }) => generateStylesByProps(ownerState),
        icon: {
          opacity: 1,
        },
      },
    },
    MuiAlertTitle: {
      styleOverrides: {
        root: {
          fontWeight: theme.typography.fontWeightSemiBold,
          marginBottom: theme.spacing(0.5),
        },
      },
    },
  };
}
