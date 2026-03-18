import { Theme } from '@mui/material/styles';
import { LoadingButtonProps, loadingButtonClasses as btnClasses } from '@mui/lab/LoadingButton';

// ----------------------------------------------------------------------

export function loadingButton(theme: Theme) {
  // Helper function to generate position styles based on size
  const createPositionStyles = (size: string | undefined) => {
    const baseStyles = {
      start: { left: 10 },
      end: { right: size === 'small' ? 10 : 14 },
    };

    return {
      [`& .${btnClasses.loadingIndicatorStart}`]: baseStyles.start,
      [`& .${btnClasses.loadingIndicatorEnd}`]: baseStyles.end,
    };
  };

  // Component customization
  const buttonOverrides = {
    MuiLoadingButton: {
      styleOverrides: {
        root: (params: { ownerState: LoadingButtonProps }) => {
          const { ownerState } = params;

          if (ownerState.variant !== 'soft') {
            return {};
          }

          return createPositionStyles(ownerState.size);
        },
      },
    },
  };

  return buttonOverrides;
}
