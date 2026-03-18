import { useMemo } from 'react';

import { Tooltip, IconButton } from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';

import Iconify from 'src/widgets/iconify';

// ----------------------------------------------------------------------

type Props = {
  onToggleCameraPanel: () => void;
  isCameraPanelOpen: boolean;
  size?: 'small' | 'medium';
  width?: number;
  height?: number;
};

export default function CameraPanelButton({
  onToggleCameraPanel,
  isCameraPanelOpen,
  size = 'small',
  width = 36,
  height = 36,
}: Props) {
  const theme = useTheme();
  const isLight = theme.palette.mode === 'light';

  const { buttonColor, buttonBg, hoverBg } = useMemo(() => {
    if (isCameraPanelOpen) {
      return {
        buttonColor: theme.palette.success.contrastText,
        buttonBg: theme.palette.success.main,
        hoverBg: theme.palette.success.dark,
      };
    }

    const neutralColor = theme.palette.text.secondary;
    const neutralBg = 'transparent';
    const hover = alpha(theme.palette.primary.main, isLight ? 0.24 : 0.32);

    return {
      buttonColor: neutralColor,
      buttonBg: neutralBg,
      hoverBg: hover,
    };
  }, [isCameraPanelOpen, isLight, theme.palette]);

  return (
    <Tooltip
      title={isCameraPanelOpen ? 'Close camera panel' : 'Open camera panel'}
      arrow
      disableInteractive
    >
      <IconButton
        size={size}
        onClick={onToggleCameraPanel}
        sx={{
          width,
          height,
          bgcolor: buttonBg,
          color: buttonColor,
          '&:hover': {
            bgcolor: hoverBg,
          },
        }}
      >
        <Iconify icon="fluent:video-32-regular" width={20} height={20} />
      </IconButton>
    </Tooltip>
  );
}
