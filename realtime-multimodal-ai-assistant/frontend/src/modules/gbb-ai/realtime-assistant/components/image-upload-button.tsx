import { useCallback } from 'react';

import { Tooltip, IconButton } from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';

import Iconify from 'src/widgets/iconify';

// ----------------------------------------------------------------------

type Props = {
  onSelectOption: () => void | Promise<void>;
  size?: 'small' | 'medium';
  width?: number;
  height?: number;
};

export default function ImageUploadButton({
  onSelectOption,
  size = 'small',
  width = 36,
  height = 36,
}: Props) {
  const theme = useTheme();
  const isLight = theme.palette.mode === 'light';

  const handleButtonClick = useCallback(async () => {
    try {
      await onSelectOption();
    } catch (error) {
      console.error('Image selection failed', error);
    }
  }, [onSelectOption]);

  const buttonBg = 'transparent';
  const buttonColor = theme.palette.text.secondary;
  const hoverBg = alpha(theme.palette.primary.main, isLight ? 0.24 : 0.32);

  return (
    <Tooltip title="Add images" arrow disableInteractive>
      <IconButton
        size={size}
        onClick={handleButtonClick}
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
        <Iconify icon="grommet-icons:attachment" width={16} height={16} color="currentColor" />
      </IconButton>
    </Tooltip>
  );
}
