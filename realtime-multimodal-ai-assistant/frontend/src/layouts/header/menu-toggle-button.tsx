import { useTheme } from '@mui/material/styles';
import IconButton, { IconButtonProps } from '@mui/material/IconButton';

import { useResponsiveUI } from 'src/hooks/responsive-ui';

import { bgBlur } from 'src/custom/css';

import Iconify from 'src/widgets/iconify';
import { useSettingsContext } from 'src/widgets/settings';

import { MENU } from '../default-layout';

// ----------------------------------------------------------------------

export default function MenuToggleButton({ sx, ...other }: IconButtonProps) {
  const theme = useTheme();

  const settings = useSettingsContext();

  const lgUp = useResponsiveUI('up', 'lg');

  if (!lgUp) {
    return null;
  }

  return (
    <IconButton
      size="small"
      onClick={() =>
        settings.onUpdate('themeLayout', settings.themeLayout === 'vertical' ? 'mini' : 'vertical')
      }
      sx={{
        p: 0.5,
        top: 24,
        position: 'fixed',
        left: MENU.W_VERTICAL - 13,
        zIndex: theme.zIndex.appBar + 1,
        border: `dashed 1px ${theme.palette.divider}`,
        ...bgBlur({ opacity: 0.48, color: theme.palette.background.default }),
        '&:hover': {
          bgcolor: 'background.default',
        },
        ...sx,
      }}
      {...other}
    >
      <Iconify
        width={16}
        icon={
          settings.themeLayout === 'vertical'
            ? 'eva:arrow-ios-back-fill'
            : 'eva:arrow-ios-forward-fill'
        }
      />
    </IconButton>
  );
}
