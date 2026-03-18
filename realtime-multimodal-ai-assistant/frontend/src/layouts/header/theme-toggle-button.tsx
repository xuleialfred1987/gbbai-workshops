import { m } from 'framer-motion';
import { Icon } from '@iconify/react';
import sunFill from '@iconify/icons-eva/sun-fill';
import moonFill from '@iconify/icons-eva/moon-fill';

import { useTheme } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';

import { varHover } from 'src/widgets/motion';
import { useSettingsContext } from 'src/widgets/settings/context';

// ----------------------------------------------------------------------

export default function ThemeToggleButton() {
  const theme = useTheme();
  const settings = useSettingsContext();

  return (
    <IconButton
      component={m.button}
      whileTap="tap"
      whileHover="hover"
      variants={varHover(1.05)}
      onClick={() =>
        settings.themeMode === 'light'
          ? settings.onUpdate('themeMode', 'dark')
          : settings.onUpdate('themeMode', 'light')
      }
      sx={{
        padding: 0,
        width: 40,
        height: 40,
        color: theme.palette.grey[600],
      }}
    >
      <Icon icon={settings.themeMode === 'light' ? sunFill : moonFill} width={20} height={20} />
    </IconButton>
  );
}
