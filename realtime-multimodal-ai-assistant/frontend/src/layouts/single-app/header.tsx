import { useMemo } from 'react';

// mui
import { useTheme } from '@mui/material/styles';
import { Stack, AppBar, Toolbar } from '@mui/material';

// project imports
import { useOffSetTop } from 'src/hooks/off-set-top';
import { useResponsiveUI } from 'src/hooks/responsive-ui';

import { bgBlur } from 'src/custom/css';

import { useSettingsContext } from 'src/widgets/settings';

import { HEADER } from '../default-layout';
import FullscreenButton from '../header/fullscreen-button';
import ThemeToggleButton from '../header/theme-toggle-button';
import StretchToggleButton from '../header/stretch-toggle-button';

// ----------------------------------------------------------------------

interface HeaderProps {
  onOpenMenu?: () => void;
}

export default function Header({ onOpenMenu }: HeaderProps) {
  const theme = useTheme();
  const settings = useSettingsContext();
  const lgUp = useResponsiveUI('up', 'lg');
  const offset = useOffSetTop(HEADER.H_DESKTOP);

  const layoutConfig = useMemo(() => {
    const isHorizontal = settings.themeLayout === 'horizontal';
    const isMini = settings.themeLayout === 'mini';
    const offsetActive = offset && !isHorizontal;

    return {
      isHorizontal,
      isMini,
      offsetActive,
    };
  }, [settings.themeLayout, offset]);

  const appBarStyles = useMemo(
    () => ({
      width: 1,
      height: HEADER.H_MOBILE,
      zIndex: theme.zIndex.appBar + 1,
      ...bgBlur({
        color: theme.palette.background.default,
      }),
      transition: theme.transitions.create(['height'], {
        duration: theme.transitions.duration.shorter,
      }),
      ...(lgUp && {
        height: HEADER.H_DESKTOP,
        ...(layoutConfig.offsetActive && {
          height: HEADER.H_DESKTOP_OFFSET,
        }),
      }),
    }),
    [theme, lgUp, layoutConfig]
  );

  return (
    <AppBar sx={appBarStyles}>
      <Toolbar sx={{ height: 1, px: { lg: 3 } }}>
        <Stack
          flexGrow={1}
          direction="row"
          alignItems="center"
          justifyContent="flex-end"
        >
          <StretchToggleButton />
          <ThemeToggleButton />
          <FullscreenButton />
        </Stack>
      </Toolbar>
    </AppBar>
  );
}
