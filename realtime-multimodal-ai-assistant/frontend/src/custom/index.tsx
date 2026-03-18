import { useMemo } from 'react';
import merge from 'lodash/merge';

// mui
import CssBaseline from '@mui/material/CssBaseline';
import { createTheme, ThemeOptions, ThemeProvider as MuiThemeProvider } from '@mui/material/styles';

// project imports
import { useSettingsContext } from 'src/widgets/settings';

import { shadows } from './shadows';
import { palette } from './color-set';
import { typography } from './styled-typography';
import { customShadows } from './styled-shadows';
import { createPresets } from './config/defaults';
import { componentsOverrides } from './customizations';

// ----------------------------------------------------------------------

type Props = {
  children: React.ReactNode;
};

export default function ThemeCustomization({ children }: Props) {
  const settings = useSettingsContext();
  const presets = createPresets(settings.themeColorPresets);

  const themeConfig = useMemo(() => {
    const combinedPalette = {
      ...palette(settings.themeMode),
      ...presets.palette,
    };

    const combinedCustomShadows = {
      ...customShadows(settings.themeMode),
      ...presets.customShadows,
    };

    return {
      palette: combinedPalette,
      customShadows: combinedCustomShadows,
      shadows: shadows(settings.themeMode),
      shape: { borderRadius: 8 },
      typography,
    };
  }, [settings.themeMode, presets.palette, presets.customShadows]);

  const muiTheme = createTheme(themeConfig as ThemeOptions);
  muiTheme.components = merge({}, componentsOverrides(muiTheme));

  return (
    <MuiThemeProvider theme={muiTheme}>
      <CssBaseline />
      {children}
    </MuiThemeProvider>
  );
}
