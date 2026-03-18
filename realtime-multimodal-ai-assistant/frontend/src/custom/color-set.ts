import { alpha } from '@mui/material/styles';

// ----------------------------------------------------------------------

// Type definitions
export type ColorSchema = 'primary' | 'secondary' | 'info' | 'success' | 'warning' | 'error';

export type ColorType = ColorSchema | 'default';

// Module augmentation for Material UI
declare module '@mui/material/styles/createPalette' {
  interface TypeBackground {
    neutral: string;
    dark: string;
  }
  interface SimplePaletteColorOptions {
    lighter: string;
    darker: string;
  }
  interface PaletteColor {
    lighter: string;
    darker: string;
  }
}

// Color definitions
const neutralColors = {
  white: '#FFFFFF',
  black: '#000000',
  slate: {
    0: '#FFFFFF',
    100: '#F9FAFB',
    150: '#F8FAFB',
    200: '#F4F6F8',
    300: '#DFE3E8',
    400: '#C4CDD5',
    500: '#919EAB',
    600: '#637381',
    700: '#454F5B',
    800: '#212B36',
    900: '#161C24',
  },
};

const themeColors = {
  aqua: {
    lightest: '#CAFCF9',
    light: '#61E7F2',
    medium: '#00A7D6',
    dark: '#00619A',
    darkest: '#003266',
    contrast: '#FFFFFF',
  },
  purple: {
    lightest: '#EFD6FF',
    light: '#C684FF',
    medium: '#8E33FF',
    dark: '#5119B7',
    darkest: '#27097A',
    contrast: '#FFFFFF',
  },
  azure: {
    lightest: '#CCF5FF',
    light: '#66D0FF',
    medium: '#0094FF',
    dark: '#0055B7',
    darkest: '#002B7A',
    contrast: '#FFFFFF',
  },
  emerald: {
    lightest: '#D8FBDE',
    light: '#86E8AB',
    medium: '#36B37E',
    dark: '#1B806A',
    darkest: '#0A5554',
    contrast: '#FFFFFF',
  },
  amber: {
    lightest: '#FFF5CC',
    light: '#FFD666',
    medium: '#FFAB00',
    dark: '#B76E00',
    darkest: '#7A4100',
    contrast: neutralColors.slate[800],
  },
  crimson: {
    lightest: '#FFE9D5',
    light: '#FFAC82',
    medium: '#FF5630',
    dark: '#B71D18',
    darkest: '#7A0916',
    contrast: '#FFFFFF',
  },
};

// Visualization palette
const dataVisualization = {
  violetSeries: ['#826AF9', '#9E86FF', '#D0AEFF', '#F7D2FF'],
  blueSeries: ['#2D99FF', '#83CFFF', '#A5F3FF', '#CCFAFF'],
  greenSeries: ['#2CD9C5', '#60F1C8', '#A4F7CC', '#C0F2DC'],
  yellowSeries: ['#FFE700', '#FFEF5A', '#FFF7AE', '#FFF3D6'],
  redSeries: ['#FF6C40', '#FF8F6D', '#FFBD98', '#FFF2D4'],
};

// Interactive states
const interactionStates = {
  hover: alpha(neutralColors.slate[500], 0.08),
  selected: alpha(neutralColors.slate[500], 0.16),
  disabled: alpha(neutralColors.slate[500], 0.8),
  disabledBackground: alpha(neutralColors.slate[500], 0.24),
  focus: alpha(neutralColors.slate[500], 0.24),
  hoverOpacity: 0.08,
  disabledOpacity: 0.48,
};

// Map to standard names
const standardPalette = {
  primary: {
    lighter: themeColors.aqua.lightest,
    light: themeColors.aqua.light,
    main: themeColors.aqua.medium,
    dark: themeColors.aqua.dark,
    darker: themeColors.aqua.darkest,
    contrastText: themeColors.aqua.contrast,
  },
  secondary: {
    lighter: themeColors.purple.lightest,
    light: themeColors.purple.light,
    main: themeColors.purple.medium,
    dark: themeColors.purple.dark,
    darker: themeColors.purple.darkest,
    contrastText: themeColors.purple.contrast,
  },
  info: {
    lighter: themeColors.azure.lightest,
    light: themeColors.azure.light,
    main: themeColors.azure.medium,
    dark: themeColors.azure.dark,
    darker: themeColors.azure.darkest,
    contrastText: themeColors.azure.contrast,
  },
  success: {
    lighter: themeColors.emerald.lightest,
    light: themeColors.emerald.light,
    main: themeColors.emerald.medium,
    dark: themeColors.emerald.dark,
    darker: themeColors.emerald.darkest,
    contrastText: themeColors.emerald.contrast,
  },
  warning: {
    lighter: themeColors.amber.lightest,
    light: themeColors.amber.light,
    main: themeColors.amber.medium,
    dark: themeColors.amber.dark,
    darker: themeColors.amber.darkest,
    contrastText: themeColors.amber.contrast,
  },
  error: {
    lighter: themeColors.crimson.lightest,
    light: themeColors.crimson.light,
    main: themeColors.crimson.medium,
    dark: themeColors.crimson.dark,
    darker: themeColors.crimson.darkest,
    contrastText: themeColors.crimson.contrast,
  },
  grey: neutralColors.slate,
  common: {
    black: neutralColors.black,
    white: neutralColors.white,
  },
  divider: alpha(neutralColors.slate[500], 0.2),
  action: interactionStates,
  chart: {
    violet: dataVisualization.violetSeries,
    blue: dataVisualization.blueSeries,
    green: dataVisualization.greenSeries,
    yellow: dataVisualization.yellowSeries,
    red: dataVisualization.redSeries,
  },
};

// Re-export for external use
export const { slate: grey } = neutralColors;
export const { primary, secondary, info, success, warning, error, common, chart, action } =
  standardPalette;

/**
 * Generates a complete color palette based on the provided mode
 * @param mode - The color mode ('light' or 'dark')
 * @returns The complete theme palette configuration
 */
export function palette(mode: 'light' | 'dark') {
  if (mode === 'light') {
    return {
      ...standardPalette,
      mode: 'light',
      text: {
        primary: grey[800],
        secondary: grey[600],
        disabled: grey[500],
      },
      background: {
        paper: '#FFFFFF',
        default: grey[150],
        neutral: grey[200],
        dark: alpha(grey[300], 0.34),
      },
      action: {
        ...standardPalette.action,
        active: grey[600],
      },
    };
  }

  return {
    ...standardPalette,
    mode: 'dark',
    text: {
      primary: '#FFFFFF',
      secondary: grey[500],
      disabled: grey[600],
    },
    background: {
      paper: grey[800],
      default: grey[900],
      neutral: alpha(grey[500], 0.12),
      dark: alpha(grey[800], 0.54),
    },
    action: {
      ...standardPalette.action,
      active: grey[500],
    },
  };
}
