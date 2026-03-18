import { alpha } from '@mui/material/styles';

// project imports
import { grey, info, error, common, primary, success, warning, secondary } from './color-set';

// ----------------------------------------------------------------------

// Shadow configuration definition
type ShadowLevel = 'z1' | 'z2' | 'z3' | 'z4' | 'z8' | 'z12' | 'z16' | 'z20' | 'z24';
type ShadowVariant = 'primary' | 'secondary' | 'info' | 'success' | 'warning' | 'error';
type ShadowSpecial = 'card' | 'dialog' | 'dropdown';

// Extend Material-UI's Theme interface
declare module '@mui/material/styles' {
  interface Theme {
    customShadows: Record<ShadowLevel | ShadowVariant | ShadowSpecial, string>;
  }
  interface ThemeOptions {
    customShadows?: Record<ShadowLevel | ShadowVariant | ShadowSpecial, string>;
  }
}

export function customShadows(mode: 'light' | 'dark') {
  // Base shadow color depends on theme mode
  const baseColor = mode === 'light' ? grey[500] : common.black;
  const standardOpacity = 0.16;
  const baseTransparent = alpha(baseColor, standardOpacity);

  // Shadow definitions configuration
  const elevationShadows: Record<ShadowLevel, string> = {
    z1: generateShadow(1, 2, 0, baseTransparent),
    z2: generateShadow(2, 3, 0, baseTransparent),
    z3: generateShadow(2, 4, 0, baseTransparent),
    z4: generateShadow(4, 8, 0, baseTransparent),
    z8: generateShadow(8, 16, 0, baseTransparent),
    z12: generateShadow(12, 24, -4, baseTransparent),
    z16: generateShadow(16, 32, -4, baseTransparent),
    z20: generateShadow(20, 40, -4, baseTransparent),
    z24: generateShadow(24, 48, 0, baseTransparent),
  };

  // Component-specific shadows
  const componentShadows: Record<ShadowSpecial, string> = {
    card: `0 0 2px 0 ${alpha(baseColor, 0.2)}, 0 12px 24px -4px ${alpha(baseColor, 0.12)}`,
    dropdown: `0 0 2px 0 ${alpha(baseColor, 0.24)}, -20px 20px 40px -4px ${alpha(baseColor, 0.24)}`,
    dialog: `-40px 40px 80px -8px ${alpha(common.black, 0.24)}`,
  };

  // Color-specific shadows
  const colorShadows: Record<ShadowVariant, string> = {
    primary: generateColorShadow(primary.main),
    secondary: generateColorShadow(secondary.main),
    info: generateColorShadow(info.main),
    success: generateColorShadow(success.main),
    warning: generateColorShadow(warning.main),
    error: generateColorShadow(error.main),
  };

  // Combine all shadow definitions
  return {
    ...elevationShadows,
    ...componentShadows,
    ...colorShadows,
  };
}

/**
 * Helper function to generate elevation shadow strings
 */
function generateShadow(x: number, y: number, blur: number, color: string): string {
  return `0 ${x}px ${y}px ${blur}px ${color}`;
}

/**
 * Helper function to generate color-specific shadow strings
 */
function generateColorShadow(colorValue: string): string {
  return `0 8px 16px 0 ${alpha(colorValue, 0.24)}`;
}
