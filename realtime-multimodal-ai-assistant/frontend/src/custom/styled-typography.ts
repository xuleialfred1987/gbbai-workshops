// ----------------------------------------------------------------------

// Helper functions for font size conversion
const convertRemToPx = (remValue: string): number => Math.round(parseFloat(remValue) * 16);
const convertPxToRem = (pxValue: number): string => `${pxValue / 16}rem`;

// Primary font configurations
const FONTS = {
  primary: 'Public Sans, sans-serif',
  secondary: 'Barlow, sans-serif',
};

// Font weight definitions
const WEIGHTS = {
  regular: 400,
  medium: 500,
  semiBold: 600,
  bold: 700,
  extraBold: 800,
};

// Generate responsive font size configurations based on breakpoints
const createResponsiveSizes = (config: {
  small: number;
  medium: number;
  large: number;
}): Record<string, any> => ({
  '@media (min-width:600px)': {
    fontSize: convertPxToRem(config.small),
  },
  '@media (min-width:900px)': {
    fontSize: convertPxToRem(config.medium),
  },
  '@media (min-width:1200px)': {
    fontSize: convertPxToRem(config.large),
  },
});

// Type extension for Material UI typography
declare module '@mui/material/styles' {
  interface TypographyVariants {
    fontSecondaryFamily: React.CSSProperties['fontFamily'];
    fontWeightSemiBold: React.CSSProperties['fontWeight'];
  }
}

// Typography configuration object
export const typography = {
  fontFamily: FONTS.primary,
  fontSecondaryFamily: FONTS.secondary,
  fontWeightRegular: WEIGHTS.regular,
  fontWeightMedium: WEIGHTS.medium,
  fontWeightSemiBold: WEIGHTS.semiBold,
  fontWeightBold: WEIGHTS.bold,

  // Heading styles
  h1: {
    fontWeight: WEIGHTS.extraBold,
    lineHeight: 80 / 64,
    fontSize: convertPxToRem(40),
    ...createResponsiveSizes({ small: 52, medium: 58, large: 64 }),
  },
  h2: {
    fontWeight: WEIGHTS.extraBold,
    lineHeight: 64 / 48,
    fontSize: convertPxToRem(32),
    ...createResponsiveSizes({ small: 40, medium: 44, large: 48 }),
  },
  h3: {
    fontWeight: WEIGHTS.bold,
    lineHeight: 1.5,
    fontSize: convertPxToRem(24),
    ...createResponsiveSizes({ small: 26, medium: 30, large: 32 }),
  },
  h4: {
    fontWeight: WEIGHTS.bold,
    lineHeight: 1.5,
    fontSize: convertPxToRem(20),
    ...createResponsiveSizes({ small: 20, medium: 24, large: 24 }),
  },
  h5: {
    fontWeight: WEIGHTS.bold,
    lineHeight: 1.5,
    fontSize: convertPxToRem(18),
    ...createResponsiveSizes({ small: 19, medium: 20, large: 20 }),
  },
  h6: {
    fontWeight: WEIGHTS.bold,
    lineHeight: 28 / 18,
    fontSize: convertPxToRem(17),
    ...createResponsiveSizes({ small: 18, medium: 18, large: 18 }),
  },

  // Text styles
  subtitle1: {
    fontWeight: WEIGHTS.semiBold,
    lineHeight: 1.5,
    fontSize: convertPxToRem(16),
  },
  subtitle2: {
    fontWeight: WEIGHTS.semiBold,
    lineHeight: 22 / 14,
    fontSize: convertPxToRem(14),
  },
  body1: {
    lineHeight: 1.5,
    fontSize: convertPxToRem(16),
  },
  body2: {
    lineHeight: 22 / 14,
    fontSize: convertPxToRem(14),
  },
  caption: {
    lineHeight: 1.5,
    fontSize: convertPxToRem(12),
  },
  overline: {
    fontWeight: WEIGHTS.bold,
    lineHeight: 1.5,
    fontSize: convertPxToRem(12),
    textTransform: 'uppercase',
  },
  button: {
    fontWeight: WEIGHTS.bold,
    lineHeight: 24 / 14,
    fontSize: convertPxToRem(14),
    textTransform: 'unset',
  },
} as const;

// Export utility functions and font constants
export const remToPx = convertRemToPx;
export const pxToRem = convertPxToRem;
export const primaryFont = FONTS.primary;
export const secondaryFont = FONTS.secondary;
export const responsiveFontSizes = (params: { sm: number; md: number; lg: number }) =>
  createResponsiveSizes({ small: params.sm, medium: params.md, large: params.lg });
