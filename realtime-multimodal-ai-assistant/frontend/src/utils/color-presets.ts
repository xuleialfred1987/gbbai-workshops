// theme
import { palette } from 'src/custom/color-set';

// ----------------------------------------------------------------------

export const modeColorSets = {
  'open-chat': 'linear-gradient(135deg, #3DA770 0%, #1F80B3 74%)',
  rag: 'linear-gradient(135deg, #B37F40 0%, #AA2E9B 74%)',
  'function-calling': 'linear-gradient(135deg, #3E88C6 0%, #8636C3 74%)',
} as any;

export const sendBtnColorSets = {
  'open-chat': '#1F80B3',
  rag: '#B37F40',
  'function-calling': '#8636C3',
} as any;

// ----------------------------------------------------------------------

const defaultColors = {
  name: 'default',
  ...palette('light').primary,
};

const purpleColors = {
  name: 'purple',
  lighter: '#EBD6FD',
  light: '#B985F4',
  main: '#7635dc',
  dark: '#431A9E',
  darker: '#200A69',
  contrastText: '#fff',
};

const cyanColors = {
  name: 'cyan',
  lighter: '#D1FFFC',
  light: '#76F2FF',
  main: '#1CCAFF',
  dark: '#0E77B7',
  darker: '#053D7A',
  contrastText: palette('light').grey[800],
};

const blueColors = {
  name: 'blue',
  lighter: '#D1E9FC',
  light: '#76B0F1',
  main: '#427BCD',
  dark: '#103996',
  darker: '#061B64',
  contrastText: '#fff',
};

const orangeColors = {
  name: 'orange',
  lighter: '#FEF4D4',
  light: '#FED680',
  main: '#fda92d',
  dark: '#B66816',
  darker: '#793908',
  contrastText: palette('light').grey[800],
};

const redColors = {
  name: 'red',
  lighter: '#FFE3D5',
  light: '#FFC1AC',
  main: '#FF3030',
  dark: '#B71833',
  darker: '#7A0930',
  contrastText: '#fff',
};

export const colorPresets = [
  defaultColors,
  purpleColors,
  cyanColors,
  blueColors,
  orangeColors,
  redColors,
];

export const defaultPreset = colorPresets[0];
export const purplePreset = colorPresets[1];
export const cyanPreset = colorPresets[2];
export const bluePreset = colorPresets[3];
export const orangePreset = colorPresets[4];
export const redPreset = colorPresets[5];

export const colorsForPiePlotting = [
  palette('light').chart.green[0],
  palette('light').chart.blue[0],
  palette('light').warning.main,
  defaultPreset.main,
  purplePreset.main,
  cyanPreset.main,
  bluePreset.main,
  orangePreset.main,
];

export const colorsForLogPlotting = [
  {
    main: palette('light').primary.main,
    lighter: palette('light').primary.lighter,
    dark: palette('light').primary.dark,
  },
  {
    main: '#00AB55',
    lighter: '#C8FACD',
    dark: '#007B55',
  },
  {
    main: palette('light').warning.main,
    lighter: palette('light').warning.lighter,
    dark: palette('light').warning.dark,
  },
  {
    main: '#1CCAFF',
    lighter: '#D1FFFC',
    dark: '#0E77B7',
  },
  {
    main: '#EBD6FD',
    lighter: '#C8FACD',
    dark: '#431A9E',
  },
  {
    main: palette('light').secondary.main,
    lighter: palette('light').secondary.lighter,
    dark: palette('light').secondary.dark,
  },
  {
    main: palette('light').success.main,
    lighter: palette('light').success.lighter,
    dark: palette('light').success.dark,
  },
  {
    main: palette('light').warning.main,
    lighter: palette('light').warning.lighter,
    dark: palette('light').warning.dark,
  },
  {
    main: palette('light').error.main,
    lighter: palette('light').error.lighter,
    dark: palette('light').error.dark,
  },
  {
    main: palette('light').info.main,
    lighter: palette('light').info.lighter,
    dark: palette('light').info.dark,
  },
  {
    main: palette('light').chart.violet[2],
    lighter: palette('light').chart.violet[0],
    dark: palette('light').chart.violet[3],
  },
  {
    main: palette('light').chart.blue[2],
    lighter: palette('light').chart.blue[0],
    dark: palette('light').chart.blue[3],
  },
  {
    main: palette('light').chart.green[2],
    lighter: palette('light').chart.green[0],
    dark: palette('light').chart.green[3],
  },
  {
    main: palette('light').chart.yellow[2],
    lighter: palette('light').chart.yellow[0],
    dark: palette('light').chart.yellow[3],
  },
  {
    main: palette('light').chart.red[2],
    lighter: palette('light').chart.red[0],
    dark: palette('light').chart.red[3],
  },
];
