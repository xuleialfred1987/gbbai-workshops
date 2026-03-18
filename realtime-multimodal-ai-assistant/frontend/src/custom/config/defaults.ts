import { alpha } from '@mui/material/styles';

// ----------------------------------------------------------------------

type PresetType = 'default' | 'cyan';

const cyanPreset = {
  lighter: '#CAFCF9',
  light: '#61E7F2',
  main: '#00A7D6',
  dark: '#00619A',
  darker: '#003266',
  contrastText: '#FFFFFF',
};

export const presetOptions = [
  { name: 'default', value: cyanPreset.main },
  { name: 'cyan', value: cyanPreset.main },
];

export const getPrimary = (preset: PresetType) => {
  const presetMapping = {
    default: cyanPreset,
    cyan: cyanPreset,
  };
  return presetMapping[preset];
};

export const createPresets = (preset: PresetType) => {
  const primary = getPrimary(preset);
  return {
    palette: {
      primary,
    },
    customShadows: {
      primary: `0 8px 16px 0 ${alpha(primary.main, 0.24)}`,
    },
  };
};
