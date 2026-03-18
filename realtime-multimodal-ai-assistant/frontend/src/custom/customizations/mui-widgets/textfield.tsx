import { Theme } from '@mui/material/styles';
import { inputBaseClasses } from '@mui/material/InputBase';
import { inputLabelClasses } from '@mui/material/InputLabel';
import { filledInputClasses } from '@mui/material/FilledInput';
import { outlinedInputClasses } from '@mui/material/OutlinedInput';

// ----------------------------------------------------------------------

export function textField(theme: Theme) {
  // Typography settings
  const typography = {
    fieldValue: theme.typography.body2,
    fieldLabel: theme.typography.body1,
  };

  // Color palette
  const palette = {
    textFocused: theme.palette.text.primary,
    textActive: theme.palette.text.secondary,
    textPlaceholder: theme.palette.text.disabled,
    errorMain: theme.palette.error.main,
    actionDisabled: theme.palette.action.disabledBackground,
    borderNormal: theme.palette.grey[500],
  };

  // Helper function to convert hex color to RGB values for rgba
  const hexToRgb = (hex: string, alpha: number) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // Create theme overrides for text fields
  const createTextFieldOverrides = () => {
    // Helper text styles
    const helperTextStyles = {
      root: {
        marginTop: theme.spacing(1),
      },
    };

    // Label styles
    const labelStyles = {
      root: {
        ...typography.fieldValue,
        color: palette.textPlaceholder,
        [`&.${inputLabelClasses.shrink}`]: {
          ...typography.fieldLabel,
          fontWeight: 600,
          color: palette.textActive,
          [`&.${inputLabelClasses.focused}`]: {
            color: palette.textFocused,
          },
          [`&.${inputLabelClasses.error}`]: {
            color: palette.errorMain,
          },
          [`&.${inputLabelClasses.disabled}`]: {
            color: palette.textPlaceholder,
          },
          [`&.${inputLabelClasses.filled}`]: {
            transform: 'translate(12px, 6px) scale(0.75)',
          },
        },
      },
    };

    // Base input styles
    const baseInputStyles = {
      root: {
        [`&.${inputBaseClasses.disabled}`]: {
          '& svg': {
            color: palette.textPlaceholder,
          },
        },
      },
      input: {
        ...typography.fieldValue,
        '&::placeholder': {
          opacity: 1,
          color: palette.textPlaceholder,
        },
      },
    };

    // Standard input styles
    const standardInputStyles = {
      underline: {
        '&:before': {
          borderBottomColor: hexToRgb(palette.borderNormal, 0.32),
        },
        '&:after': {
          borderBottomColor: palette.textFocused,
        },
      },
    };

    // Outlined input styles
    const outlinedInputStyles = {
      root: {
        [`&.${outlinedInputClasses.focused}`]: {
          [`& .${outlinedInputClasses.notchedOutline}`]: {
            borderColor: palette.textFocused,
          },
        },
        [`&.${outlinedInputClasses.error}`]: {
          [`& .${outlinedInputClasses.notchedOutline}`]: {
            borderColor: palette.errorMain,
          },
        },
        [`&.${outlinedInputClasses.disabled}`]: {
          [`& .${outlinedInputClasses.notchedOutline}`]: {
            borderColor: palette.actionDisabled,
          },
        },
      },
      notchedOutline: {
        borderColor: hexToRgb(palette.borderNormal, 0.2),
        transition: theme.transitions.create(['border-color'], {
          duration: theme.transitions.duration.shortest,
        }),
      },
    };

    // Filled input styles
    const filledInputStyles = {
      root: {
        borderRadius: theme.shape.borderRadius,
        backgroundColor: hexToRgb(palette.borderNormal, 0.08),
        '&:hover': {
          backgroundColor: hexToRgb(palette.borderNormal, 0.16),
        },
        [`&.${filledInputClasses.focused}`]: {
          backgroundColor: hexToRgb(palette.borderNormal, 0.16),
        },
        [`&.${filledInputClasses.error}`]: {
          backgroundColor: hexToRgb(palette.errorMain, 0.08),
          [`&.${filledInputClasses.focused}`]: {
            backgroundColor: hexToRgb(palette.errorMain, 0.16),
          },
        },
        [`&.${filledInputClasses.disabled}`]: {
          backgroundColor: palette.actionDisabled,
        },
      },
    };

    return {
      MuiFormHelperText: { styleOverrides: helperTextStyles },
      MuiFormLabel: { styleOverrides: labelStyles },
      MuiInputBase: { styleOverrides: baseInputStyles },
      MuiInput: { styleOverrides: standardInputStyles },
      MuiOutlinedInput: { styleOverrides: outlinedInputStyles },
      MuiFilledInput: { styleOverrides: filledInputStyles },
    };
  };

  return createTextFieldOverrides();
}
