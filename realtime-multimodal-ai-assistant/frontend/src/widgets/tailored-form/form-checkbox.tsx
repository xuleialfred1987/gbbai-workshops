import React from 'react';
import { Controller, useFormContext } from 'react-hook-form';

// mui
import {
  Theme,
  SxProps,
  Checkbox,
  FormGroup,
  FormLabel,
  FormControl,
  FormHelperText,
  FormControlLabel,
  FormControlLabelProps,
  formControlLabelClasses,
} from '@mui/material';

// ----------------------------------------------------------------------

// ========== Helper Components ==========
interface HelperTextProps {
  error?: any;
  helperText?: React.ReactNode;
}

const FieldHelperText = ({ error, helperText }: HelperTextProps) =>
  !error && !helperText ? null : (
    <FormHelperText error={Boolean(error)} sx={{ mx: 0 }}>
      {error?.message || helperText}
    </FormHelperText>
  );

// ========== Single Checkbox Component ==========
interface CheckboxFieldProps extends Omit<FormControlLabelProps, 'control'> {
  name: string;
  helperText?: React.ReactNode;
}

function CheckboxField({ name, helperText, ...labelProps }: CheckboxFieldProps) {
  const formMethods = useFormContext();

  if (!formMethods) {
    console.error('CheckboxField must be used within a FormProvider');
    return null;
  }

  const { control } = formMethods;

  return (
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState }) => (
        <div>
          <FormControlLabel
            {...labelProps}
            control={
              <Checkbox
                checked={Boolean(field.value)}
                onChange={(e) => field.onChange(e.target.checked)}
                onBlur={field.onBlur}
                name={field.name}
                inputRef={field.ref}
              />
            }
          />
          <FieldHelperText error={fieldState.error} helperText={helperText} />
        </div>
      )}
    />
  );
}

// ========== Multi Checkbox Component ==========
interface Option {
  label: string;
  value: any;
}

interface MultiCheckboxFieldProps extends Omit<FormControlLabelProps, 'control' | 'label'> {
  name: string;
  options: Option[];
  label?: string;
  row?: boolean;
  spacing?: number;
  helperText?: React.ReactNode;
  sx?: SxProps<Theme>;
}

function MultiCheckboxField({
  name,
  label,
  options,
  row = false,
  spacing,
  helperText,
  sx,
  ...checkboxProps
}: MultiCheckboxFieldProps) {
  const { control } = useFormContext();

  // Update array value by adding or removing the selected item
  const updateArrayValue = (currentArray: any[] | undefined, item: any): any[] => {
    const array = currentArray || [];
    return array.includes(item) ? array.filter((value) => value !== item) : [...array, item];
  };

  // Properly typed sx prop
  const formGroupSx: SxProps<Theme> = {
    ...(row ? { flexDirection: 'row' } : {}),
    [`& .${formControlLabelClasses.root}`]: {
      '&:not(:last-of-type)': { mb: spacing || 0 },
      ...(row
        ? {
            mr: 0,
            '&:not(:last-of-type)': { mr: spacing || 2 },
          }
        : {}),
    },
    ...(sx || {}),
  };

  return (
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState }) => (
        <FormControl component="fieldset" fullWidth>
          {label && (
            <FormLabel component="legend" sx={{ typography: 'body2' }}>
              {label}
            </FormLabel>
          )}

          <FormGroup sx={formGroupSx}>
            {options.map((option) => (
              <FormControlLabel
                key={option.value}
                control={
                  <Checkbox
                    checked={Array.isArray(field.value) && field.value.includes(option.value)}
                    onChange={() => {
                      const newValue = updateArrayValue(field.value || [], option.value);
                      field.onChange(newValue);
                    }}
                  />
                }
                label={option.label}
                {...checkboxProps}
              />
            ))}
          </FormGroup>

          <FieldHelperText error={fieldState.error} helperText={helperText} />
        </FormControl>
      )}
    />
  );
}

// Export with more descriptive names
export { CheckboxField as FormCheckbox, MultiCheckboxField as FormMultiCheckbox };
