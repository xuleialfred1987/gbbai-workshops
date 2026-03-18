import { Controller, useFormContext } from 'react-hook-form';

// mui
import {
  Box,
  Radio,
  RadioGroup,
  Typography,
  RadioGroupProps,
  FormControlLabel,
} from '@mui/material';

// ----------------------------------------------------------------------

interface RadioOption {
  value: any;
  label: string;
}

interface FormRadioGroupProps extends Omit<RadioGroupProps, 'name'> {
  name: string;
  options: RadioOption[];
  label?: string;
  spacing?: number;
  helperText?: React.ReactNode;
}

export default function FormRadioGroup({
  name,
  label,
  options = [],
  row = false,
  spacing,
  helperText,
  ...otherProps
}: FormRadioGroupProps) {
  // Access form methods from context
  const { control } = useFormContext();

  // Generate unique ID for accessibility
  const accessibilityId = label ? `radio-group-${name}-${label}` : undefined;

  return (
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState }) => {
        const { error } = fieldState;
        const hasError = !!error;

        return (
          <Box sx={{ width: '100%' }}>
            {label && (
              <Typography
                variant="body2"
                component="label"
                id={accessibilityId}
                sx={{ mb: 1, display: 'block' }}
              >
                {label}
              </Typography>
            )}

            <RadioGroup {...field} {...otherProps} row={row} aria-labelledby={accessibilityId}>
              {options.map((option) => (
                <FormControlLabel
                  key={option.value}
                  value={option.value}
                  label={option.label}
                  control={<Radio />}
                  sx={{
                    ...(row
                      ? { mr: 0, '&:not(:last-child)': { mr: spacing || 2 } }
                      : { '&:not(:last-child)': { mb: spacing || 0 } }),
                  }}
                />
              ))}
            </RadioGroup>

            {(hasError || helperText) && (
              <Typography
                variant="caption"
                color={hasError ? 'error' : 'text.secondary'}
                sx={{ mt: 0.5, display: 'block' }}
              >
                {hasError ? error.message : helperText}
              </Typography>
            )}
          </Box>
        );
      }}
    />
  );
}
