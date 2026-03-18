import { Controller, useFormContext } from 'react-hook-form';

// mui
import { TextField, TextFieldProps } from '@mui/material';

// ----------------------------------------------------------------------

interface EnhancedTextFieldProps extends Omit<TextFieldProps, 'name'> {
  name: string;
  transformValue?: (value: any) => any;
}

const FormTextField = ({
  name,
  helperText,
  type = 'text',
  transformValue,
  ...restProps
}: EnhancedTextFieldProps) => {
  const { control } = useFormContext();

  const handleValueChange =
    (onChange: (value: any) => void) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const rawValue = event.target.value;

      if (transformValue) {
        onChange(transformValue(rawValue));
        return;
      }

      switch (type) {
        case 'number':
          onChange(rawValue === '' ? 0 : Number(rawValue));
          break;
        default:
          onChange(rawValue);
      }
    };

  const formatDisplayValue = (value: any) => {
    if (type === 'number' && value === 0) {
      return '';
    }
    return value;
  };

  return (
    <Controller
      name={name}
      control={control}
      render={({ field: { value, onChange, ...fieldProps }, fieldState: { error } }) => (
        <TextField
          {...fieldProps}
          {...restProps}
          fullWidth
          type={type}
          value={formatDisplayValue(value)}
          onChange={handleValueChange(onChange)}
          error={Boolean(error)}
          helperText={error?.message || helperText}
        />
      )}
    />
  );
};

export default FormTextField;
