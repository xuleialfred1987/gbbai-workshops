import React from 'react';
import {
  Controller,
  useFormContext,
  ControllerFieldState,
  ControllerRenderProps,
} from 'react-hook-form';

// mui
import {
  Switch,
  FormHelperText,
  FormControlLabel,
  type FormControlLabelProps,
} from '@mui/material';

// ----------------------------------------------------------------------

type FormSwitchProps = {
  name: string;
  size?: 'small' | 'medium';
  helperText?: React.ReactNode;
} & Omit<FormControlLabelProps, 'control'>;

/**
 * React Hook Form integrated Switch component
 * Provides form control with validation support
 */
const FormSwitch: React.FC<FormSwitchProps> = (props) => {
  const { name, size = 'medium', helperText, ...restProps } = props;

  const formContext = useFormContext();

  if (!formContext) {
    console.error('FormSwitch must be used within a FormProvider');
    return null;
  }

  const { control } = formContext;

  const renderSwitch = ({
    field,
    fieldState,
  }: {
    field: ControllerRenderProps<any, string>;
    fieldState: ControllerFieldState;
  }) => {
    const { error } = fieldState;
    const showHelperText = Boolean(error || helperText);

    return (
      <div className="rhf-switch-wrapper">
        <FormControlLabel
          {...restProps}
          control={
            <Switch
              size={size}
              checked={Boolean(field.value)}
              onChange={field.onChange}
              onBlur={field.onBlur}
              name={field.name}
              ref={field.ref}
            />
          }
        />

        {showHelperText && (
          <FormHelperText error={Boolean(error)}>{error?.message || helperText}</FormHelperText>
        )}
      </div>
    );
  };

  return <Controller name={name} control={control} render={renderSwitch} />;
};

export default FormSwitch;
