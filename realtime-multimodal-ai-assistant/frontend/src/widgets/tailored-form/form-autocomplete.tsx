import React from 'react';
import { Controller, useFormContext } from 'react-hook-form';

// mui
import { TextField, Autocomplete } from '@mui/material';
import type { AutocompleteProps } from '@mui/material/Autocomplete';

// ----------------------------------------------------------------------

// Define the component types and interfaces
type FormAutocompleteProps<
  ItemType,
  IsMultiple extends boolean | undefined,
  IsClearableDisabled extends boolean | undefined,
  IsFreeSolo extends boolean | undefined,
> = Omit<
  AutocompleteProps<ItemType, IsMultiple, IsClearableDisabled, IsFreeSolo> & {
    name: string;
    label?: string;
    placeholder?: string;
    helperText?: React.ReactNode;
  },
  'renderInput'
>;

/**
 * Enhanced Autocomplete component that integrates with React Hook Form
 */
function FormAutocomplete<
  ItemType,
  IsMultiple extends boolean | undefined,
  IsClearableDisabled extends boolean | undefined,
  IsFreeSolo extends boolean | undefined,
>(props: FormAutocompleteProps<ItemType, IsMultiple, IsClearableDisabled, IsFreeSolo>) {
  // Extract specific props
  const { name, label, placeholder, helperText, ...autocompleteProps } = props;

  // Access form methods
  const formMethods = useFormContext();
  const { control, setValue } = formMethods;

  // Handle value changes
  const handleChange = (_: any, newValue: any) => {
    setValue(name, newValue, { shouldValidate: true });
  };

  // Render text field configuration
  const configureTextField = (params: any, errorState: any) => (
    <TextField
      {...params}
      label={label}
      placeholder={placeholder}
      error={Boolean(errorState)}
      helperText={errorState ? errorState.message : helperText}
    />
  );

  return (
    <Controller
      control={control}
      name={name}
      render={({ field, fieldState }) => {
        const { value, ref, ...restFieldProps } = field;
        const { error } = fieldState;

        return (
          <Autocomplete
            {...restFieldProps}
            value={value}
            ref={ref}
            onChange={handleChange}
            renderInput={(params) => configureTextField(params, error)}
            {...autocompleteProps}
          />
        );
      }}
    />
  );
}

export default FormAutocomplete;
