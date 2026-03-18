import React from 'react';
import { Controller, useFormContext } from 'react-hook-form';

// mui
import { Box } from '@mui/material';
import FormHelperText from '@mui/material/FormHelperText';

// project imports
import { Upload, UploadBox, UploadProps, UploadAvatar, UploadAvatarRectangular } from '../upload';

// ----------------------------------------------------------------------

// Types
type FileUploadProps = Omit<UploadProps, 'file'> & {
  name: string;
  multiple?: boolean;
  rectangular?: boolean;
  helperText?: React.ReactNode;
};

// Component for avatar uploads
export const ProfileImageUploader = ({
  name,
  rectangular = false,
  ...restProps
}: FileUploadProps) => {
  const { control } = useFormContext();

  return (
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState }) => {
        const hasError = !!fieldState.error;

        return (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            {rectangular ? (
              <UploadAvatarRectangular error={hasError} file={field.value} {...restProps} />
            ) : (
              <UploadAvatar error={hasError} file={field.value} {...restProps} />
            )}

            {hasError && (
              <FormHelperText error sx={{ mt: 1, textAlign: 'center', width: '100%' }}>
                {fieldState.error?.message}
              </FormHelperText>
            )}
          </Box>
        );
      }}
    />
  );
};

// Component for boxed upload area
export const BoxUploader = ({ name, ...restProps }: FileUploadProps) => {
  const { control } = useFormContext();

  return (
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState }) => (
        <Box sx={{ width: '100%' }}>
          <UploadBox files={field.value} error={!!fieldState.error} {...restProps} />
        </Box>
      )}
    />
  );
};

// Main upload component with flexible options
export const MediaUploader = ({
  name,
  multiple = false,
  helperText,
  ...restProps
}: FileUploadProps) => {
  const { control } = useFormContext();

  return (
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState }) => {
        const hasError = !!fieldState.error;
        const displayMessage = hasError ? fieldState.error?.message : helperText;
        const acceptedTypes = { 'image/*': [] };

        const renderHelperText = () =>
          (hasError || helperText) && (
            <FormHelperText error={hasError} sx={{ mt: 0.5, mx: 0 }}>
              {displayMessage}
            </FormHelperText>
          );

        if (multiple) {
          return (
            <Box sx={{ width: '100%' }}>
              <Upload
                multiple
                accept={acceptedTypes}
                files={field.value}
                error={hasError}
                helperText={renderHelperText()}
                {...restProps}
              />
            </Box>
          );
        }

        return (
          <Box sx={{ width: '100%' }}>
            <Upload
              accept={acceptedTypes}
              file={field.value}
              error={hasError}
              helperText={renderHelperText()}
              {...restProps}
            />
          </Box>
        );
      }}
    />
  );
};

export const FormUpload = MediaUploader;
export const FormUploadBox = BoxUploader;
export const FormUploadAvatar = ProfileImageUploader;
