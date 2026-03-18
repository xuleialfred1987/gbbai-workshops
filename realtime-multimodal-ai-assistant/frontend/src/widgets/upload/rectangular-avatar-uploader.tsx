import { useDropzone } from 'react-dropzone';

// mui
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import { alpha, Theme, SxProps } from '@mui/material/styles';

// project imports
import Image from '../img-wrap';
import Iconify from '../iconify';
import { UploadProps } from './types';
import RejectedFiles from './rejected-files';

// ----------------------------------------------------------------------

// Extracted component for image preview
type ImagePreviewProps = {
  fileExists: boolean;
  imageSource?: string;
};

const ImagePreview = ({ fileExists, imageSource }: ImagePreviewProps) => {
  if (!fileExists) return null;

  return (
    <Image
      alt="avatar"
      src={imageSource}
      sx={{
        width: 1,
        height: 1,
        borderRadius: 1,
      }}
    />
  );
};

// Extracted component for placeholder
type PlaceholderProps = {
  showError: boolean;
  fileExists: boolean;
};

const PlaceholderComponent = ({ showError, fileExists }: PlaceholderProps) => (
  <Stack
    alignItems="center"
    justifyContent="center"
    spacing={1}
    className="upload-placeholder"
    sx={{
      top: 0,
      left: 0,
      width: 1,
      height: 1,
      zIndex: 9,
      borderRadius: 1,
      position: 'absolute',
      color: 'text.disabled',
      bgcolor: (theme: Theme) => alpha(theme.palette.grey[500], 0.08),
      transition: (theme: Theme) =>
        theme.transitions.create(['opacity'], {
          duration: theme.transitions.duration.shorter,
        }),
      '&:hover': { opacity: 0.72 },
      ...(showError && {
        color: 'error.main',
        bgcolor: (theme: Theme) => alpha(theme.palette.error.main, 0.08),
      }),
      ...(fileExists && {
        zIndex: 9,
        opacity: 0,
        color: 'common.white',
        bgcolor: (theme: Theme) => alpha(theme.palette.grey[900], 0.64),
      }),
    }}
  >
    <Iconify icon="fluent:camera-add-24-filled" width={20} />
  </Stack>
);

// Main component
export default function UploadAvatarRectangular({
  error,
  file,
  disabled,
  helperText,
  sx,
  ...other
}: UploadProps) {
  const { getRootProps, getInputProps, isDragActive, isDragReject, fileRejections } = useDropzone({
    multiple: false,
    disabled,
    accept: { 'image/*': [] },
    ...other,
  });

  const fileExists = Boolean(file);
  const showError = isDragReject || Boolean(error);
  const imageSource = typeof file === 'string' ? file : file?.preview;

  const containerStyle: SxProps<Theme> = {
    p: 0,
    m: 'auto',
    width: 144,
    height: 144,
    cursor: disabled ? 'default' : 'pointer',
    overflow: 'hidden',
    borderRadius: '50%',
    border: (theme: Theme) => `1px dashed ${alpha(theme.palette.grey[500], 0.2)}`,
    ...(isDragActive && { opacity: 0.72 }),
    ...(disabled && { opacity: 0.48, pointerEvents: 'none' }),
    ...(showError && { borderColor: 'error.main' }),
    ...(fileExists && {
      ...(showError && {
        bgcolor: (theme: Theme) => alpha(theme.palette.error.main, 0.08),
      }),
      '&:hover .upload-placeholder': { opacity: 1 },
      border: 'None',
    }),
    ...sx,
  };

  return (
    <>
      <Box {...getRootProps()} sx={containerStyle}>
        <input {...getInputProps()} />

        <Box
          sx={{
            width: 1,
            height: 1,
            overflow: 'hidden',
            borderRadius: 1,
            position: 'relative',
          }}
        >
          <ImagePreview fileExists={fileExists} imageSource={imageSource} />
          <PlaceholderComponent showError={showError} fileExists={fileExists} />
        </Box>
      </Box>

      {helperText && helperText}

      <RejectedFiles fileRejections={fileRejections} />
    </>
  );
}
