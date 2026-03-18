import { useDropzone } from 'react-dropzone';

// mui
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { alpha, Theme, SxProps } from '@mui/material/styles';

// project imports
import Image from '../img-wrap';
import Iconify from '../iconify';
import { UploadProps } from './types';
import RejectedFiles from './rejected-files';

// ----------------------------------------------------------------------

export default function UploadAvatar({
  error,
  file,
  disabled,
  helperText,
  sx,
  ...other
}: UploadProps) {
  // Dropzone configuration
  const dropzoneConfig = {
    multiple: false,
    disabled,
    accept: {
      'image/*': [],
    },
    ...other,
  };

  // Initialize dropzone
  const { getRootProps, getInputProps, isDragActive, isDragReject, fileRejections } =
    useDropzone(dropzoneConfig);

  // File state management
  const hasFile = !!file;
  const hasError = isDragReject || !!error;
  const imgUrl = typeof file === 'string' ? file : file?.preview;
  const uploadText = file ? 'Update photo' : 'Upload photo';

  // Component styles as SxProps
  const rootSx: SxProps<Theme> = {
    p: 1,
    m: 'auto',
    width: 144,
    height: 144,
    cursor: 'pointer',
    overflow: 'hidden',
    borderRadius: '50%',
    border: (theme) => `1px dashed ${alpha(theme.palette.grey[500], 0.2)}`,
    ...(isDragActive && {
      opacity: 0.72,
    }),
    ...(disabled && {
      opacity: 0.48,
      pointerEvents: 'none',
    }),
    ...(hasError && {
      borderColor: 'error.main',
    }),
    ...(hasFile && {
      ...(hasError && {
        bgcolor: (theme) => alpha(theme.palette.error.main, 0.08),
      }),
      '&:hover .upload-placeholder': {
        opacity: 1,
      },
    }),
    ...(sx || {}),
  };

  const containerSx: SxProps<Theme> = {
    width: 1,
    height: 1,
    overflow: 'hidden',
    borderRadius: '50%',
    position: 'relative',
  };

  const imageSx: SxProps<Theme> = {
    width: 1,
    height: 1,
    borderRadius: '50%',
  };

  const placeholderSx: SxProps<Theme> = {
    top: 0,
    left: 0,
    width: 1,
    height: 1,
    zIndex: 9,
    borderRadius: '50%',
    position: 'absolute',
    color: 'text.disabled',
    bgcolor: (theme) => alpha(theme.palette.grey[500], 0.08),
    transition: (theme) =>
      theme.transitions.create(['opacity'], {
        duration: theme.transitions.duration.shorter,
      }),
    '&:hover': {
      opacity: 0.72,
    },
    ...(hasError && {
      color: 'error.main',
      bgcolor: (theme) => alpha(theme.palette.error.main, 0.08),
    }),
    ...(hasFile && {
      zIndex: 9,
      opacity: 0,
      color: 'common.white',
      bgcolor: (theme) => alpha(theme.palette.grey[900], 0.64),
    }),
  };

  // Render component
  return (
    <>
      <Box {...getRootProps()} sx={rootSx}>
        <input {...getInputProps()} />

        <Box sx={containerSx}>
          {hasFile && <Image alt="avatar" src={imgUrl} sx={imageSx} />}

          <Stack
            alignItems="center"
            justifyContent="center"
            spacing={1}
            className="upload-placeholder"
            sx={placeholderSx}
          >
            <Iconify icon="fluent:camera-add-24-filled" width={32} />
            <Typography variant="caption">{uploadText}</Typography>
          </Stack>
        </Box>
      </Box>

      {helperText && helperText}

      <RejectedFiles fileRejections={fileRejections} />
    </>
  );
}
