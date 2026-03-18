import { useDropzone } from 'react-dropzone';
import roundAddPhotoAlternate from '@iconify/icons-ic/round-add-photo-alternate';

// mui
import Box from '@mui/material/Box';
import { alpha } from '@mui/material/styles';

// project imports
import Iconify from '../iconify';
import { UploadProps } from './types';

// ----------------------------------------------------------------------

export default function UploadIconButton({
  placeholder,
  error,
  disabled,
  sx,
  ...other
}: UploadProps) {
  const { getRootProps, getInputProps, isDragReject } = useDropzone({
    disabled,
    ...other,
  });

  const hasError = isDragReject || error;

  return (
    <Box
      {...getRootProps()}
      sx={{
        // m: 0.5,
        width: 30,
        height: 30,
        flexShrink: 0,
        display: 'flex',
        borderRadius: '50%',
        cursor: 'pointer',
        alignItems: 'center',
        color: 'text.secondary',
        justifyContent: 'center',
        // bgcolor: (theme) => alpha(theme.palette.grey[500], 0.08),
        // border: (theme) => `dashed 1px ${alpha(theme.palette.grey[500], 0.16)}`,
        // ...(isDragActive && {
        //   opacity: 0.72,
        // }),
        ...(disabled && {
          opacity: 0.48,
          pointerEvents: 'none',
        }),
        ...(hasError && {
          color: 'error.main',
          borderColor: 'error.main',
          bgcolor: (theme) => alpha(theme.palette.error.main, 0.08),
        }),
        '&:hover': {
          bgcolor: (theme) => alpha(theme.palette.grey[500], 0.18),
          // opacity: 0.72,
        },
        ...sx,
      }}
    >
      <input {...getInputProps()} />

      {placeholder || <Iconify icon={roundAddPhotoAlternate} width={22} />}
    </Box>
  );
}
