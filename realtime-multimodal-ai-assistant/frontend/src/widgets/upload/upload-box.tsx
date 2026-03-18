import { useDropzone } from 'react-dropzone';

// mui
import Box from '@mui/material/Box';
import { alpha, Theme, SxProps, useTheme } from '@mui/material/styles';

// project imports
import Iconify from '../iconify';
import { UploadProps } from './types';

// ----------------------------------------------------------------------

/**
 * UploadBox component for file uploads with drag-and-drop functionality
 */
export default function UploadBox({ placeholder, error, disabled, sx, ...other }: UploadProps) {
  const theme = useTheme();

  const dropzoneConfig = {
    disabled,
    ...other,
  };

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone(dropzoneConfig);

  // Determine if there's an error state
  const isErrorState = error || isDragReject;

  // Style configurations
  const baseStyles: SxProps<Theme> = {
    margin: 0.5,
    width: 64,
    height: 64,
    flexShrink: 0,
    display: 'flex',
    borderRadius: 1,
    alignItems: 'center',
    justifyContent: 'center',
    cursor: disabled ? 'default' : 'pointer',
    color: theme.palette.text.disabled,
    backgroundColor: alpha(theme.palette.grey[500], 0.08),
    border: `dashed 1px ${alpha(theme.palette.grey[500], 0.16)}`,
    transition: 'all 0.2s ease-in-out',
  };

  const activeStyles: SxProps<Theme> = isDragActive
    ? {
        opacity: 0.72,
      }
    : {};

  const disabledStyles: SxProps<Theme> = disabled
    ? {
        opacity: 0.48,
        pointerEvents: 'none',
      }
    : {};

  const errorStyles: SxProps<Theme> = isErrorState
    ? {
        color: theme.palette.error.main,
        borderColor: theme.palette.error.main,
        backgroundColor: alpha(theme.palette.error.main, 0.08),
      }
    : {};

  const hoverStyles: SxProps<Theme> = {
    '&:hover': {
      opacity: 0.72,
    },
  };

  // Combine all styles
  const combinedStyles: SxProps<Theme> = {
    ...baseStyles,
    ...activeStyles,
    ...disabledStyles,
    ...errorStyles,
    ...hoverStyles,
    ...(sx || {}),
  };

  const renderContent = () => {
    if (placeholder) {
      return placeholder;
    }
    return <Iconify icon="eva:cloud-upload-fill" width={28} />;
  };

  return (
    <Box {...getRootProps()} sx={combinedStyles}>
      <input {...getInputProps()} />
      {renderContent()}
    </Box>
  );
}
