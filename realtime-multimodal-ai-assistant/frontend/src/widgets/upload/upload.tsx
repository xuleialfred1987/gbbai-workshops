import { useDropzone } from 'react-dropzone';

// mui
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import { alpha, styled } from '@mui/material/styles';

// project imports
import { UploadImage } from 'src/media';

import Iconify from '../iconify';
import { UploadProps } from './types';
import RejectedFiles from './rejected-files';
import MultiFilePreview from './multi-file-preview';
import SingleFilePreview from './single-file-preview';

// ----------------------------------------------------------------------

const DropZoneContainer = styled(Box, {
  shouldForwardProp: (prop) =>
    prop !== 'hasError' && prop !== 'hasFile' && prop !== 'isActive' && prop !== 'isDisabled',
})<{ hasError?: boolean; hasFile?: boolean; isActive?: boolean; isDisabled?: boolean }>(
  ({ theme, hasError, hasFile, isActive, isDisabled }) => ({
    padding: 40,
    outline: 'none',
    borderRadius: 8,
    cursor: 'pointer',
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: alpha(theme.palette.grey[500], 0.08),
    border: `1px dashed ${alpha(theme.palette.grey[500], 0.2)}`,
    transition: theme.transitions.create(['opacity', 'padding']),
    '&:hover': {
      opacity: 0.72,
    },
    ...(isActive && {
      opacity: 0.72,
    }),
    ...(isDisabled && {
      opacity: 0.48,
      pointerEvents: 'none',
    }),
    ...(hasError && {
      color: theme.palette.error.main,
      borderColor: theme.palette.error.main,
      backgroundColor: alpha(theme.palette.error.main, 0.08),
    }),
    ...(hasFile && {
      padding: '24% 0',
    }),
  })
);

const DeleteButton = styled(IconButton)(({ theme }) => ({
  top: 16,
  right: 16,
  zIndex: 9,
  position: 'absolute',
  color: alpha(theme.palette.common.white, 0.8),
  backgroundColor: alpha(theme.palette.grey[900], 0.72),
  '&:hover': {
    backgroundColor: alpha(theme.palette.grey[900], 0.48),
  },
}));

export default function Upload({
  disabled,
  multiple = false,
  error,
  helperText,
  file,
  onDelete,
  files,
  thumbnail,
  onUpload,
  onRemove,
  onRemoveAll,
  sx,
  ...other
}: UploadProps) {
  const { getRootProps, getInputProps, isDragActive, isDragReject, fileRejections } = useDropzone({
    multiple,
    disabled,
    ...other,
  });

  const hasFile = !!file && !multiple;
  const hasFiles = !!files && multiple && !!files.length;
  const hasError = isDragReject || !!error;

  const renderDropzoneContent = () => {
    if (hasFile) {
      return <SingleFilePreview imgUrl={typeof file === 'string' ? file : file?.preview} />;
    }

    return (
      <Stack spacing={3} alignItems="center" justifyContent="center" flexWrap="wrap">
        <UploadImage sx={{ width: 1, maxWidth: 200 }} />
        <Stack spacing={1} sx={{ textAlign: 'center' }}>
          <Typography variant="h6">Drop or Select file</Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Drop files here or click
            <Box
              component="span"
              sx={{ mx: 0.5, color: 'primary.main', textDecoration: 'underline' }}
            >
              browse
            </Box>
            thorough your machine
          </Typography>
        </Stack>
      </Stack>
    );
  };

  const renderFileManagementControls = () => {
    if (!hasFiles) return null;

    return (
      <>
        <Box sx={{ my: 3 }}>
          <MultiFilePreview files={files} thumbnail={thumbnail} onRemove={onRemove} />
        </Box>

        <Stack direction="row" justifyContent="flex-end" spacing={1.5}>
          {onRemoveAll && (
            <Button color="inherit" variant="outlined" size="small" onClick={onRemoveAll}>
              Remove All
            </Button>
          )}

          {onUpload && (
            <Button
              size="small"
              variant="contained"
              onClick={onUpload}
              startIcon={<Iconify icon="eva:cloud-upload-fill" />}
            >
              Upload
            </Button>
          )}
        </Stack>
      </>
    );
  };

  return (
    <Box sx={{ width: 1, position: 'relative', ...sx }}>
      <DropZoneContainer
        {...getRootProps()}
        hasError={hasError}
        hasFile={hasFile}
        isActive={isDragActive}
        isDisabled={disabled}
      >
        <input {...getInputProps()} />
        {renderDropzoneContent()}
      </DropZoneContainer>

      {hasFile && onDelete && (
        <DeleteButton size="small" onClick={onDelete}>
          <Iconify icon="mingcute:close-line" width={18} />
        </DeleteButton>
      )}

      {helperText && helperText}

      <RejectedFiles fileRejections={fileRejections} />

      {renderFileManagementControls()}
    </Box>
  );
}
