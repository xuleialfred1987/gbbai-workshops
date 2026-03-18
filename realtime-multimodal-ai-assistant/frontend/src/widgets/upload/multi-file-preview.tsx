import { m, AnimatePresence } from 'framer-motion';

// mui
import Stack from '@mui/material/Stack';
import { alpha } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';
import ListItemText from '@mui/material/ListItemText';

// project imports
import { fData } from 'src/utils/number-formatter';

import Iconify from '../iconify';
import { UploadProps } from './types';
import { varDissolve } from '../motion';
import MediaPreview, { fileData } from '../media-preview';

// ----------------------------------------------------------------------

export default function MultiFilePreview({
  files,
  thumbnail,
  onRemove,
  onClick,
  imageView = true,
  sx,
}: UploadProps) {
  const renderThumbnailView = (file: any) => {
    const { key } = fileData(file);

    return (
      <Stack
        key={key}
        component={m.div}
        {...varDissolve().inUp}
        sx={{
          position: 'relative',
          display: 'inline-flex',
          justifyContent: 'center',
          alignItems: 'center',
          width: 80,
          height: 80,
          m: 0.5,
          overflow: 'hidden',
          borderRadius: 1.25,
          border: (theme) => `solid 1px ${alpha(theme.palette.grey[500], 0.16)}`,
          ...sx,
        }}
      >
        <MediaPreview
          file={file}
          tooltip
          imageView={imageView}
          sx={{ position: 'absolute' }}
          imgSx={{ position: 'absolute' }}
          onClick={onClick}
        />

        {onRemove && (
          <IconButton
            onClick={() => onRemove(file)}
            size="small"
            sx={{
              position: 'absolute',
              top: 4,
              right: 4,
              p: 0.5,
              color: 'common.white',
              bgcolor: (theme) => alpha(theme.palette.grey[900], 0.48),
              '&:hover': {
                bgcolor: (theme) => alpha(theme.palette.grey[900], 0.72),
              },
            }}
          >
            <Iconify icon="mingcute:close-line" width={14} />
          </IconButton>
        )}
      </Stack>
    );
  };

  const renderListView = (file: any) => {
    const { key, name = '', size = 0 } = fileData(file);
    const isStringFile = typeof file === 'string';

    return (
      <Stack
        key={key}
        component={m.div}
        {...varDissolve().inUp}
        direction="row"
        alignItems="center"
        spacing={2}
        sx={{
          px: 1.5,
          py: 1,
          my: 1,
          borderRadius: 1,
          border: (theme) => `solid 1px ${alpha(theme.palette.grey[500], 0.16)}`,
          ...sx,
        }}
      >
        <MediaPreview file={file} />

        <ListItemText
          primary={isStringFile ? file : name}
          secondary={isStringFile ? '' : fData(size)}
          secondaryTypographyProps={{
            component: 'span',
            typography: 'caption',
          }}
        />

        {onRemove && (
          <IconButton onClick={() => onRemove(file)} size="small">
            <Iconify icon="mingcute:close-line" width={16} />
          </IconButton>
        )}
      </Stack>
    );
  };

  return (
    <AnimatePresence initial={false}>
      {files?.map((file) => (thumbnail ? renderThumbnailView(file) : renderListView(file)))}
    </AnimatePresence>
  );
}
