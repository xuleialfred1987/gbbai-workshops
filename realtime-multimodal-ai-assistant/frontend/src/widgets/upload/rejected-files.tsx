import React from 'react';
import { FileRejection } from 'react-dropzone';

// mui
import Chip from '@mui/material/Chip';
import List from '@mui/material/List';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import ListItem from '@mui/material/ListItem';
import AlertTitle from '@mui/material/AlertTitle';
import ListItemText from '@mui/material/ListItemText';

// project imports
import { fData } from 'src/utils/number-formatter';

import Iconify from 'src/widgets/iconify';

import { fileData } from '../media-preview';

// ----------------------------------------------------------------------

interface RejectedFilesProps {
  fileRejections: FileRejection[];
}

const RejectedFiles: React.FC<RejectedFilesProps> = ({ fileRejections }) => {
  // Early return if no rejections
  if (fileRejections.length === 0) return null;

  return (
    <Alert
      severity="error"
      variant="outlined"
      icon={<Iconify icon="mdi:alert-circle-outline" />}
      sx={{
        mt: 3,
        borderStyle: 'dashed',
        backgroundColor: 'error.lighter',
      }}
    >
      <AlertTitle>Failed to upload the following files</AlertTitle>

      <List dense sx={{ pt: 1 }}>
        {fileRejections.map(({ file, errors }) => {
          const { path, size } = fileData(file);

          return (
            <ListItem key={path} divider sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
              <ListItemText
                primary={
                  <Stack direction="row" spacing={1} alignItems="center">
                    <span>{path}</span>
                    {size && <Chip size="small" label={fData(size)} />}
                  </Stack>
                }
              />

              <Stack spacing={0.5} sx={{ width: '100%', mt: 1 }}>
                {errors.map((error) => (
                  <Stack
                    key={error.code}
                    direction="row"
                    spacing={1}
                    sx={{ color: 'error.main', fontSize: '0.75rem' }}
                  >
                    <span>•</span>
                    <span>{error.message}</span>
                  </Stack>
                ))}
              </Stack>
            </ListItem>
          );
        })}
      </List>
    </Alert>
  );
};

export default RejectedFiles;
