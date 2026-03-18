// mui
import { useTheme } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';

// project imports
import { bgBlur } from 'src/custom/css';

import Iconify from '../iconify';

// ----------------------------------------------------------------------

type SaveButtonProps = {
  onDownload?: VoidFunction;
};

export default function SaveButton({ onDownload }: SaveButtonProps) {
  const theme = useTheme();

  return (
    <IconButton
      onClick={onDownload}
      sx={{
        padding: 0,
        position: 'absolute',
        top: 0,
        right: 0,
        width: '100%',
        height: '100%',
        zIndex: 9,
        opacity: 0,
        borderRadius: 0,
        justifyContent: 'center',
        backgroundColor: theme.palette.grey[800],
        color: theme.palette.common.white,
        transition: theme.transitions.create('opacity'),

        '&:hover': {
          opacity: 1,
          ...bgBlur({
            opacity: 0.64,
            color: theme.palette.grey[900],
          }),
        },
      }}
    >
      <Iconify icon="eva:arrow-circle-down-fill" width={24} />
    </IconButton>
  );
}
